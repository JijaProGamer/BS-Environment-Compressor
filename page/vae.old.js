function get1dGaussianKernel(sigma, size) {
    let x = tf.range(Math.floor(-size / 2) + 1, Math.floor(size / 2) + 1)
    x = tf.pow(x, 2)
    x = tf.exp(x.div(-2.0 * (sigma * sigma)))
    x = x.div(tf.sum(x))

    return x
}

function get2dGaussianKernel(size, sigma) {
    sigma = sigma || (0.3 * ((size - 1) * 0.5 - 1) + 0.8)

    let kerne1d = get1dGaussianKernel(sigma, size)
    return tf.outerProduct(kerne1d, kerne1d)
}

function getGaussianKernel(size, sigma) {
    return tf.tidy(() => {
        var kerne2d = get2dGaussianKernel(size, sigma)
        var kerne3d = tf.stack([kerne2d, kerne2d, kerne2d])
        return tf.reshape(kerne3d, [size, size, 3, 1])
    })
}

function ssim(img1, img2, L = 1.0, filterSize = 11, sigma = 1.5, K1 = 0.01, K2 = 0.03) {
    return tf.tidy(() => {
        const C1 = (K1 * L) ** 2;
        const C2 = (K2 * L) ** 2;

        const kernel = getGaussianKernel(filterSize, sigma);

        const mu1 = tf.depthwiseConv2d(img1, kernel, 1, 'same').squeeze();
        const mu2 = tf.depthwiseConv2d(img2, kernel, 1, 'same').squeeze();
        
        const mu1_sq = tf.mul(mu1, mu1);
        const mu2_sq = tf.mul(mu2, mu2);
        const mu1_mu2 = tf.mul(mu1, mu2);
        
        const sigma1_sq = tf.depthwiseConv2d(tf.mul(img1, img1), kernel, 1, 'same').squeeze().sub(mu1_sq);
        const sigma2_sq = tf.depthwiseConv2d(tf.mul(img2, img2), kernel, 1, 'same').squeeze().sub(mu2_sq);
        const sigma12 = tf.depthwiseConv2d(tf.mul(img1, img2), kernel, 1, 'same').squeeze().sub(mu1_mu2);
        
        const ssim_map_numerator = tf.mul(tf.add(tf.mul(2, mu1_mu2), C1), tf.add(tf.mul(2, sigma12), C2));
        const ssim_map_denominator = tf.mul(tf.add(mu1_sq, mu2_sq).add(C1), tf.add(sigma1_sq, sigma2_sq).add(C2));
        const ssim_map = tf.div(ssim_map_numerator, ssim_map_denominator);
        
        return tf.mean(ssim_map).clipByValue(0, 1);
    })
}

class SamplingLayer extends tf.layers.Layer {
    constructor() {
        super({});
    }

    computeOutputShape(inputShape) {
        return inputShape[0];
    }

    call(inputs) {
        return tf.tidy(() => {
            const [zMean, zLogVar] = inputs;
            const batch = zMean.shape[0];
            const dim = zMean.shape[1];
        
            const mean = 0;
            const std = 1.0;
            // sample epsilon = N(0, I)
            const epsilon = tf.randomNormal([batch, dim], mean, std);
            return zMean.add(zLogVar.mul(0.5).exp().mul(epsilon));
        });
    }

    getConfig() {
        return {};
    }

    static get className() {
        return 'SamplingLayer';
    }
}

tf.serialization.registerClass(SamplingLayer);

function conv2d(input, { filters, kernelSize, strides }) {
    const conv = tf.layers.conv2d({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.glorotNormal(), useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) }).apply(input)
    const leakyRelu = tf.layers.leakyReLU().apply(conv);

    const batchNorm = tf.layers.batchNormalization( { momentum: 0.9 } ).apply(leakyRelu);
    //const dropout = tf.layers.dropout({ rate: 0.2 }).apply(batchNorm)
    return batchNorm;
}

function conv2dTranspose(input, { filters, kernelSize, strides, activation = "relu" }) {
    const conv = tf.layers.conv2dTranspose({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.glorotNormal(), activation, useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) }).apply(input)
    const leakyRelu = tf.layers.leakyReLU().apply(conv);

    const batchNorm = tf.layers.batchNormalization( { momentum: 0.9 } ).apply(leakyRelu);
    //const dropout = tf.layers.dropout({ rate: 0.2 }).apply(batchNorm)
    return batchNorm;
}

function relu(input, units) {
    let dense = tf.layers.dense({ units, kernelInitializer: tf.initializers.glorotNormal(), useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) }).apply(input);
    const leakyRelu = tf.layers.leakyReLU().apply(dense);

    const batchNorm = tf.layers.batchNormalization( { momentum: 0.9 } ).apply(leakyRelu);
    //const dropout = tf.layers.dropout({ rate: 0.2 }).apply(batchNorm)
    return batchNorm;
}

const pi = tf.scalar(Math.PI);
const log2pi = tf.log(pi.mul(2));

function gaussianLogDensity(samples, mean, logSquaredScale) {
    const invSigma = tf.exp(tf.neg(logSquaredScale));
    const tmp = samples.sub(mean);

    return tmp.pow(2).mul(invSigma).mul(tf.scalar(-0.5))
           .add(logSquaredScale)
           .add(log2pi)
           .mul(tf.scalar(-1));
}

function logNormalPdf(sample, mean, logvar){
  return tf.sum(
    (sample.sub(mean)).pow(2).mul(-0.5)
  .mul(tf.exp(logvar.neg()).add(logvar).add(log2pi))
  , 1)
}

class VAE {
    LR;
    beta;
    LatentDims;
    inputShape;

    betaValue;
    lastLR;

    init() {
        [this.encoder, this.decoder, this.apply] = this.build();
    }

    build() {        
        const encoderInput = tf.input({ shape: this.inputShape });
        let x = conv2d(encoderInput, { filters: 32, kernelSize: 3, strides: 1 });
        x = conv2d(x, { filters: 64, kernelSize: 3, strides: 2 });
        x = conv2d(x, { filters: 128, kernelSize: 3, strides: 2 });
        x = conv2d(x, { filters: 256, kernelSize: 3, strides: 2 });
        x = conv2d(x, { filters: 64, kernelSize: 3, strides: 2 });
        x = tf.layers.flatten().apply(x);
        x = relu(x, 1024);

        const zMean = tf.layers.dense({ units: this.LatentDims }).apply(x);
        const zLogVar = tf.layers.dense({ units: this.LatentDims }).apply(x);
        const z = new SamplingLayer().apply([zMean, zLogVar]);

        const encoder = tf.model({ inputs: encoderInput, outputs: [zMean, zLogVar, z], name: 'encoder' });



        const decoderInput = tf.input({ shape: [this.LatentDims] });
        let y = relu(decoderInput, Math.floor(this.inputShape[0] /168) * Math.floor(this.inputShape[1] / 16) * 48);
        y = tf.layers.reshape({ targetShape: [Math.floor(this.inputShape[0] / 16), Math.floor(this.inputShape[1] / 16), 48] }).apply(y);
        y = conv2dTranspose(y, { filters: 256, kernelSize: 3, strides: 2 });
        y = conv2dTranspose(y, { filters: 128, kernelSize: 3, strides: 2 });
        y = conv2dTranspose(y, { filters: 64, kernelSize: 3, strides: 2 });
        y = conv2dTranspose(y, { filters: 32, kernelSize: 3, strides: 2 });
        const decoderOutput = conv2dTranspose(y, { filters: this.inputShape[2], kernelSize: 3, strides: 1, activation: 'sigmoid' });

        const decoder = tf.model({ inputs: decoderInput, outputs: decoderOutput, name: 'decoder' });




        /*const discriminatorInput = tf.input({ shape: this.inputShape });
        let j = conv2d(encoderInput, { filters: 32, kernelSize: 3, strides: 1 });
        j = conv2d(x, { filters: 128, kernelSize: 3, strides: 2 });
        j = conv2d(x, { filters: 256, kernelSize: 3, strides: 2 });
        j = tf.layers.flatten().apply(x);
        j = relu(x, 512);
        
        const discriminatorOutput = tf.layers.dense({ units: 1 }).apply(j);
        const discriminator = tf.model({ inputs: discriminatorInput, outputs: discriminatorOutput, name: 'discriminator' });*/



        const vaeModel = (inputs) => {
            return tf.tidy(() => {
                const [zMean, zLogVar, z] = this.encoder.apply(inputs);
                const outputs = this.decoder.apply(z);
                return [zMean, zLogVar, z, outputs];
            });
        };

        return [encoder, decoder, vaeModel];
    }



    loss(xTrue, yPred) {
        /*return tf.tidy(() => {
            const [zMean, zLogVar, z, xPred] = yPred;

            // reconstruction

            const mae = xTrue.sub(xPred).abs().mean();
            const ssim_loss = tf.scalar(1).sub(
                ssim(xTrue, xPred, 1, 11, 1.5, 0.01, 0.03)
            );

            /*const log_sigma_opt = mse.log().mul(0.5);
    
            const diff = xTrue.sub(xPred);
            const normalized = diff.div(log_sigma_opt.exp());
            const r_loss = normalized.square().mul(0.5).add(log_sigma_opt);

            return r_loss.sum();/

            const reconstruction = mae.mul(0.25).add( ssim_loss.mul(0.75) ).mul(xTrue.shape[1] * xTrue.shape[2]);

            // kl

            const kl = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp()).sum(-1).mul(-0.5).mean()
                .mul(this.betaValue)

            // tc

            let logQzProb = gaussianLogDensity(
                    z.expandDims(1),
                    zMean.expandDims(0),
                    zLogVar.expandDims(0)
                );

        
            const logQzProduct = logQzProb.sum(1).logSumExp(1);
            const logQz = logQzProb.sum(2).logSumExp(1);
        
            const tc = logQz.sub(logQzProduct).mean()
                .mul(0.01);

            return { reconstruction, kl, tc };
        });*/

        return tf.tidy(() => {
            const [zMean, zLogVar, z, xPred] = yPred;

            // reconstruction

            const mae = xTrue.sub(xPred).abs().mean();
            const ssim_loss = tf.scalar(1).sub(
                ssim(xTrue, xPred, 1, 11, 1.5, 0.01, 0.03)
            );

            /*const log_sigma_opt = mse.log().mul(0.5);
    
            const diff = xTrue.sub(xPred);
            const normalized = diff.div(log_sigma_opt.exp());
            const r_loss = normalized.square().mul(0.5).add(log_sigma_opt);

            return r_loss.sum();*/

            const reconstruction = mae.mul(0.25).add( ssim_loss.mul(0.75) ).mul(xTrue.shape[1] * xTrue.shape[2]);

            // kl

            const logpz = logNormalPdf(z, tf.scalar(0), tf.scalar(0))
            const logqz_x = logNormalPdf(z, zMean, zLogVar)
            const kl = (logpz.sub(logqz_x)).neg().mean();

            // tc

            const logQzProb = gaussianLogDensity(
                z.expandDims(1),
                zMean.expandDims(0),
                zLogVar.expandDims(0)
            );

        
            const logQzProduct = logQzProb.sum(1).logSumExp(1);
            const logQz = logQzProb.sum(2).logSumExp(1);
        
            const tc = logQz.sub(logQzProduct).mean()
                .mul(0.01);

            return { reconstruction, kl, tc };
        });
    }

    cosineAnnealingSchedule(epoch, batchProgress) {
        epoch += batchProgress;
    
        const cycle = Math.floor(epoch / this.beta.cycle);
        const progress = (epoch % this.beta.cycle) / this.beta.cycle;
        let scheduleValue;

        if (cycle % 2 === 0) {
            scheduleValue = 0.5 * (1 - Math.cos(Math.PI * progress));
        } else {
            scheduleValue = 0.5 * (1 - Math.cos(Math.PI * (1 - progress)));
        }
    
        return this.beta.min + (this.beta.max - this.beta.min) * scheduleValue;
    }

    getLR(epoch) {
        let closest = null;
    
        for (let i = 0; i < this.LR.length; i++) {
            if (this.LR[i].epoch <= epoch) {
                closest = this.LR[i];
            }
        }
        
        return closest;
    }

    async train(batch, epoch, batchProgress) {
        //if (this.beta.mode === 'cosine') {
        //    this.betaValue = this.cosineAnnealingSchedule(epoch, batchProgress);
        //}

        this.betaValue = 1;

        let learningRate = this.getLR(epoch);
        if(this.lastLR !== JSON.stringify(learningRate)){
            this.optimizer = tf.train.adam(learningRate.LR_Autoencoder);
            this.lastLR = JSON.stringify(learningRate);
        }

        let discriminatorLossGlobal, reconstructionLossGlobal, adversarialLossGlobal, klLossGlobal, tcLossGlobal;
        adversarialLossGlobal = 0;
        discriminatorLossGlobal = 0;

        await tf.tidy(() => {
            this.optimizer.minimize(() => {
                const fakeImages = this.apply(batch);
                const loss = this.loss(batch, fakeImages);

                reconstructionLossGlobal = loss.reconstruction.dataSync()[0];
                klLossGlobal = loss.kl.dataSync()[0];
                tcLossGlobal = loss.tc.dataSync()[0];

                return loss.reconstruction.add(loss.kl)//.add(loss.tc);
            }, false, [...this.encoder.trainableWeights.map(w => w.val), ...this.decoder.trainableWeights.map(w => w.val)]);
        })

        tf.dispose(batch);

        return { 
            batchLoss: {
                discriminator: discriminatorLossGlobal, 
                reconstruction: reconstructionLossGlobal,
                adversarial: adversarialLossGlobal,
                tc: tcLossGlobal,
                kl: klLossGlobal
            }, 
            beta: this.betaValue,
            lr: learningRate 
        };
    }

    compressImage(image) {
        return tf.tidy(() => {
            const inputImage = image.expandDims(0);

            const [zMean, zLogVar, z] = this.encoder.apply(inputImage);

            return z.squeeze();
        });
    }

    generateImage(latentVector) {
        return tf.tidy(() => {
            const input = latentVector.expandDims(0);
            const generatedImage = this.decoder.apply(input);

            return generatedImage.squeeze().mul(255).cast('int32');
        })
    }

    async saveModel() {
        await this.encoder.save(`http://localhost:${location.port}/encoder`);
        await this.decoder.save(`http://localhost:${location.port}/decoder`);
    }

    async loadModel() {
        this.encoder = await tf.loadLayersModel(`http://localhost:${location.port}/encoder/model.json`);
        this.decoder = await tf.loadLayersModel(`http://localhost:${location.port}/decoder/model.json`);
    }
}

/*class VAE {
    LR;
    beta;
    LatentDims;
    inputShape;

    betaValue;
    lastLR;

    init() {
        [this.encoder, this.decoder, this.apply] = this.build();
    }

    build() {        
        const encoderInput = tf.input({ shape: this.inputShape });
        let x = conv2d(encoderInput, { filters: 32, kernelSize: 3, strides: 1 });
        x = conv2d(x, { filters: 64, kernelSize: 3, strides: 2 });
        x = conv2d(x, { filters: 128, kernelSize: 3, strides: 2 });
        x = conv2d(x, { filters: 256, kernelSize: 3, strides: 2 });
        x = conv2d(x, { filters: 64, kernelSize: 3, strides: 2 });
        x = tf.layers.flatten().apply(x);
        x = relu(x, 1024);

        const zMean = tf.layers.dense({ units: this.LatentDims }).apply(x);
        const zLogVar = tf.layers.dense({ units: this.LatentDims }).apply(x);
        const z = new SamplingLayer().apply([zMean, zLogVar]);

        const encoder = tf.model({ inputs: encoderInput, outputs: [zMean, zLogVar, z], name: 'encoder' });



        const decoderInput = tf.input({ shape: [this.LatentDims] });
        let y = relu(decoderInput, Math.floor(this.inputShape[0] / 8) * Math.floor(this.inputShape[1] / 8) * 48);
        y = tf.layers.reshape({ targetShape: [Math.floor(this.inputShape[0] / 8), Math.floor(this.inputShape[1] / 8), 48] }).apply(y);
        y = conv2dTranspose(y, { filters: 256, kernelSize: 3, strides: 2 });
        y = conv2dTranspose(y, { filters: 128, kernelSize: 3, strides: 2 });
        y = conv2dTranspose(y, { filters: 64, kernelSize: 3, strides: 2 });
        const decoderOutput = conv2dTranspose(y, { filters: this.inputShape[2], kernelSize: 3, strides: 1, activation: 'sigmoid' });

        const decoder = tf.model({ inputs: decoderInput, outputs: decoderOutput, name: 'decoder' });




        /*const discriminatorInput = tf.input({ shape: this.inputShape });
        let j = conv2d(encoderInput, { filters: 32, kernelSize: 3, strides: 1 });
        j = conv2d(x, { filters: 128, kernelSize: 3, strides: 2 });
        j = conv2d(x, { filters: 256, kernelSize: 3, strides: 2 });
        j = tf.layers.flatten().apply(x);
        j = relu(x, 512);
        
        const discriminatorOutput = tf.layers.dense({ units: 1 }).apply(j);
        const discriminator = tf.model({ inputs: discriminatorInput, outputs: discriminatorOutput, name: 'discriminator' });*



        const vaeModel = (inputs) => {
            return tf.tidy(() => {
                const [zMean, zLogVar, z] = this.encoder.apply(inputs);
                const outputs = this.decoder.apply(z);
                return [zMean, zLogVar, z, outputs];
            });
        };

        return [encoder, decoder, vaeModel];
    }



    loss(xTrue, yPred) {
        /*return tf.tidy(() => {
            const [zMean, zLogVar, z, xPred] = yPred;

            // reconstruction

            const mae = xTrue.sub(xPred).abs().mean();
            const ssim_loss = tf.scalar(1).sub(
                ssim(xTrue, xPred, 1, 11, 1.5, 0.01, 0.03)
            );

            /*const log_sigma_opt = mse.log().mul(0.5);
    
            const diff = xTrue.sub(xPred);
            const normalized = diff.div(log_sigma_opt.exp());
            const r_loss = normalized.square().mul(0.5).add(log_sigma_opt);

            return r_loss.sum();/

            const reconstruction = mae.mul(0.25).add( ssim_loss.mul(0.75) ).mul(xTrue.shape[1] * xTrue.shape[2]);

            // kl

            const kl = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp()).sum(-1).mul(-0.5).mean()
                .mul(this.betaValue)

            // tc

            let logQzProb = gaussianLogDensity(
                    z.expandDims(1),
                    zMean.expandDims(0),
                    zLogVar.expandDims(0)
                );

        
            const logQzProduct = logQzProb.sum(1).logSumExp(1);
            const logQz = logQzProb.sum(2).logSumExp(1);
        
            const tc = logQz.sub(logQzProduct).mean()
                .mul(0.01);

            return { reconstruction, kl, tc };
        });*

        return tf.tidy(() => {
            const [zMean, zLogVar, z, xPred] = yPred;

            // reconstruction

            const mae = xTrue.sub(xPred).abs().mean();
            const ssim_loss = tf.scalar(1).sub(
                ssim(xTrue, xPred, 1, 11, 1.5, 0.01, 0.03)
            );

            /*const log_sigma_opt = mse.log().mul(0.5);
    
            const diff = xTrue.sub(xPred);
            const normalized = diff.div(log_sigma_opt.exp());
            const r_loss = normalized.square().mul(0.5).add(log_sigma_opt);

            return r_loss.sum();*

            const reconstruction = mae.mul(0.25).add( ssim_loss.mul(0.75) ).mul(xTrue.shape[1] * xTrue.shape[2]);

            // kl

            let logpz = logNormalPdf(z, tf.scalar(0), tf.scalar(0))
            let logqz_x = logNormalPdf(z, zMean, zLogVar)
            //return (reconstruction.add(logpz).sub(logqz_x)).neg().mean()

            const kl = (logpz.sub(logqz_x)).neg().mean();

            //const kl = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp()).sum(-1).mul(-0.5).mean()
            //    .mul(this.betaValue)

            // tc

            let logQzProb = gaussianLogDensity(
                    z.expandDims(1),
                    zMean.expandDims(0),
                    zLogVar.expandDims(0)
                );

        
            const logQzProduct = logQzProb.sum(1).logSumExp(1);
            const logQz = logQzProb.sum(2).logSumExp(1);
        
            const tc = logQz.sub(logQzProduct).mean()
                .mul(0.01);

            return { reconstruction, kl, tc };
        });
    }

    cosineAnnealingSchedule(epoch, batchProgress) {
        epoch += batchProgress;
    
        const cycle = Math.floor(epoch / this.beta.cycle);
        const progress = (epoch % this.beta.cycle) / this.beta.cycle;
        let scheduleValue;

        if (cycle % 2 === 0) {
            scheduleValue = 0.5 * (1 - Math.cos(Math.PI * progress));
        } else {
            scheduleValue = 0.5 * (1 - Math.cos(Math.PI * (1 - progress)));
        }
    
        return this.beta.min + (this.beta.max - this.beta.min) * scheduleValue;
    }

    getLR(epoch) {
        let closest = null;
    
        for (let i = 0; i < this.LR.length; i++) {
            if (this.LR[i].epoch <= epoch) {
                closest = this.LR[i];
            }
        }
        
        return closest;
    }

    async train(batch, epoch, batchProgress) {
        //if (this.beta.mode === 'cosine') {
        //    this.betaValue = this.cosineAnnealingSchedule(epoch, batchProgress);
        //}

        this.betaValue = 1;

        let learningRate = this.getLR(epoch);
        if(this.lastLR !== JSON.stringify(learningRate)){
            this.optimizer = tf.train.adam(learningRate.LR_Autoencoder);
            this.lastLR = JSON.stringify(learningRate);
        }

        let discriminatorLossGlobal, reconstructionLossGlobal, adversarialLossGlobal, klLossGlobal, tcLossGlobal;
        adversarialLossGlobal = 0;
        discriminatorLossGlobal = 0;

        await tf.tidy(() => {
            this.optimizer.minimize(() => {
                const fakeImages = this.apply(batch);
                const loss = this.loss(batch, fakeImages);

                /*const fakePred = this.discriminator.apply(fakeImages);
                const adversarialLoss = this.adversarialLoss(fakePred);

                adversarialLossGlobal = adversarialLoss.dataSync()[0];*
                reconstructionLossGlobal = loss.reconstruction.dataSync()[0];
                klLossGlobal = loss.kl.dataSync()[0];
                tcLossGlobal = loss.tc.dataSync()[0];

                return loss.reconstruction.add(loss.kl)//.add(loss.tc);
            }, false, [...this.encoder.trainableWeights.map(w => w.val), ...this.decoder.trainableWeights.map(w => w.val)]);

            /*this.discriminatorOptimizer.minimize(() => {
                const fakeImages = this.apply(batch);
                const fakePred = this.discriminator.apply(fakeImages);
                const realPred = this.discriminator.apply(batch);
    
                const realLoss = tf.losses.sigmoidCrossEntropy(tf.onesLike(realPred), realPred);
                const fakeLoss = tf.losses.sigmoidCrossEntropy(tf.zerosLike(fakePred), fakePred);
    
                let discriminatorLoss = realLoss.add(fakeLoss).div(2);
                discriminatorLossGlobal = discriminatorLoss.dataSync()[0];
    
                return discriminatorLoss;//.mul(batch.shape[1] * batch.shape[2]);           
            }, false, this.discriminator.trainableWeights.map(w => w.val));*
        })

        tf.dispose(batch);

        return { 
            batchLoss: {
                discriminator: discriminatorLossGlobal, 
                reconstruction: reconstructionLossGlobal,
                adversarial: adversarialLossGlobal,
                tc: tcLossGlobal,
                kl: klLossGlobal
            }, 
            beta: this.betaValue,
            lr: learningRate 
        };
    }

    compressImage(image) {
        return tf.tidy(() => {
            const inputImage = image.expandDims(0);

            const [zMean, zLogVar, z] = this.encoder.apply(inputImage);

            return z.squeeze();
        });
    }

    generateImage(latentVector) {
        return tf.tidy(() => {
            const input = latentVector.expandDims(0);
            const generatedImage = this.decoder.apply(input);

            return generatedImage.squeeze().mul(255).cast('int32');
        })
    }

    async saveModel() {
        await this.encoder.save(`http://localhost:${location.port}/encoder`);
        await this.decoder.save(`http://localhost:${location.port}/decoder`);
    }

    async loadModel() {
        this.encoder = await tf.loadLayersModel(`http://localhost:${location.port}/encoder/model.json`);
        this.decoder = await tf.loadLayersModel(`http://localhost:${location.port}/decoder/model.json`);
    }
}*/