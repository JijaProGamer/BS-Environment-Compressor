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

        return tf.mean(ssim_map)//.clipByValue(0, 1);
    })
}

function reparameterize(zMean, zLogVar) {
    const batch = zMean.shape[0];
    const dim = zMean.shape[1];

    const mean = 0;
    const std = 1.0;
    // sample epsilon = N(0, I)
    const epsilon = tf.randomNormal([batch, dim], mean, std);
    return zMean.add(zLogVar.mul(0.5).exp().mul(epsilon));
}

function conv2d({ filters, kernelSize, strides, opts = {} }) {
    const conv = tf.layers.conv2d({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.glorotNormal(), useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), ...opts })
    const leakyRelu = tf.layers.leakyReLU();

    const batchNorm = tf.layers.batchNormalization({ momentum: 0.9 });
    return [conv, leakyRelu, batchNorm];
}

function conv2dLN({ filters, kernelSize, strides, opts = {} }) {
    const conv = tf.layers.conv2d({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.glorotNormal(), useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), ...opts })
    const leakyRelu = tf.layers.leakyReLU();

    //const layerNorm = tf.layers.layerNormalization();
    //return [conv, leakyRelu, layerNorm];
    return [conv, leakyRelu];
}

function conv2dTranspose({ filters, kernelSize, strides }) {
    const conv = tf.layers.conv2dTranspose({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.glorotNormal(), useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) });
    const leakyRelu = tf.layers.leakyReLU();

    const batchNorm = tf.layers.batchNormalization({ momentum: 0.9 });
    return [conv, leakyRelu, batchNorm];
}

function relu(units, inputShape) {
    let dense = tf.layers.dense({ units, kernelInitializer: tf.initializers.glorotNormal(), useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), inputShape });
    const leakyRelu = tf.layers.leakyReLU();

    const batchNorm = tf.layers.batchNormalization({ momentum: 0.9 });
    return [dense, leakyRelu, batchNorm];

}

function reluLN(units, inputShape) {
    let dense = tf.layers.dense({ units, kernelInitializer: tf.initializers.glorotNormal(), useBias: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), inputShape });
    const leakyRelu = tf.layers.leakyReLU();

    //const layerNorm = tf.layers.layerNormalization();
    //return [dense, leakyRelu, layerNorm];
    return [dense, leakyRelu];
}

const pi = Math.PI;
const log2pi = Math.log(2.0 * pi);

function gaussianLogDensity(samples, mean, logSquaredScale) {
    const invSigma = tf.exp(tf.neg(logSquaredScale));
    const tmp = tf.sub(samples, mean);

    return tmp.square().mul(invSigma).add(logSquaredScale).add(log2pi).mul(-0.5)
}

class VAE {
    LR;
    beta;
    LatentDims;
    Lambda;
    CriticUpdates;
    inputShape;

    betaValue;
    lastLR;

    init() {
        [this.encoder, this.decoder, this.discriminator] = this.build();
    }

    build() {
        const kernelInitializer = tf.initializers.glorotNormal(); //tf.initializers.randomNormal({ mean: 0, stdDev: 0.03 })
        const kernelRegularizer = tf.regularizers.l2({ l2: 0.01 });

        const encoder = tf.sequential({
            layers: [
                tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, inputShape: this.inputShape }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),

                tf.layers.conv2d({ filters: 256, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),

                tf.layers.conv2d({ filters: 128, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),

                tf.layers.conv2d({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),

                tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),


                tf.layers.flatten(),


                tf.layers.dense({ units: 1024, useBias: false, kernelInitializer, kernelRegularizer }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),

                tf.layers.dense({ units: this.LatentDims * 2 })
            ],
            name: "encoder"
        });

        const resolutionDecline = 8;
        const resolutionDeclineFilters = 48;
        const resolutionDeclineShape = [Math.floor(this.inputShape[0] / resolutionDecline), Math.floor(this.inputShape[1] / resolutionDecline), resolutionDeclineFilters]
        const resolutionDeclineNeurons = resolutionDeclineShape[0] * resolutionDeclineShape[1] * resolutionDeclineShape[2];

        const decoder = tf.sequential({
            layers: [
                tf.layers.dense({ units: resolutionDeclineNeurons, inputShape: [this.LatentDims], useBias: false, kernelInitializer, kernelRegularizer }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),


                tf.layers.reshape({ targetShape: resolutionDeclineShape }),


                tf.layers.conv2dTranspose({ filters: 64, kernelSize: 3, strides: 2, padding: 'same', useBias: false, kernelInitializer, kernelRegularizer }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),

                tf.layers.conv2dTranspose({ filters: 256, kernelSize: 3, strides: 2, padding: 'same', useBias: false, kernelInitializer, kernelRegularizer }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),

                tf.layers.conv2dTranspose({ filters: 128, kernelSize: 3, strides: 2, padding: 'same', useBias: false, kernelInitializer, kernelRegularizer }),
                tf.layers.batchNormalization({ momentum: 0.95 }),
                tf.layers.leakyReLU(),


                tf.layers.conv2dTranspose({ filters: this.inputShape[2], kernelSize: 3, strides: 1, activation: "tanh", padding: 'same', kernelInitializer, kernelRegularizer }),
            ],
            name: "decoder"
        });

        /*const discriminator = tf.sequential({
            layers: [
                tf.layers.conv2d({ filters: 64, kernelSize: 3, strides: 2, inputShape: this.inputShape, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.leakyReLU(),
                //tf.layers.layerNormalization(),

                tf.layers.conv2d({ filters: 256, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.leakyReLU(),
                //tf.layers.layerNormalization(),

                tf.layers.conv2d({ filters: 128, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.leakyReLU(),
                //tf.layers.layerNormalization(),

                tf.layers.conv2d({ filters: 32, kernelSize: 3, strides: 2, padding: 'same', kernelInitializer, kernelRegularizer, useBias: false, }),
                tf.layers.leakyReLU(),
                //tf.layers.layerNormalization(),


                tf.layers.flatten(),


                tf.layers.dense({ units: 1024, useBias: false, kernelInitializer, kernelRegularizer }),
                tf.layers.leakyReLU(),
                //tf.layers.layerNormalization(),


                tf.layers.dense({ units: 1 })
            ],
            name: "discriminator"
        });

        return [encoder, decoder, discriminator];*/
        return [encoder, decoder, null];
    }



    loss(batch, zMean, zLogVar, z, fakeImages) {
        return tf.tidy(() => {
            // reconstruction

            const ssim_loss = tf.scalar(1).sub(ssim(batch, fakeImages, 2, 7, 1.5, 0.01, 0.03).sqrt());
            const mae = batch.sub(fakeImages).abs().mean();
            //let adversial = this.discriminator.apply(fakeImages, { training: true }).mean();

            const reconstruction = ssim_loss.mul(0.9).add(mae.mul(0.1))
                .mul(batch.shape[1] * batch.shape[2]);

            //adversial = adversial.mul(tf.scalar(1).sub(reconstruction));

            // kl

            let kl = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp()).sum(-1).mul(-0.5).mean().mul(this.betaValue / 2);

            // tc

            const logQzProb = gaussianLogDensity(
                z.expandDims(1), 
                zMean.expandDims(0), 
                zLogVar.expandDims(0)
            );
            
            const logQzProduct = logQzProb.logSumExp(1, false).sum(1, false)
            const logQz = logQzProb.sum(2, false).logSumExp(1, false);

            const tc = logQz.sub(logQzProduct).mean().mul(this.betaValue / 2);

            return { reconstruction, kl, tc, adversial: tf.tensor1d([1]) };
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
        const batchMul = batch.mul(2);
        const scaledBatch = batchMul.sub(1);
        
        
        if (this.beta.mode === 'cosine') {
            this.betaValue = this.cosineAnnealingSchedule(epoch, batchProgress);
        }

        let learningRate = this.getLR(epoch);
        if (this.lastLR !== JSON.stringify(learningRate)) {
            this.optimizer = tf.train.adam(learningRate.LR_Autoencoder);
            //this.discriminatorOptimizer = tf.train.rmsprop(learningRate.LR_Discriminator, undefined, 0);
            //this.discriminatorOptimizer = tf.train.adam(learningRate.LR_Discriminator);
            this.lastLR = JSON.stringify(learningRate);
        }

        let discriminatorLossGlobal, reconstructionLossGlobal, adversarialLossGlobal, klLossGlobal, tcLossGlobal;
        discriminatorLossGlobal = 0;

        // generator training

        await this.optimizer.minimize(() => {
            const zData = this.encoder.apply(scaledBatch, { training: true });
            const [zMean, zLogVar] = tf.split(zData, 2, 1)

            const z = reparameterize(zMean, zLogVar);
            const fakeImages = this.decoder.apply(z, { training: true });

            const losses = this.loss(scaledBatch, zMean, zLogVar, z, fakeImages);

            reconstructionLossGlobal = losses.reconstruction.dataSync()[0];
            klLossGlobal = losses.kl.dataSync()[0];
            tcLossGlobal = losses.tc.dataSync()[0];
            adversarialLossGlobal = losses.adversial.dataSync()[0];

            let elbo = losses.reconstruction.add(losses.kl);
            let loss = elbo.add(losses.tc);

            return loss;
        }, false, [...this.encoder.trainableWeights.map(w => w.val), ...this.decoder.trainableWeights.map(w => w.val)]);

        // discriminator training

        /*for (let i = 0; i < this.CriticUpdates; i++) {
            await this.discriminatorOptimizer.minimize(() => {
                const epsilon = tf.randomUniform([scaledBatch.shape[0], 1, 1, 1], -0.3, 0.3);
                const fakeImages = this.reconstructImages(scaledBatch, true);

                const fakeImagesMixed = tf.add(
                    tf.mul(epsilon, scaledBatch),
                    tf.mul(tf.sub(1, epsilon), fakeImages)
                ).clipByValue(0, 1);

                const gradientsFn = tf.grad(x => this.discriminator.apply(x, { training: true }))(fakeImagesMixed);
                const grads = tf.tensor(gradientsFn.arraySync())

                const gradNorms = grads.square().sum([1, 2]).sqrt();
                const gradientPenalty = gradNorms.sub(1).square().mean();
                //gradientPenaltyGlobal = gradientPenalty.dataSync()[0];

                const fakePred = this.discriminator.apply(fakeImages, { training: true });
                const realPred = this.discriminator.apply(scaledBatch, { training: true });

                const criticLoss = fakePred.mean().sub(realPred.mean()).add(gradientPenalty.mul(this.Lambda));
                discriminatorLossGlobal += criticLoss.dataSync()[0];

                return criticLoss;
            }, false, this.discriminator.trainableWeights.map(w => w.val))
        }

        discriminatorLossGlobal /= this.CriticUpdates;*/


        tf.dispose([ batch, scaledBatch, batchMul ]);

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
            const inputImage = image.expandDims(0).mul(2).sub(1);

            const zData = this.encoder.apply(inputImage, { training: false });
            const [zMean, zLogVar] = tf.split(zData, 2, 1)

            //const z = reparameterize(zMean, zLogVar);
            //return z.squeeze();
            return zMean.squeeze();
        });
    }

    generateImage(latentVector) {
        return tf.tidy(() => {
            const input = latentVector.expandDims(0);

            const generatedImage = this.decoder.apply(input, { training: false });
            return generatedImage.add(1).div(2).squeeze().mul(255).cast('int32');
        })
    }

    reconstructImages(images, training) {
        return tf.tidy(() => {
            const zData = this.encoder.apply(images, { training });
            const [zMean, zLogVar] = tf.split(zData, 2, 1)

            const z = reparameterize(zMean, zLogVar);
            const reconstructed = this.decoder.apply(z, { training });

            return reconstructed.add(1).div(2).clipByValue(0, 1);
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