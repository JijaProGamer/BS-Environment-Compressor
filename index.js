const tf = require("@tensorflow/tfjs-node")
const fs = require('fs');
const path = require("path");
const { createCanvas } = require('canvas');


let defaultImageList = fs.readdirSync(path.join(__dirname, "images")).map(v => {return {v: path.join(__dirname, "images", v), r: Math.random()}}).sort((a, b) => a.r - b.r).map(v => v.v);
let imagesList;

function ReadImageList(){
    imagesList = fs.readdirSync(path.join(__dirname, "images")).map(v => {return {v: path.join(__dirname, "images", v), r: Math.random()}}).sort((a, b) => a.r - b.r).map(v => v.v);
}

ReadImageList();

function emptyDirectory(directoryPath) {
    const files = fs.readdirSync(directoryPath);
    files.map(file => fs.unlinkSync(path.join(directoryPath, file)));
}

emptyDirectory(path.join(__dirname, `/tensorboard`));

const summaryWriter = tf.node.summaryFileWriter(path.join(__dirname, `/tensorboard`));
const VAEClass = require("./page/main/server.js");
const VAE = new VAEClass();

VAE.savePath = path.join(__dirname, "model")
VAE.LatentDims = 1024;
VAE.inputShape = [96, 224, 3]; 128,288
//VAE.inputShape = [64, 144, 3];
//VAE.beta = 1;
VAE.beta = {
    min: 1,
    max: 1,
    cycle: 2,
    mode: "cosine"
}
VAE.LR = [
    { epoch: 0  , LR_Autoencoder: 0.0005,  LR_Discriminator: 0.001   },
    { epoch: 25 , LR_Autoencoder: 0.00005,  LR_Discriminator: 0.0001   },
    { epoch: 100 , LR_Autoencoder: 0.00001, LR_Discriminator: 0.00001   }
];

let fullShape = [224, 448, 3];

const epochs = 1000;//VAE.beta.cycle * 20;
const batchSize = 16;

VAE.launchModel().then(async () => {
    if(fs.existsSync(path.join(VAE.savePath, "encoder/model.json"))){
        //await VAE.loadModel()
    }

    /*//await makeImage()
    await makeReconstructionImage()*/


    await train();

    //await makeImage()
    //await makeReconstructionImage()
});

let trainLossStep = 0;
let trainBetaStep = 0;
let trainLrStep = 0;

async function train(){
    let batches = Math.ceil(imagesList.length / batchSize);
    let destroyed = false;

    for (let epoch = VAE.epoch; epoch < epochs; epoch++) {
        ReadImageList();



        VAE.epoch = epoch;

        let epochLoss = 0;
        let datasetPass = dataset();

        for (let batch = 0; batch < batches; batch++) {
            const rawBatch = datasetPass.next().value; 
            const batchInput = tf.concat(rawBatch);
            tf.dispose(rawBatch);

            //const trainLoss = 0;
            const trainData = await VAE.train(batchInput, batch / batches);

            //let batchLoss = trainData.batchLoss.discriminator + trainData.batchLoss.reconstruction + trainData.batchLoss.adversarial + trainData.batchLoss.kl //+ trainData.batchLoss.tc;
            let batchLoss = trainData.batchLoss.reconstruction;
            epochLoss += Math.min(Math.max(0, batchLoss), 100000);

            console.log(trainData.batchLoss)
            if(trainData.batchLoss.discriminator + trainData.batchLoss.reconstruction + trainData.batchLoss.adversarial + trainData.batchLoss.kl + trainData.batchLoss.tc < 0.0001){
                destroyed = true;
                console.log("Training got destroyed.")
                break;
            }

            summaryWriter.scalar('lr_autoencoder', trainData.lr.LR_Autoencoder, trainLrStep);
            summaryWriter.scalar('lr_discriminator', trainData.lr.LR_Discriminator, trainLrStep);
            trainLrStep++;
            summaryWriter.scalar('beta', trainData.beta, trainBetaStep++);
            summaryWriter.scalar('loss', batchLoss, trainLossStep);
            summaryWriter.scalar('loss_discriminator', trainData.batchLoss.discriminator, trainLossStep);
            summaryWriter.scalar('loss_reconstruction', trainData.batchLoss.reconstruction, trainLossStep);
            summaryWriter.scalar('loss_adversarial', trainData.batchLoss.adversarial, trainLossStep);
            summaryWriter.scalar('loss_kl', trainData.batchLoss.kl, trainLossStep);
            summaryWriter.scalar('loss_tc', trainData.batchLoss.tc, trainLossStep);
            trainLossStep++

            tf.dispose(batchInput);

            console.log(`[Epoch ${epoch + 1}] Loss: ${epochLoss / (batch + 1)}, progress: ${(batch / batches).toFixed(2)}`);
        }

        if(destroyed){
            break;
        }

        await VAE.saveModel()
        await makeReconstructionImage()
    }
    //await makeReconstructionImage()
}

function* dataset() {
    for(let i = 0; i < Math.ceil(imagesList.length / batchSize); i++){
        const images = [];

        for(let j = 0; j < batchSize; j++){
            let index = (i * batchSize) + j;

            if(!fs.existsSync(imagesList[index])){
                break;
            }

            const img = fs.readFileSync(imagesList[index]);
    
            const decodedImg = tf.tensor(img, fullShape, 'int32');
            const pixels = processImage(decodedImg);

            tf.dispose([ decodedImg ]);
    
            images.push(pixels);
        }

        yield images;
    }
}

function processImage(img) {
    return tf.tidy(() => {
        return img.resizeBilinear([VAE.inputShape[0], VAE.inputShape[1]]).toFloat().div(255.0).expandDims();
    })
}

function loadSingleImage(path){
    return tf.tidy(() => {
        const img = fs.readFileSync(path);
        const decodedImg = tf.tensor(img, fullShape, 'int32');

        return decodedImg
            .resizeBilinear([VAE.inputShape[0], VAE.inputShape[1]])            
            .toFloat()
            .div(tf.scalar(255.0));
    })
}





const valuesToCheck = 25

async function saveImageAsPNG(ctx, x, y, imageTensor) {
    const imageData = new Uint8ClampedArray(await imageTensor.data());

    const imgData = ctx.createImageData(VAE.inputShape[1], VAE.inputShape[0]);
    
    for (let i = 0; i < imageData.length / 3; i += 1) {
        let index = i * 4;
        let colorIndex = i * 3;

        /*imgData.data[index] = imageData[i];
        imgData.data[index + 1] = imageData[i];
        imgData.data[index + 2] = imageData[i];*/
        imgData.data[index] = imageData[colorIndex];
        imgData.data[index + 1] = imageData[colorIndex + 1];
        imgData.data[index + 2] = imageData[colorIndex + 2];
        imgData.data[index + 3] = 255;
    }

    ctx.putImageData(imgData, (VAE.inputShape[1] + 4) * x, (VAE.inputShape[0] + 4) * y);
}

let valuesToReconstruct = 50;

async function saveImageAsPNG(ctx, x, y, imageTensor) {
    const imageData = new Uint8ClampedArray(await imageTensor.data());

    const imgData = ctx.createImageData(VAE.inputShape[1], VAE.inputShape[0]);
    
    for (let i = 0; i < imageData.length / 3; i += 1) {
        let index = i * 4;
        let colorIndex = i * 3;

        imgData.data[index] = imageData[colorIndex];
        imgData.data[index + 1] = imageData[colorIndex + 1];
        imgData.data[index + 2] = imageData[colorIndex + 2];
        imgData.data[index + 3] = 255;
    }

    ctx.putImageData(imgData, (VAE.inputShape[1] + 4) * x, (VAE.inputShape[0] + 4) * y);
}

async function makeReconstructionImage(){
    const canvas = createCanvas((VAE.inputShape[1] + 4) * valuesToReconstruct, (VAE.inputShape[0] + 4) * 2);
    const ctx = canvas.getContext('2d');

    for(let x = 0; x < valuesToReconstruct; x++){
        const image = loadSingleImage(defaultImageList[x]);

        const latent = await VAE.compressImage(image);
        const generatedImageTensor = await VAE.generateImage(latent);

        const imageMul = image.mul(255)
        const imageInt = imageMul.cast("int32")

        await saveImageAsPNG(ctx, x, 0, imageInt);
        await saveImageAsPNG(ctx, x, 1, generatedImageTensor);

        tf.dispose([image, imageMul, imageInt, latent, generatedImageTensor])
    }

    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync('reconstructed_image.png', buffer);
}

/*const valuesToCheck = 10;

async function saveImageAsPNG(ctx, t, y, imageTensor) {
    const imageData = new Uint8ClampedArray(await imageTensor.data());

    const imgData = ctx.createImageData(modelConfig.inputShape[0], modelConfig.inputShape[1]);
    
    for (let i = 0; i < imageData.length; i += 1) {
        let index = i * 4;

        imgData.data[index] = imageData[i];
        imgData.data[index + 1] = imageData[i];
        imgData.data[index + 2] = imageData[i];
        imgData.data[index + 3] = 255;
    }

    ctx.putImageData(imgData, (modelConfig.inputShape[0] + 4) * (t * valuesToCheck), (modelConfig.inputShape[1] + 4) * y);
}

async function makeImage(){
    const canvas = createCanvas((modelConfig.inputShape[0] + 4) * valuesToCheck, (modelConfig.inputShape[1] + 4) * modelConfig.latentDim);
    const ctx = canvas.getContext('2d');

    for(let y = 0; y < modelConfig.latentDim; y++){
        for(let t = 0; t <= 1; t += 1/valuesToCheck){
            const latent = tf.oneHot(tf.tensor1d([y], 'int32'), modelConfig.latentDim)
                        .asType("float32")
                        .mul((t * 2) - 1)
                        .reshape([modelConfig.latentDim])
                        .arraySync();

            const generatedImageTensor = vae.generateImage(latent);
            await saveImageAsPNG(ctx, t, y, generatedImageTensor);
        }
    }

    const buffer = canvas.toBuffer('image/png');
    fs.writeFileSync('generated_image.png', buffer);
}*/