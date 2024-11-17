const path = require("path");

const VAEClass = require("./server.js");
const VAE = new VAEClass();

VAE.savePath = path.join(__dirname, "../../model")
VAE.LatentDims = 1024;
VAE.inputShape = [96, 224, 3];

VAE.start().then(async () => {

});

