let modelHyperparameters;

let model;
const socket = new WebSocket(`ws://localhost:${parseInt(location.port) + 1}`);

socket.addEventListener("open", (event) => {

});

socket.addEventListener("message", async (event) => {
    if (!model){
        await tf.setBackend("webgl");

        modelHyperparameters = await (await fetch("/hyperparameters.json")).json();
    
        model = new VAE();

        //model.epsilon = parseFloat(await (await fetch("/epsilon")).text());

        model.LR = modelHyperparameters.LR;
        model.LatentDims = modelHyperparameters.LatentDims;
        model.inputShape = modelHyperparameters.inputShape
        model.beta = modelHyperparameters.beta;

        model.init();
    }

    let data = JSON.parse(event.data);

    switch (data.type) {
        case "save":
            await model.saveModel();
            socket.send(JSON.stringify({ id: data.id }));
            break;
        case "load":
            await model.loadModel();
            socket.send(JSON.stringify({ id: data.id }));
            break;
        case "train":
            var imgTensor = tf.tensor(data.data.img, data.data.shape);

            let trainResults = await model.train(imgTensor, data.epoch, data.batchProgress);
            socket.send(JSON.stringify({ id: data.id, trainResults }));
            break;
        case "compressImage":
            var imgTensor = tf.tensor(data.data.img, data.data.shape);

            let compressionResult = await model.compressImage(imgTensor);
            socket.send(JSON.stringify({ id: data.id, compressionResult: {shape: compressionResult.shape, data: [...compressionResult.dataSync()]} }));

            tf.dispose([imgTensor, compressionResult]);
            break;
        case "generateImage":
            var latentVector = tf.tensor(data.data.latent, data.data.shape);

            let latentResult = await model.generateImage(latentVector);
            socket.send(JSON.stringify({ id: data.id, latentResult: {shape: latentResult.shape, data: [...latentResult.dataSync()]} }));

            tf.dispose([latentVector, latentResult]);
            break;
        /*case "updateTargetModel":
            model.updateTargetModel();
            break;
        case "updateEpsilon":
            model.updateEpsilon();
            break;*/
        case "act":
            let result = await model.calculateActions(data.state, data.ignoreEpsilon);
            socket.send(JSON.stringify({ id: data.id, predictions: result }));

            break;
    }
});