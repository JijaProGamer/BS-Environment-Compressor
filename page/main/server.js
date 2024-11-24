const playwright = require("playwright")
const express = require('express');
const path = require("path")
const uuid = require("uuid")
const multer = require("multer")
const { WebSocketServer } = require('ws');
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const port = 4664;

class VAEModel {
  LR            = 1e-3;
  LatentDims    = 0;
  Lambda        = 0;
  CriticUpdates = 0;
  inputShape    = 0;
  beta          = 20;
  epoch         = 0;

  savePath;

  saveModel() {
    fs.writeFileSync(path.join(this.savePath, "/epoch"), this.epoch.toString());
    return this.sendRequest("save", {}, true);
  }

  loadModel() {
    if(fs.existsSync(path.join(this.savePath, "/epoch"))){
      this.epoch = parseInt(fs.readFileSync(path.join(this.savePath, "/epoch")));
    }

    return this.sendRequest("load", {}, true);
  }

  async train(data, batchProgress) {
    let rawTrainResults = (await this.sendRequest("train", { data: {shape: data.shape, img: [...data.dataSync()]}, epoch: this.epoch, batchProgress}, true));
    return rawTrainResults.trainResults;
  }

  async compressImage(img) {
    let rawTensor = (await this.sendRequest("compressImage", { data: {shape: img.shape, img: [...img.dataSync()]} }, true)).compressionResult;
    return tf.tensor(rawTensor.data, rawTensor.shape);
  }

  async generateImage(latent) {
    let rawTensor = (await this.sendRequest("generateImage", { data: {shape: latent.shape, latent: [...latent.dataSync()]} }, true)).latentResult;
    return tf.tensor(rawTensor.data, rawTensor.shape);
  }

  quit(){
    this.browser.close()
  }

  launchModel() {
    return new Promise(async (resolve, reject) => {
      this.expressServer = express();

      this.webSocketServer = new WebSocketServer({
        port: port + 1,
        perMessageDeflate: false
      });

      this.expressServer.get('/', (req, res) => {
        res.sendFile(path.join(__dirname, '/index.html'));
      });

      this.expressServer.get('/index.js', (req, res) => {
        res.sendFile(path.join(__dirname, '/index.js'));
      });

      this.expressServer.get('/vae.js', (req, res) => {
        res.sendFile(path.join(__dirname, '../vae.js'));
      });

      /*this.expressServer.get('/epsilon', (req, res) => {
        if(!fs.existsSync(path.join(this.savePath, "/epsilon"))){
          fs.writeFileSync(path.join(this.savePath, "/epsilon"), this.epsilon.toPrecision(8).toString())
          return res.send(this.epsilon.toPrecision(8).toString());
        }
        
        res.send(this.epsilon.toPrecision(8).toString());
      });

      this.expressServer.post('/epsilon', (req, res) => {
        fs.writeFileSync(path.join(this.savePath, "/epsilon"), parseInt(req.query.epsilon).toPrecision(8).toString())
        res.sendStatus(200)
      });*/

      this.expressServer.get('/hyperparameters.json', (req, res) => {
        res.json({
          LR: this.LR,
          LatentDims: this.LatentDims,
          inputShape: this.inputShape,
          beta: this.beta,
          Lambda: this.Lambda,
          CriticUpdates: this.CriticUpdates,
        })
      });

      const modelSaver = (model) => {
        fs.mkdirSync(path.join(this.savePath, `/${model}`), { recursive: true });

        this.expressServer.get(`/${model}/:name.bin`, (req, res) => {
          const name = req.params.name;
          res.sendFile(path.join(this.savePath, `/${model}/${name}.bin`));
        });

        this.expressServer.get(`/${model}/model.json`, (req, res) => {
          res.sendFile(path.join(this.savePath, `/${model}/model.json`));
        });
  
        const storage = multer.diskStorage({
          destination: (req, file, cb) => {
            cb(null, path.join(this.savePath, `/${model}`));
          },
          filename: (req, file, cb) => {
            //if(file.originalname.includes(".bin")){
            //  file.originalname = file.split("model.").pop();
            //}
  
            cb(null, file.originalname);
          }
        });
  
        const upload = multer({ storage: storage });
        this.expressServer.post(`/${model}`, upload.any(), (req, res) => {
          if (!req.files) {
            return res.status(400).send('No files uploaded.');
          }
  
          res.send('Files uploaded successfully.');
        });
  
      }

      modelSaver("discriminator");
      modelSaver("decoder");
      modelSaver("encoder");

      this.expressServer.listen(port, () => {
        //console.log(`Environment GPU ML Model is running at http://localhost:${port}, and WS at ws://localhost:${port + 1}`);
      });

      this.webSocketServer.on('connection', (ws) => {
        this.websocket = ws;

        this.websocket.on('error', console.error);
        //this.warmup().then(resolve)
        resolve()
      });

      this.browser = await playwright.chromium.launch({
        headless: false,
        args: [
          "--no-sandbox",
          "--use-angle=default",
          //'--use-gl=egl'
        ]
      })

      this.page = await this.browser.newPage()
      await this.page.goto(`http://localhost:${port}/`)

      //await this.page.screenshot({path: "gputest.png", fullPage: true})
    })
  }

  sendRequest(type, data, awaitResponse) {
    return new Promise((resolve, reject) => {
      let messageId = uuid.v4();
      let begin = performance.now();
      const websocket = this.websocket;

      if (awaitResponse) {
        function onMessage(message) {
          message = JSON.parse(message)
          if (message.id == messageId) {
            websocket.off("message", onMessage)

            resolve({ ...message, duration: performance.now() - begin })
          }
        }

        websocket.on("message", onMessage);
      } else {
        resolve();
      }

      websocket.send(JSON.stringify({ type, id: messageId, ...data }));
    })
  }
}

module.exports = VAEModel;