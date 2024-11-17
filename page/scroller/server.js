const express = require('express');
const path = require("path")
const tf = require("@tensorflow/tfjs");
const fs = require("fs");

const port = 5555;
let fullShape = [224, 448, 3];

class VAEModel {
    LR = 1e-3;
    LatentDims = 0;
    inputShape = 0;
    beta = 20;
    epoch = 0;

    quit() {
        this.browser.close()
    }

    loadSingleImage(path) {
        return tf.tidy(() => {
            const img = fs.readFileSync(path);
            const decodedImg = tf.tensor(img, fullShape, 'int32');

            return decodedImg
                .resizeBilinear([this.inputShape[0], this.inputShape[1]])
                .toFloat()
                .div(tf.scalar(255.0));
        })
    }

    start() {
        return new Promise(async (resolve, reject) => {
            this.expressServer = express();

            this.expressServer.get('/', (req, res) => {
                res.sendFile(path.join(__dirname, '/index.html'));
            });

            this.expressServer.get('/index.js', (req, res) => {
                res.sendFile(path.join(__dirname, '/index.js'));
            });

            this.expressServer.get('/vae.js', (req, res) => {
                res.sendFile(path.join(__dirname, '../vae.js'));
            });

            this.expressServer.get('/images', (req, res) => {
                res.json(fs.readdirSync(path.join(__dirname, "../../images")));
            });

            this.expressServer.get('/image/:img', (req, res) => {
                let imgPath = path.join(__dirname, "../../images", req.params.img);
                let img = this.loadSingleImage(imgPath);

                res.json([...img.dataSync()]);

                img.dispose();
            });


            this.expressServer.get('/hyperparameters.json', (req, res) => {
                res.json({
                    LR: this.LR,
                    LatentDims: this.LatentDims,
                    inputShape: this.inputShape,
                    beta: this.beta,
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

            }

            modelSaver("decoder");
            modelSaver("encoder");

            this.expressServer.listen(port, () => {
                console.log(`Environment GPU ML Model is running at http://localhost:${port}`);
            });
        })
    }
}

module.exports = VAEModel;