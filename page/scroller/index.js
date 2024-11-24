const slidersDiv = document.querySelector(".sliders")
const randomizer = document.querySelector(".randomizer")

const canvas = document.querySelector('canvas');
const ctx = canvas.getContext("2d")


canvas.width = window.screen.width;
canvas.height = window.screen.height;




let model;
let latentDimSliders = []
let latentDimValues = []

async function init() {
  await tf.setBackend("webgl");

  const modelHyperparameters = await (await fetch("/hyperparameters.json")).json();

  model = new VAE();

  model.LR = modelHyperparameters.LR;
  model.LatentDims = modelHyperparameters.LatentDims;
  model.inputShape = modelHyperparameters.inputShape
  model.beta = modelHyperparameters.beta;

  model.init();
  await model.loadModel();

  makeSliders();
}

function sampleGaussian(mean = 0, stdDev = 1) {
  const u1 = Math.random();
  const u2 = Math.random();

  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

randomizer.addEventListener("click", async function() {
  for (let i = 0; i < model.LatentDims; i++) {
    latentDimValues[i] = sampleGaussian(0, 1);
  }

  setSlidersValues()
  await renderLatent()
})

function makeSliders() {
  for (let i = 0; i < model.LatentDims; i++) {
    latentDimValues.push(0)


    const slide = document.createElement('div');
    slide.className = 'slide';

    const rangeInput = document.createElement('input');
    rangeInput.type = 'range';
    rangeInput.min = -3;
    rangeInput.max = 3;
    rangeInput.step = 0.01;

    const valueDisplay = document.createElement('div');
    valueDisplay.className = 'value_display';
    valueDisplay.innerText = `#${i}: `;

    rangeInput.addEventListener('input', async function () {
      latentDimValues[i] = rangeInput.value;
      await renderLatent()
    });

    slide.appendChild(valueDisplay);
    slide.appendChild(rangeInput);

    slidersDiv.appendChild(slide);


    latentDimSliders.push(rangeInput)
  }
}

function setSlidersValues() {
  for (let i = 0; i < model.LatentDims; i++) {
    latentDimSliders[i].value = latentDimValues[i]
  }
}

async function listImages() {
  return await (await fetch("/images")).json();
}

async function getImage(path) {
  let imageData = await (await fetch(`/image/${path}`)).json();

  return tf.tensor(imageData, model.inputShape);
}

async function imageToLatent(imageTensor) {
  return await model.compressImage(imageTensor);
}

async function latentToImage(latentTensor) {
  return await model.generateImage(latentTensor);
}

async function drawCanvas(tensor) {
  const data = await tensor.data();
  const imageDataArray = new Uint8ClampedArray(model.inputShape[0] * model.inputShape[1] * 4);

  for (let i = 0; i < model.inputShape[0] * model.inputShape[1]; i++) {
    const offset = i * 4;
    const index = i * 3;

    imageDataArray[offset] = data[index];
    imageDataArray[offset + 1] = data[index + 1];
    imageDataArray[offset + 2] = data[index + 2];
    imageDataArray[offset + 3] = 255;
  }

  const imageData = new ImageData(imageDataArray, model.inputShape[1], model.inputShape[0]);
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.putImageData(imageData, 0, 0);
  ctx.drawImage(canvas, 0, 0, model.inputShape[1], model.inputShape[0], 0, 0, canvas.width, canvas.height)
}

async function renderLatent() {
  let latentTensor = tf.tensor(latentDimValues);

  let imageDecoded = await latentToImage(latentTensor);

  drawCanvas(imageDecoded);
}

async function start() {
  let images = await listImages();

  let imageTensor = await getImage(images[Math.floor(Math.random() * images.length)]);
  let latentTensor = await imageToLatent(imageTensor);
  latentDimValues = await latentTensor.data();

  tf.dispose([imageTensor, latentTensor]);

  setSlidersValues();
  await renderLatent();
}

init().then(start)