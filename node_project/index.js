// try doing "npm install @tensorflow/tfjs-node" in the terminal
//then run "node index.js" in the terminal

const tf = require("@tensorflow/tfjs-core");
const modelPath = './model.json';

//should be (replace with this if download works):
//const tf = require('@tensorflow/tfjs-node');
//const modelPath = './path/to/tfjs/model/model.json';


async function loadModel() {
  const loadedModel = await tf.loadLayersModel(`file://${modelPath}`);
  return loadedModel;
}

async function predict() {
  const loadedModel = await loadModel();
  const inputData = tf.tensor2d([[...your_input_data]], [1, ...input_shape]);
  const outputTensor = loadedModel.predict(inputData);
  const outputData = Array.from(await outputTensor.data());
  console.log('Predictions:', outputData);
}

predict();
console.log('hi');
