// frontend/src/utils/model.js
import * as tf from '@tensorflow/tfjs';

const MODEL_URL = process.env.PUBLIC_URL + '/models/model.json';
const VOCAB_URL = process.env.PUBLIC_URL + '/models/vocabulary.json';
const IMG_SIZE   = 299;
const MAX_LENGTH = 40;

let tfModel, w2i, i2w, startToken, endToken;

/**
 * Load the TF.js model and vocabulary JSON.
 * Call this once on app startup.
 */
export async function initModel() {
  // 1) Load the model
  tfModel = await tf.loadLayersModel(MODEL_URL);

  // 2) Fetch vocab
  const vocab = await fetch(VOCAB_URL).then(r => r.json());
  w2i = vocab.word_to_index;
  // index_to_word comes in as strings; convert keys to ints
  i2w = Object.fromEntries(
    Object.entries(vocab.index_to_word)
      .map(([k,v]) => [parseInt(k, 10), v])
  );

  startToken = w2i['<start>'];
  endToken   = w2i['<end>'];
}

/**
 * Given a File (from <input type="file"/>), preprocess it,
 * run the model autoregressively, and return the final caption.
 */
export async function predictCaption(imageFile) {
  if (!tfModel) {
    throw new Error('Model not loaded yet');
  }

  // 1) Turn File → HTMLImageElement → Tensor
  const img = new Image();
  img.crossOrigin = 'anonymous';
  const imgLoad = new Promise((res, rej) => {
    img.onload  = () => res();
    img.onerror = err => rej(err);
  });
  img.src = URL.createObjectURL(imageFile);
  await imgLoad;

  let tensor = tf.browser.fromPixels(img)
    .resizeBilinear([IMG_SIZE, IMG_SIZE])
    .toFloat()
    .div(tf.scalar(127.5))
    .sub(tf.scalar(1))      // matches InceptionV3 preprocess_input
    .expandDims();          // shape [1,299,299,3]

  // 2) Autoregressive decode
  let seq = tf.tensor([[startToken]], [1,1], 'int32');
  const tokens = [];
  for (let i = 0; i < MAX_LENGTH - 1; i++) {
    const preds = tfModel.predict([tensor, seq]);
    // grab the logits at time-step i
    const stepLogits = preds
      .slice([0, i, 0], [1, 1, -1])
      .squeeze();
    const predId = (await stepLogits.argMax().data())[0];

    if (predId === endToken) break;
    tokens.push(predId);

    // append predicted id to the input seq for next step
    seq = seq.concat(
      tf.tensor([[predId]], [1,1], 'int32'),
      1
    );
  }

  // map token IDs back to words
  return tokens.map(id => i2w[id]).join(' ');
}
