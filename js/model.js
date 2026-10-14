// The forward pass of training/train.py, implemented without ML dependencies.
// Tensor layout: channels first (channel, row, column), weights: output first.
export function decodeModel(metadata, buffer) {
  const expected = [
    ['conv1.weight', [8, 1, 3, 3]], ['conv1.bias', [8]],
    ['conv2.weight', [16, 8, 3, 3]], ['conv2.bias', [16]],
    ['fc1.weight', [64, 400]], ['fc1.bias', [64]],
    ['fc2.weight', [10, 64]], ['fc2.bias', [10]],
  ];
  if (metadata.formatVersion !== 1 || metadata.layers?.length !== expected.length) {
    throw new Error('Неизвестный формат модели');
  }
  const view = new DataView(buffer);
  const weights = {};
  let offset = 0;
  expected.forEach(([name, shape], index) => {
    const layer = metadata.layers[index];
    const length = shape.reduce((a, b) => a * b, 1);
    if (layer.name !== name || JSON.stringify(layer.shape) !== JSON.stringify(shape)
        || layer.offset !== offset || layer.length !== length) {
      throw new Error('Повреждена структура модели');
    }
    weights[name] = Float32Array.from({ length }, (_, i) => view.getFloat32((offset + i) * 4, true));
    if (weights[name].some(value => !Number.isFinite(value))) throw new Error('Повреждены веса модели');
    offset += length;
  });
  if (buffer.byteLength !== offset * 4 || metadata.parameters !== offset) throw new Error('Неверный размер весов');
  return { metadata, weights };
}

function convolution(input, size, inChannels, outChannels, weights, biases) {
  const nextSize = size - 2;
  const output = new Float32Array(outChannels * nextSize * nextSize);
  for (let oc = 0; oc < outChannels; oc++) {
    for (let y = 0; y < nextSize; y++) {
      for (let x = 0; x < nextSize; x++) {
        let value = biases[oc];
        for (let ic = 0; ic < inChannels; ic++) {
          for (let ky = 0; ky < 3; ky++) {
            for (let kx = 0; kx < 3; kx++) {
              value += input[ic * size * size + (y + ky) * size + x + kx]
                * weights[(oc * inChannels + ic) * 9 + ky * 3 + kx];
            }
          }
        }
        output[oc * nextSize * nextSize + y * nextSize + x] = Math.max(0, value);
      }
    }
  }
  return output;
}

function maxPool(input, size, channels) {
  const nextSize = Math.floor(size / 2);
  const output = new Float32Array(channels * nextSize * nextSize);
  for (let c = 0; c < channels; c++) {
    for (let y = 0; y < nextSize; y++) {
      for (let x = 0; x < nextSize; x++) {
        const i = c * size * size + y * 2 * size + x * 2;
        output[c * nextSize * nextSize + y * nextSize + x] = Math.max(
          input[i], input[i + 1], input[i + size], input[i + size + 1]);
      }
    }
  }
  return output;
}

function dense(input, weights, biases, relu) {
  return Float32Array.from(biases, (bias, neuron) => {
    let value = bias;
    for (let i = 0; i < input.length; i++) value += input[i] * weights[neuron * input.length + i];
    return relu ? Math.max(0, value) : value;
  });
}

export function predict(model, pixels) {
  if (pixels.length !== 784 || pixels.some(value => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error('Ожидаются 784 пикселя от 0 до 1');
  }
  const w = model.weights;
  const features = convolution(pixels, 28, 1, 8, w['conv1.weight'], w['conv1.bias']);
  const pool1 = maxPool(features, 26, 8);
  const conv2 = convolution(pool1, 13, 8, 16, w['conv2.weight'], w['conv2.bias']);
  const pool2 = maxPool(conv2, 11, 16);
  const hidden = dense(pool2, w['fc1.weight'], w['fc1.bias'], true);
  const logits = dense(hidden, w['fc2.weight'], w['fc2.bias'], false);
  const max = Math.max(...logits);
  const exp = Array.from(logits, value => Math.exp(value - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  const probabilities = exp.map(value => value / sum);
  const digit = probabilities.indexOf(Math.max(...probabilities));
  return { digit, probabilities, features, hidden };
}
