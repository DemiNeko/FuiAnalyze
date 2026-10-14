import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { decodeModel, predict } from '../js/model.js';

const metadata = JSON.parse(await readFile(new URL('../assets/model.json', import.meta.url), 'utf8'));
const bytes = await readFile(new URL('../assets/weights.bin', import.meta.url));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const examples = JSON.parse(await readFile(new URL('../assets/examples.json', import.meta.url), 'utf8'));
const model = decodeModel(metadata, buffer);

test('browser forward pass matches independent PyTorch outputs on 30 MNIST examples', () => {
  assert.equal(examples.length, 30);
  for (const example of examples) {
    const result = predict(model, Float32Array.from(example.pixels, value => value / 255));
    example.probabilities.forEach((expected, digit) => {
      assert.ok(Math.abs(expected - result.probabilities[digit]) < 0.00002,
        `Example ${example.index}, class ${digit}: JS ${result.probabilities[digit]} vs PyTorch ${expected}`);
    });
    assert.ok(Math.abs(result.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-10);
    assert.equal(result.digit, example.probabilities.indexOf(Math.max(...example.probabilities)));
    assert.equal(result.features.length, 8 * 26 * 26);
    assert.equal(result.hidden.length, 64);
  }
});

test('model checksum and held-out metrics are consistent', () => {
  assert.equal(createHash('sha256').update(bytes).digest('hex'), metadata.weightsSha256);
  assert.equal(metadata.testSize, 10000);
  assert.equal(metadata.trainSize + metadata.validationSize, 60000);
  const total = metadata.confusionMatrix.flat().reduce((a, b) => a + b, 0);
  const correct = metadata.confusionMatrix.reduce((sum, row, digit) => sum + row[digit], 0);
  assert.equal(total, metadata.testSize);
  assert.equal(correct, metadata.testCorrect);
  assert.ok(Math.abs(correct / total - metadata.testAccuracy) < 1e-5);
  assert.ok(metadata.testAccuracy >= 0.97, 'The exported model must meet the demo quality threshold');
  assert.equal(metadata.validationAccuracy, Math.max(...metadata.history.map(epoch => epoch.validationAccuracy)));
});

test('damaged weights, incompatible architecture and invalid inputs are rejected', () => {
  assert.throws(() => decodeModel(metadata, buffer.slice(0, 100)));
  assert.throws(() => decodeModel({ ...metadata, formatVersion: 20 }, buffer));
  const incompatible = structuredClone(metadata); incompatible.layers[0].shape = [8, 9];
  assert.throws(() => decodeModel(incompatible, buffer));
  assert.throws(() => predict(model, new Float32Array(783)));
  const invalid = new Float32Array(784); invalid[12] = NaN;
  assert.throws(() => predict(model, invalid));
  invalid[12] = 1.1; assert.throws(() => predict(model, invalid));
});
