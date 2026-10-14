import { decodeModel, predict } from './model.js';
import { normalizeDrawing, paintPixels } from './preprocess.js';

const $ = id => document.getElementById(id);
const canvas = $('drawing');
const context = canvas.getContext('2d', { willReadFrequently: true });
let model = null, examples = [], inputPixels = null, drawing = false, pointer = null;
let history = [], activeSample = null;
const sampleCounters = Array(10).fill(0);
const percent = value => (value * 100).toLocaleString('ru-RU', { maximumFractionDigits: 1, minimumFractionDigits: 1 });

const rows = Array.from({ length: 10 }, (_, digit) => {
  const row = document.createElement('div');
  row.className = 'probability-row';
  row.innerHTML = `<span>${digit}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="probability-value">—</span>`;
  $('probabilities').append(row);
  const button = document.createElement('button');
  button.type = 'button'; button.textContent = digit; button.disabled = true;
  button.setAttribute('aria-label', `Пример цифры ${digit}`);
  button.addEventListener('click', () => useExample(digit));
  $('example-buttons').append(button);
  return { row, fill: row.querySelector('.bar-fill'), value: row.querySelector('.probability-value'), button };
});
const featureCanvases = Array.from({ length: 4 }, (_, i) => {
  const c = document.createElement('canvas'); c.width = c.height = 52;
  c.setAttribute('aria-label', `Карта активаций фильтра ${i + 1}`);
  $('feature-maps').append(c); return c;
});
const neuronElements = Array.from({ length: 64 }, () => {
  const span = document.createElement('span'); $('neurons').append(span); return span;
});

function clearResults() {
  $('result-digit').textContent = '?'; $('result-digit').classList.remove('has-result');
  $('result-caption').textContent = 'Пока чистый лист'; $('result-score').textContent = 'Твой ход';
  $('result-detail').textContent = 'Нарисуй цифру или выбери пример';
  $('pipeline-result').textContent = '?'; $('inference-time').textContent = '0–9';
  rows.forEach(({ row, fill, value }) => { row.classList.remove('winner'); fill.style.width = '0%'; value.textContent = '—'; });
  paintPixels($('input-preview'), new Float32Array(784));
  featureCanvases.forEach(c => paintPixels(c, new Float32Array(676), 26, true));
  neuronElements.forEach(span => { span.style.background = ''; });
}

function reset() {
  if (drawing) { drawing = false; pointer = null; }
  context.fillStyle = '#000'; context.fillRect(0, 0, canvas.width, canvas.height);
  history = []; inputPixels = null; activeSample = null;
  $('placeholder').hidden = false; $('undo').disabled = true;
  $('input-message').textContent = 'Рисуй по центру, не касаясь краёв холста.';
  rows.forEach(({ button }) => button.classList.remove('active'));
  clearResults();
}

function saveUndo() {
  history.push({ image: context.getImageData(0, 0, canvas.width, canvas.height),
    pixels: inputPixels?.slice() ?? null, sample: activeSample });
  if (history.length > 20) history.shift();
  $('undo').disabled = false;
}

function runPrediction() {
  if (!model) return;
  const pixels = inputPixels ?? normalizeDrawing(canvas);
  if (!pixels) {
    clearResults();
    $('input-message').textContent = 'Сначала нарисуй крупную цифру или выбери пример ниже.';
    return;
  }
  paintPixels($('input-preview'), pixels);
  const start = performance.now();
  const result = predict(model, pixels);
  const milliseconds = performance.now() - start;
  const score = result.probabilities[result.digit];
  $('result-digit').textContent = result.digit; $('result-digit').classList.add('has-result');
  $('result-caption').textContent = score < 0.75 ? 'Есть несколько версий' : `Похоже на цифру ${result.digit}`;
  $('result-score').textContent = `${percent(score)}%`;
  $('result-detail').textContent = 'Оценка самого вероятного класса';
  $('inference-time').textContent = `${milliseconds.toLocaleString('ru-RU', { maximumFractionDigits: 1 })} мс`;
  $('pipeline-result').textContent = result.digit;
  rows.forEach(({ row, fill, value }, digit) => {
    row.classList.toggle('winner', digit === result.digit);
    fill.style.width = `${result.probabilities[digit] * 100}%`;
    value.textContent = `${percent(result.probabilities[digit])}%`;
  });
  featureCanvases.forEach((c, i) => paintPixels(c, result.features.subarray(i * 676, (i + 1) * 676), 26, true));
  const maximum = Math.max(...result.hidden, 0.001);
  neuronElements.forEach((span, i) => {
    const v = result.hidden[i] / maximum;
    span.style.background = `rgb(${Math.round(223 - v * 160)},${Math.round(231 - v * 111)},${Math.round(214 - v * 171)})`;
    span.title = `Нейрон ${i + 1}: ${result.hidden[i].toFixed(3)}`;
  });
  $('input-message').textContent = activeSample === null
    ? 'Можно дорисовать цифру — ответ обновится автоматически.'
    : `Пример MNIST №${activeSample.index}. Правильный ответ: ${activeSample.label}.`;
}

function useExample(digit) {
  if (!model || drawing) return;
  saveUndo();
  const samples = examples.filter(example => example.label === digit);
  const sample = samples[sampleCounters[digit]++ % samples.length];
  activeSample = sample;
  inputPixels = Float32Array.from(sample.pixels, value => value / 255);
  paintPixels(canvas, inputPixels);
  $('placeholder').hidden = true;
  rows.forEach(({ button }, n) => button.classList.toggle('active', n === digit));
  runPrediction();
}

function position(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height };
}
canvas.addEventListener('pointerdown', event => {
  if (drawing || !event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault(); saveUndo(); drawing = true; pointer = event.pointerId;
  inputPixels = null; activeSample = null; $('placeholder').hidden = true;
  rows.forEach(({ button }) => button.classList.remove('active'));
  canvas.setPointerCapture(event.pointerId);
  const { x, y } = position(event);
  context.strokeStyle = '#fff'; context.fillStyle = '#fff'; context.lineWidth = 21;
  context.lineCap = context.lineJoin = 'round';
  context.beginPath(); context.arc(x, y, 10.5, 0, Math.PI * 2); context.fill();
  context.beginPath(); context.moveTo(x, y);
});
canvas.addEventListener('pointermove', event => {
  if (!drawing || event.pointerId !== pointer) return;
  const events = event.getCoalescedEvents?.() || [];
  for (const point of events.length ? events : [event]) {
    const { x, y } = position(point); context.lineTo(x, y); context.stroke();
  }
});
function finishDrawing(event) {
  if (!drawing || event.pointerId !== pointer) return;
  drawing = false; pointer = null; context.closePath();
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  runPrediction();
}
canvas.addEventListener('pointerup', finishDrawing);
canvas.addEventListener('pointercancel', finishDrawing);
canvas.addEventListener('lostpointercapture', finishDrawing);
$('clear').addEventListener('click', reset);
$('recognize').addEventListener('click', runPrediction);
$('undo').addEventListener('click', () => {
  if (drawing) return;
  const state = history.pop(); if (!state) return;
  context.putImageData(state.image, 0, 0); inputPixels = state.pixels; activeSample = state.sample;
  $('undo').disabled = history.length === 0;
  const pixels = inputPixels ?? normalizeDrawing(canvas);
  $('placeholder').hidden = Boolean(pixels);
  rows.forEach(({ button }, digit) => button.classList.toggle('active', activeSample?.label === digit));
  if (pixels) runPrediction(); else { clearResults(); $('input-message').textContent = 'Рисуй по центру, не касаясь краёв холста.'; }
});

async function fetchAsset(path, binary = false) {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`Не загружен ${path}: ${response.status}`);
  return binary ? response.arrayBuffer() : response.json();
}
async function load() {
  $('load-error').hidden = true; $('model-status').textContent = 'Загружаем модель…';
  $('model-status').className = 'status'; $('retry').disabled = true;
  try {
    const [metadata, buffer, samples] = await Promise.all([
      fetchAsset('../assets/model.json'), fetchAsset('../assets/weights.bin', true), fetchAsset('../assets/examples.json'),
    ]);
    if (crypto.subtle) {
      const hash = await crypto.subtle.digest('SHA-256', buffer);
      const hex = Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
      if (hex !== metadata.weightsSha256) throw new Error('Контрольная сумма модели не совпадает');
    }
    model = decodeModel(metadata, buffer); examples = samples;
    $('model-status').textContent = 'Модель готова к работе'; $('model-status').className = 'status ready';
    $('recognize').disabled = false; rows.forEach(({ button }) => { button.disabled = false; });
    $('accuracy').textContent = `${(metadata.testAccuracy * 100).toLocaleString('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
    $('accuracy').title = `${metadata.testCorrect} правильных ответов из ${metadata.testSize}`;
    $('parameter-count').textContent = metadata.parameters.toLocaleString('ru-RU');
    if (normalizeDrawing(canvas)) runPrediction();
  } catch (error) {
    console.error(error); model = null;
    $('load-error').hidden = false;
    $('model-status').textContent = 'Модель не загрузилась'; $('model-status').className = 'status error';
    $('recognize').disabled = true; rows.forEach(({ button }) => { button.disabled = true; });
  } finally { $('retry').disabled = false; }
}
$('retry').addEventListener('click', load);
reset(); load();
