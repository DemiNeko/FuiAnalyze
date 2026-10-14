// Match the MNIST convention: preserve aspect ratio, fit into 20x20,
// then translate the center of mass to the center of a 28x28 image.
export function normalizeDrawing(canvas) {
  const { width, height } = canvas;
  const data = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, width, height).data;
  let left = width, right = -1, top = height, bottom = -1, mass = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = data[(y * width + x) * 4];
      if (value > 30) {
        left = Math.min(left, x); right = Math.max(right, x);
        top = Math.min(top, y); bottom = Math.max(bottom, y);
        mass += value / 255;
      }
    }
  }
  if (right < left || mass < 15 || Math.max(right - left, bottom - top) < 10) return null;
  const temp = document.createElement('canvas');
  temp.width = temp.height = 28;
  const ctx = temp.getContext('2d', { willReadFrequently: true });
  const cropWidth = right - left + 1, cropHeight = bottom - top + 1;
  const scale = 20 / Math.max(cropWidth, cropHeight);
  const targetWidth = Math.max(1, Math.round(cropWidth * scale));
  const targetHeight = Math.max(1, Math.round(cropHeight * scale));
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 28, 28);
  ctx.drawImage(canvas, left, top, cropWidth, cropHeight,
    Math.floor((28 - targetWidth) / 2), Math.floor((28 - targetHeight) / 2), targetWidth, targetHeight);
  const small = ctx.getImageData(0, 0, 28, 28).data;
  const pixels = Float32Array.from({ length: 784 }, (_, i) => small[i * 4] / 255);
  let sum = 0, centerX = 0, centerY = 0;
  pixels.forEach((value, i) => { sum += value; centerX += (i % 28) * value; centerY += Math.floor(i / 28) * value; });
  if (sum === 0) return null;
  const dx = Math.round(13.5 - centerX / sum), dy = Math.round(13.5 - centerY / sum);
  const centered = new Float32Array(784);
  pixels.forEach((value, i) => {
    const x = i % 28 + dx, y = Math.floor(i / 28) + dy;
    if (x >= 0 && x < 28 && y >= 0 && y < 28) centered[y * 28 + x] = value;
  });
  return centered;
}

export function paintPixels(canvas, pixels, size = 28, color = false) {
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(size, size);
  const max = color ? Math.max(...pixels, 0.0001) : 1;
  pixels.forEach((value, i) => {
    const v = Math.max(0, Math.min(1, value / max));
    image.data[i * 4] = color ? Math.round(20 + v * 98) : Math.round(v * 255);
    image.data[i * 4 + 1] = color ? Math.round(30 + v * 208) : Math.round(v * 255);
    image.data[i * 4 + 2] = color ? Math.round(45 + v * 165) : Math.round(v * 255);
    image.data[i * 4 + 3] = 255;
  });
  const temp = document.createElement('canvas');
  temp.width = temp.height = size;
  temp.getContext('2d').putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(temp, 0, 0, canvas.width, canvas.height);
}
