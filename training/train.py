"""Train a small MNIST CNN and export dependency-free browser inference assets.

Run from anywhere: python training/train.py --epochs 8
The official MNIST test set is evaluated once, after model selection on validation.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


def augment(x):
    n = x.shape[0]
    angle = (torch.rand(n) - 0.5) * 0.30
    scale = 0.9 + torch.rand(n) * 0.2
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = angle.cos() * scale
    theta[:, 0, 1] = -angle.sin() * scale
    theta[:, 1, 0] = angle.sin() * scale
    theta[:, 1, 1] = angle.cos() * scale
    theta[:, :, 2] = (torch.rand(n, 2) - 0.5) * 0.20
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)


@torch.no_grad()
def evaluate(model, x, y):
    logits = torch.cat([model(batch) for batch in x.split(512)])
    return float((logits.argmax(1) == y).float().mean()), logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    data_dir = ROOT / "training" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "mnist.npz"
    if not data_path.exists():
        print("Downloading MNIST...", flush=True)
        urllib.request.urlretrieve(DATA_URL, data_path)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f"MNIST checksum mismatch: {digest}")
    data = np.load(data_path, allow_pickle=False)
    x = torch.from_numpy(data["x_train"].copy()).unsqueeze(1).float() / 255
    y = torch.from_numpy(data["y_train"].astype(np.int64))
    order = torch.randperm(len(x))
    train_idx, val_idx = order[:55000], order[55000:]
    model = DigitCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_accuracy, best_state, best_epoch = -1, None, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for idx in shuffled.split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(augment(x[idx])), y[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        accuracy, _ = evaluate(model, x[val_idx], y[val_idx])
        history.append({"epoch": epoch, "loss": round(total_loss / len(train_idx), 5),
                        "validationAccuracy": round(accuracy, 5)})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        print(f"Epoch {epoch}/{args.epochs}: loss {history[-1]['loss']:.4f}, "
              f"validation {accuracy:.2%}, elapsed {time.perf_counter()-started:.0f}s", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.from_numpy(data["x_test"].copy()).unsqueeze(1).float() / 255
    test_y = torch.from_numpy(data["y_test"].astype(np.int64))
    test_accuracy, logits = evaluate(model, test_x, test_y)
    predictions = logits.argmax(1)
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for actual, predicted in zip(test_y, predictions):
        confusion[actual, predicted] += 1
    output = ROOT / "assets"
    output.mkdir(exist_ok=True)
    offset, layers, arrays = 0, [], []
    for name, tensor in model.state_dict().items():
        array = tensor.detach().numpy().astype("<f4").ravel()
        layers.append({"name": name, "shape": list(tensor.shape), "offset": offset, "length": len(array)})
        offset += len(array)
        arrays.append(array)
    weights_bytes = np.concatenate(arrays).tobytes()
    (output / "weights.bin").write_bytes(weights_bytes)
    metadata = {
        "formatVersion": 1, "architecture": "Conv8(3) → Pool2 → Conv16(3) → Pool2 → Dense64 → Dense10",
        "dataset": "MNIST", "datasetUrl": DATA_URL, "datasetSha256": digest,
        "seed": 42, "trainSize": 55000, "validationSize": 5000, "testSize": 10000,
        "epochs": args.epochs, "selectedEpoch": best_epoch, "parameters": offset,
        "testAccuracy": round(test_accuracy, 5), "testCorrect": int((predictions == test_y).sum()),
        "validationAccuracy": round(best_accuracy, 5), "history": history,
        "confusionMatrix": confusion.tolist(), "layers": layers,
        "weightsSha256": hashlib.sha256(weights_bytes).hexdigest(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "preprocessing": "28x28 grayscale, white digit on black, pixels / 255",
    }
    (output / "model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # First three test examples per class, selected by label, never by correctness.
    indices = [int(i) for digit in range(10) for i in np.flatnonzero(data["y_test"] == digit)[:3]]
    examples = [{"label": int(data["y_test"][i]), "index": i,
                 "pixels": data["x_test"][i].flatten().tolist(),
                 "probabilities": logits[i].softmax(0).tolist()} for i in indices]
    (output / "examples.json").write_text(json.dumps(examples, separators=(",", ":")), encoding="utf-8")
    print(f"Test: {metadata['testCorrect']}/10000 ({test_accuracy:.2%}); "
          f"{offset:,} parameters; selected epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
