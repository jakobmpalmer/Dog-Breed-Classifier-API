"""Check that the ONNX model matches the Keras model, and measure clean test accuracy.

Preprocessing matches Jakob's training notebook exactly:
  load as RGB, resize to 224x224 (nearest, the Keras generator default),
  subtract the ImageNet channel means, then divide by 255. No std division.

Usage: python verify.py model.keras dog_breed_densenet.onnx data/test
"""
import os
import sys
import time
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.NEAREST)
    x = np.asarray(img, dtype=np.float32)
    return (x - MEAN) / 255.0


keras_path, onnx_path, test_dir = sys.argv[1:4]
breeds = sorted(os.listdir(test_dir))  # same order as the Keras generator's class indices
files, labels = [], []
for i, b in enumerate(breeds):
    for f in sorted(os.listdir(os.path.join(test_dir, b))):
        files.append(os.path.join(test_dir, b, f))
        labels.append(i)
labels = np.array(labels)
X = np.stack([preprocess(f) for f in files])
print(f"{len(breeds)} breeds, {len(files)} test images")

import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
t0 = time.time()
p_onnx = np.concatenate([sess.run(None, {"image": X[i:i + 32]})[0] for i in range(0, len(X), 32)])
onnx_time = time.time() - t0

import keras
model = keras.models.load_model(keras_path, compile=False)
p_keras = model.predict(X, batch_size=32, verbose=0)

diff = np.abs(p_onnx - p_keras)
print(f"Max abs difference in probabilities: {diff.max():.2e}  (mean {diff.mean():.2e})")
print(f"Top-1 agreement Keras vs ONNX: {(p_onnx.argmax(1) == p_keras.argmax(1)).mean() * 100:.2f}%")
for name, p in (("Keras", p_keras), ("ONNX", p_onnx)):
    top1 = (p.argmax(1) == labels).mean() * 100
    top5 = np.mean([labels[i] in np.argsort(p[i])[-5:] for i in range(len(p))]) * 100
    print(f"{name}: clean test accuracy top-1 {top1:.2f}%  top-5 {top5:.2f}%")

single = X[:1]
sess.run(None, {"image": single})
t1 = time.time()
for _ in range(10):
    sess.run(None, {"image": single})
print(f"ONNX Runtime CPU: {(time.time() - t1) / 10 * 1000:.0f} ms per image (batch of 1); "
      f"{onnx_time:.1f} s for all {len(X)}")
