"""Flask API that serves Jakob's DenseNet121 dog breed model with ONNX Runtime."""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from werkzeug.exceptions import HTTPException

VERSION = "1.0.0"
MODEL_PATH = os.environ.get("MODEL_PATH", "model/dog_breed_densenet.onnx")
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://dog-breed-classifier-frontend.vercel.app,http://localhost:3000",
).split(",")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# ImageNet channel means, subtracted before dividing by 255 to match training.
# The model was trained without std division, so none is applied here.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("dog-breed-api")

LABELS = json.loads(Path(__file__).with_name("labels.json").read_text())


def load_model(path):
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception:
        log.exception("Could not load model from %s", path)
        return None
    log.info("Loaded model from %s", path)
    return session


session = load_model(MODEL_PATH)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app, origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()])


def preprocess(img):
    """Match training: RGB, nearest resize to 224x224, minus means, /255, NHWC."""
    img = img.convert("RGB").resize((224, 224), Image.Resampling.NEAREST)
    x = np.asarray(img, dtype=np.float32)
    return ((x - MEAN) / 255.0)[np.newaxis]


@app.errorhandler(HTTPException)
def http_error(e):
    return jsonify(error=e.description), e.code


@app.get("/")
def index():
    return f"Dog Breed Classifier API {VERSION}\n", 200, {"Content-Type": "text/plain"}


@app.get("/health")
def health():
    if session is None:
        return jsonify(status="unavailable"), 503
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    if session is None:
        return jsonify(error="Model is not loaded"), 500

    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image provided (use multipart field 'image')"), 400
    try:
        img = Image.open(file.stream)
        img.load()
    except Exception:
        return jsonify(error="Could not read the image"), 400

    start = time.perf_counter()
    probs = session.run(None, {"image": preprocess(img)})[0][0]
    top = np.argsort(probs)[::-1][:5]
    latency_ms = (time.perf_counter() - start) * 1000

    breed, accuracy = LABELS[top[0]], float(probs[top[0]])
    log.info("Predicted %s (%.3f) in %.0f ms", breed, accuracy, latency_ms)
    return jsonify(
        breed=breed,
        accuracy=accuracy,
        top_5=[{"breed": LABELS[i], "confidence": float(probs[i])} for i in top],
    )
