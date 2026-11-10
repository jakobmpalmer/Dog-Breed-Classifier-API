# Dog Breed Classifier API

A Flask API that serves my DenseNet121 dog breed model with ONNX Runtime. It runs on Google
Cloud Run and is the backend for https://dogclassifier.jakobmpalmer.com.

The model was trained on Kaggle and classifies 70 breeds with 96.4% test accuracy. It's hosted on
Hugging Face at [jakobmp/dog-breed-densenet121](https://huggingface.co/jakobmp/dog-breed-densenet121),
whose model card covers training and evaluation.

## API

| Method | Path | Response |
|---|---|---|
| `GET` | `/` | `Dog Breed Classifier API 1.0.0` (plain text) |
| `GET` | `/health` | `{"status": "ok"}`, or 503 if the model didn't load |
| `POST` | `/predict` | the prediction for an image sent as the multipart field `image` |

```bash
curl -F "image=@beagle.jpg" https://<service-url>/predict
```

```json
{
  "breed": "Beagle",
  "accuracy": 0.976,
  "top_5": [
    {"breed": "Beagle", "confidence": 0.976},
    {"breed": "Basset", "confidence": 0.006},
    ...
  ]
}
```

`accuracy` is the top prediction's confidence. Errors return JSON like `{"error": "..."}`:
400 for a missing or unreadable image, 413 for uploads over 10 MB, and 500 if the model isn't
loaded.

Images are converted to RGB, resized to 224×224 and preprocessed exactly as in training, so
predictions match the original Keras model.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | the Vercel site and `http://localhost:3000` | comma-separated origins allowed by CORS |
| `MODEL_PATH` | `model/dog_breed_densenet.onnx` | the ONNX model file |
| `PORT` | `8080` in the container | set by Cloud Run |

## Running locally

Requires Python 3.11. Download `dog_breed_densenet.onnx` from Hugging Face into `model/`, then:

```bash
pip install -r requirements-dev.txt
flask --app app.main run --port 8080
```

## Tests

```bash
pytest
```

The API tests use the model in `model/`. `tests/test_parity.py` also checks accuracy on the
Kaggle test set when `TEST_IMAGES_DIR` points at its `test` folder, and is skipped otherwise.

## Deployment

The `Dockerfile` downloads the model from Hugging Face at build time and checks its SHA-256.
[DEPLOY.md](DEPLOY.md) has the commands to deploy to Cloud Run, test the service and keep costs
at zero when idle.

## Repo layout

- `app/`: the Flask API and the 70 breed labels
- `tests/`: API and accuracy tests
- `scripts/`: converting the Keras model to ONNX (`convert.py`), checking the two match
  (`verify.py`) and fetching the model during the Docker build (`fetch_model.py`)
- `huggingface/`: the model card published on Hugging Face

I built this with AI coding tools in November 2026 to serve the model I trained, moving it from
AWS EC2 to ONNX Runtime on Cloud Run.
