FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The model is downloaded from the Hugging Face Hub at build time and checked
# against its SHA-256. For a local build, put it in model/ and it is used instead.
ARG MODEL_URL=https://huggingface.co/jakobmp/dog-breed-densenet121/resolve/main/dog_breed_densenet.onnx
ARG MODEL_SHA256=87f56c8216282521ac7b770d5a717a4b5c2fe3c495c7bd98fa780f95c93c54e9
COPY scripts/fetch_model.py /tmp/fetch_model.py
COPY model/ model/
RUN python /tmp/fetch_model.py "$MODEL_URL" "$MODEL_SHA256" model/dog_breed_densenet.onnx \
    && rm /tmp/fetch_model.py

COPY app/ app/

# One worker keeps a single copy of the model in memory; threads handle concurrency.
CMD exec gunicorn --bind ":$PORT" --workers 1 --threads 4 --timeout 0 app.main:app
