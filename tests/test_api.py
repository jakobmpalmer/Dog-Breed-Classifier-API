import pytest

from app import main
from conftest import image_bytes, post_image

needs_model = pytest.mark.skipif(
    main.session is None, reason=f"model not found at {main.MODEL_PATH}"
)


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.mimetype == "text/plain"
    assert main.VERSION in r.get_data(as_text=True)


@needs_model
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


@needs_model
@pytest.mark.parametrize("mode, fmt", [("RGB", "JPEG"), ("RGBA", "PNG"), ("L", "PNG")])
def test_predict_response_shape(client, mode, fmt):
    r = post_image(client, image_bytes(mode, fmt=fmt), f"dog.{fmt.lower()}")
    assert r.status_code == 200
    data = r.get_json()
    assert set(data) == {"breed", "accuracy", "top_5"}
    assert data["breed"] in main.LABELS
    assert 0 <= data["accuracy"] <= 1

    top_5 = data["top_5"]
    assert len(top_5) == 5
    assert all(set(p) == {"breed", "confidence"} for p in top_5)
    assert len({p["breed"] for p in top_5}) == 5
    confidences = [p["confidence"] for p in top_5]
    assert confidences == sorted(confidences, reverse=True)
    assert top_5[0] == {"breed": data["breed"], "confidence": data["accuracy"]}


def test_predict_without_image(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert "error" in r.get_json()


@needs_model
def test_predict_unreadable_image(client):
    r = post_image(client, b"this is not an image", "dog.jpg")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_too_large(client):
    r = post_image(client, b"\0" * (main.MAX_UPLOAD_BYTES + 1))
    assert r.status_code == 413
    assert "error" in r.get_json()


def test_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(main, "session", None)
    assert client.get("/health").status_code == 503
    r = post_image(client, image_bytes())
    assert r.status_code == 500
    assert "error" in r.get_json()


def test_cors_allows_only_configured_origins(client):
    allowed = "http://localhost:3000"
    r = client.get("/", headers={"Origin": allowed})
    assert r.headers.get("Access-Control-Allow-Origin") == allowed

    r = client.get("/", headers={"Origin": "https://evil.example.com"})
    assert "Access-Control-Allow-Origin" not in r.headers
