import io

import pytest
from PIL import Image

from app import main


@pytest.fixture
def client():
    return main.app.test_client()


def image_bytes(mode="RGB", size=(320, 240), fmt="JPEG"):
    """A small generated gradient image, so no dataset images live in the repo."""
    img = Image.linear_gradient("L").resize(size).convert(mode)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def post_image(client, data, filename="dog.jpg", **kwargs):
    return client.post(
        "/predict",
        data={"image": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
        **kwargs,
    )
