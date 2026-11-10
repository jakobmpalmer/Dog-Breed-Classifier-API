"""Accuracy through the app's own preprocessing on the Kaggle test set.

Runs only when TEST_IMAGES_DIR points at the "70 Dog Breed Images/test" folder.
The exported model scores 96.43% on it (see the model card).
"""
import os
from pathlib import Path

import pytest

from app import main
from conftest import post_image

TEST_IMAGES_DIR = os.environ.get("TEST_IMAGES_DIR")


@pytest.mark.skipif(not TEST_IMAGES_DIR, reason="TEST_IMAGES_DIR not set")
@pytest.mark.skipif(main.session is None, reason="model not loaded")
def test_test_set_accuracy(client):
    test_dir = Path(TEST_IMAGES_DIR)
    breeds = sorted(p.name for p in test_dir.iterdir() if p.is_dir())
    assert breeds == main.LABELS

    correct = total = 0
    for breed in breeds:
        for path in sorted((test_dir / breed).iterdir()):
            r = post_image(client, path.read_bytes(), path.name)
            assert r.status_code == 200, path
            correct += r.get_json()["breed"] == breed
            total += 1

    accuracy = correct / total
    print(f"\nTest accuracy: {correct}/{total} = {accuracy:.2%}")
    assert total == 700
    assert accuracy >= 0.95
