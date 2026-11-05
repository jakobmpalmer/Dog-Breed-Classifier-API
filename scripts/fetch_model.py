"""Put the ONNX model in place during the Docker build and verify its SHA-256.

Uses a local copy if one is already at the destination (local builds);
otherwise downloads it from the URL. Standard library only.

Usage: python fetch_model.py URL SHA256 DEST
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

url, expected, dest = sys.argv[1], sys.argv[2].lower(), Path(sys.argv[3])

if dest.exists():
    print(f"Using local model at {dest}")
elif url:
    print(f"Downloading model from {url}")
    urllib.request.urlretrieve(url, dest)
else:
    sys.exit(f"No model at {dest} and no MODEL_URL given")

actual = hashlib.sha256(dest.read_bytes()).hexdigest()
if actual != expected:
    sys.exit(f"SHA-256 mismatch for {dest}: expected {expected}, got {actual}")
print(f"SHA-256 OK: {actual}")
