---
license: apache-2.0
library_name: onnx
pipeline_tag: image-classification
tags:
  - image-classification
  - dog-breeds
  - densenet
  - onnx
  - keras
---

# Dog breed classifier (DenseNet121, 70 breeds)

Classifies a photo of a dog as one of 70 breeds. It powers the
[Dog Breed Classifier](https://dog-breed-classifier-frontend.vercel.app) web app.

## Model

- **Architecture:** DenseNet121 (ImageNet weights, no top) → global average pooling →
  Dense(1024, `selu`, L1(0.001) kernel regularization) → Dense(70, softmax).
- **Fine-tuning:** all layers trained end to end with Adam (learning rate 1e-4, clipnorm 1.0)
  and categorical cross-entropy, for 10 epochs at batch size 128.
- **Training:** by Jakob Paulson-Palmer. The notebook was written in April 2024:
  https://www.kaggle.com/code/jakobmp/dog-breed-classification-densenet.
  These weights come from rerunning it on Kaggle on November 2, 2026, with the code unchanged.

## Data

The Kaggle [70 Dog Breeds-Image Data Set](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set):
7,946 training, 700 validation and 700 test images (10 per breed in validation and test).

## Results

| Evaluation | Accuracy |
|---|---|
| Test set, clean (no augmentation), top-1 | **96.14%** (673/700) |
| Test set, clean, top-5 | 100.00% |
| Test set, Kaggle `model.evaluate` (with augmentation) | 96.29% |
| Validation set, Kaggle `model.evaluate` (with augmentation) | 92.57% |

The ONNX export gives the same top-1 prediction as the Keras model on all 700 test images
(largest probability difference 1.6e-06).

## Preprocessing

Inputs must be prepared exactly as in training:

1. Decode the image and convert it to RGB.
2. Resize to 224x224 with nearest-neighbor resampling.
3. Convert to float32 and divide by 255.
4. Subtract the ImageNet channel means `[0.485, 0.456, 0.406]`. Do **not** divide by a standard deviation.
5. Add a batch dimension: shape `[1, 224, 224, 3]` (NHWC).

```python
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("dog_breed_densenet.onnx")
labels = json.load(open("labels.json"))

img = Image.open("dog.jpg").convert("RGB").resize((224, 224), Image.Resampling.NEAREST)
x = np.asarray(img, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)
probs = session.run(None, {"image": x[np.newaxis]})[0][0]
print(labels[int(probs.argmax())], float(probs.max()))
```

## Files

| File | Description |
|---|---|
| `dog_breed_densenet.onnx` | ONNX (opset 17), 32 MB. Input `image`: float32 `[N, 224, 224, 3]`; output `probabilities`: `[N, 70]`. SHA-256 `87f56c8216282521ac7b770d5a717a4b5c2fe3c495c7bd98fa780f95c93c54e9`. |
| `labels.json` | The 70 breed names in output-index order (alphabetical). |
| `dog_breed_densenet.keras` | The original Keras 3 model the ONNX file was converted from. |

## Intended use and limitations

- For identifying dog breeds in photos, for fun and learning.
- It only knows these 70 breeds. Any photo, including mixed breeds, other breeds, other animals or
  no animal at all, gets mapped to one of them, often with high confidence.
- Its training images are mostly clear, centered photos; accuracy will be lower on unusual angles,
  lighting or crops.
- Not for safety, legal, veterinary or other consequential decisions (for example, breed-specific
  regulations).

## License

Apache-2.0, the same as the Kaggle notebook.
