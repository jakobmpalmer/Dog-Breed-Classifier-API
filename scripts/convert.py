"""Convert Jakob's DenseNet121 Keras model to ONNX.

Usage: python convert.py model.keras dog_breed_densenet.onnx
"""
import sys
import tensorflow as tf
import keras
import tf2onnx

src, dst = sys.argv[1], sys.argv[2]

model = keras.models.load_model(src)
print("Loaded:", model.name, "| params:", f"{model.count_params():,}")

# The saved model accepts any image size; fix it to 224x224 RGB,
# the size it was trained on, so the ONNX graph has a static shape.
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image"),)


@tf.function(input_signature=spec)
def serve(image):
    return {"probabilities": model(image, training=False)}


onnx_model, _ = tf2onnx.convert.from_function(
    serve, input_signature=spec, opset=17, output_path=dst
)
print("Wrote", dst)
print("Inputs:", [i.name for i in onnx_model.graph.input])
print("Outputs:", [o.name for o in onnx_model.graph.output])
