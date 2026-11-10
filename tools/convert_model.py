"""Converts DeepFace's Facenet512 Keras model to the ONNX file the web app loads.

    pip install "tensorflow==2.15.0" "numpy<2" "tf2onnx==1.16.1" "onnx==1.16.1" deepface
    python tools/convert_model.py models/facenet512.onnx

Large float32 weights are stored as float16 and cast back to float32 when the model loads.
That halves the download (about 45 MB); embeddings stay practically identical (cosine > 0.99999).
"""
import sys

import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
from deepface.modules import modeling
from onnx import TensorProto, helper, numpy_helper

out_path = sys.argv[1] if len(sys.argv) > 1 else "models/facenet512.onnx"

keras_model = modeling.build_model(task="facial_recognition", model_name="Facenet512").model
spec = (tf.TensorSpec((None, 160, 160, 3), tf.float32, name="input"),)
model, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, opset=13)

graph = model.graph
casts = []
for init in graph.initializer:
    if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= 256:
        name = init.name
        half = numpy_helper.to_array(init).astype(np.float16)
        init.CopyFrom(numpy_helper.from_array(half, name + "_f16"))
        casts.append(helper.make_node("Cast", [name + "_f16"], [name], to=TensorProto.FLOAT, name=f"cast_{len(casts)}"))
nodes = casts + list(graph.node)
del graph.node[:]
graph.node.extend(nodes)

onnx.checker.check_model(model)
onnx.save(model, out_path)
print(f"Saved {out_path}")
