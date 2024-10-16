import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load the ONNX model
onnx_model = onnx.load("model_modified.onnx")

# Convert ONNX model to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model
tf_rep.export_graph("model.pb")
