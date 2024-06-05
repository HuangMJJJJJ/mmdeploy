_base_ = ["../_base_/base_static.py", "../../_base_/backends/onnxruntime.py"]
onnx_config = dict(input_shape=[800, 1344])
