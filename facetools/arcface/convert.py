"""
convert model to onnx again
I don't use the official model in page because its opsets version is too old, will generate a wrong quantized model

"""
import os
import numpy as np
import mxnet as mx
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from modify import modify

model_name = 'resnet100.onnx'
if not os.path.exists(model_name):
    mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx')

print('build model')
sym, arg_params, aux_params = import_model(model_name)
ctx = mx.cpu()
model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
image_size = (112, 112)
model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
model.set_params(arg_params, aux_params)
in_shapes = [(1, 3, 112, 112)]
in_types = [np.float32]
print('export model')
converted_model_path = mx.onnx.export_model(sym, [arg_params, aux_params], in_shapes, in_types, 'arcface.onnx')

print('modify model')
# https://zhuanlan.zhihu.com/p/165294876
modify('arcface.onnx', 'arcface_modified.onnx')


from onnxruntime.quantization import quantize_dynamic, QuantType

print('quantize model')
model_fp32 = 'arcface_modified.onnx'
model_quant = 'arcface-int.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
