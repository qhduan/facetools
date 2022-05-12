"""
modify model because bug of mxnet/onnx
reference: https://zhuanlan.zhihu.com/p/165294876
"""

import onnx


def modify(input_path, output_path):
    model = onnx.load(input_path)
    onnx.checker.check_model(model)
    graph = model.graph

    node_to_change = []
    for node in graph.node:
        if node.op_type == 'PRelu':
            # print(node.name)
            # print(node.op_type)
            # print(node.input)
            # print(node.output)
            for _input in node.input:
                if 'gamma' in _input:
                    if _input not in node_to_change:
                        node_to_change.append(_input)

    input_map = {
        x.name: x
        for x in graph.input
    }

    new_initializers = []
    for initializer in graph.initializer:
        # Data type:
        # https://github.com/onnx/onnx/blob/rel-1.9.0/onnx/onnx.proto
        if initializer.name in node_to_change:
            # breakpoint()
            ni = onnx.helper.make_tensor(
                initializer.name,
                initializer.data_type,
                [initializer.dims[0], 1, 1],
                initializer.float_data
            )
            new_initializers.append(ni)
            graph.initializer.remove(initializer)
    graph.initializer.extend(new_initializers)


    new_nvs = []
    for input_name in node_to_change:
        dim = input_map[input_name].type.tensor_type.shape.dim
        if len(dim) == 1:
            input_dim_val = dim[0].dim_value
            # print('change', input_name, input_dim_val)
            new_nv = onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, [input_dim_val, 1, 1]
            )
            new_nvs.append(new_nv)
            graph.input.remove(input_map[input_name])

    graph.input.extend(new_nvs)

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
