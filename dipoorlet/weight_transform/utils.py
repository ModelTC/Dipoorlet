import numpy as np
import torch
from onnx import numpy_helper

from ..quantize import get_qnode_by_param

LEARNABLE_LAYER_TYPES = ['Conv', 'Gemm', 'ConvTranspose']
__all__ = ['LEARNABLE_LAYER_TYPES', 'follow_relu', 'following_relu', 'update_weight', 'get_quant_tensor', 'get_block_from_first']


def follow_relu(graph, node):
    conv_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(conv_out)
    return len(nxt_node) == 1 and not isinstance(nxt_node[0], str) and nxt_node[0].op_type == 'Relu'


def following_relu(graph, node):
    conv_out = node.output[0]
    nxt_node = graph.get_tensor_consumer(conv_out)
    assert nxt_node[0].op_type == 'Relu'
    return nxt_node[0]


def update_weight(graph, weight_tensor, weight_name):
    name = graph.initializer[weight_name][0].name
    graph.set_initializer(name, weight_tensor)


def get_quant_tensor(weight_shape, param, weight_range):
    q_nodes, q_min, q_max = get_qnode_by_param(param, 'tmp', weight_shape, weight_range)
    scale = None
    for init in q_nodes.initializer:
        if init.name == 'tmp_scale':
            scale = numpy_helper.to_array(init)

    if 'per_channel' in param and param['per_channel']:
        c_num = weight_shape[0]
        scale = torch.from_numpy(np.array(scale).astype(np.float32)).view(
            [c_num, *[1] * (len(weight_shape) - 1)]).cuda()
        q_min = torch.from_numpy(np.array(q_min).astype(np.float32)).view(
            [c_num, *[1] * (len(weight_shape) - 1)]).cuda()
        q_max = torch.from_numpy(np.array(q_max).astype(np.float32)).view(
            [c_num, *[1] * (len(weight_shape) - 1)]).cuda()
    else:
        scale = torch.from_numpy(np.array(scale).astype(np.float32)).cuda()
        q_min = torch.from_numpy(np.array(q_min).astype(np.float32)).cuda()
        q_max = torch.from_numpy(np.array(q_max).astype(np.float32)).cuda()
    scale.requires_grad = False
    q_min.requires_grad = False
    q_max.requires_grad = False
    return scale, q_min, q_max


def get_block_from_first(graph, node, args):
    res = [node]
    while True:
        next_node = graph.get_tensor_consumer(node.output[0])
        if len(next_node) != 1 or isinstance(next_node[0], str) or next_node[0].op_type not in LEARNABLE_LAYER_TYPES + ['Relu']:
            return res
        if next_node[0].op_type != 'Relu':
            res.append(next_node[0])
            # We set max len=3.
            if len(res) == 3:
                return res
        node = next_node[0]
