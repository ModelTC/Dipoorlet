import numpy as np
from onnx import numpy_helper

from ..forward_net import ActivationCache
from ..quantize import quant_graph
from ..utils import ONNXGraph, logger


def update_conv_node_bias(graph_bc, node, fp_activations, q_activations):
    bias_diff = np.stack(fp_activations, axis=0) \
        - np.stack(q_activations, axis=0)
    axis = (0, 2, 3) if node.op_type == 'Conv' else (0)
    bias_diff = np.squeeze(bias_diff, axis=1).mean(axis=axis)
    if len(node.input) > 2:
        ori_bias = numpy_helper.to_array(graph_bc.initializer[node.input[2]][0])
        corrected_bias = ori_bias + bias_diff
        corrected_bias_name = graph_bc.initializer[node.input[2]][0].name
        graph_bc.set_initializer(corrected_bias_name, corrected_bias)
        graph_bc.tensor_name_shape_map[corrected_bias_name] = \
            graph_bc.tensor_name_shape_map.pop(graph_bc.initializer[node.input[2]][0].name)
        graph_bc.input.append(corrected_bias_name)
    else:
        bias = bias_diff
        bias_name = node.name + '_bias'
        graph_bc.set_initializer(bias_name, bias)
        graph_bc.tensor_name_shape_map[bias_name] = list(bias.shape)
        graph_bc.input.append(bias_name)
        for bc_node in graph_bc.graph.node:
            if bc_node.name == node.name:
                bc_node.input.append(bias_name)
                return


def bias_correction(graph, act_clip_val, weight_clip_val, args):
    bias_correction_node_type = ['Conv', 'Gemm']
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_bc = ONNXGraph()
    graph_bc.copy_from(graph)
    fp_cache = ActivationCache(graph, args)
    prev_act = None
    for node in graph.graph.node:
        if node.op_type in bias_correction_node_type:
            logger.info("Update bias for node: {}".format(node.name))
            # We should do incremental update here.
            graph_q, _ = quant_graph(graph_bc, clip_val, args)
            q_cache = ActivationCache(graph_q, args)
            if prev_act is not None:
                q_cache.activation_cache = prev_act
            _ = q_cache[node.input[0]]
            prev_act = q_cache.activation_cache.copy()
            update_conv_node_bias(graph_bc, node, fp_cache[node.output[0]], q_cache[node.output[0]])
            graph_bc.update_model()

    graph_bc.save_onnx_model('update_bias_model')
