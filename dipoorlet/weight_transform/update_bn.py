import copy

import numpy as np
from onnx import numpy_helper

from ..forward_net import ActivationCache
from ..quantize import quant_graph
from ..tensor_cali import tensor_calibration
from ..utils import ONNXGraph, logger


def update_bn_node(graph, node, in_tensor, momentum=0.9):
    running_mean = numpy_helper.to_array(graph.initializer[node.input[3]][0])
    running_var = numpy_helper.to_array(graph.initializer[node.input[4]][0])
    for i in range(len(in_tensor)):
        running_mean = momentum * running_mean + (1.0 - momentum) * np.mean(in_tensor[i], axis=(0, 2, 3))
        running_var = momentum * running_var + (1.0 - momentum) * np.std(in_tensor[i], axis=(0, 2, 3))
    update_mean = running_mean
    update_var = running_var
    update_mean_name = graph.initializer[node.input[3]][0].name
    update_var_name = graph.initializer[node.input[4]][0].name
    graph.set_initializer(update_mean_name, update_mean)
    graph.set_initializer(update_var_name, update_var)


def update_bn_multipass(graph, act_clip_val, weight_clip_val, args):
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_tuning_bn = ONNXGraph()
    graph_tuning_bn.copy_from(graph)
    bn_list = []
    for node in graph_tuning_bn.graph.node:
        if node.op_type == "BatchNormalization":
            bn_list.append(node)
    pre_act_cache = None
    for node in bn_list:
        logger.info("Update BN for node: {}".format(node.name))
        graph_q, quant_node_list = quant_graph(graph_tuning_bn, clip_val, args)
        q_cache = ActivationCache(graph_q, args)
        if pre_act_cache:
            q_cache.activation_cache = pre_act_cache
        update_bn_node(graph_tuning_bn, node, q_cache[node.input[0]])
        pre_act_cache = copy.deepcopy(q_cache.activation_cache)
        graph_tuning_bn.update_model()
    graph_tuning_bn.save_onnx_model('update_bn_model')

    act_clip_val, weight_clip_val = tensor_calibration(graph_tuning_bn, args)
    return graph_tuning_bn, act_clip_val, weight_clip_val


def update_bn_onepass(graph, act_clip_val, weight_clip_val, args):
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_tuning_bn = copy.deepcopy(graph)
    graph_q, quant_node_list = quant_graph(graph, clip_val, args)
    q_cache = ActivationCache(graph_q, args)
    for node in graph_tuning_bn.nodes:
        if node.op_type == "BatchNormalization":
            logger.info("Update BN for node: {}".format(node.name))
            update_bn_node(graph_tuning_bn, node, q_cache[node.input[0]])

    graph_tuning_bn.update_model()
    graph_tuning_bn.save_onnx_model('update_bn_model')


update_bn = update_bn_multipass
