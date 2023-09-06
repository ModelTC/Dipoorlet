import copy

import numpy as np
from onnx import numpy_helper

from ..utils import ONNXGraph, logger
from .utils import update_weight


def find_successor(cur_node, graph):
    # Conv -> Relu -> Conv or Conv -> Conv pattern supported.
    result = []
    out_tensor = cur_node.output[0]
    nxt_node = graph.get_tensor_consumer(out_tensor)
    for node in nxt_node:
        if isinstance(node, str):
            return []
        if node.op_type in ['Relu', 'PRelu']:
            relu_out = node.output[0]
            nxt_nxt_node = graph.get_tensor_consumer(relu_out)
            for _node in nxt_nxt_node:
                if not isinstance(_node, str) and _node.op_type == 'Conv':
                    result.append(_node)
                else:
                    return []
        elif node.op_type == 'Conv':
            result.append(node)
        else:
            return []
    return result


def node_has_equalized(graph, node):
    # Helper function for other algos.
    return len(find_successor(node, graph)) == 1


def weight_equalization(graph, args):
    graph_we = ONNXGraph()
    graph_we.copy_from(graph)

    for node in graph_we.graph.node:
        if node.op_type == 'Conv':
            succ = find_successor(node, graph_we)
            if len(succ) != 1:
                continue
            iter = 1
            while True:
                weight_first = numpy_helper.to_array(graph_we.initializer[node.input[1]][0])
                new_weight_first = copy.deepcopy(weight_first)
                if len(node.input) == 3:
                    bias_first = numpy_helper.to_array(graph_we.initializer[node.input[2]][0])
                    new_bias_first = copy.deepcopy(bias_first)
                next_node = succ[0]
                weight_second = numpy_helper.to_array(graph_we.initializer[next_node.input[1]][0])
                new_weight_second = copy.deepcopy(weight_second)
                num_group = weight_first.shape[0] // weight_second.shape[1]
                logger.info('Cross Layer WE: {} --- {} Groups: {} Iter: {}'.format(node.name, next_node.name, num_group, iter))
                group_channels_i = weight_first.shape[0] // num_group
                group_channels_o = weight_second.shape[0] // num_group
                for g in range(num_group):
                    c_start_i = g * group_channels_i
                    c_end_i = (g + 1) * group_channels_i
                    weight_first_group = weight_first[c_start_i:c_end_i]
                    c_start_o = g * group_channels_o
                    c_end_o = (g + 1) * group_channels_o
                    weight_second_group = weight_second[c_start_o:c_end_o]
                    for ii in range(weight_second_group.shape[1]):
                        range_1 = np.abs(weight_first_group)[ii].max()
                        range_2 = np.abs(weight_second_group)[:, ii].max()
                        if range_1 < 1e-6:
                            range_1 = 0.
                        if range_2 < 1e-6:
                            range_2 = 0.
                        s = range_1 / np.sqrt(range_1 * range_2)
                        if np.isinf(s) or np.isnan(s):
                            s = 1.0
                        new_weight_first[c_start_i + ii] /= s
                        new_weight_second[c_start_o:c_end_o, ii] *= s
                        if len(node.input) == 3:
                            new_bias_first[c_start_i + ii] /= s

                if converged([weight_first, weight_second], [new_weight_first, new_weight_second]):
                    break
                iter += 1
                # Update layer.
                update_weight(graph_we, new_weight_first, node.input[1])
                graph_we.update_model()
                update_weight(graph_we, new_weight_second, next_node.input[1])
                graph_we.update_model()
                if len(node.input) == 3:
                    update_weight(graph_we, new_bias_first, node.input[2])
                    graph_we.update_model()
    graph_we.save_onnx_model('weight_equal_model')


def converged(cur_weight, prev_weight, threshold=1e-4):
    norm_sum = 0
    norm_sum += np.linalg.norm(cur_weight[0] - prev_weight[0])
    norm_sum += np.linalg.norm(cur_weight[1] - prev_weight[1])
    return norm_sum < threshold
