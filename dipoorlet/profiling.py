import heapq
import math
import os

import numpy as np
import onnx
import copy
import torch.distributed as dist
from onnx import numpy_helper
from tqdm import tqdm

from .forward_net import forward_get_tensor, ActivationCache
from .platform_settings import platform_setting_table
from .quantize import DQTENSORSUFFIX, QUANT_NODE_NAME_LIST, quant_graph
from .utils import cos_similarity, abs_gap, logger


def update_node_quant_profiling(graph_q, node, fp_cache, q_cache, layer_cosine_dict, args):
    # Layer till now.
    for out_tensor in node.output:
        out_quant_node = graph_q.get_tensor_consumer(out_tensor)[0]
        q_out_tensor = out_tensor
        if not isinstance(out_quant_node, str) and out_quant_node.op_type == QUANT_NODE_NAME_LIST[-1]:
            q_out_tensor = out_tensor + DQTENSORSUFFIX
        cos_tol = 0.
        cur_batch_size = len(fp_cache[out_tensor])
        for i in range(cur_batch_size):
            cos_tol += cos_similarity(fp_cache[out_tensor][i], q_cache[q_out_tensor][i])
        if out_tensor not in layer_cosine_dict:
            layer_cosine_dict[out_tensor] = (0, 0.)
        num_till_now, cos_till_now = layer_cosine_dict[out_tensor]
        layer_cosine_dict[out_tensor] = (num_till_now + cur_batch_size,
                                         (num_till_now * cos_till_now + cos_tol) / (num_till_now + cur_batch_size))


def quantize_profiling_multipass(graph_after_wt, graph_ori, act_clip_val, weight_clip_val, args):
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_q, quant_node_list = quant_graph(graph_after_wt, clip_val, args)

    rank = dist.get_rank()
    if rank == 0:
        onnx_model = graph_q.model
        onnx.save(onnx_model, os.path.join(args.output_dir, 'quant_model.onnx'))

    layer_cosine_dict = {}
    model_cosine_dict = {}
    single = get_output_single_map(graph_after_wt)
    fp_net = graph_ori.model
    q_net = graph_q.model
    rank_data_size = math.ceil(args.data_num / args.world_size)
    rank_st = rank * rank_data_size
    rank_ed = min(rank * rank_data_size + rank_data_size, args.data_num)
    rank_data_size = rank_ed - rank_st
    if rank == 0:
        data_gen = tqdm(range(rank_st, rank_ed))
    else:
        data_gen = range(rank_st, rank_ed)
    for i in data_gen:
        fp_tensors = forward_get_tensor(graph_ori, fp_net, i, args)
        q_tensors = forward_get_tensor(graph_q, q_net, i, args)
        for node in quant_node_list:
            for tensor_name in node.output:
                if tensor_name not in layer_cosine_dict:
                    layer_cosine_dict[tensor_name] = cos_similarity(fp_tensors[tensor_name], q_tensors[tensor_name])
                else:
                    layer_cosine_dict[tensor_name] += cos_similarity(fp_tensors[tensor_name], q_tensors[tensor_name])
        for tensor_name in graph_after_wt.network_outputs:
            q_tensor_name = tensor_name
            if tensor_name + DQTENSORSUFFIX in q_tensors:
                q_tensor_name = tensor_name + DQTENSORSUFFIX
            if single[tensor_name]:
                if tensor_name not in model_cosine_dict:
                    model_cosine_dict[tensor_name] = {'res_tol': [q_tensors[q_tensor_name]], 'fp_tol': [fp_tensors[tensor_name]]}
                else:
                    model_cosine_dict[tensor_name]['res_tol'].append(q_tensors[q_tensor_name])
                    model_cosine_dict[tensor_name]['fp_tol'].append(fp_tensors[tensor_name])
            else:
                _cos = cos_similarity(fp_tensors[tensor_name], q_tensors[q_tensor_name])
                if tensor_name not in model_cosine_dict:
                    model_cosine_dict[tensor_name] = [_cos, _cos]
                else:
                    model_cosine_dict[tensor_name][0] += _cos
                    model_cosine_dict[tensor_name][1] = min(_cos, model_cosine_dict[tensor_name][1])
            if args.savefp and rank == 0:
                save_path = os.path.join(args.output_dir, 'output', tensor_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fp_tensors[tensor_name].astype(np.float32).tofile(os.path.join(save_path, 'onnx-output-{}.bin'.format(i)))

    for k, v in layer_cosine_dict.items():
        layer_cosine_dict[k] = v / rank_data_size

    for k, v in model_cosine_dict.items():
        if single[k]:
            _cos = cos_similarity(np.stack(model_cosine_dict[k]['res_tol']),
                                  np.stack(model_cosine_dict[k]['fp_tol']))
            model_cosine_dict[k] = [_cos, _cos]
        else:
            model_cosine_dict[k] = [v[0] / rank_data_size, v[1]]

    return layer_cosine_dict, model_cosine_dict, quant_node_list


def update_quant_model_cosine(graph, fp_cache, q_cache, model_cosine_dict, single, args):
    for name in graph.network_outputs:
        cos_tol = 0.
        min_cos = 1.0
        res_tol = []
        fp_tol = []
        # Output name may not change.
        if name + DQTENSORSUFFIX in q_cache.graph.network_outputs:
            q_network_output_act = q_cache[name + DQTENSORSUFFIX]
        else:
            q_network_output_act = q_cache[name]
        fp_network_output_act = fp_cache[name]
        if args.savefp:
            save_path = os.path.join(args.output_dir, 'output', name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        cur_batch_size = len(fp_network_output_act)
        for i in range(cur_batch_size):
            if not single[name]:
                _cos = cos_similarity(q_network_output_act[i], fp_network_output_act[i])
                min_cos = min(min_cos, _cos)
                cos_tol += _cos
            else:
                res_tol.append(q_network_output_act[i])
                fp_tol.append(fp_network_output_act[i])
            if args.savefp:
                fp_network_output_act[i].astype(np.float32).tofile(
                    os.path.join(save_path, 'onnx-output-{}.bin'.format(i + fp_cache.st)))
        if name not in model_cosine_dict:
            if single[name]:
                model_cosine_dict[name] = {'res_tol': [], 'fp_tol': []}
            else:
                model_cosine_dict[name] = (0, 0., 1.0)
        if single[name]:
            model_cosine_dict[name]['res_tol'].append(res_tol)
            model_cosine_dict[name]['fp_tol'].append(fp_tol)
        else:
            num_till_now, cos_till_now, min_till_now = model_cosine_dict[name]
            model_cosine_dict[name] = (num_till_now + cur_batch_size,
                                       (num_till_now * cos_till_now + cos_tol) / (num_till_now + cur_batch_size),
                                       min(min_till_now, min_cos))


def get_output_single_map(graph):
    single = {}
    for out_tensor in graph.network_outputs:
        shape = graph.get_tensor_shape(out_tensor)
        single[out_tensor] = np.prod(shape[1:]) <= 10
    return single


def show_model_ranges(graph, act_clip_val, weight_clip_val, args):
    logger.info("Model ranges:")
    ranges_all = act_clip_val.copy()
    ranges_all.update(weight_clip_val)
    for name, range in ranges_all.items():
        tensor_shape = graph.get_tensor_shape(name)
        if isinstance(range[0], np.ndarray):
            per_channel = ""
            if 'per_channel' in platform_setting_table[args.deploy]['qw_params'] and \
                    platform_setting_table[args.deploy]['qw_params']['per_channel']:
                per_channel = "per channel "
            logger.info("{:<30} Shape: {:<20} Range: {}[{:<10f} {:<10f}]".format(name, str(tensor_shape),
                                                                                 per_channel, range[0].min(), range[1].max()))
        else:
            logger.info("{:<30} Shape: {:<20} Range: [{:<10f} {:<10f}]".format(name, str(tensor_shape), range[0], range[1]))


def weight_need_perchannel(graph, args):
    qw_params = platform_setting_table[args.deploy]['qw_params']
    if 'per_channel' in qw_params and qw_params['per_channel']:
        return
    logger.info("Layer degradate by per layer: ")
    heap = []
    for node in graph.graph.node:
        if node.op_type == 'Conv':
            weight = numpy_helper.to_array(graph.initializer[node.input[1]][0])
            c_num = weight.shape[0]
            per_channel_min = np.min(weight.reshape((c_num, -1)), -1)
            per_channel_max = np.max(weight.reshape((c_num, -1)), -1)
            per_channel_range = per_channel_max - per_channel_min
            per_layer_range = np.max(weight) - np.min(weight)
            heapq.heappush(heap, (per_channel_range.mean() / per_layer_range, node.name))
    for tuple_cos_name in heapq.nsmallest(len(heap), heap):
        logger.info("{:40} ratio : {:<.5f}".format(tuple_cos_name[1], tuple_cos_name[0]))


def show_model_profiling_res(graph_after_wt, layer_cosine_dict, model_cosine_dict, quant_node_list, args):
    quant_heapq = []
    single = get_output_single_map(graph_after_wt)
    if not args.skip_prof_layer:
        for node in quant_node_list:
            logger.info(node.name)
            for out_tensor in node.output:
                logger.info("Layer  cos: {:.5f}".format(layer_cosine_dict[out_tensor]))
                heapq.heappush(quant_heapq, (layer_cosine_dict[out_tensor], node.name + '-' + out_tensor))
        logger.info("The smallest cos value of 10 layers: ")
        for tuple_cos_name in heapq.nsmallest(10, quant_heapq):
            logger.info("{:40} cos : {:<.5f}".format(tuple_cos_name[1], tuple_cos_name[0]))
    logger.info("Quant model output cos: ")
    for name in graph_after_wt.network_outputs:
        if not single[name]:
            logger.info("{:40} avgcos : {:<.5f}    mincos : {:<.5f}".format(name, model_cosine_dict[name][0],
                                                                            model_cosine_dict[name][1]))
        else:
            logger.info("{:40} tolcos : {:<.5f}".format(name, model_cosine_dict[name][0]))


def show_layerwise_profiling_res(graph_after_wt, model_error_dict, quant_node_list, args):
    quant_heapq = []
    single = get_output_single_map(graph_after_wt)
    if args.eval == "cosine":
        logger.info("Quant model output cos by skip quant a layer: ")
    elif args.eval == 'abs_gap':
        logger.info("Quant model output abs_gap by skip quant a layer: ")

    for name in graph_after_wt.network_outputs:
        logger.info(name)
        quant_heapq = []
        for node in quant_node_list:
            if not single[name]:
                if args.sort_way == 'avg':
                    heapq.heappush(quant_heapq, (model_error_dict["Skip_" + node.name + "_" + name][0], "Skip " + node.name))
                elif args.sort_way == 'min':
                    assert args.eval == "cosine", "args.eval must be 'cosine' when args.sort_way is 'min'"
                    heapq.heappush(quant_heapq, (model_error_dict["Skip_" + node.name + "_" + name][0], "Skip " + node.name))
                elif args.sort_way == 'max':
                    assert args.eval == "abs_gap", "args.eval must be 'abs_gap' when args.sort_way is 'max'"
                    heapq.heappush(quant_heapq, (model_error_dict["Skip_" + node.name + "_" + name][0], "Skip " + node.name))
            else:
                heapq.heappush(quant_heapq, (model_error_dict["Skip_" + node.name + "_" + name][0], "Skip " + node.name))
        if args.eval == "cosine":
            for tuple_cos_name in heapq.nlargest(len(quant_node_list), quant_heapq):
                logger.info("{:40} cos : {:<.5f}".format(tuple_cos_name[1], tuple_cos_name[0]))
        elif args.eval == 'abs_gap':
            for tuple_cos_name in heapq.nsmallest(len(quant_node_list), quant_heapq):
                logger.info("{:40} abs_gap : {:<.5f}".format(tuple_cos_name[1], tuple_cos_name[0]))


def update_cache(graph_ori, node, tmp_act_cache, q_act_cache):
    q_act_cache.clear_cache()
    tmp_act = {}
    for input in node.input:
        if input.endswith(DQTENSORSUFFIX):
            ori_input = input[:-1 * len(DQTENSORSUFFIX)]
            if ori_input in graph_ori.initializer:
                continue
            tmp_act[ori_input] = copy.deepcopy(tmp_act_cache[ori_input])
            tmp_act_cache.activation_cache[ori_input] = q_act_cache[ori_input]
    for output in node.output:
        q_act_cache.activation_cache[output] = copy.deepcopy(tmp_act_cache[output])
    for name in tmp_act:
        tmp_act_cache.activation_cache[name] = copy.deepcopy(tmp_act[name])
    for output in node.output:
        del tmp_act_cache.activation_cache[output]


def quantize_profiling_layerwise(graph_after_wt, graph_ori, act_clip_val, weight_clip_val, args):
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_q, quant_node_list = quant_graph(graph_after_wt, clip_val, args)

    rank = dist.get_rank()
    model_res_dict = {}
    single = get_output_single_map(graph_after_wt)
    rank_data_size = math.ceil(args.data_num / args.world_size)
    rank_st = rank * rank_data_size
    rank_ed = min(rank * rank_data_size + rank_data_size, args.data_num)
    rank_data_size = rank_ed - rank_st

    if rank == 0:
        node_gen = tqdm(range(0, len(quant_node_list)))
    else:
        node_gen = range(0, len(quant_node_list))

    fp_act_cache = ActivationCache(graph_ori, args, rank_st, rank_ed)
    tmp_act_cache = ActivationCache(graph_ori, args, rank_st, rank_ed)
    q_act_cache = ActivationCache(graph_q, args, rank_st, rank_ed)

    prefix_tensor_name_map = {}
    for skip_node_idx in node_gen:
        node = quant_node_list[skip_node_idx]
        prefix = "Skip_" + node.name + "_"
        update_cache(graph_ori, node, tmp_act_cache, q_act_cache)
        for i in range(rank_ed - rank_st):
            for tensor_name in graph_after_wt.network_outputs:
                q_tensor_name = tensor_name
                if tensor_name + DQTENSORSUFFIX in q_act_cache.graph.network_outputs:
                    q_tensor_name = tensor_name + DQTENSORSUFFIX

                fp_tensor = fp_act_cache[tensor_name][i]
                q_tensor = q_act_cache[q_tensor_name][i]

                if i == 0:
                    prefix_tensor_name_map[prefix + tensor_name] = tensor_name

                if single[tensor_name]:
                    if prefix + tensor_name not in model_res_dict:
                        model_res_dict[prefix + tensor_name] = {'res_tol': [q_tensor], 'fp_tol': [fp_tensor]}
                    else:
                        model_res_dict[prefix + tensor_name]['res_tol'].append(q_tensor)
                        model_res_dict[prefix + tensor_name]['fp_tol'].append(fp_tensor)
                else:
                    if args.eval == "cosine":
                        _res = cos_similarity(fp_tensor, q_tensor)
                    elif args.eval == "abs_gap":
                        _res = abs_gap(fp_tensor, q_tensor)

                    if prefix + tensor_name not in model_res_dict:
                        model_res_dict[prefix + tensor_name] = [_res, _res]
                    else:
                        model_res_dict[prefix + tensor_name][0] += _res
                        if args.eval == 'cosine':
                            model_res_dict[prefix + tensor_name][1] = min(_res, model_res_dict[prefix + tensor_name][1])
                        elif args.eval == 'abs_gap':
                            model_res_dict[prefix + tensor_name][1] = max(_res, model_res_dict[prefix + tensor_name][1])

    for k, v in model_res_dict.items():
        tensor_name = prefix_tensor_name_map[k]
        if single[tensor_name]:
            if args.eval == 'cosine':
                _res = cos_similarity(np.stack(model_res_dict[k]['res_tol']),
                                      np.stack(model_res_dict[k]['fp_tol']))
            elif args.eval == 'l1':
                _res = abs_gap(np.stack(model_res_dict[k]['res_tol']),
                               np.stack(model_res_dict[k]['fp_tol']))
            else:
                pass
            model_res_dict[k] = [_res, _res]
        else:
            model_res_dict[k] = [v[0] / rank_data_size, v[1]]
    return model_res_dict