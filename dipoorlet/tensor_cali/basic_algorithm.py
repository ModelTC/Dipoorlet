import numpy as np

from ..forward_net import *
from ..platform_settings import LAYER_HAS_WEIGHT
from ..utils import dispatch_functool, logger


@dispatch_functool
def tensor_cali_dispatcher(*args, **kwargs):
    logger.info("Calibration Algorithm Not Found!")


@tensor_cali_dispatcher.register('minmax')
def find_clip_val_minmax(onnx_graph, args, **kwargs):
    if args.optim_transformer:
        stats_min_max = forward_get_minmax_transformer(onnx_graph, args)
    else:
        stats_min_max = forward_get_minmax(onnx_graph, args)
    clip_val = {}
    for name, tensor in stats_min_max.items():
        clip_val[name] = [np.min(tensor['min']), np.max(tensor['max'])]
    return clip_val


@tensor_cali_dispatcher.register('hist')
def find_clip_val_hist(onnx_graph, args, store_stats=None, **kwargs):
    if store_stats:
        stats_min_max = store_stats['minmax']
        act_stats_hist = store_stats['hist']
    else:
        if args.optim_transformer:
            stats_min_max = forward_get_minmax_transformer(onnx_graph, args)
            act_stats_hist = forward_get_hist_transformer(onnx_graph, stats_min_max, args)
        else:
            stats_min_max = forward_get_minmax(onnx_graph, args)
            act_stats_hist = forward_get_hist(onnx_graph, stats_min_max, args)
        for name, hist in act_stats_hist.items():
            act_stats_hist[name] = np.stack(hist).sum(0)
    clip_val = {}
    for name, hist in act_stats_hist.items():
        hist = hist.astype(np.float32) / hist.sum()
        data_max = max(-np.min(stats_min_max[name]['min']), np.max(stats_min_max[name]['max']))
        accum = 0
        for i in range(len(hist)):
            accum += hist[i]
            if accum >= args.threshold:
                clip_value = (i + 0.5) * (data_max / args.bins)
                clip_val[name] = [max(-clip_value, np.min(stats_min_max[name]['min'])),
                                  min(clip_value, np.max(stats_min_max[name]['max']))]
                break
        if name not in clip_val:
            clip_val[name] = [np.min(stats_min_max[name]['min']),
                              np.max(stats_min_max[name]['max'])]
    return clip_val


@tensor_cali_dispatcher.register('mse')
def find_clip_val_octav(onnx_graph, args, **kwargs):
    if args.optim_transformer:
        optimal_s = forward_net_octav_transformer(onnx_graph, args)
    else:
        optimal_s = forward_net_octav(onnx_graph, args)
    clip_val = {}
    for k, v in optimal_s.items():
        data_max = np.array(v['max']).max()
        data_min = np.array(v['min']).min()
        clip_val[k] = [max(data_min, -np.array(v['optimal_s']).mean()),
                       min(data_max, np.array(v['optimal_s']).mean())]
    return clip_val


def find_clip_val_minmax_weight(onnx_graph, args):
    weight_tensor = {}
    need_transpose = []
    for node in onnx_graph.graph.node:
        if node.op_type in LAYER_HAS_WEIGHT:
            for in_tensor in node.input[1:]:
                weight_tensor[in_tensor] = onnx_graph.get_initializer(in_tensor)
            if node.op_type == 'ConvTranspose':
                need_transpose.append(node.input[1])
    weight_clip_val = {}
    for name, tensor in weight_tensor.items():
        # BN tracked param do not have shape.
        if len(tensor.shape) < 1:
            continue
        if name in need_transpose:
            tensor = tensor.transpose([1, 0, 2, 3])
        c_num = tensor.shape[0]
        weight_clip_val[name] = [np.min(tensor.reshape((c_num, -1)), -1),
                                 np.max(tensor.reshape((c_num, -1)), -1)]
    return weight_clip_val
