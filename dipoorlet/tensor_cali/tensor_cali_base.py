from .basic_algorithm import *


def tensor_calibration(onnx_graph, args):
    weight_clip_val = find_clip_val_minmax_weight(onnx_graph, args)
    act_clip_val = tensor_cali_dispatcher(args.act_quant, onnx_graph, args)
    return act_clip_val, weight_clip_val
