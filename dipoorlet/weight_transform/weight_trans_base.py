import onnx
import torch.distributed as dist

from ..tensor_cali import tensor_calibration, find_clip_val_minmax_weight
from ..utils import (ONNXGraph, load_clip_val, logger, save_clip_val,
                     update_model_path)
from .adaround import adaround
from .bias_correction import bias_correction
from .brecq import brecq
from .update_bn import update_bn
from .weight_equalization import weight_equalization
from .sparse_quant import sparse_quant


def weight_calibration(onnx_graph, act_clip_val, weight_clip_val, args):
    '''It must be sure that after weight calibration, model/args/clip_val must be
    exactly the same on every GPUs.
    '''
    graph_after_wt = ONNXGraph()
    onnx_graph.copy_to(graph_after_wt)
    if args.bc:
        if dist.get_rank() == 0:
            bias_correction(graph_after_wt, act_clip_val, weight_clip_val, args)
        dist.barrier()
        update_model_path('update_bias_model', args)
        model = onnx.load(args.model)
        graph_after_wt = ONNXGraph(model, args.output_dir)
        # Update bias range.
        weight_clip_val = find_clip_val_minmax_weight(graph_after_wt, args)

    if args.we:
        if dist.get_rank() == 0:
            weight_equalization(graph_after_wt, args)
        dist.barrier()
        update_model_path('weight_equal_model', args)
        model = onnx.load(args.model)
        graph_after_wt = ONNXGraph(model, args.output_dir)
        act_clip_val, weight_clip_val = tensor_calibration(graph_after_wt, args)

    if args.update_bn:
        if dist.get_rank() == 0:
            update_bn(graph_after_wt, act_clip_val, weight_clip_val, args)
        dist.barrier()
        update_model_path('update_bn_model', args)
        model = onnx.load(args.model)
        graph_after_wt = ONNXGraph(model, args.output_dir)
        if dist.get_rank() == 0:
            logger.info("Re calibration...")
            act_clip_val, weight_clip_val = tensor_calibration(graph_after_wt, args)
            save_clip_val(act_clip_val, weight_clip_val, args)
        dist.barrier()
        act_clip_val, weight_clip_val = load_clip_val(args)

    if not args.sparse:
        if args.adaround:
            args.acti_quant = False
            graph_after_wt = adaround(onnx_graph, graph_after_wt, act_clip_val, weight_clip_val, args)

        if args.brecq:
            if args.drop is True:
                args.acti_quant = True
            else:
                args.acti_quant = False
            graph_after_wt = brecq(onnx_graph, graph_after_wt, act_clip_val, weight_clip_val, args)
    else:
        graph_after_wt = sparse_quant(onnx_graph, graph_after_wt, act_clip_val, weight_clip_val, args)

    return graph_after_wt, onnx_graph, act_clip_val, weight_clip_val
