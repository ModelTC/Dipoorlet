import json
import os.path as osp

from ..platform_settings import platform_setting_table
from .deploy_default import deploy_dispatcher

ATLAS_QUANT_LAYER = platform_setting_table['atlas']['quant_nodes']


def get_step_zeropoint(clip_val):
    ret = dict()
    range_min = min(0, clip_val[0])
    range_max = max(0, clip_val[1])
    step = (range_max - range_min) / 255.
    # Zero point range [-128, 127] to support sym/asym in same time.
    if step == 0.0:
        step = 1.0
    zero_point = round(-range_min / step) - 128
    ret.update({'scale': step, 'offset': int(zero_point)})
    return ret


@deploy_dispatcher.register("atlas")
def gen_atlas_quant_param(graph, clip_val, args, **kwargs):
    res = {}
    for node in graph.graph.node:
        if node.op_type in ATLAS_QUANT_LAYER:
            tensor_name = node.input[0]
            res[tensor_name] = get_step_zeropoint(clip_val[tensor_name])

    with open(osp.join(args.output_dir, 'atlas_quant_param.json'), 'w') as f:
        json.dump(res, f, indent=4)
