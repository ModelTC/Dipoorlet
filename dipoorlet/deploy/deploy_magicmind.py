import json
import os

import numpy as np

from .deploy_default import deploy_dispatcher


@deploy_dispatcher.register("magicmind")
def gen_magicmind_proto(graph, clip_val, args, **kwargs):
    cambricom_quant_param = {}
    for k, v in clip_val.items():
        cambricom_quant_param[k] = {
            "min": float(np.min(clip_val[k][0])),
            "max": float(np.max(clip_val[k][1]))
        }
    blob_range_json = dict()
    blob_range_json['blob_range'] = cambricom_quant_param
    with open(os.path.join(args.output_dir, 'magicmind_quant_param.json'), 'wt') as f:
        json.dump(blob_range_json, f, indent=4)
