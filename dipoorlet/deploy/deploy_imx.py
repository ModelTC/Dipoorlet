import json
import os
import numpy as np

from .deploy_default import deploy_dispatcher


@deploy_dispatcher.register("imx")
def gen_imx_range(graph, clip_val, args, **kwargs):
    bit_width = 8
    for k, v in clip_val.items():
        clip_max = np.max(np.abs(clip_val[k]), axis=0)
        q_max = [2 ** (bit_width - 1) - 1]
        scale = np.array(clip_max) / q_max
        if np.any(scale == 0):
            scale = np.where(scale == 0, 1., scale)

        scale = 2 ** np.round(np.log2(scale))
        clip_val[k] = scale.tolist()
    imx_blob_json = dict()
    imx_blob_json['blob_range'] = clip_val
    with open(os.path.join(args.output_dir, 'imx_clip_val.json'), 'w') as f:
        json.dump(imx_blob_json, f, indent=4)
