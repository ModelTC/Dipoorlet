import json
import os

from .deploy_default import deploy_dispatcher


@deploy_dispatcher.register("trt")
def gen_trt_range(graph, clip_val, args, **kwargs):
    for k, v in clip_val.items():
        # max(-clip_min, clip_max)
        clip_val[k] = max(-clip_val[k][0].astype(float), clip_val[k][1].astype(float))

    tensorrt_blob_json = dict()
    tensorrt_blob_json['blob_range'] = clip_val
    with open(os.path.join(args.output_dir, 'trt_clip_val.json'), 'w') as f:
        json.dump(tensorrt_blob_json, f, indent=4)
