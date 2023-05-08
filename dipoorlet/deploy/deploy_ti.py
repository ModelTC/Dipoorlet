import json
import os

from .deploy_default import deploy_dispatcher


@deploy_dispatcher.register("ti")
def gen_ti_json(graph, clip_val, args, **kwargs):
    # Deploy for raw range.
    with open(os.path.join(args.output_dir, 'ti_blob_range.txt'), 'w') as f:
        for k, v in clip_val.items():
            f.write('{} {} {}\n'.format(k, v[0], v[1]))
    # Deploy for nart.
    ti_blob_range = dict()
    for k, v in clip_val.items():
        clip_val[k] = [float(_v) for _v in v]
    ti_blob_range['blob_range'] = clip_val
    with open(os.path.join(args.output_dir, 'ti_blob_range.json'), 'w') as f:
        json.dump(ti_blob_range, f, indent=4)
