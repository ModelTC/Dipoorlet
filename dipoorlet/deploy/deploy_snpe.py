import json
import os

from .deploy_default import deploy_dispatcher


@deploy_dispatcher.register("snpe")
def gen_snpe_encodings(graph, clip_val, args, **kwargs):
    activation_encodings = {}
    # We can set param_encodings actually, but we only set activation_encodings.
    # https://developer.qualcomm.com/docs/snpe/quantized_models.html
    for node in graph.graph.node:
        for idx, in_tensor in enumerate(node.input):
            if in_tensor == '':
                continue
            if in_tensor in graph.initializer:
                continue
            activation_encodings[in_tensor] = [{
                'bitwidth': 8,
                'min': float(clip_val[in_tensor][0]),
                'max': max(max(0.0, float(clip_val[in_tensor][1])), float(clip_val[in_tensor][0]) + 0.01)
            }]
    for output_tensor in graph.network_outputs:
        activation_encodings[output_tensor] = [{
            'bitwidth': 8,
            'min': float(clip_val[output_tensor][0]),
            'max': max(max(0.0, float(clip_val[output_tensor][1])), float(clip_val[output_tensor][0]) + 0.01)
        }]
    encodings = {
        'activation_encodings': activation_encodings,
        'param_encodings': {}
    }
    with open(os.path.join(args.output_dir, 'snpe_encodings.json'), 'wt') as f:
        json.dump(encodings, f, indent=4)
