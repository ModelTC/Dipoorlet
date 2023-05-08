import json
import os.path as osp

import numpy as np
import yaml

from ..platform_settings import LAYER_HAS_WEIGHT
from .deploy_default import deploy_dispatcher


def step_zeropoint(clip_val):
    ret = dict()
    range_min = min(0, np.min(clip_val[0]))
    range_max = max(0, np.max(clip_val[1]))
    step = (range_max - range_min) / 255.
    if step == 0.0:
        step = 1.0 / 255.
    zero_point = round(-range_min / step)
    ret.update({'scale': [float(step)], 'zero_point': [int(zero_point)]})
    return ret


@deploy_dispatcher.register("rv")
def gen_rv_yaml(graph, clip_val, args, **kwargs):
    def gen1126(graph, clip_val, args, **kwargs):
        res = {'customized_quantize_layers': {},
               'quantize_parameters': {}}
        # Pass concat qparam to input
        for node in graph.graph.node:
            if node.op_type == 'Concat':
                for input_tensor in node.input:
                    clip_val[input_tensor][0] = clip_val[node.output[0]][0]
                    clip_val[input_tensor][1] = clip_val[node.output[0]][1]
            next_node = graph.get_tensor_consumer(node.output[0])
        for i in graph.network_inputs:
            tensor_dict = {
                'dtype': 'asymmetric_affine',
                'method': 'layer',
                'max_value': [max(0., float(clip_val[i][1]))],
                'min_value': [min(0., float(clip_val[i][0]))],
                'qtype': 'u8'
            }
            key = f'@{i}:out0'
            res['quantize_parameters'][key] = tensor_dict
            res['quantize_parameters'][key].update(step_zeropoint(clip_val[i]))
        for node in graph.graph.node:
            # Sigmoid input has specific range -6.3-6.3
            next_node = graph.get_tensor_consumer(node.output[0])
            if len(next_node) == 1 and not isinstance(next_node[0], str) and next_node[0].op_type == 'Sigmoid':
                continue
            if node.op_type in LAYER_HAS_WEIGHT:
                for idx, input_tensor in enumerate(node.input[1:]):
                    qtype = 'u8'
                    if idx == 0:
                        # weight
                        key = f'@{node.name}:weight'
                        tensor_dict = {
                            'dtype': 'asymmetric_affine',
                            'method': 'layer',
                            'max_value': [max(0.0, float(np.max(clip_val[input_tensor][1])))],
                            'min_value': [min(0.0, float(np.min(clip_val[input_tensor][0])))],
                            'qtype': qtype
                        }
                        tensor_dict.update(step_zeropoint(clip_val[input_tensor]))
                    elif idx == 1:
                        key = f'@{node.name}:bias'
                        qtype = 'i32'
                        acts = step_zeropoint(clip_val[node.input[0]])['scale']
                        ws = step_zeropoint(clip_val[node.input[1]])['scale']
                        tensor_dict = {
                            'dtype': 'asymmetric_affine',
                            'method': 'layer',
                            'max_value': [],
                            'min_value': [],
                            'zero_point': [0],
                            'scale': [ws[0] * acts[0]],
                            'qtype': qtype
                        }
                    else:
                        print("We meet unsupported node{}, skip.".format(node.name))
                    res['quantize_parameters'][key] = tensor_dict
            for idx, output_tensor in enumerate(node.output):
                tensor_dict = {
                    'dtype': 'asymmetric_affine',
                    'method': 'layer',
                    'max_value': [max(0., float(np.max(clip_val[output_tensor][1])))],
                    'min_value': [min(0., float(np.min(clip_val[output_tensor][0])))],
                    'qtype': 'u8'
                }
                key = f'@{node.name}:out{idx}'
                res['quantize_parameters'][key] = tensor_dict
                res['quantize_parameters'][key].update(step_zeropoint(clip_val[output_tensor]))
            # We need to merge relu.
            if node.op_type == 'Relu':
                prev_node = graph.get_tensor_producer(node.input[0])
                for prev_key in res['quantize_parameters']:
                    if prev_node.name in prev_key and 'out' in prev_key:
                        res['quantize_parameters'][prev_key] = res['quantize_parameters'][key]
            # We need to merge BatchNorm and Scale.
            if node.op_type == 'CaffeScale':
                prev_node = graph.get_tensor_producer(node.input[0])
                if prev_node.op_type == 'CaffeBatchNorm':
                    for prev_key in res['quantize_parameters']:
                        if prev_node.name in prev_key and 'out' in prev_key:
                            res['quantize_parameters'][prev_key] = res['quantize_parameters'][key]
                    del res['quantize_parameters'][key]
        with open(osp.join(args.output_dir, 'rv_quantized_param.yaml'), 'w') as f:
            f.write(yaml.dump(res))
        with open(osp.join(args.output_dir, 'rv_quantized_param.json'), 'w') as f:
            json.dump(res, f, indent=4)

    def gen3568(graph, clip_val, args, **kwargs):
        res = {'custom_quantize_layers': {},
               'quantize_parameters': {}}
        # Pass concat qparam to input
        for node in graph.graph.node:
            if node.op_type == 'Concat':
                for input_tensor in node.input:
                    clip_val[input_tensor][0] = clip_val[node.output[0]][0]
                    clip_val[input_tensor][1] = clip_val[node.output[0]][1]
            next_node = graph.get_tensor_consumer(node.output[0])
        for i in graph.network_inputs:
            tensor_dict = {
                'max': [max(0., float(clip_val[i][1]))],
                'min': [min(0., float(clip_val[i][0]))],
            }
            key = f'{i}'
            res['quantize_parameters'][key] = tensor_dict
        for node in graph.graph.node:
            # Sigmoid input has specific range -6.3-6.3
            next_node = graph.get_tensor_consumer(node.output[0])
            if len(next_node) == 1 and not isinstance(next_node[0], str) and next_node[0].op_type == 'Sigmoid':
                continue
            if node.op_type in LAYER_HAS_WEIGHT:
                for idx, input_tensor in enumerate(node.input[1:]):
                    if idx == 0:
                        # weight
                        key = f'{node.name}_W'
                        tensor_dict = {
                            'max': [max(0.0, float(np.max(clip_val[input_tensor][1])))],
                            'min': [min(0.0, float(np.min(clip_val[input_tensor][0])))],
                        }
                    elif idx == 1:
                        key = f'{node.name}_b'
                        max_val = np.max(clip_val[node.input[2]])
                        min_val = np.min(clip_val[node.input[2]])
                        tensor_dict = {
                            'max': [float(max(abs(max_val), abs(min_val)))],
                            'min': [float(-max(abs(max_val), abs(min_val)))],
                        }
                    else:
                        print("We meet unsupported node{}, skip.".format(node.name))
                    res['quantize_parameters'][key] = tensor_dict
            for idx, output_tensor in enumerate(node.output):
                tensor_dict = {
                    'max': [max(0., float(np.max(clip_val[output_tensor][1])))],
                    'min': [min(0., float(np.min(clip_val[output_tensor][0])))],
                }
                key = f'{output_tensor}'
                res['quantize_parameters'][key] = tensor_dict
            # We need to merge relu.
            if node.op_type == 'Relu':
                prev_key = node.input[0]
                res['quantize_parameters'][prev_key] = res['quantize_parameters'][key]
            # We need to merge BatchNorm and Scale.
            if node.op_type == 'CaffeScale':
                prev_node = graph.get_tensor_producer(node.input[0])[0]
                if prev_node.op_type == 'CaffeBatchNorm':
                    prev_key = node.input[0]
                    res['quantize_parameters'][prev_key] = res['quantize_parameters'][key]
                    del res['quantize_parameters'][key]
        with open(osp.join(args.output_dir, 'rk_quantized_param.yaml'), 'w') as f:
            f.write(yaml.dump(res))
        with open(osp.join(args.output_dir, 'rk_quantized_param.json'), 'w') as f:
            json.dump(res, f, indent=4)

    gen1126(graph, clip_val, args, **kwargs)
    gen3568(graph, clip_val, args, **kwargs)
