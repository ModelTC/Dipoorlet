import json
import os

import numpy as np
from onnx import numpy_helper

from ..platform_settings import LAYER_HAS_WEIGHT
from .deploy_default import deploy_dispatcher

dtINT8 = 1
dtINT16 = 2
dtINT32 = 3
dtINT64 = 4
dtUINT8 = 5
dtUINT16 = 6
dtUINT32 = 7
dtUINT64 = 8
dtFLOAT16 = 9
dtFLOAT32 = 10
dtFLOAT64 = 11


@deploy_dispatcher.register("stpu")
def gen_stpu_minmax(graph, clip_val, args, **kwargs):
    param = {}

    quant_weight(clip_val, graph, param)
    quant_activation(clip_val, graph, param)
    merge_relu_layer(graph, param)
    if args.stpu_wg:
        conv_wg_layer(graph, param)
    layer_emin_state(graph, param)
    quant_bias(graph, param, args)

    with open(os.path.join(args.output_dir, 'stpu_minmax.json'), 'wt') as f:
        json.dump(param, f, indent=4)


def quant_weight(clip_val, graph, param):
    for node in graph.graph.node:
        if node.op_type in LAYER_HAS_WEIGHT:
            abs_max = max(np.abs(np.min(clip_val[node.input[1]][0])),
                          np.max(clip_val[node.input[1]][1]))
            param[node.name + '_weights'] = {
                'min': float(-abs_max),
                'max': float(abs_max)
            }


def quant_activation(clip_val, graph, param):
    for in_tensor in graph.network_inputs:
        abs_max = max(np.abs(clip_val[in_tensor][0]), clip_val[in_tensor][1])
        param[in_tensor] = {
            'min': float(-abs_max),
            'max': float(abs_max)
        }
    for node in graph.graph.node:
        for out_tensor in node.output:
            abs_max = max(np.abs(clip_val[out_tensor][0]), clip_val[out_tensor][1])
            param[out_tensor] = {
                'min': float(-abs_max),
                'max': float(abs_max)
            }


def merge_relu_layer(graph, param):
    for node in graph.graph.node:
        if node.op_type in ['Relu', 'Clip']:
            param[node.input[0]] = param[node.output[0]].copy()


def conv_wg_filter(node):
    if node.op_type != 'Conv':
        return False
    if node.get_attribute_value('group', 1) != 1:
        return False
    kernel_h, kernel_w = node.get_attribute_value('kernel_shape')
    stride_h, stride_w = node.get_attribute_value('strides')
    if (kernel_h == 3 and kernel_w == 3) and (stride_h == 1 and stride_w == 1):
        return True
    return False


def wg_weight_convt(ker):
    wu_ = np.zeros((*ker.shape[:2], 4, 4))
    g = np.array([[2, 0, 0], [1, 1, 1], [1, -1, 1], [0, 0, 2]], dtype='float32')
    for i in range(ker.shape[0]):
        for j in range(ker.shape[1]):
            wu_[i, j, :, :] = g.dot(ker[i, j, :, :]).dot(g.T)
    return max(wu_.max(), 0), min(wu_.min(), 0)


def conv_wg_layer(graph, param):
    for node in graph.graph.node:
        if conv_wg_filter(node):
            if not 'layer_' + node.name in param.keys():
                param['layer_' + node.name] = {'wg': True}
                weight = numpy_helper.to_array(graph.initializer[node.input[1]][0])
                vmax, vmin = wg_weight_convt(weight)
                abs_vmax = max(vmax, -vmin)
                param[node.name + '_weights']['max'] = float(abs_vmax)
                param[node.name + '_weights']['min'] = float(-abs_vmax)


def find_e(v):
    v_ = abs(v)
    if v_ == 0:
        return 0

    for e in range(1, 254):
        r_e = e - 127
        if (v >= 2 ** r_e) and (v < 2 ** (r_e + 1)):
            return e

    if v < 2 ** (-126):
        return 1
    return 254


def find_interp_emin(vmax, r):
    emax = find_e(vmax)
    return emax - (22 - r)


def find_pool_ave_emin(i_vmax, o_vmax, n, r):
    n = n ** .5
    vmax = max(i_vmax, o_vmax) * n
    emax = find_e(vmax)
    n_e = find_e(n * 4) - 127
    return emax - (22 - r - n_e)


def find_conv_emin(i_vmax, w_vmax, o_vmax, n, r):
    n = n ** .5
    vmax = max(n * i_vmax * w_vmax, o_vmax)
    emax = find_e(vmax)
    return emax - (12 - r)


def find_corr_emin(o_vmax, n, r):
    n = n ** .5
    vmax = o_vmax * n
    emax = find_e(vmax)
    return emax - (12 - r)


def find_softmax_emin(i_vmax, n):
    vmax = np.exp(i_vmax) * n
    emax = find_e(vmax)
    return emax - 22


def find_psroipooling_emin(i_vmax, o_vmax, r):
    # vmax = max(i_vmax, o_vmax)
    emax = find_e(i_vmax) + 6
    return emax - (22 - r)


def layer_emin_state(graph, param):
    for l in graph.graph.node:
        if l.op_type in ['Upsample', 'DynamicUpsample']:
            emin = find_interp_emin(param[l.output[0]]['max'], 2)
            param[l.output[0]]['emin'] = emin
        '''
        if l.op_type == 'Softmax':
            axis = l.get_attribute_value('axis')
            i_vmax = param[l.input[0]]['max']
            c = graph.get_tensor_shape(l.input[0])[axis]
            emin = find_softmax_emin(i_vmax, c)
            param[l.output[0]]['emin'] = emin
        if l.op_type in ['GlobalAveragePool', 'GlobalMaxPool'] + ['MaxPool', 'AveragePool']:
            if l.op_type in ['GlobalAveragePool', 'GlobalMaxPool']:
                kernel_h, kernel_w = l.get_attribute_value('kernel_shape')
                n = kernel_h * kernel_w
            else:
                n, c, kernel_h, kernel_w = graph.get_tensor_shape(l.input[0])
            i_vmax = param[l.input[0]]['max']
            o_vmax = param[l.output[0]]['max']
            emin = find_pool_ave_emin(i_vmax, o_vmax, n, 2)
            param[l.output[0]]['emin'] = emin
        '''
        if l.op_type in ['Conv', 'ConvTranspose']:
            weight_shape = graph.get_tensor_shape(l.input[1])
            n = weight_shape[1] * weight_shape[2] * weight_shape[3]
            i_vmax = param[l.input[0]]['max']
            o_vmax = param[l.output[0]]['max']
            w_vmax = param[l.name + '_weights']['max']
            emin = find_conv_emin(i_vmax, w_vmax, o_vmax, n, 2)
            param[l.output[0]]['emin'] = emin
        if l.op_type == 'Gemm':
            i_vmax = param[l.input[0]]['max']
            o_vmax = param[l.output[0]]['max']
            w_vmax = param[l.name + '_weights']['max']
            n = np.prod(graph.get_tensor_shape(l.input[0]))
            emin = find_conv_emin(i_vmax, w_vmax, o_vmax, n, 2)
            param[l.output[0]]['emin'] = emin
        '''
        if l.op_type == 'PSROIPool':
            i_vmax = param[l.input[0]]['max']
            o_vmax = param[l.output[0]]['max']
            emin = find_psroipooling_emin(i_vmax, o_vmax, 1)
            param[l.output[0]]['emin'] = emin
        '''
        if l.op_type == 'Corr':
            co = l.get_attribute_value('groups')
            n = np.prod(graph.get_tensor_shape(l.input[0])) / co
            o_vmax = param[l.output[0]]['max']
            emin = find_corr_emin(o_vmax, n, 4)
            param[l.output[0]]['emin'] = emin


def quant_bias(graph, param, args):
    for l in graph.graph.node:
        if l.op_type in ['Conv', 'ConvTranspose', 'Gemm'] and len(l.input) == 3:
            wmax = param[l.name + '_weights']['max']
            wmin = param[l.name + '_weights']['min']
            imax = param[l.input[0]]['max']
            imin = param[l.input[0]]['min']

            walpha = (wmax - wmin) / (2 ** 8 - 2)
            ialpha = (imax - imin) / (2 ** 8 - 2)
            param[l.name + '_bias'] = {'alpha': walpha * ialpha, 'zero_point': 0}
