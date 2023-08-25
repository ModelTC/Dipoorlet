import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper

__all__ = ["SparseQLayer", "L2_norm", "quant_weight_wo_roundmask", "prune_weight"]


class STE(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        input = input.round()
        return input

    @staticmethod
    def backward(self, grad_output):
        return grad_output


def quant_weight_wo_roundmask(weight, scale, q_min, q_max, per_channel):
    weight = STE.apply(weight / scale)
    if not per_channel:
        weight.clamp(q_min.item(), q_max.item())
    else:
        weight = torch.max(weight, q_min)
        weight = torch.min(weight, q_max)
    weight = weight * scale
    return weight


def create_unstruction_mask(weight, sparsity):
    revised_weight = weight.abs()
    prune_num = int(sparsity * revised_weight.numel())
    if prune_num == 0:
        threshold = revised_weight.min() - 1
    else:
        threshold = torch.topk(revised_weight.view(-1), prune_num, largest=False)[0].max()
    mask = torch.gt(revised_weight, threshold).type_as(revised_weight)
    return mask


def prune_weight(weight, sparse_info):
    if sparse_info["pattern"] == "unstruction":
        mask = create_unstruction_mask(weight, sparse_info["rate"])
    return weight * mask


def L2_norm(pred, tgt):
    return (pred - tgt).pow(2.0).sum(1).mean()


class SparseQLayer(torch.nn.Module):
    def __init__(self, node, weight, bias, qw_tensor, qi_tensor, relu_flag, type, sparse_info=None):
        super(SparseQLayer, self).__init__()
        self.qw_tensor = qw_tensor
        self.qi_tensor = qi_tensor
        self.type = type
        self.transposed = False
        self.attr_name_map = {}
        self.get_attr_name_map(node)
        if self.type == 'Conv':
            self.layer = self.build_torch_conv(node, weight, bias)
        elif self.type == 'Gemm':
            self.layer = self.build_torch_linear(node, weight, bias)
        else:
            self.layer = self.build_torch_deconv(node, weight, bias)
            self.transposed = True
        self.relu_flag = relu_flag
        if relu_flag:
            self.relu = nn.ReLU()
        # Sparse
        self.sparse_info = sparse_info

    def get_attr_name_map(self, node):
        for attr in node.attribute:
            self.attr_name_map[attr.name] = attr

    def build_torch_conv(self, node, weight, bias):
        dialiations = helper.get_attribute_value(self.attr_name_map['dilations'])
        groups = helper.get_attribute_value(self.attr_name_map['group'])
        kernel_size = helper.get_attribute_value(self.attr_name_map['kernel_shape'])
        padding = helper.get_attribute_value(self.attr_name_map['pads'])[:2]
        stride = helper.get_attribute_value(self.attr_name_map['strides'])
        o_c = weight.shape[0]
        i_c = weight.shape[1] * groups
        bias_flag = bias is not None
        conv = torch.nn.Conv2d(i_c, o_c, kernel_size, stride, padding, dialiations, groups, bias_flag)
        conv.weight.data = weight.data
        conv.weight.requires_grad = True
        if bias is not None:
            conv.bias.data = torch.from_numpy(bias).cuda().data
            conv.bias.requires_grad = True
        return conv

    def build_torch_linear(self, node, weight, bias):
        o_c = weight.shape[0]
        i_c = weight.shape[1]
        bias_flag = bias is not None
        linear = torch.nn.Linear(i_c, o_c, bias_flag)
        linear.weight.data = weight.data
        linear.weight.requires_grad = True
        if bias is not None:
            linear.bias.data = torch.from_numpy(bias).cuda().data
            linear.bias.requires_grad = True
        return linear

    def build_torch_deconv(self, node, weight, bias):
        dialiations = helper.get_attribute_value(self.attr_name_map['dilations'])
        groups = helper.get_attribute_value(self.attr_name_map['group'])
        kernel_size = helper.get_attribute_value(self.attr_name_map['kernel_shape'])
        padding = helper.get_attribute_value(self.attr_name_map['pads'])[:2]
        stride = helper.get_attribute_value(self.attr_name_map['strides'])
        if "output_padding" in self.attr_name_map:
            output_padding = helper.get_attribute_value(self.attr_name_map['output_padding'])
        else:
            output_padding = 0
        o_c = weight.shape[0]
        i_c = weight.shape[1] * groups
        bias_flag = bias is not None
        deconv = torch.nn.ConvTranspose2d(
            i_c, o_c, kernel_size, stride, padding, output_padding, groups, bias_flag, dialiations)
        deconv.weight.data = weight.data
        deconv.weight.requires_grad = True
        if bias is not None:
            deconv.bias.data = torch.from_numpy(bias).cuda().data
            deconv.bias.requires_grad = True
        return deconv

    def forward(self, x):
        s_weight = prune_weight(self.layer.weight, self.sparse_info)
        q_weight = quant_weight_wo_roundmask(s_weight,
                                self.qw_tensor['scale'], self.qw_tensor['q_min'], self.qw_tensor['q_max'],
                                self.qw_tensor['per_channel'])
        if self.transposed:
            q_weight = q_weight.transpose(0, 1)
        if self.type == 'Conv':
            x = F.conv2d(
                x,
                q_weight, self.layer.bias,
                self.layer.stride,
                self.layer.padding,
                self.layer.dilation,
                self.layer.groups)
        elif self.type == 'Gemm':
            x = F.linear(
                x,
                q_weight, self.layer.bias)
        else:
            x = F.conv_transpose2d(
                x,
                q_weight, self.layer.bias,
                self.layer.stride,
                self.layer.padding,
                self.layer.output_padding,
                self.layer.groups,
                self.layer.dilation)
        if self.relu_flag:
            x = F.relu(x)
        return x
