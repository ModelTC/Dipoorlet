import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import helper

__all__ = ["nnie_rest_init", "quant_acti", "quant_weight", "quant_weight_nnie", "quant_acti_nnie",
           "adaround_reg", "AdaQLayer", "L2_norm"]


def nnie_rest_init(weight):
    def zero_point(x, z):
        out = 16 * torch.log2(x) - z
        out = torch.round(out)
        return out
    max_value = weight.abs().max()
    z = zero_point(max_value, 127).cuda()
    pos_idx = weight > 2 ** ((z - 16) / 16)
    neg_idx = weight < - 2 ** ((z + 1 - 16) / 16)
    zero_idx = (weight >= - 2 ** ((z + 1 - 16) / 16)) & (weight < 2 ** ((z - 16) / 16))
    res = weight.clone()
    res[zero_idx] = 0
    res[pos_idx] = (16 * torch.log2(weight[pos_idx]) - z) - torch.floor(16 * torch.log2(weight[pos_idx]) - z)
    res[neg_idx] = (16 * torch.log2(-weight[neg_idx]) - z) - torch.floor(16 * torch.log2(-weight[neg_idx]) - z)
    return res


def quant_acti(x, scale, q_min, q_max, prob):
    x_ori = x
    x = (x / scale).round()
    x = torch.max(x, q_min)
    x = torch.min(x, q_max)
    x *= scale
    if prob < 1.0:
        x = torch.where(torch.rand_like(x) < prob, x, x_ori)
    return x


def quant_weight(weight, round_mask, scale, q_min, q_max, per_channel, soft=True):
    if soft:
        weight = (weight / scale).floor() + adaround_reg().rectified_sigmoid(round_mask)
    else:
        weight = (weight / scale).floor() + (round_mask >= 0).float()
    if not per_channel:
        weight.clamp(q_min.item(), q_max.item())
    else:
        weight = torch.max(weight, q_min)
        weight = torch.min(weight, q_max)
    weight = weight * scale
    return weight


def quant_weight_nnie(weight, round_mask, soft=True):
    def zero_point(x, z):
        out = 16 * torch.log2(x) - z
        out = torch.round(out)
        return out
    max_value = weight.abs().max()
    z = zero_point(max_value, 127).cuda()
    pos_idx = weight > 2 ** ((z - 16) / 16)
    neg_idx = weight < - 2 ** ((z + 1 - 16) / 16)
    zero_idx = (weight >= - 2 ** ((z + 1 - 16) / 16)) & (weight <= 2 ** ((z - 16) / 16))
    res = weight.clone()
    res[zero_idx] = 0
    if soft:
        wp = torch.floor(16 * torch.log2(weight[pos_idx]) - z) + adaround_reg().rectified_sigmoid(round_mask[pos_idx])
        wn = torch.floor(16 * torch.log2(-weight[neg_idx]) - z) + adaround_reg().rectified_sigmoid(round_mask[neg_idx])
    else:
        wp = torch.floor(16 * torch.log2(weight[pos_idx]) - z) + (round_mask[pos_idx] >= 0).float()
        wn = torch.floor(16 * torch.log2(-weight[neg_idx]) - z) + (round_mask[neg_idx] >= 0).float()
    res[pos_idx] = 2 ** ((torch.clamp(wp, 0, 127) + z) / 16)
    res[neg_idx] = - 2 ** ((torch.clamp(wn, 1, 127) + z) / 16)
    return res


def quant_acti_nnie(acti, max_value, prob):
    def zero_point(x, z):
        out = 16 * torch.log2(x) - z
        out = torch.round(out)
        return out
    z = zero_point(max_value, 127).cuda()
    pos_idx = acti > 2 ** ((z - 16) / 16)
    neg_idx = acti < - 2 ** ((z + 1 - 16) / 16)
    zero_idx = (acti >= -2 ** ((z + 1 - 16) / 16)) & (acti <= 2 ** ((z - 16) / 16))
    acti_q = acti.clone()
    ap = torch.round(16 * torch.log2(acti[pos_idx]) - z)
    an = torch.round(16 * torch.log2(-acti[neg_idx]) - z)
    acti_q[zero_idx] = 0.
    acti_q[pos_idx] = 2 ** ((torch.clamp(ap, 0, 127) + z) / 16)
    acti_q[neg_idx] = - 2 ** ((torch.clamp(an, 1, 127) + z) / 16)
    if prob < 1.0:
        acti_q = torch.where(torch.rand_like(acti) < prob, acti_q, acti)
    return acti_q


class adaround_reg(nn.Module):
    def __init__(self, max_iter=10000, zeta=1.1, gamma=-0.1, alpha=0.01, beta=20):
        self.zeta = zeta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.temp_anneal = TempDecay(max_iter)
        super().__init__()

    def rectified_sigmoid(self, round_mask):
        return ((self.zeta - self.gamma) * torch.sigmoid(round_mask) + self.gamma).clamp(0, 1)

    def forward(self, round_mask, iter):
        self.beta = self.temp_anneal(iter)
        return self.alpha * (1 - torch.pow((self.rectified_sigmoid(round_mask) - 0.5).abs() * 2, self.beta)).sum()


def L2_norm(pred, tgt):
    return (pred - tgt).pow(2.0).sum(1).mean()


class TempDecay:
    def __init__(self, t_max, rel_start_decay=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b
        self.type = type

    def __call__(self, t):
        if t < self.start_decay:
            return 0.0
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))


class AdaQLayer(torch.nn.Module):
    def __init__(self, node, weight, bias, rest, reg, qw_tensor, qi_tensor, relu_flag, type, acti_quant):
        super(AdaQLayer, self).__init__()
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
        # Init alpha.
        rest = -torch.log((reg.zeta - reg.gamma) / (rest - reg.gamma) - 1)
        self.round_mask = torch.nn.Parameter(rest.cuda(), True)
        # Init drop ratio.
        self.drop_ratio = 0.5
        # Init activation quantization mode
        self.acti_quant = acti_quant

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
        conv.weight.requires_grad = False
        if bias is not None:
            conv.bias.data = torch.from_numpy(bias).cuda().data
            conv.bias.requires_grad = False
        return conv

    def build_torch_linear(self, node, weight, bias):
        o_c = weight.shape[0]
        i_c = weight.shape[1]
        bias_flag = bias is not None
        linear = torch.nn.Linear(i_c, o_c, bias_flag)
        linear.weight.data = weight.data
        linear.weight.requires_grad = False
        if bias is not None:
            linear.bias.data = torch.from_numpy(bias).cuda().data
            linear.bias.requires_grad = False
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
        deconv.weight.requires_grad = False
        if bias is not None:
            deconv.bias.data = torch.from_numpy(bias).cuda().data
            deconv.bias.requires_grad = False
        return deconv

    def forward(self, x):
        if self.qw_tensor['type'] == 'Linear':
            q_weight = quant_weight(self.layer.weight, self.round_mask,
                                    self.qw_tensor['scale'], self.qw_tensor['q_min'], self.qw_tensor['q_max'],
                                    self.qw_tensor['per_channel'])
            if self.transposed:
                q_weight = q_weight.transpose(0, 1)
        else:
            q_weight = quant_weight_nnie(self.layer.weight, self.round_mask)
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
        if self.acti_quant and self.qi_tensor['type'] == 'Linear':
            x = quant_acti(x, self.qi_tensor['scale'], self.qi_tensor['q_min'],
                           self.qi_tensor['q_max'], self.drop_ratio)
        elif self.acti_quant and self.qi_tensor['type'] == 'NNIE':
            x = quant_acti_nnie(x, self.qi_tensor['max_value'], self.drop_ratio)
        return x
