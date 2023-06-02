import copy
import time
from collections import OrderedDict

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.distributed as dist
from onnx.helper import make_graph, make_model
from onnx.helper import make_tensor_value_info as mtvi
from tqdm import tqdm

from .platform_settings import platform_setting_table
from .quantize import QUANT_NODE_NAME_LIST
from .utils import ONNXGraph, logger

ort.set_default_logger_severity(3)


class ActivationCache(object):
    # We assume get tensor by sequence.
    def __init__(self, graph, args, st=None, ed=None):
        self.graph = copy.deepcopy(graph)
        self.graph_list = []
        self.ref_cnt = {}
        self.name_to_net = {}
        self.name_to_graph_id = {}
        self.activation_cache = {}
        self.args = args
        self.st = st
        self.ed = ed
        self.providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
        if len(self.graph.value_name_type_map) == 0:
            self.graph.get_inp_oup()
            self.graph.get_shape_type()
        self.fetch_input()
        self._split_network()
        self.fill_ref_cnt()

    def reset(self):
        self.activation_cache.clear()
        self.fetch_input()
        self.fill_ref_cnt()

    def fetch_input(self, in_tensor=None):
        if in_tensor is None:
            # Means We are initializing.
            for name in self.graph.network_inputs:
                self.activation_cache[name] = []
            if self.st is None:
                self.st = 0
                self.ed = self.args.data_num
            for data in input_data_generator(self.args.input_dir, self.graph.network_inputs, self.st, self.ed):
                for name in self.graph.network_inputs:
                    self.activation_cache[name].append(
                        data[name][:].reshape(*self.graph.get_tensor_shape(name)).copy())
        else:
            # Means We need specific tensor.
            self.activation_cache[in_tensor] = []
            for data in input_data_generator(self.args.input_dir, self.graph.network_inputs, self.st, self.ed):
                self.activation_cache[in_tensor].append(
                    data[in_tensor][:].reshape(*self.graph.get_tensor_shape(in_tensor)).copy())

    def clear_cache(self):
        names = list(self.activation_cache.keys())
        for name in names:
            if name in self.graph.network_inputs:
                continue
            del self.activation_cache[name]

    def input_generator(self, tensor_name_list):
        # TODO batch generator.
        data = {}
        for i in range(self.ed - self.st):
            for tensor in tensor_name_list:
                data[tensor] = self.activation_cache[tensor][i]
            yield data

    def __getitem__(self, tensor_name):
        if tensor_name in self.graph.initializer:
            return self.graph.initializer[tensor_name][0]
        if tensor_name not in self.activation_cache:
            node = self.graph.get_tensor_producer(tensor_name)
            # quantize_output(self.name, 'get item: ', tensor_name, self.activation_cache.keys())
            self.forward_subnet(node.name, node.input)
        return self.activation_cache[tensor_name]

    def forward_subnet(self, subnet_name, input_list):
        sub_graph = self.graph_list[self.name_to_graph_id[subnet_name]]
        for input_tensor in input_list:
            if input_tensor == '':
                continue
            if input_tensor not in sub_graph.initializer and input_tensor not in self.activation_cache:
                node = self.graph.get_tensor_producer(input_tensor)
                if isinstance(node, str):
                    # Means We need network input.
                    self.fetch_input(input_tensor)
                else:
                    self.forward_subnet(node.name, node.input)

        input_generator = self.input_generator(sub_graph.network_inputs)
        sub_graph = self.graph_list[self.name_to_graph_id[subnet_name]]
        sub_net = sub_graph.model
        ort_inputs = {}
        ort_session = ort.InferenceSession(sub_net.SerializeToString(), providers=self.providers)
        for data in input_generator:
            for name in sub_graph.network_inputs:
                ort_inputs[name] = data[name][:].reshape(*sub_graph.get_tensor_shape(name))
            outputs = [output.name for output in ort_session.get_outputs()]
            ort_outputs = ort_session.run(outputs, ort_inputs)
            ort_outs = OrderedDict(zip(outputs, ort_outputs))

            for i in ort_outs:
                # There may be dummy outputs, which
                # do not needed by any other layers neither is network output.
                if i in self.ref_cnt or i in self.graph.network_outputs:
                    if i in self.activation_cache:
                        self.activation_cache[i].append(ort_outs[i].copy())
                    else:
                        self.activation_cache[i] = [ort_outs[i].copy()]
        # Tensor Wont Be used in this forward.
        for input_tensor in input_list:
            if input_tensor in sub_graph.initializer:
                continue
            if input_tensor == '':
                continue
            self.ref_cnt[input_tensor] -= 1
            if self.ref_cnt[input_tensor] == 0:
                del (self.activation_cache[input_tensor])

    def fill_ref_cnt(self):
        for node in self.graph.graph.node:
            for in_tensor in node.input:
                if in_tensor in self.ref_cnt:
                    self.ref_cnt[in_tensor] += 1
                else:
                    self.ref_cnt[in_tensor] = 1

    def _split_network(self):
        for i, node in enumerate(self.graph.graph.node):
            inputs = []
            outputs = []
            inits = []
            network_inputs = []
            network_outputs = []
            for input in node.input:
                if input == '':
                    continue
                if input not in self.graph.initializer:
                    in_type = self.graph.get_value_type(input)
                    input_value = mtvi(input, in_type, self.graph.get_tensor_shape(input))
                    inputs.append(input_value)
                    network_inputs.append(input)
                else:
                    inits.append(self.graph.initializer[input][0])

            for output in node.output:
                out_type = self.graph.get_value_type(output)
                output_value = mtvi(output, out_type, self.graph.get_tensor_shape(output))
                outputs.append(output_value)
                network_outputs.append(output)

            graph = make_graph(nodes=[node], name=node.name, inputs=inputs,
                               outputs=outputs, initializer=inits)
            opset_import = self.graph.model.opset_import
            sub_net = make_model(graph, producer_name=node.name, opset_imports=opset_import)
            sub_graph = ONNXGraph(sub_net, self.args.output_dir)
            sub_graph.tensor_name_shape_map = self.graph.tensor_name_shape_map
            sub_graph.network_inputs = network_inputs
            sub_graph.network_outputs = network_outputs
            self.graph_list.append(sub_graph)
        for idx, sub_graph in enumerate(self.graph_list):
            self.name_to_graph_id[sub_graph.graph.name] = idx

    def update_graph(self, graph):
        for i, sub_graph in enumerate(self.graph_list):
            for init_name in self.graph_list[i].initializer:
                tensor = graph.get_initializer(init_name)
                self.graph_list[i].set_initializer(init_name, tensor)
            self.graph_list[i].update_model()
        self.ref_cnt = {}
        self.fill_ref_cnt()


def forward_get_minmax(onnx_graph, args):
    net = copy.deepcopy(onnx_graph.model)
    graph = net.graph
    for node in reversed(graph.node):
        for output_name in reversed(node.output):
            if output_name not in [_o.name for _o in graph.output]:
                graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
        logger.warning("CUDA may not used. Please check your ort/cuda/cudnn version.")
    # Start activation quantization.
    statistics = {}
    t1 = 0
    ort_inputs = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    for data in tqdm(input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx),
                     desc='Minmax update'):
        for name in onnx_graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(onnx_graph.get_tensor_shape(name))
        st = time.time()
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ed = time.time()
        t1 += ed - st
        ort_outs = OrderedDict(zip(outputs, ort_outputs))
        for i in ort_inputs:
            if i in statistics:
                statistics[i]['max'].append(ort_inputs[i].max())
                statistics[i]['min'].append(ort_inputs[i].min())
            else:
                statistics[i] = {}
                statistics[i]['max'] = [ort_inputs[i].max()]
                statistics[i]['min'] = [ort_inputs[i].min()]
        for i in ort_outs:
            if i in statistics:
                statistics[i]['max'].append(ort_outs[i].max())
                statistics[i]['min'].append(ort_outs[i].min())
            else:
                statistics[i] = {}
                statistics[i]['max'] = [ort_outs[i].max()]
                statistics[i]['min'] = [ort_outs[i].min()]
    logger.info("Forward time: {:.2f} seconds".format(t1))
    return statistics


def forward_get_hist(onnx_graph, stats_min_max, args):
    net = copy.deepcopy(onnx_graph.model)
    graph = net.graph
    for node in reversed(graph.node):
        for output_name in reversed(node.output):
            if output_name not in [_o.name for _o in graph.output]:
                graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
        logger.warning("CUDA may not used. Please check your ort/cuda/cudnn version.")
    # Start activation quantization.
    statistics = {}
    ort_inputs = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    for data in tqdm(input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx),
                     desc='Hist update: {}'.format(args.rank)):
        for name in onnx_graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(onnx_graph.get_tensor_shape(name))
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ort_outs = OrderedDict(zip(outputs, ort_outputs))

        for i in ort_inputs:
            data_max = max(np.max(stats_min_max[i]['max']),
                           -np.min(stats_min_max[i]['min']))
            hist, _ = np.histogram(np.abs(ort_inputs[i]), int(args.bins), (0, data_max))
            if i in statistics:
                statistics[i].append(hist)
            else:
                statistics[i] = [hist]
        for i in ort_outs:
            data_max = max(np.max(stats_min_max[i]['max']),
                           -np.min(stats_min_max[i]['min']))
            hist, _ = np.histogram(np.abs(ort_outs[i]), int(args.bins), (0, data_max))
            if i in statistics:
                statistics[i].append(hist)
            else:
                statistics[i] = [hist]
    return statistics


def forward_net_octav(onnx_graph, args):
    # Generate Graph and Net
    net = copy.deepcopy(onnx_graph.model)
    graph = net.graph
    for node in reversed(graph.node):
        for output_name in reversed(node.output):
            if output_name not in [_o.name for _o in graph.output]:
                graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
        logger.warning("CUDA may not used. Please check your ort/cuda/cudnn version.")
    # Start activation quantization.
    statistics = {}
    t1 = 0
    ort_inputs = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    for data in tqdm(input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx),
                     desc='OCTAV update rank: {}'.format(args.rank)):
        ort_inputs = {}
        for name in onnx_graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(onnx_graph.get_tensor_shape(name))
        st = time.time()
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ed = time.time()
        t1 += ed - st
        ort_outs = OrderedDict(zip(outputs, ort_outputs))
        ort_inputs.update(ort_outs)
        for i in ort_inputs:
            data_max = ort_inputs[i].max()
            data_min = ort_inputs[i].min()
            # If dynamic_sym = True, Means one more bit.
            if np.abs(data_min - 0) < 1e-6 and 'dynamic_sym' in platform_setting_table[args.deploy]['qi_params']:
                unsigned = 4
            else:
                unsigned = 1
            abs_x = np.abs(ort_inputs[i])
            s_n = abs_x.sum() / abs_x[abs_x > 0].size
            for _ in range(20):
                s_n_plus_1 = abs_x[abs_x > s_n].sum() / \
                    (1 / (4 ** 8) / 3 / unsigned * abs_x[abs_x <= s_n].size + abs_x[abs_x > s_n].size)
                if np.abs(s_n_plus_1 - s_n) < 1e-6:
                    break
                s_n = s_n_plus_1
            if i in statistics:
                statistics[i]['optimal_s'].append(s_n)
                statistics[i]['min'].append(data_min)
                statistics[i]['max'].append(data_max)
            else:
                statistics[i] = {
                    'optimal_s': [s_n],
                    'min': [data_min],
                    'max': [data_max]
                }
    logger.info("Forward time: {:.2f} seconds".format(t1))
    return statistics


def input_data_generator(input_dir, input_name_list, data_st_idx, data_ed_idx):
    for idx in range(data_st_idx, data_ed_idx):
        data = {}
        for i in input_name_list:
            data[i] = np.fromfile(f'{input_dir}/{i}/{idx}.bin', 'float32')
        yield data


def forward_get_tensor(graph, net, index, args):
    for node in graph.graph.node:
        if node.op_type in QUANT_NODE_NAME_LIST:
            continue
        for output_name in node.output:
            if output_name not in [_o.name for _o in net.graph.output]:
                net.graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    providers = [("CUDAExecutionProvider", {'device_id': device})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    ort_inputs = {}
    for data in input_data_generator(args.input_dir, graph.network_inputs, index, index + 1):
        for name in graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(graph.get_tensor_shape(name))
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ort_outs = OrderedDict(zip(outputs, ort_outputs))
    return copy.deepcopy(ort_outs)