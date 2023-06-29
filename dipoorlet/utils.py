import copy
import json
import logging
import os
import time

import numpy as np
import onnx
import torch.distributed as dist
from onnx import TensorProto, numpy_helper
from onnx.shape_inference import infer_shapes

from .platform_settings import platform_setting_table

logger = logging.getLogger("dipoorlet")


class ONNXGraph(object):
    def __init__(self, model=None, output_dir=""):
        self.model = model
        if self.model:
            self.graph = self.model.graph
        self.output_dir = output_dir
        self.initializer = {}
        self.input_map = {}
        self.output_map = {}
        self.network_inputs = []
        self.network_outputs = []
        self.tensor_name_shape_map = {}
        self.value_name_type_map = {}
        self.name_idx_map = {}
        self.input = []
        self.output = []
        if self.model:
            self.replace_upsample_op_with_resize()
            self.topologize_graph()
            self.prepare_initializer()
            self.set_index()
            self.get_inp_oup()
            self.get_shape_type()

    def replace_upsample_op_with_resize(self):
        self.model = infer_shapes(self.model)
        for i, _node in enumerate(self.model.graph.node):
            if _node.op_type == "Upsample":
                inputs = []
                inputs.append(_node.input[0])
                inputs.append('')
                inputs.append(_node.input[1])
                inputs.append('')
                node = onnx.helper.make_node(
                    name="Resize_" + str(i),
                    op_type="Resize",
                    inputs=inputs,
                    outputs=_node.output,
                )
                self.model.graph.node.remove(_node)
                self.model.graph.node.insert(i, node)
        self.graph = self.model.graph
        self.update_model()

    def prepare_initializer(self):
        self.initializer.clear()
        for idx, init in enumerate(self.graph.initializer):
            self.initializer[init.name] = (init, idx)

    def get_inp_oup(self):
        self.network_inputs.clear()
        self.network_outputs.clear()
        self.tensor_name_shape_map.clear()
        self.input.clear()
        self.output.clear()
        for input in self.graph.input:
            if isinstance(self.get_tensor_producer(input.name), str) and \
                    input.name not in self.initializer:
                self.network_inputs.append(input.name)
        for output in self.graph.output:
            self.network_outputs.append(output.name)
        self.input = self.network_inputs.copy()
        self.output = self.network_outputs.copy()

        for node in self.model.graph.node:
            for inp in node.input:
                if inp in self.initializer and inp not in self.input:
                    self.input.append(inp)
            for oup in node.output:
                if oup not in self.output:
                    self.output.append(oup)

    def get_shape_type(self):
        for input in self.graph.input:
            if input.name in self.network_inputs:
                self.tensor_name_shape_map[input.name] = [x.dim_value for x in input.type.tensor_type.shape.dim]
                self.value_name_type_map[input.name] = input.type.tensor_type.elem_type
        for output in self.graph.output:
            self.tensor_name_shape_map[output.name] = [x.dim_value for x in output.type.tensor_type.shape.dim]
            self.value_name_type_map[output.name] = output.type.tensor_type.elem_type

        model = infer_shapes(self.model)

        for init in self.initializer:
            self.tensor_name_shape_map[init] = list(self.get_initializer(init).shape)
        inferred_value_info = model.graph.value_info
        for info in inferred_value_info:
            shape = [x.dim_value for x in info.type.tensor_type.shape.dim]
            value_type = info.type.tensor_type.elem_type
            self.tensor_name_shape_map[info.name] = shape
            self.value_name_type_map[info.name] = value_type

    def get_tensor_shape(self, tensor_name):
        return self.tensor_name_shape_map[tensor_name]

    def get_value_type(self, value_name):
        return self.value_name_type_map[value_name]

    def get_constant(self, name):
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                if node.output[0] == name:
                    return numpy_helper.to_array(node.attribute[0].t).tolist()

    def get_initializer(self, initializer_name):
        return numpy_helper.to_array(self.initializer[initializer_name][0])

    def set_initializer(self, initializer_name, value_tensor, raw=True):
        idx = None
        data_type = None
        if initializer_name in self.initializer:
            idx = self.initializer[initializer_name][1]
        if raw:
            initializer = numpy_helper.from_array(value_tensor)
        else:
            if value_tensor.dtype == np.float32:
                data_type = TensorProto.FLOAT
            if value_tensor.dtype == np.uint8:
                data_type = TensorProto.UINT8
            if value_tensor.dtype == np.int8:
                data_type = TensorProto.INT8
            initializer = onnx.helper.make_tensor(name=initializer_name,
                                                  data_type=data_type,
                                                  dims=[] if value_tensor.size == 1 else list(value_tensor.shape),
                                                  vals=value_tensor,
                                                  raw=False)
        initializer.name = initializer_name
        if idx is not None:
            self.graph.initializer.remove(self.graph.initializer[idx])
            self.graph.initializer.insert(idx, initializer)
        else:
            self.graph.initializer.append(initializer)
        self.prepare_initializer()

    def topologize_graph(self):
        self.input_map.clear()
        self.output_map.clear()
        for idx, node in enumerate(self.graph.node):
            for output_name in node.output:
                self.output_map[output_name] = node
            for input_name in node.input:
                if input_name not in self.input_map:
                    self.input_map[input_name] = []
                self.input_map[input_name].append(node)

    def get_tensor_producer(self, output_name):
        if output_name not in self.output_map:
            return 'INPUT_TOKEN'
        return self.output_map[output_name]

    def get_tensor_consumer(self, input_name):
        if input_name not in self.input_map:
            return ['OUTPUT_TOKEN']
        return self.input_map[input_name]

    def save_onnx_model(self, name='tmp'):
        model_path = os.path.join(self.output_dir, '{}.onnx'.format(name))
        onnx.save(self.model, model_path)

    def remove_node_purely(self, node):
        self.graph.node.remove(node)

    def insert_node_purely(self, node, idx=0):
        self.graph.node.insert(idx, node)

    def insert_qnodes_purely(self, q_nodes, idx=0, node=None):
        node_list = reversed(q_nodes.node)
        if node:
            idx = self.index(node)
        for node in node_list:
            self.graph.node.insert(idx, node)
        for init in q_nodes.initializer:
            self.graph.initializer.append(init)
        self.set_index()

    def del_network_output(self, out_name):
        idx = self.network_outputs.index(out_name)
        self.graph.output.pop(idx)
        self.network_outputs.remove(out_name)

    def add_network_output(self, out_put):
        self.graph.output.append(out_put)
        self.network_outputs.append(out_put.name)

    def del_initializer(self, initializer_name):
        if initializer_name in self.initializer:
            del self.initializer[initializer_name]

    def set_index(self):
        for node_idx, node in enumerate(self.graph.node):
            self.name_idx_map[node.name] = node_idx

    def index(self, node):
        return self.name_idx_map[node.name]

    def update_model(self):
        self.set_index()
        opset_info = copy.deepcopy(self.model.opset_import[0])
        if opset_info.version < 13:
            opset_info.version = 13
        self.model = onnx.helper.make_model(self.graph,
                                            producer_name='updated_model',
                                            opset_imports=[opset_info])
        self.prepare_initializer()

    def copy_to(self, target_graph):
        target_graph.model = copy.deepcopy(self.model)
        target_graph.graph = copy.deepcopy(self.graph)
        target_graph.initializer = copy.deepcopy(self.initializer)
        target_graph.input_map = copy.deepcopy(self.input_map)
        target_graph.output_map = copy.deepcopy(self.output_map)
        target_graph.network_inputs = copy.deepcopy(self.network_inputs)
        target_graph.network_outputs = copy.deepcopy(self.network_outputs)
        target_graph.tensor_name_shape_map = copy.deepcopy(self.tensor_name_shape_map)
        target_graph.input = copy.deepcopy(self.input)
        target_graph.output = copy.deepcopy(self.output)
        target_graph.name_idx_map = self.name_idx_map.copy()
        target_graph.output_dir = self.output_dir


def setup_logger(args):
    global logger
    logger.setLevel(logging.INFO)
    logger_file = os.path.join(args.output_dir,
                               'log-{}.txt'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    with open(logger_file, 'w') as f:
        f.write(str(args) + '\n')
    file_log = logging.FileHandler(logger_file)
    file_log.setLevel(logging.INFO)
    logger.addHandler(file_log)
    std_log = logging.StreamHandler()
    logger.addHandler(std_log)


def cos_similarity(ta, tb):
    assert ta.shape == tb.shape
    if np.sum(ta * tb) == 0:
        return 0.
    return np.sum(ta * tb) / np.sqrt(np.square(ta).sum()) \
        / np.sqrt(np.square(tb).sum())


def dispatch_functool(func):
    registry = {}

    def dispatch(value):
        try:
            return registry[value]
        except KeyError:
            return func

    def register(value, func=None):
        if func is None:
            return lambda f: register(value, f)
        registry[value] = func
        return func

    def wrapper(*args, **kw):
        return dispatch(args[0])(*(args[1:]), **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


def update_model_path(name, args):
    '''Update model path saved in args. Often sync among GPUs.
    Always followed load_graph.
    '''
    args.model = os.path.join(args.output_dir, '{}.onnx'.format(name))


def save_clip_val(act_clip_val, weight_clip_val, args, act_fname='act_clip_val.json', weight_fname='weight_clip_val.json'):
    for k, v in act_clip_val.items():
        act_clip_val[k][0] = act_clip_val[k][0].tolist()
        act_clip_val[k][1] = act_clip_val[k][1].tolist()
    for k, v in weight_clip_val.items():
        weight_clip_val[k][0] = weight_clip_val[k][0].tolist()
        weight_clip_val[k][1] = weight_clip_val[k][1].tolist()
    with open(os.path.join(args.output_dir, act_fname), 'w') as f:
        json.dump(act_clip_val, f, indent=4)
    with open(os.path.join(args.output_dir, weight_fname), 'w') as f:
        json.dump(weight_clip_val, f, indent=4)


def reduce_clip_val(rank_size, args, act_fname='act_clip_val.json', weight_fname='weight_clip_val.json'):
    '''Collect activation clip val from each GPU and reduce. Weight range use rank0.
    '''
    act_clip_val, weight_clip_val = load_clip_val(args, act_fname + '.rank0', weight_fname + '.rank0')
    for k, v in act_clip_val.items():
        if args.act_quant != 'minmax':
            v[0] /= float(rank_size)
            v[1] /= float(rank_size)
    for i in range(1, rank_size):
        with open(os.path.join(args.output_dir, act_fname + '.rank{}'.format(i)), 'r') as f:
            _act_clip_val = json.load(f)
            for k, v in _act_clip_val.items():
                if args.act_quant != 'minmax':
                    act_clip_val[k][0] += v[0] / float(rank_size)
                    act_clip_val[k][1] += v[1] / float(rank_size)
                else:
                    act_clip_val[k] = [
                        np.array(min(v[0], act_clip_val[k][0])),
                        np.array(max(v[1], act_clip_val[k][1]))]
    save_clip_val(act_clip_val, weight_clip_val, args)


def load_clip_val(args, act_fname='act_clip_val.json', weight_fname='weight_clip_val.json'):
    act_clip_val = {}
    weight_clip_val = {}
    with open(os.path.join(args.output_dir, act_fname), 'r') as f:
        act_clip_val = json.load(f)
        for k, v in act_clip_val.items():
            # We need scalar here.
            act_clip_val[k][0] = np.float64(act_clip_val[k][0])
            act_clip_val[k][1] = np.float64(act_clip_val[k][1])
    with open(os.path.join(args.output_dir, weight_fname), 'r') as f:
        per_channel = False
        if 'per_channel' in platform_setting_table[args.deploy]['qw_params']:
            per_channel = platform_setting_table[args.deploy]['qw_params']['per_channel']
        weight_clip_val = json.load(f)
        for k, v in weight_clip_val.items():
            weight_clip_val[k][0] = np.array(weight_clip_val[k][0])
            weight_clip_val[k][1] = np.array(weight_clip_val[k][1])
            if not per_channel:
                weight_clip_val[k][0] = np.float64(weight_clip_val[k][0])
                weight_clip_val[k][1] = np.float64(weight_clip_val[k][1])
    return act_clip_val, weight_clip_val


def save_profiling_res(layer_cosine_dict, model_cosine_dict, args,
                       layer_res_fname='layer_res.json', model_res_fname='model_res.json'):
    rank = dist.get_rank()
    for k, v in layer_cosine_dict.items():
        layer_cosine_dict[k] = float(v)
    for k, v in model_cosine_dict.items():
        model_cosine_dict[k][0] = float(v[0])
        model_cosine_dict[k][1] = float(v[1])
    with open(os.path.join(args.output_dir, layer_res_fname + '.rank{}'.format(rank)), 'w') as f:
        json.dump(layer_cosine_dict, f, indent=4)
    with open(os.path.join(args.output_dir, model_res_fname + '.rank{}'.format(rank)), 'w') as f:
        json.dump(model_cosine_dict, f, indent=4)


def reduce_profiling_res(rank_size, args, layer_res_fname='layer_res.json', model_res_fname='model_res.json'):
    '''Collect profiling res from each GPU and reduce.
    '''
    with open(os.path.join(args.output_dir, layer_res_fname + '.rank0'), 'r') as f:
        layer_cosine_dict = json.load(f)
    with open(os.path.join(args.output_dir, model_res_fname + '.rank0'), 'r') as f:
        model_cosine_dict = json.load(f)
    for k, v in layer_cosine_dict.items():
        layer_cosine_dict[k] = v / float(rank_size)
    for i in range(1, rank_size):
        with open(os.path.join(args.output_dir, layer_res_fname + '.rank{}'.format(i)), 'r') as f:
            _layer_cosine_dict = json.load(f)
            for k, v in _layer_cosine_dict.items():
                layer_cosine_dict[k] += v / float(rank_size)
    for k, v in model_cosine_dict.items():
        model_cosine_dict[k][0] = v[0] / float(rank_size)
    for i in range(1, rank_size):
        with open(os.path.join(args.output_dir, model_res_fname + '.rank{}'.format(i)), 'r') as f:
            _model_cosine_dict = json.load(f)
            for k, v in _model_cosine_dict.items():
                model_cosine_dict[k][0] += v[0] / float(rank_size)
                model_cosine_dict[k][1] = min(model_cosine_dict[k][1], v[1])
    return layer_cosine_dict, model_cosine_dict
