import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from onnx import numpy_helper
from torch.nn.parallel import DistributedDataParallel as DDP

from ..forward_net import ActivationCache
from ..platform_settings import platform_setting_table
from ..quantize import QUANT_NODE_NAME_LIST, quant_graph
from ..utils import logger
from .sparse_quant_layer import *
from .utils import *
from .weight_equalization import node_has_equalized


def sparse_quant(graph_ori, graph, act_clip_val, weight_clip_val, args):
    dist.barrier()
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_ada = copy.deepcopy(graph)
    rank = dist.get_rank()
    num_per_rank = args.data_num // dist.get_world_size()
    rank_st = rank * num_per_rank
    rank_ed = rank_st + num_per_rank
    sparse_info = {
                    "sparse": True,
                    "rate": args.sparse_rate,
                    "pattern": args.pattern
                }
    fp_act_cache = ActivationCache(graph_ori, args, rank_st, rank_ed)
    prev_act_cache = None
    for node in graph_ori.graph.node:
        if node.name in args.skip_layers:
            continue
        if node.op_type in LEARNABLE_LAYER_TYPES:
            # We can not mimic when node has weight equalized.
            if args.we and node_has_equalized(graph, node):
                continue
            if dist.get_rank() == 0:
                logger.info("sparse_quant for: {}".format(node.name))
            # Using graph_ada and restore act cache for incremental update.
            if not prev_act_cache:
                graph_q, quant_node_list = quant_graph(graph_ada, clip_val, args)
                q_act_cache = ActivationCache(graph_q, args, rank_st, rank_ed)
            else:
                q_act_cache.update_graph(graph_q)
                q_act_cache.activation_cache = prev_act_cache
            prev_node = graph_q.get_tensor_consumer(node.input[0])[0]
            prev_node = graph_q.get_tensor_consumer(prev_node.output[0])[0]
            in_tensor_name = node.input[0]
            if prev_node.op_type == QUANT_NODE_NAME_LIST[-1]:
                in_tensor_name = prev_node.output[0]

            q_in_tensor = np.stack(q_act_cache[in_tensor_name])
            fp_out_tensor = np.stack(fp_act_cache[node.output[0]])
            prev_act_cache = q_act_cache.activation_cache.copy()

            # Get weight and build torch conv.
            weight = numpy_helper.to_array(graph_ada.initializer[node.input[1]][0])
            bias = None
            if len(node.input) == 3:
                bias = numpy_helper.to_array(graph_ada.initializer[node.input[2]][0])

            weight = torch.from_numpy(weight).cuda()
            # Get quantization param.
            weight_range = clip_val[node.input[1]]
            qw_param = platform_setting_table[args.deploy]['qw_params']
            if node.op_type == 'ConvTranspose':
                weight = weight.transpose(0, 1)
            scale, q_min, q_max = get_quant_tensor(weight.shape, qw_param, weight_range)
            qw_tensor = {'scale': scale,
                            'q_min': q_min,
                            'q_max': q_max,
                            'per_channel': qw_param['per_channel'],
                            'type': 'Linear'}
            # Learning.
            relu_flag = follow_relu(graph, node)
            if relu_flag:
                fp_tensor = torch.nn.Parameter(F.relu(torch.from_numpy(fp_out_tensor)), False)
            else:
                fp_tensor = torch.nn.Parameter(torch.from_numpy(fp_out_tensor), False)
            # Learning sparse quant.
            sq_layer = SparseQLayer(node, weight, bias, qw_tensor, None,
                                  relu_flag, node.op_type, sparse_info)
            weight = learning_sparse_quant(
                torch.nn.Parameter(torch.from_numpy(q_in_tensor).cuda(), False),
                fp_tensor.cuda(),
                sq_layer, args.ada_bs, args.ada_epoch)
            # Deploy new weight.
            new_weight = prune_weight(weight, sparse_info)
            new_weight = quant_weight_wo_roundmask(
                new_weight,
                scale, q_min, q_max,
                qw_param['per_channel'])
            if node.op_type == 'ConvTranspose':
                new_weight = new_weight.transpose(0, 1)
            new_weight = new_weight.cpu().detach().numpy()
            update_weight(graph_ada, new_weight, node.input[1])
            update_weight(graph_q, new_weight, node.input[1])
            graph_ada.update_model()
            graph_q.update_model()
    if dist.get_rank() == 0:
        graph_ada.save_onnx_model('sparse_quant')
    # We must use original ranges.
    return graph_ada


def learning_sparse_quant(in_tensor, fp_out_tensor, ada_layer, batch_size, max_epoch):
    optimizer = torch.optim.SGD([ada_layer.layer.weight], lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_epoch)
    ada_layer = DDP(ada_layer, [torch.cuda.current_device()])
    # New train precedure
    cur_iter = 0
    for epoch in range(max_epoch):
        for idx in range(np.ceil(len(in_tensor) / batch_size).astype(int)):
            st = idx * batch_size
            ed = st + batch_size
            input = in_tensor[st:ed].squeeze(1)
            fp_output = fp_out_tensor[st:ed].squeeze(1)
            output = ada_layer(input)
            loss = L2_norm(output, fp_output)
            cur_iter += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 50 == 0 and dist.get_rank() == 0:
            logger.info("Epoch: {:<4} L2 Loss: {:>10.6f}, LR: {:>10.6f}".format(epoch, loss, scheduler.get_lr()[0]))
    if dist.get_rank() == 0:
        logger.info("Loss: {:>10.6f}".format(loss))
    return ada_layer.module.layer.weight
