import argparse
import os
import sys
import time

import onnx
import torch
import torch.distributed as dist

from onnxsim import simplify

from .deploy import to_deploy
from .dist_helper import init_from_mpi, init_from_slurm
from .profiling import (quantize_profiling_multipass, quantize_profiling_transformer,
                        show_model_profiling_res, show_model_ranges, weight_need_perchannel)
from .tensor_cali import tensor_calibration
from .utils import (ONNXGraph, load_clip_val, logger, reduce_clip_val,
                    reduce_profiling_res, save_clip_val, save_profiling_res,
                    setup_logger)
from .weight_transform import weight_calibration

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", help="onnx model")
parser.add_argument("-I", "--input_dir", help="calibration data", required=True)
parser.add_argument("-O", "--output_dir", help="output data path")
parser.add_argument("-N", "--data_num", help="num of calibration pics", type=int, required=True)
parser.add_argument("--we", help="weight euqalization", action="store_true")
parser.add_argument("--bc", help="bias correction", action="store_true")
parser.add_argument("--update_bn", help="update BN", action="store_true")
parser.add_argument("--adaround", help="Adaround", action="store_true")
parser.add_argument("--brecq", help="BrecQ", action="store_true")
parser.add_argument("--drop", help="QDrop", action="store_true")
parser.add_argument("-A", "--act_quant", help="algorithm of activation quantization",
                    choices=['minmax', 'hist', 'mse'], default='mse')
parser.add_argument("-D", "--deploy", help="deploy platform",
                    choices=['trt', 'stpu', 'magicmind', 'rv', 'atlas',
                             'snpe', 'ti', 'imx'], required=True)
parser.add_argument("--bins", help="bins for histogram and kl", default=2048)
parser.add_argument("--threshold", help="threshold for histogram", default=0.99999, type=float)
parser.add_argument("--savefp", help="Save FP output of model.", action="store_true")
parser.add_argument("--ada_bs", help="Batch size for adaround.", type=int, default=64)
parser.add_argument("--ada_epoch", help="Epoch for adaround.", type=int, default=5000)
parser.add_argument("--skip_layers", help="Skip layer name", default=[], type=str, nargs='+')
parser.add_argument("--stpu_wg", help="Enable winograd for stpu.", action="store_true")
parser.add_argument("--skip_prof_layer", help="Skip profiling by layer.", default=False, action='store_true')
parser.add_argument("--slurm", help="Launch task from slurm", default=False, action='store_true')
parser.add_argument("--mpirun", help="Launch task from mpirun", default=False, action='store_true')
parser.add_argument("--sparse", help="Sparse on/off", default=False, action="store_true")
parser.add_argument("--sparse_rate", help="Sparse rate", type=float, default=0.5)
parser.add_argument("--pattern", help="Sparse pattern", choices=["unstruction", "nv24"], default="unstruction")
parser.add_argument("--optim_transformer", help="Transformer model optimization", default=False, action='store_true')
parser.add_argument("--model_type", help="Transformer model type", choices=["unet"], default=None)
args = parser.parse_args()

if args.slurm:
    init_from_slurm()
elif args.mpirun:
    init_from_mpi()
else:
    dist.init_process_group(backend='nccl')
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)

if args.output_dir is None:
    model_path = ('/').join(args.model.split('/')[:-1])
    output_dir = os.path.join(os.path.abspath(model_path), 'results')
    args.output_dir = output_dir

if args.model_type is not None:
    args.optim_transformer = True
    args.skip_prof_layer = True

if dist.get_rank() == 0:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    setup_logger(args)

    if args.optim_transformer:
        model_path = ('/').join(args.model.split('/')[:-1])
        args.infer_shape_dir = os.path.join(os.path.abspath(model_path), "infer_shape.onnx")
        onnx.shape_inference.infer_shapes_path(args.model, args.infer_shape_dir)
        args.optimzed_model_dir = os.path.join(args.output_dir, 'optim_model.onnx')
        os.system("python -m onnxruntime.transformers.optimizer \
                   --input {} --output {} --model_type={} \
                   --use_external_data_format --disable_packed_qkv \
                   --disable_packed_kv --use_gpu --disable_nhwc_conv"
                   .format(args.infer_shape_dir, args.optimzed_model_dir, args.model_type))

logger.parent = None

start = time.time()
if args.optim_transformer:
    model = onnx.load(args.optimzed_model_dir)
else:
    model = onnx.load(args.model)
    model = onnx.version_converter.convert_version(model, 13)
    model, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
onnx_graph = ONNXGraph(model, args.output_dir, args.deploy, args.model_type)

if dist.get_rank() == 0 and not args.optim_transformer:
    try:
        onnx.checker.check_model(onnx_graph.model)
    except onnx.checker.ValidationError as e:
        logger.info("The onnx model is invalid:{}, please rectifie your model and restart Dipoorlet.".format(e))
        sys.exit()

# Assgin rank index to calibration GPU wise.
# Split the dataset averagly.
setattr(args, 'rank', dist.get_rank())
setattr(args, 'local_rank', dist.get_rank() % torch.cuda.device_count())
setattr(args, 'world_size', dist.get_world_size())
if dist.get_rank() == 0:
    logger.info("Do tensor calibration...")
act_clip_val, weight_clip_val = tensor_calibration(onnx_graph, args)
save_clip_val(act_clip_val, weight_clip_val, args,
              act_fname='act_clip_val.json.rank{}'.format(args.rank),
              weight_fname='weight_clip_val.json.rank{}'.format(args.rank))
dist.barrier()
if dist.get_rank() == 0:
    reduce_clip_val(dist.get_world_size(), args)
dist.barrier()
act_clip_val, weight_clip_val = load_clip_val(args)

# Weight Transform.
if dist.get_rank() == 0:
    logger.info("Weight transform...")
graph, graph_ori, act_clip_val, weight_clip_val = \
    weight_calibration(onnx_graph, act_clip_val, weight_clip_val, args)
dist.barrier()

# Profiling Distributed.
if dist.get_rank() == 0:
    logger.info("Profiling...")
if args.model_type is not None:
    layer_cosine_dict, model_cosine_dict, quant_node_list = quantize_profiling_transformer(
        graph, graph_ori, act_clip_val, weight_clip_val, args)
else:
    layer_cosine_dict, model_cosine_dict, quant_node_list = quantize_profiling_multipass(
        graph, graph_ori, act_clip_val, weight_clip_val, args)
save_profiling_res(layer_cosine_dict, model_cosine_dict, args)
dist.barrier()
if dist.get_rank() == 0:
    layer_cosine_dict, model_cosine_dict = reduce_profiling_res(dist.get_world_size(), args)
    show_model_profiling_res(graph, layer_cosine_dict, model_cosine_dict, quant_node_list, args)
    show_model_ranges(graph, act_clip_val, weight_clip_val, args)
    weight_need_perchannel(graph, args)

# Deploy
if dist.get_rank() == 0:
    logger.info("Deploy to " + args.deploy + '...')
    to_deploy(graph, act_clip_val, weight_clip_val, args)
    end = time.time()
    logger.info("Total time cost: {} seconds.".format(int(end - start)))