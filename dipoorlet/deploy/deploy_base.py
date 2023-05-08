from ..platform_settings import platform_setting_table
from .deploy_atlas import gen_atlas_quant_param
from .deploy_default import deploy_dispatcher
from .deploy_magicmind import gen_magicmind_proto
from .deploy_rv import gen_rv_yaml
from .deploy_snpe import gen_snpe_encodings
from .deploy_stpu import gen_stpu_minmax
from .deploy_ti import gen_ti_json
from .deploy_trt import gen_trt_range


def to_deploy(graph, act_clip_val, weight_clip_val, args, **kwargs):
    if platform_setting_table[args.deploy]['deploy_weight']:
        clip_val = act_clip_val.copy()
        clip_val.update(weight_clip_val)
    else:
        clip_val = act_clip_val
    deploy_dispatcher(args.deploy, graph, clip_val, args, **kwargs)
