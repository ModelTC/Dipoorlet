LAYER_HAS_WEIGHT = ['Conv', 'Gemm', 'ConvTranspose', 'PRelu', 'BatchNormalization']
basic_quant_node = ['Relu', 'Eltwise', 'MaxPool', 'Conv', 'Gemm', 'ConvTranspose', 'PRelu',
                    'AveragePool', 'Concat', 'Split', 'Add', 'Mul', 'Abs', 'Reciprocal', 'Sigmoid']


trt_platform_settings = {
    'deploy_exclude_layers': [],
    'quant_nodes': ['Relu', 'MaxPool', 'Conv', 'Gemm', 'ConvTranspose', 'PRelu', 'AveragePool', 'Add', 'Sigmoid'],
    'qw_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': True,
        'per_channel': True
    },
    'qi_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': True
    },
    'quantize_network_output': False,
    'deploy_weight': False
}


stpu_platform_settings = {
    'deploy_exclude_layers': [],
    'quant_nodes': basic_quant_node + ['Clip'],
    'qi_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': True
    },
    'qw_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': True,
        'per_channel': False
    },
    'quantize_network_output': False,
    'deploy_weight': True
}


magicmind_platform_settings = {
    'deploy_exclude_layers': [],
    'quant_nodes': ['Gemm', 'Conv', 'ConvTranspose', 'MatMul'],
    'qw_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': False,
        'log_scale': False,
        'per_channel': True
    },
    'qi_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': False,
        'log_scale': False
    },
    'quantize_network_output': False,
    'deploy_weight': False
}


rv_platform_settings = {
    'deploy_exclude_layers': [],
    'quant_nodes': basic_quant_node,
    'qi_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': False
    },
    'qw_params': {
        'bit_width': 8,
        'type': 'Linear',
        'per_channel': False,
        'symmetric': False
    },
    'quantize_network_output': True,
    'deploy_weight': True
}


# Set rely on Atlas manual.
# https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/graphdevg/graphdevg_000029.html
# Todo: Pool cannot be global.
atlas_platform_settings = {
    'quant_nodes': ['Conv', 'Gemm', 'AveragePool'],
    'qw_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': True,
        'per_channel': True
    },
    'qi_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': False
    },
    'quantize_network_output': False,
    'deploy_weight': False
}


# SNPE docs
# https://developer.qualcomm.com/sites/default/files/docs/snpe/quantized_models.html
snpe_platform_settings = {
    'deploy_exclude_layers': [],
    'quant_nodes': basic_quant_node + ['Sigmoid'],
    'qw_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': False,
        'per_channel': False
    },
    'qi_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': False
    },
    'quantize_network_output': True,
    'deploy_weight': False
}


# TI docs https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7 \
# /07_03_00_07/exports/docs/tidl_j7_02_00_00_07/ti_dl/docs/ \
# user_guide_html/md_tidl_fsg_quantization.html
# If calibrationOption = 13
# dw conv perchannel + log2=True
# odinary conv perlayer + log2=False
# If calibrationOption = 16
# dw conv weight bit_width = 16
ti_platform_settings = {
    'deploy_exclude_layers': [],
    'quant_nodes': basic_quant_node,
    'qw_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': True,
        'per_channel': False,
        'log_scale': False
    },
    'qi_params': {
        'bit_width': 8,
        'type': 'Linear',
        'symmetric': True,
        'dynamic_sym': True,
        'log_scale': True
    },
    'quantize_network_output': False,
    'deploy_weight': False
}


platform_setting_table = {
    'trt': trt_platform_settings,
    'stpu': stpu_platform_settings,
    'magicmind': magicmind_platform_settings,
    'rv': rv_platform_settings,
    'atlas': atlas_platform_settings,
    'snpe': snpe_platform_settings,
    'ti': ti_platform_settings,
}
