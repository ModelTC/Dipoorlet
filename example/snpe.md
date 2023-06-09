# Deploy on Snpe

We provide an example of writing the quantization parameters generated by the "adaround" algorithm into the snpe platform model.

For the ONNX model "model.onnx", quantification is first performed through Dipoorlet. The activation calibration method here uses "mse" and fine-tuning its weights using the "adaround" algorithm.

```
python -m torch.distributed.launch --use_env -m dipoorlet -M model.onnx -I workdir/ -N 100 -A mse -adaround -D snpe
```

Dipoorlet will generate calibrated model "adaround.onnx" and quantitative configuration information "snpe_encodings.json":


```
snpe_encodings.json:
{
    "activation_encodings": {
        "0": [
            {
                "bitwidth": 8,
                "min": -2.1179039478302,
                "max": 2.4663430328313014
            }
        ],
        "43": [
            {
                "bitwidth": 8,
                "min": -2.0301631384284935,
                "max": 2.0301631384284935
            }
        ],
        "44": [
            {
                "bitwidth": 8,
                "min": 0.0,
                "max": 2.004604876945847
            }
        ],
        ...
    },
    "param_encodings": {}
}
```

Subsequently, convert the calibrated model "adaround.onnx" to a spne type network and write "snpe_encodings.json" to the network

```
from subprocess import PIPE, Popen
import json
import onnx

snpemodel_path = 'path-to-dlc_model/model.dlc'
q_overrides_path = 'path-to-snpe_encodings/snpe_encodings.json'
model_fp = onnx.load('adaround.onnx')
cmd_args = ['snpe-onnx-to-dlc', '-i', model_fp, '-o', snpemodel_path]
cmd_args.extend(['--quantization_overrides', q_overrides_path])
p = Popen(cmd_args, stdout=PIPE, stderr=PIPE)
log = p.communicate()
ret = p.returncode
if (ret != 0):
    print("call snpe-dlc-quantize failed.)
    exit(1)
quant_snpemodel_path = 'path-to-quant_model/quant_model.dlc'
cmd_args = ['snpe-dlc-quantize', '--input_dlc', snpemodel_path, '--input_list', raw_list.txt, '--output_dlc', quant_snpemodel_path]
cmd_args.append('--override_params')
p = Popen(cmd_args, stdout=PIPE, stderr=PIPE)
log = p.communicate()
ret = p.returncode    
if (ret != 0):
    print("call snpe-dlc-quantize failed)
    exit(1)
```