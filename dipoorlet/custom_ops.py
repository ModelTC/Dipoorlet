from onnxruntime_extensions import onnx_op, PyCustomOpDef
import numpy as np


@onnx_op(op_type="QuantDequantFP8", inputs=[PyCustomOpDef.dt_float],
         attrs={"exponent": PyCustomOpDef.dt_float, "mantissa": PyCustomOpDef.dt_float})
def QuantDequantFP8(x, **kwargs):
    exponent = kwargs["exponent"]
    mantissa = kwargs["mantissa"]
    b = 2 ** (exponent - 1) - 1
    ret = np.zeros_like(x, dtype=x.dtype)
    c = (2 - 2 * 2 ** - mantissa) * 2 ** (2 ** exponent - b - 1)
    ret = np.floor(np.log2(np.abs(x) + 1e-8)) - mantissa
    ret = 2 ** np.clip(ret, 1 - b - mantissa, ret)
    ret = ret * np.round(x / ret)
    ret = np.clip(ret, -c, c)
    return ret
