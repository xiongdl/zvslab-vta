"""Reproducible first quantized KWS depthwise layer from a real WAV."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import tvm
from tvm import relay
from . import model

APP_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DwcSample:
    activation: np.ndarray
    weight: np.ndarray
    strides: tuple
    padding: tuple
    reference: np.ndarray
    hashes: dict


def scalar_reference(activation, weight, strides, padding):
    """Signed integer convolution independent of Relay/TOP/layout packing."""
    pt, pl, pb, pr = padding
    x = np.pad(activation.astype('int32'), ((0, 0), (pt, pb), (pl, pr), (0, 0)))
    kh, kw, channels, multiplier = weight.shape
    assert multiplier == 1
    oh = (x.shape[1] - kh) // strides[0] + 1
    ow = (x.shape[2] - kw) // strides[1] + 1
    result = np.zeros((activation.shape[0], oh, ow, channels), dtype='int32')
    for n in range(result.shape[0]):
        for h in range(oh):
            for w in range(ow):
                for c in range(channels):
                    result[n, h, w, c] = sum(int(x[n, h * strides[0] + r, w * strides[1] + s, c]) * int(weight[r, s, c, 0]) for r in range(kh) for s in range(kw))
    return result


def extract_dwc_sample(model_path=None, sample_path=None):
    """Evaluate operator 1's quantized operands and int32 output on CPU."""
    model_path = Path(model_path or APP_ROOT / 'model/kws_ref_model_float32.tflite')
    sample_path = Path(sample_path or APP_ROOT / 'samples/right-00b01445_nohash_0.wav')
    imported = model.import_model(model_path)
    module = relay.transform.InferType()(model.quantize_model(imported))
    depthwise = []
    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == 'nn.conv2d' and int(node.attrs.groups) == 64:
            depthwise.append(node)
    relay.analysis.post_order_visit(module['main'].body, visit)
    if not depthwise:
        raise ValueError('quantized model has no 64-channel depthwise convolution')
    layer = depthwise[0]
    attrs = layer.attrs
    assert str(attrs.data_layout) == 'NHWC' and str(attrs.kernel_layout) == 'HWOI'
    assert tuple(map(int, layer.args[0].checked_type.shape)) == (1, 25, 5, 64)
    assert tuple(map(int, layer.args[1].checked_type.shape)) == (3, 3, 64, 1)
    assert tuple(map(int, layer.checked_type.shape)) == (1, 25, 5, 64)
    assert str(layer.checked_type.dtype) == 'int32'
    expression = relay.Tuple([layer.args[0], layer.args[1], layer])
    function = relay.Function(relay.analysis.free_vars(expression), expression)
    extracted = relay.transform.InferType()(tvm.IRModule.from_expr(function))
    with tvm.transform.PassContext(opt_level=3):
        executor = relay.create_executor('vm', mod=extracted, device=tvm.cpu(), target='llvm').evaluate()
        values = executor(**{imported.input_name: tvm.nd.array(model.load_sample(sample_path))})
    activation, weight, expected = (values[index].numpy().copy() for index in range(3))
    strides, padding = tuple(map(int, attrs.strides)), tuple(map(int, attrs.padding))
    reference = scalar_reference(activation, weight, strides, padding)
    np.testing.assert_array_equal(reference, expected)
    hashes = {'model': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'wav': hashlib.sha256(sample_path.read_bytes()).hexdigest()}
    for name, value in [('activation', activation), ('weight', weight), ('reference', reference)]:
        hashes[name] = hashlib.sha256(value.tobytes()).hexdigest()
    return DwcSample(activation, weight, strides, padding, reference, hashes)
