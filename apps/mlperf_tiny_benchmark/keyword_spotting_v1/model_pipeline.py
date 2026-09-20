"""Deterministic KWS preprocessing and Relay/VTA model preparation."""

from dataclasses import dataclass
import hashlib
import math
from functools import lru_cache
from pathlib import Path
import wave

import numpy as np
import tflite
import tvm
import vta
from tvm import relay


MODEL_SHA256 = "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
INPUT_NAME = "input_1"
INPUT_SHAPE = (1, 49, 10, 1)
INPUT_DTYPE = "int8"
INPUT_SCALE = 0.5847029
INPUT_ZERO_POINT = 83
OUTPUT_NAME = "Identity"
OUTPUT_SHAPE = (1, 12)
OUTPUT_DTYPE = "int8"
SAMPLE_RATE = 16000
CLIP_FRAMES = 16000
WINDOW_FRAMES = 480
STRIDE_FRAMES = 320
FFT_LENGTH = 512
MEL_BINS = 40
MFCCS = 10
VTA_MODULE_NAME = "mlperf_kws"
EXPECTED_TFLITE_OPERATORS = (
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "AVERAGE_POOL_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
)


@dataclass(frozen=True)
class ImportedModel:
    """The authenticated TFLite model and its typed Relay import."""

    module: tvm.IRModule
    params: dict
    model_sha256: str
    input_name: str
    input_shape: tuple
    input_dtype: str
    input_scale: float
    input_zero_point: int
    output_name: str
    output_shape: tuple
    output_dtype: str
    tflite_operator_names: tuple


@dataclass(frozen=True)
class RoutingSummary:
    """Structural summary of the fixed VTA/host partition boundary."""

    symbols: tuple
    convolutions_per_partition: tuple
    host_operator_names: tuple
    composite_names: tuple


@dataclass(frozen=True)
class PreparedModel:
    """One quantized graph forked into reference and VTA-partitioned forms."""

    imported: ImportedModel
    quantized_module: tvm.IRModule
    reference_module: tvm.IRModule
    mixed_module: tvm.IRModule
    routing: RoutingSummary


def _shape(expr):
    return tuple(int(dimension) for dimension in expr.checked_type.shape)


def _tensor_shape(tensor):
    return tuple(int(dimension) for dimension in tensor.ShapeAsNumpy())


def _tensor_name(tensor):
    return tensor.Name().decode("utf-8")


def _operator_name_map():
    return {
        value: name
        for name, value in vars(tflite.BuiltinOperator).items()
        if not name.startswith("_") and isinstance(value, int)
    }


def _quantization(tensor):
    quantization = tensor.Quantization()
    if quantization is None or quantization.ScaleLength() != 1 or quantization.ZeroPointLength() != 1:
        raise ValueError(f"tensor { _tensor_name(tensor)!r} must have one quantization scale and zero point")
    return float(quantization.Scale(0)), int(quantization.ZeroPoint(0))


def _flatbuffer_contract(model_bytes):
    try:
        model = tflite.Model.GetRootAsModel(model_bytes, 0)
    except Exception as error:
        raise ValueError("model is not a valid TFLite FlatBuffer") from error

    if model.Version() != 3 or model.SubgraphsLength() != 1:
        raise ValueError("model must contain exactly one TFLite v3 subgraph")
    graph = model.Subgraphs(0)
    if graph.InputsLength() != 1 or graph.OutputsLength() != 1:
        raise ValueError("model must have exactly one input and one output")

    input_tensor = graph.Tensors(graph.Inputs(0))
    output_tensor = graph.Tensors(graph.Outputs(0))
    input_contract = (_tensor_name(input_tensor), _tensor_shape(input_tensor), int(input_tensor.Type()))
    output_contract = (_tensor_name(output_tensor), _tensor_shape(output_tensor), int(output_tensor.Type()))
    int8_type = int(tflite.TensorType.INT8)
    if input_contract != (INPUT_NAME, INPUT_SHAPE, int8_type):
        raise ValueError(f"unexpected model input contract: {input_contract}")
    if output_contract != (OUTPUT_NAME, OUTPUT_SHAPE, int8_type):
        raise ValueError(f"unexpected model output contract: {output_contract}")
    input_scale, input_zero_point = _quantization(input_tensor)
    if not math.isclose(input_scale, INPUT_SCALE, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError(f"unexpected input quantization scale: {input_scale}")
    if input_zero_point != INPUT_ZERO_POINT:
        raise ValueError(f"unexpected input quantization zero point: {input_zero_point}")

    names_by_code = _operator_name_map()
    operator_names = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())
        operator_names.append(names_by_code.get(code, f"UNKNOWN_{code}"))
    operator_names = tuple(operator_names)
    if operator_names != EXPECTED_TFLITE_OPERATORS:
        raise ValueError(f"unexpected TFLite operator topology: {operator_names}")
    return model, input_scale, input_zero_point, operator_names


def _relay_operator_names(function):
    operator_names = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            operator_names.append(node.op.name)

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(operator_names)


def import_model(model_path):
    """Authenticate and import the committed int8 TFLite model exactly once."""
    model_path = Path(model_path)
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if model_sha256 != MODEL_SHA256:
        raise ValueError(
            f"model SHA-256 mismatch: expected {MODEL_SHA256}, received {model_sha256}"
        )

    model, input_scale, input_zero_point, operator_names = _flatbuffer_contract(model_bytes)
    module, params = relay.frontend.from_tflite(
        model,
        shape_dict={INPUT_NAME: INPUT_SHAPE},
        dtype_dict={INPUT_NAME: INPUT_DTYPE},
    )
    module = relay.transform.InferType()(module)
    main = module["main"]
    relay_input = (_shape(main.params[0]), main.params[0].checked_type.dtype)
    relay_output = (tuple(int(dimension) for dimension in main.ret_type.shape), main.ret_type.dtype)
    if relay_input != (INPUT_SHAPE, INPUT_DTYPE):
        raise ValueError(f"unexpected Relay input contract: {relay_input}")
    if relay_output != (OUTPUT_SHAPE, OUTPUT_DTYPE):
        raise ValueError(f"unexpected Relay output contract: {relay_output}")

    return ImportedModel(
        module=module,
        params=dict(params),
        model_sha256=model_sha256,
        input_name=INPUT_NAME,
        input_shape=INPUT_SHAPE,
        input_dtype=INPUT_DTYPE,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        output_name=OUTPUT_NAME,
        output_shape=OUTPUT_SHAPE,
        output_dtype=OUTPUT_DTYPE,
        tflite_operator_names=operator_names,
    )


class _VTAReadyMutator(relay.ExprMutator):
    """Normalize TFLite qnn edges to the repository VTA composite contract."""

    def visit_call(self, call):
        rewritten = super().visit_call(call)
        if not isinstance(rewritten.op, tvm.ir.Op):
            return rewritten
        operator_name = rewritten.op.name

        if operator_name == "fixed_point_multiply_per_axis":
            shifts = rewritten.args[-1].data.numpy().reshape(-1)
            return relay.right_shift(rewritten.args[0], relay.const(int(max(shifts)), "int32"))
        if operator_name == "fixed_point_multiply":
            shift = int(rewritten.attrs.shift)
            if shift < 0:
                return relay.left_shift(rewritten.args[0], relay.const(-shift, "int32"))
            return relay.right_shift(rewritten.args[0], relay.const(shift, "int32"))

        if operator_name == "cast" and str(rewritten.attrs.dtype) == "int32":
            original_type = getattr(call.args[0], "checked_type", None)
            if original_type is not None and str(original_type.dtype) == "int32":
                return rewritten.args[0]

        if operator_name == "nn.conv2d":
            data, weight = rewritten.args
            if (
                isinstance(weight, relay.Call)
                and isinstance(weight.op, tvm.ir.Op)
                and weight.op.name == "cast"
                and isinstance(weight.args[0], relay.Constant)
            ):
                attrs = rewritten.attrs
                return relay.nn.conv2d(
                    data,
                    weight.args[0],
                    channels=int(attrs.channels),
                    kernel_size=tuple(int(value) for value in attrs.kernel_size),
                    strides=tuple(int(value) for value in attrs.strides),
                    padding=tuple(int(value) for value in attrs.padding),
                    dilation=tuple(int(value) for value in attrs.dilation),
                    groups=int(attrs.groups),
                    data_layout=str(attrs.data_layout),
                    kernel_layout=str(attrs.kernel_layout),
                    out_layout=str(attrs.out_layout) or None,
                    out_dtype=str(attrs.out_dtype),
                )

        if operator_name == "subtract":
            left = rewritten.args[0]
            if isinstance(left, relay.Call) and isinstance(left.op, tvm.ir.Op) and left.op.name == "nn.conv2d":
                return left

        if operator_name == "add":
            left, right = rewritten.args
            if isinstance(left, relay.Call) and isinstance(left.op, tvm.ir.Op) and left.op.name == "right_shift":
                return left
            if isinstance(right, relay.Call) and isinstance(right.op, tvm.ir.Op) and right.op.name == "right_shift":
                return right

        return rewritten


def _normalize_for_vta(module):
    canonical = relay.qnn.transform.CanonicalizeOps()(module)
    canonical = relay.transform.InferType()(canonical)
    main = canonical["main"]
    body = _VTAReadyMutator().visit(main.body)
    normalized = tvm.IRModule.from_expr(
        relay.Function(main.params, body, type_params=main.type_params, attrs=main.attrs)
    )
    return relay.transform.InferType()(normalized)


def quantize_model(imported):
    """Run the approved Relay quantization pass once, then normalize VTA edges."""
    missing = object()
    previous_math = getattr(np, "math", missing)
    np.math = math
    try:
        with relay.quantize.qconfig(
            calibrate_mode="global_scale",
            global_scale=8.0,
            skip_conv_layers=[0],
        ):
            quantized = relay.quantize.quantize(imported.module, params=imported.params)
    finally:
        if previous_math is missing:
            delattr(np, "math")
        else:
            np.math = previous_math
    if not isinstance(quantized, tvm.IRModule):
        return quantized
    return _normalize_for_vta(quantized)


def _external_functions(module):
    external = []
    for global_var, function in module.functions.items():
        if (
            isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
            and function.attrs.get_str("Compiler") == "vta"
        ):
            external.append((function.attrs.get_str("global_symbol"), global_var, function))
    return tuple(sorted(external, key=lambda item: int(item[0].rsplit("_", 1)[1])))


def _composite_names(function):
    names = []

    def visit(node):
        if isinstance(node, relay.Function) and node.attrs is not None and "Composite" in node.attrs:
            names.append(node.attrs.get_str("Composite"))

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(names)


def inspect_partitioning(reference_module, mixed_module):
    """Validate non-empty deterministic VTA regions and host routing."""
    external = _external_functions(mixed_module)
    if not external:
        raise ValueError("KWS model produced no VTA partitions")
    symbols = tuple(item[0] for item in external)
    expected_prefix = f"tvmgen_{VTA_MODULE_NAME}_vta_main_"
    if not all(symbol.startswith(expected_prefix) for symbol in symbols):
        raise ValueError(f"unexpected VTA symbols: {symbols}")
    convolutions = tuple(_relay_operator_names(item[2]).count("nn.conv2d") for item in external)
    if not all(count > 0 for count in convolutions):
        raise ValueError(f"VTA partitions must contain convolution regions: {convolutions}")

    host_operator_names = tuple(sorted(set(_relay_operator_names(mixed_module["main"]))))
    required_host = {"nn.avg_pool2d", "nn.dense", "nn.softmax", "reshape"}
    if not required_host <= set(host_operator_names):
        missing = sorted(required_host - set(host_operator_names))
        raise ValueError(f"mixed main is missing required host operators: {missing}")
    composite_names = tuple(name for item in external for name in _composite_names(item[2]))
    routing = RoutingSummary(
        symbols=symbols,
        convolutions_per_partition=convolutions,
        host_operator_names=host_operator_names,
        composite_names=composite_names,
    )
    if _shape(reference_module["main"].params[0]) != INPUT_SHAPE:
        raise ValueError("reference graph input shape changed during partitioning")
    if tuple(int(dimension) for dimension in mixed_module["main"].ret_type.shape) != OUTPUT_SHAPE:
        raise ValueError("mixed graph output shape changed during partitioning")
    return routing


def prepare_model(model_path):
    """Import, quantize once, partition once, and validate the routing."""
    imported = import_model(model_path)
    quantized_module = quantize_model(imported)
    reference_module = quantized_module
    mixed_module = vta.relay.partition_for_vta(quantized_module, mod_name=VTA_MODULE_NAME)
    routing = inspect_partitioning(reference_module, mixed_module)
    return PreparedModel(
        imported=imported,
        quantized_module=quantized_module,
        reference_module=reference_module,
        mixed_module=mixed_module,
        routing=routing,
    )


@lru_cache(maxsize=1)
def _mel_filterbank():
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    points = mel_to_hz(np.linspace(hz_to_mel(20.0), hz_to_mel(4000.0), MEL_BINS + 2))
    bins = np.floor((FFT_LENGTH + 1) * points / SAMPLE_RATE).astype(np.int32)
    filters = np.zeros((MEL_BINS, FFT_LENGTH // 2 + 1), dtype=np.float32)
    for index in range(MEL_BINS):
        left, center, right = bins[index : index + 3]
        if center > left:
            filters[index, left:center] = np.arange(left, center) / float(center - left)
        if right > center:
            filters[index, center:right] = np.arange(center, right) / float(right - center)
    return filters


def _read_wav(sample_path):
    sample_path = Path(sample_path)
    try:
        with wave.open(str(sample_path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
                raise ValueError(f"{sample_path} must be mono 16-bit {SAMPLE_RATE} Hz WAV")
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    except (OSError, EOFError) as error:
        raise ValueError(f"unable to read WAV sample {sample_path}") from error
    if samples.size == 0:
        raise ValueError(f"{sample_path} contains no PCM frames")
    samples = samples[:CLIP_FRAMES]
    if samples.size < CLIP_FRAMES:
        samples = np.pad(samples, (0, CLIP_FRAMES - samples.size))
    return samples.astype(np.float32) / np.float32(32768.0)


def _mfcc(samples):
    frame_starts = range(0, CLIP_FRAMES - WINDOW_FRAMES + 1, STRIDE_FRAMES)
    frames = np.stack([samples[start : start + WINDOW_FRAMES] for start in frame_starts])
    window = np.hanning(WINDOW_FRAMES + 1)[:-1].astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(frames * window, n=FFT_LENGTH)).astype(np.float32)
    mel = np.maximum(magnitudes @ _mel_filterbank().T, np.float32(1e-6))
    log_mel = np.log(mel)
    positions = np.arange(MEL_BINS, dtype=np.float32) + np.float32(0.5)
    coefficients = np.arange(MFCCS, dtype=np.float32)[:, None]
    dct = np.cos(np.pi / MEL_BINS * coefficients * positions)
    dct[0] *= np.float32(1.0 / math.sqrt(MEL_BINS))
    dct[1:] *= np.float32(math.sqrt(2.0 / MEL_BINS))
    return log_mel @ dct.T


def quantize_features(features, scale=INPUT_SCALE, zero_point=INPUT_ZERO_POINT):
    """Quantize MFCCs once using the committed TFLite input contract."""
    if features.shape != (49, 10) or not np.isfinite(features).all():
        raise ValueError(f"unexpected MFCC feature matrix: {features.shape}")
    quantized = np.rint(features / np.float32(scale) + np.float32(zero_point))
    return np.clip(quantized, -128, 127).astype(np.int8)[None, :, :, None]


def load_sample(sample_path):
    """Load one WAV and return the model's deterministic int8 MFCC tensor."""
    return quantize_features(_mfcc(_read_wav(sample_path)))
