"""Deterministic streaming wakeword preprocessing.

The feature path is a NumPy transcription of the training ``get_lfbe_func``
graph.  Model import and Relay preparation are kept below this self-contained
audio path so runtime preprocessing never needs the training environment.
"""

from functools import lru_cache
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import wave

import numpy as np
import tflite
import tvm
import vta
from tvm import relay


MODEL_SHA256 = "3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b"
SAMPLE_RATE = 16000
CLIP_FRAMES = SAMPLE_RATE
WINDOW_SIZE_SAMPLES = 1024  # 64 ms; TensorFlow's implicit FFT length is 1024.
WINDOW_STRIDE_SAMPLES = 512  # 32 ms
FFT_LENGTH = 1024
MEL_BINS = 40
POWER_OFFSET = 52.0
INPUT_SCALE = 0.003701042616739869
INPUT_ZERO_POINT = -128
INPUT_SHAPE = (1, 30, 1, 40)
INPUT_NAME = "serving_default_input_1:0"
INPUT_DTYPE = "int8"
OUTPUT_NAME = "StatefulPartitionedCall:0"
OUTPUT_SHAPE = (1, 3)
OUTPUT_DTYPE = "int8"
OUTPUT_SCALE = 0.00390625
OUTPUT_ZERO_POINT = -128
VTA_MODULE_NAME = "mlperf_streaming_wakeword"
EXPECTED_TFLITE_OPERATORS = (
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
)


@dataclass(frozen=True)
class ImportedModel:
    """Authenticated TFLite bytes and their typed Relay import."""

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
    output_scale: float
    output_zero_point: int
    tflite_operator_names: tuple


@dataclass(frozen=True)
class RoutingSummary:
    """Structural summary of the reference/mixed VTA boundary."""

    symbols: tuple
    convolutions_per_partition: tuple
    host_operator_names: tuple
    composite_names: tuple


@dataclass(frozen=True)
class PreparedModel:
    """One normalized reference graph and its once-partitioned mixed graph."""

    imported: ImportedModel
    normalized_module: tvm.IRModule
    reference_module: tvm.IRModule
    mixed_module: tvm.IRModule
    routing: RoutingSummary


@lru_cache(maxsize=1)
def _mel_filterbank():
    """Return TensorFlow's linear-to-mel triangular filterbank in float32."""
    def hertz_to_mel(hertz):
        return np.float64(1127.0) * np.log1p(np.float64(hertz) / 700.0)

    def mel_to_hertz(mel):
        return 700.0 * np.expm1(np.asarray(mel, dtype=np.float64) / 1127.0)

    lower_edge_mel = hertz_to_mel(0.0)
    upper_edge_mel = hertz_to_mel(SAMPLE_RATE / 2.0)
    band_edges_mel = np.linspace(lower_edge_mel, upper_edge_mel, MEL_BINS + 2)
    band_edges_hertz = mel_to_hertz(band_edges_mel)
    spectrogram_hertz = np.linspace(
        0.0, SAMPLE_RATE / 2.0, FFT_LENGTH // 2 + 1, dtype=np.float64
    )

    lower_edges = band_edges_hertz[:-2, None]
    center_edges = band_edges_hertz[1:-1, None]
    upper_edges = band_edges_hertz[2:, None]
    lower_slopes = (spectrogram_hertz[None, :] - lower_edges) / (
        center_edges - lower_edges
    )
    upper_slopes = (upper_edges - spectrogram_hertz[None, :]) / (
        upper_edges - center_edges
    )
    return np.maximum(0.0, np.minimum(lower_slopes, upper_slopes)).astype(
        np.float32
    )


def _read_wav(sample_path):
    """Read strict mono PCM16/16 kHz WAV and normalize to float32."""
    sample_path = Path(sample_path)
    try:
        with wave.open(str(sample_path), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getframerate() != SAMPLE_RATE
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError(
                    f"{sample_path} must be mono 16-bit {SAMPLE_RATE} Hz WAV"
                )
            samples = np.frombuffer(
                wav.readframes(wav.getnframes()), dtype="<i2"
            ).copy()
    except ValueError:
        raise
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"unable to read WAV sample {sample_path}") from error

    if samples.size == 0:
        raise ValueError(f"{sample_path} contains no PCM frames")
    samples = samples[:CLIP_FRAMES]
    if samples.size < CLIP_FRAMES:
        samples = np.pad(samples, (0, CLIP_FRAMES - samples.size))
    return samples.astype(np.float32) / np.float32(32768.0)


def _log_mel_features(samples):
    """Compute the training graph's 30x40 normalized log-mel matrix."""
    if samples.shape != (CLIP_FRAMES,) or samples.dtype != np.float32:
        raise ValueError(f"unexpected normalized audio shape/dtype: {samples.shape}, {samples.dtype}")

    preemphasis = np.float32(1.0 - 2.0 ** -5)
    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - preemphasis * samples[:-1]

    starts = np.arange(
        0, CLIP_FRAMES - WINDOW_SIZE_SAMPLES + 1, WINDOW_STRIDE_SAMPLES
    )
    frames = np.stack(
        [emphasized[start : start + WINDOW_SIZE_SAMPLES] for start in starts]
    )
    window = np.hamming(WINDOW_SIZE_SAMPLES).astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(frames * window, n=FFT_LENGTH)).astype(
        np.float32
    )
    power = (np.square(magnitudes) / np.float32(WINDOW_SIZE_SAMPLES)).astype(
        np.float32
    )
    peak = max(float(power.max()), 1e-30)
    power = np.clip(power, np.float32(1e-30), np.float32(peak))
    mel = np.tensordot(power, _mel_filterbank(), axes=([-1], [1])).astype(
        np.float32
    )
    mel = np.maximum(mel, np.float32(1e-30))
    log_mel = np.float32(10.0) * np.log10(mel).astype(np.float32)
    log_mel = (log_mel + np.float32(POWER_OFFSET) - 32.0 + 32.0) / 64.0
    return np.clip(log_mel, 0.0, 1.0).astype(np.float32)


def quantize_features(features, scale=INPUT_SCALE, zero_point=INPUT_ZERO_POINT):
    """Quantize a validated 30x40 normalized log-mel matrix once."""
    features = np.asarray(features)
    if features.shape != (30, 40) or features.dtype.kind not in "fc":
        raise ValueError(f"unexpected log-mel feature matrix: {features.shape}, {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("log-mel features must be finite")
    if float(features.min()) < 0.0 or float(features.max()) > 1.0:
        raise ValueError("log-mel features must be clipped to [0, 1]")
    if scale <= 0:
        raise ValueError("input quantization scale must be positive")
    quantized = np.rint(features / np.float32(scale) + np.float32(zero_point))
    return np.clip(quantized, -128, 127).astype(np.int8)[None, :, None, :]


def load_sample(sample_path):
    """Load one strict WAV and return the model's int8 input tensor."""
    return quantize_features(_log_mel_features(_read_wav(sample_path)))


def _shape(expr):
    return tuple(int(dimension) for dimension in expr.checked_type.shape)


def _return_shape(function):
    return tuple(int(dimension) for dimension in function.ret_type.shape)


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
    if (
        quantization is None
        or quantization.ScaleLength() != 1
        or quantization.ZeroPointLength() != 1
    ):
        raise ValueError(
            f"tensor {_tensor_name(tensor)!r} must have one quantization scale and zero point"
        )
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
    int8_type = int(tflite.TensorType.INT8)
    input_contract = (
        _tensor_name(input_tensor),
        _tensor_shape(input_tensor),
        int(input_tensor.Type()),
    )
    output_contract = (
        _tensor_name(output_tensor),
        _tensor_shape(output_tensor),
        int(output_tensor.Type()),
    )
    if input_contract != (INPUT_NAME, INPUT_SHAPE, int8_type):
        raise ValueError(f"unexpected model input contract: {input_contract}")
    if output_contract != (OUTPUT_NAME, OUTPUT_SHAPE, int8_type):
        raise ValueError(f"unexpected model output contract: {output_contract}")

    input_scale, input_zero_point = _quantization(input_tensor)
    output_scale, output_zero_point = _quantization(output_tensor)
    if not math.isclose(input_scale, INPUT_SCALE, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"unexpected input quantization scale: {input_scale}")
    if input_zero_point != INPUT_ZERO_POINT:
        raise ValueError(
            f"unexpected input quantization zero point: {input_zero_point}"
        )
    if not math.isclose(output_scale, OUTPUT_SCALE, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"unexpected output quantization scale: {output_scale}")
    if output_zero_point != OUTPUT_ZERO_POINT:
        raise ValueError(
            f"unexpected output quantization zero point: {output_zero_point}"
        )

    names_by_code = _operator_name_map()
    operator_names = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())
        operator_names.append(names_by_code.get(code, f"UNKNOWN_{code}"))
    operator_names = tuple(operator_names)
    if operator_names != EXPECTED_TFLITE_OPERATORS:
        raise ValueError(f"unexpected TFLite operator topology: {operator_names}")
    return (
        model,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        operator_names,
    )


def _relay_operator_names(function):
    operator_names = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            operator_names.append(node.op.name)

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(operator_names)


def import_model(model_path):
    """Authenticate and import the committed int8 TFLite model."""
    model_path = Path(model_path)
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if model_sha256 != MODEL_SHA256:
        raise ValueError(
            f"model SHA-256 mismatch: expected {MODEL_SHA256}, received {model_sha256}"
        )

    (
        model,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        operator_names,
    ) = _flatbuffer_contract(model_bytes)
    module, params = relay.frontend.from_tflite(
        model,
        shape_dict={INPUT_NAME: INPUT_SHAPE},
        dtype_dict={INPUT_NAME: INPUT_DTYPE},
    )
    module = relay.transform.InferType()(module)
    main = module["main"]
    relay_input = (_shape(main.params[0]), main.params[0].checked_type.dtype)
    relay_output = (_return_shape(main), main.ret_type.dtype)
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
        output_scale=output_scale,
        output_zero_point=output_zero_point,
        tflite_operator_names=operator_names,
    )


class _VTAReadyMutator(relay.ExprMutator):
    """Lower imported qnn arithmetic into the VTA composite contract."""

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
            if (
                isinstance(left, relay.Call)
                and isinstance(left.op, tvm.ir.Op)
                and left.op.name == "nn.conv2d"
            ):
                return left

        if operator_name == "add":
            left, right = rewritten.args
            if (
                isinstance(left, relay.Call)
                and isinstance(left.op, tvm.ir.Op)
                and left.op.name == "right_shift"
            ):
                return left
            if (
                isinstance(right, relay.Call)
                and isinstance(right.op, tvm.ir.Op)
                and right.op.name == "right_shift"
            ):
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


def normalize_model(imported):
    """Apply the required qnn-to-VTA normalization exactly once."""
    return _normalize_for_vta(imported.module)


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
    return tuple(sorted(external, key=lambda item: item[0]))


def _composite_names(function):
    names = []

    def visit(node):
        if isinstance(node, relay.Function) and node.attrs is not None and "Composite" in node.attrs:
            names.append(node.attrs.get_str("Composite"))

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(names)


def inspect_partitioning(reference_module, mixed_module):
    """Validate VTA regions, symbols, convolution content, and graph IO."""
    external = _external_functions(mixed_module)
    if not external:
        raise ValueError("streaming wakeword model produced no VTA partitions")
    symbols = tuple(item[0] for item in external)
    expected_prefix = f"tvmgen_{VTA_MODULE_NAME}_vta_main_"
    if not all(
        symbol.startswith(expected_prefix)
        and symbol[len(expected_prefix) :].isdigit()
        for symbol in symbols
    ):
        raise ValueError(f"unexpected VTA symbols: {symbols}")
    convolutions = tuple(
        _relay_operator_names(item[2]).count("nn.conv2d") for item in external
    )
    if not all(count > 0 for count in convolutions):
        raise ValueError(f"VTA partitions must contain convolution regions: {convolutions}")

    host_operator_names = tuple(sorted(set(_relay_operator_names(mixed_module["main"]))))
    required_host = {"nn.dense", "nn.softmax", "reshape"}
    if not required_host <= set(host_operator_names):
        missing = sorted(required_host - set(host_operator_names))
        raise ValueError(f"mixed main is missing required host operators: {missing}")
    composite_names = tuple(name for item in external for name in _composite_names(item[2]))
    if not composite_names or not all(name.startswith("vta.") for name in composite_names):
        raise ValueError(f"unexpected VTA composites: {composite_names}")

    reference_main = reference_module["main"]
    mixed_main = mixed_module["main"]
    reference_input = (_shape(reference_main.params[0]), reference_main.params[0].checked_type.dtype)
    reference_output = (_return_shape(reference_main), reference_main.ret_type.dtype)
    mixed_output = (_return_shape(mixed_main), mixed_main.ret_type.dtype)
    if reference_input != (INPUT_SHAPE, INPUT_DTYPE):
        raise ValueError(f"reference graph input contract changed: {reference_input}")
    if reference_output != (OUTPUT_SHAPE, OUTPUT_DTYPE):
        raise ValueError(f"reference graph output contract changed: {reference_output}")
    if mixed_output != (OUTPUT_SHAPE, OUTPUT_DTYPE):
        raise ValueError(f"mixed graph output contract changed: {mixed_output}")

    return RoutingSummary(
        symbols=symbols,
        convolutions_per_partition=convolutions,
        host_operator_names=host_operator_names,
        composite_names=composite_names,
    )


def prepare_model(model_path):
    """Import, normalize, partition once, and validate the routing boundary."""
    imported = import_model(model_path)
    normalized_module = normalize_model(imported)
    reference_module = normalized_module
    mixed_input = normalized_module.clone()
    mixed_module = vta.relay.partition_for_vta(
        mixed_input,
        params=imported.params,
        mod_name=VTA_MODULE_NAME,
    )
    routing = inspect_partitioning(reference_module, mixed_module)
    return PreparedModel(
        imported=imported,
        normalized_module=normalized_module,
        reference_module=reference_module,
        mixed_module=mixed_module,
        routing=routing,
    )
