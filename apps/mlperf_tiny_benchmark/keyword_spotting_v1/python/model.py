"""Deterministic float-feature preprocessing and quantized Relay preparation."""

from dataclasses import dataclass
import hashlib
import math
from functools import lru_cache
from pathlib import Path
import wave

import numpy as np
import tflite
import tvm
from tvm import relay


MODEL_ID = "keyword_spotting_v1"
CLASS_NAMES = (
    "Down", "Go", "Left", "No", "Off", "On", "Right", "Stop", "Up", "Yes",
    "Silence", "Unknown",
)
PREPROCESSING_POLICY = (
    "mono PCM16 16 kHz, pad/trim to one second, deterministic 49x10 MFCC, "
    "float32 MFCC features"
)
QUANTIZATION = {
    "calibrate_mode": "global_scale",
    "global_scale": 8.0,
    "skip_conv_layers": [0],
}
MODEL_SHA256 = "738a9f29d175aaa3928db9c8281265be5ec3406598fd3d30018b26084a3d5536"
INPUT_SHAPE = (1, 49, 10, 1)
INPUT_NAME = "serving_default_input_1:0"
INPUT_DTYPE = "float32"
OUTPUT_SHAPE = (1, 12)
OUTPUT_NAME = "StatefulPartitionedCall:0"
OUTPUT_DTYPE = "float32"
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
    """The TFLite model and its typed Relay import."""

    module: tvm.IRModule
    params: dict
    model_sha256: str
    input_name: str
    input_shape: tuple
    input_dtype: str
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
    """Arithmetic-preserving CPU graph and optional real VTA partition."""

    imported: ImportedModel
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
    float_type = int(tflite.TensorType.FLOAT32)
    if input_contract != (INPUT_NAME, INPUT_SHAPE, float_type):
        raise ValueError(f"unexpected model input contract: {input_contract}")
    if output_contract != (OUTPUT_NAME, OUTPUT_SHAPE, float_type):
        raise ValueError(f"unexpected model output contract: {output_contract}")

    names_by_code = _operator_name_map()
    operator_names = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())
        operator_names.append(names_by_code.get(code, f"UNKNOWN_{code}"))
    operator_names = tuple(operator_names)
    if operator_names != EXPECTED_TFLITE_OPERATORS:
        raise ValueError(f"unexpected TFLite operator topology: {operator_names}")
    return model, operator_names


def _relay_operator_names(function):
    operator_names = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            operator_names.append(node.op.name)

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(operator_names)


def import_model(model_path):
    """Import the supported all-float TFLite model and record its content hash."""
    model_path = Path(model_path)
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    model, operator_names = _flatbuffer_contract(model_bytes)
    graph = model.Subgraphs(0)
    input_name = _tensor_name(graph.Tensors(graph.Inputs(0)))
    output_name = _tensor_name(graph.Tensors(graph.Outputs(0)))
    module, params = relay.frontend.from_tflite(
        model, shape_dict={input_name: INPUT_SHAPE}, dtype_dict={input_name: INPUT_DTYPE}
    )
    module = relay.transform.InferType()(module)
    main = module["main"]
    relay_input = (_shape(main.params[0]), main.params[0].checked_type.dtype)
    relay_output = (tuple(int(dimension) for dimension in main.ret_type.shape), main.ret_type.dtype)
    if relay_input != (INPUT_SHAPE, INPUT_DTYPE) or relay_output != (OUTPUT_SHAPE, OUTPUT_DTYPE):
        raise ValueError(f"unexpected Relay tensor contract: {relay_input} -> {relay_output}")

    return ImportedModel(
        module=module,
        params=dict(params),
        model_sha256=model_sha256,
        input_name=input_name,
        input_shape=INPUT_SHAPE,
        input_dtype=INPUT_DTYPE,
        output_name=output_name,
        output_shape=OUTPUT_SHAPE,
        output_dtype=OUTPUT_DTYPE,
        tflite_operator_names=operator_names,
    )


def quantize_model(imported):
    """Quantize float Relay once using the reference image-classification policy."""
    with relay.quantize.qconfig(
        calibrate_mode="global_scale", global_scale=8.0, skip_conv_layers=[0]
    ):
        missing = object()
        previous_math = getattr(np, "math", missing)
        np.math = math
        try:
            return relay.quantize.quantize(imported.module, params=imported.params)
        finally:
            if previous_math is missing:
                delattr(np, "math")
            else:
                np.math = previous_math


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
    """Count real VTA partitions and preserve unsupported work on CPU."""
    external = _external_functions(mixed_module)
    symbols = tuple(item[0] for item in external)
    expected_prefix = f"tvmgen_{VTA_MODULE_NAME}_vta_main_"
    if not all(symbol.startswith(expected_prefix) for symbol in symbols):
        raise ValueError(f"unexpected VTA symbols: {symbols}")
    convolutions = tuple(_relay_operator_names(item[2]).count("nn.conv2d") for item in external)
    if any(count <= 0 for count in convolutions):
        raise ValueError(f"VTA partitions must contain real convolutions: {convolutions}")
    host_operator_names = tuple(sorted(set(_relay_operator_names(mixed_module["main"]))))
    composite_names = tuple(name for item in external for name in _composite_names(item[2]))
    if _shape(reference_module["main"].params[0]) != INPUT_SHAPE:
        raise ValueError("reference graph input shape changed during partitioning")
    if tuple(int(dimension) for dimension in mixed_module["main"].ret_type.shape) != OUTPUT_SHAPE:
        raise ValueError("mixed graph output shape changed during partitioning")
    return RoutingSummary(
        symbols=symbols,
        convolutions_per_partition=convolutions,
        host_operator_names=host_operator_names,
        composite_names=composite_names,
    )


def prepare_model(model_path, *, use_vta=True):
    """Prepare an exact CPU graph and optionally inspect real VTA partitions."""
    imported = import_model(model_path)
    reference_module = quantize_model(imported)
    if use_vta:
        import vta

        mixed_module = vta.relay.partition_for_vta(reference_module, mod_name=VTA_MODULE_NAME)
        routing = inspect_partitioning(reference_module, mixed_module)
    else:
        mixed_module = routing = None
    return PreparedModel(
        imported=imported,
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
    except (OSError, EOFError, wave.Error) as error:
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


def load_sample(sample_path):
    """Return unchanged float32 MFCC features for the float TFLite input."""
    features = _mfcc(_read_wav(sample_path))
    if features.shape != (49, 10) or not np.isfinite(features).all():
        raise ValueError(f"unexpected MFCC feature matrix: {features.shape}")
    return np.asarray(features, dtype=np.float32)[None, :, :, None]
