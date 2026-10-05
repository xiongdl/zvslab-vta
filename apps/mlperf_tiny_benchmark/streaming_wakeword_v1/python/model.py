"""Streaming wakeword input preparation and semantically exact Relay import."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import wave

import numpy as np
import tflite
import tvm
from tvm import relay


SAMPLE_RATE = 16000
CLIP_FRAMES = SAMPLE_RATE
WINDOW_SIZE_SAMPLES = 1024
WINDOW_STRIDE_SAMPLES = 512
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
LABELS = ("Marvin", "Silence", "Unknown")
EXPECTED_TFLITE_OPERATORS = (
    "DEPTHWISE_CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D",
    "DEPTHWISE_CONV_2D", "CONV_2D", "DEPTHWISE_CONV_2D", "CONV_2D",
    "RESHAPE", "FULLY_CONNECTED", "SOFTMAX",
)


@dataclass(frozen=True)
class ImportedModel:
    module: tvm.IRModule
    params: dict
    model_sha256: str
    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int
    tflite_operator_names: tuple


@dataclass(frozen=True)
class PreparedModel:
    imported: ImportedModel
    reference_module: tvm.IRModule
    mixed_module: tvm.IRModule | None
    routing: "RoutingSummary"
    fallback_reason: str | None


@dataclass(frozen=True)
class RoutingSummary:
    symbols: tuple


@lru_cache(maxsize=1)
def _mel_filterbank():
    def hertz_to_mel(hertz):
        return np.float64(1127.0) * np.log1p(np.float64(hertz) / 700.0)

    def mel_to_hertz(mel):
        return 700.0 * np.expm1(np.asarray(mel, dtype=np.float64) / 1127.0)

    lower = hertz_to_mel(0.0)
    upper = hertz_to_mel(SAMPLE_RATE / 2.0)
    edges = mel_to_hertz(np.linspace(lower, upper, MEL_BINS + 2))
    spectrum = np.linspace(0.0, SAMPLE_RATE / 2.0, FFT_LENGTH // 2 + 1)
    lower_slope = (spectrum[None, :] - edges[:-2, None]) / (
        edges[1:-1, None] - edges[:-2, None]
    )
    upper_slope = (edges[2:, None] - spectrum[None, :]) / (
        edges[2:, None] - edges[1:-1, None]
    )
    return np.maximum(0.0, np.minimum(lower_slope, upper_slope)).astype(np.float32)


def _read_wav(sample_path):
    sample_path = Path(sample_path)
    try:
        with wave.open(str(sample_path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (
                1, 2, SAMPLE_RATE, "NONE"
            ):
                raise ValueError(f"{sample_path} must be mono 16-bit {SAMPLE_RATE} Hz WAV")
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
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
    preemphasis = np.float32(1.0 - 2.0 ** -5)
    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - preemphasis * samples[:-1]
    starts = np.arange(0, CLIP_FRAMES - WINDOW_SIZE_SAMPLES + 1, WINDOW_STRIDE_SAMPLES)
    frames = np.stack([emphasized[start:start + WINDOW_SIZE_SAMPLES] for start in starts])
    window = np.hamming(WINDOW_SIZE_SAMPLES).astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(frames * window, n=FFT_LENGTH)).astype(np.float32)
    power = (np.square(magnitudes) / np.float32(WINDOW_SIZE_SAMPLES)).astype(np.float32)
    peak = max(float(power.max()), 1e-30)
    power = np.clip(power, np.float32(1e-30), np.float32(peak))
    mel = np.tensordot(power, _mel_filterbank(), axes=([-1], [1])).astype(np.float32)
    mel = np.maximum(mel, np.float32(1e-30))
    log_mel = np.float32(10.0) * np.log10(mel).astype(np.float32)
    log_mel = (log_mel + np.float32(POWER_OFFSET)) / 64.0
    return np.clip(log_mel, 0.0, 1.0).astype(np.float32)


def quantize_features(features, scale=INPUT_SCALE, zero_point=INPUT_ZERO_POINT):
    features = np.asarray(features)
    if features.shape != (30, 40) or features.dtype.kind not in "fc":
        raise ValueError(f"unexpected log-mel feature matrix: {features.shape}, {features.dtype}")
    if not np.isfinite(features).all() or float(features.min()) < 0.0 or float(features.max()) > 1.0:
        raise ValueError("log-mel features must be finite and clipped to [0, 1]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("input quantization scale must be positive and finite")
    quantized = np.rint(features / np.float32(scale) + np.float32(zero_point))
    return np.clip(quantized, -128, 127).astype(np.int8)[None, :, None, :]


def load_sample(sample_path):
    return quantize_features(_log_mel_features(_read_wav(sample_path)))


def _tensor_shape(tensor):
    return tuple(int(dimension) for dimension in tensor.ShapeAsNumpy())


def _operator_name_map():
    return {value: name for name, value in vars(tflite.BuiltinOperator).items()
            if not name.startswith("_") and isinstance(value, int)}


def _tensor_quantization(tensor):
    quant = tensor.Quantization()
    if quant is None or quant.ScaleLength() != 1 or quant.ZeroPointLength() != 1:
        raise ValueError(f"tensor {tensor.Name()!r} must have one scale and zero point")
    return float(quant.Scale(0)), int(quant.ZeroPoint(0))


def import_model(model_path):
    model_path = Path(model_path).expanduser().resolve()
    model_bytes = model_path.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    try:
        model = tflite.Model.GetRootAsModel(model_bytes, 0)
        if model.Version() != 3 or model.SubgraphsLength() != 1:
            raise ValueError("model must contain exactly one TFLite v3 subgraph")
        graph = model.Subgraphs(0)
        if graph.InputsLength() != 1 or graph.OutputsLength() != 1:
            raise ValueError("model must have exactly one input and one output")
        input_tensor = graph.Tensors(graph.Inputs(0))
        output_tensor = graph.Tensors(graph.Outputs(0))
        input_contract = (input_tensor.Name().decode(), _tensor_shape(input_tensor), int(input_tensor.Type()))
        output_contract = (output_tensor.Name().decode(), _tensor_shape(output_tensor), int(output_tensor.Type()))
        if input_contract != (INPUT_NAME, INPUT_SHAPE, int(tflite.TensorType.INT8)):
            raise ValueError(f"unexpected model input contract: {input_contract}")
        if output_contract != (OUTPUT_NAME, OUTPUT_SHAPE, int(tflite.TensorType.INT8)):
            raise ValueError(f"unexpected model output contract: {output_contract}")
        input_scale, input_zero = _tensor_quantization(input_tensor)
        output_scale, output_zero = _tensor_quantization(output_tensor)
        if not math.isclose(input_scale, INPUT_SCALE, rel_tol=1e-6, abs_tol=1e-9) or input_zero != INPUT_ZERO_POINT:
            raise ValueError("model input quantization differs from the supported contract")
        if not math.isclose(output_scale, OUTPUT_SCALE, rel_tol=1e-6, abs_tol=1e-9) or output_zero != OUTPUT_ZERO_POINT:
            raise ValueError("model output quantization differs from the supported contract")
        operator_names = []
        names = _operator_name_map()
        for index in range(graph.OperatorsLength()):
            operator = graph.Operators(index)
            code = int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())
            operator_names.append(names.get(code, f"UNKNOWN_{code}"))
        operator_names = tuple(operator_names)
        if operator_names != EXPECTED_TFLITE_OPERATORS:
            raise ValueError(f"unexpected model operator topology: {operator_names}")
        module, params = relay.frontend.from_tflite(
            model, shape_dict={INPUT_NAME: INPUT_SHAPE}, dtype_dict={INPUT_NAME: INPUT_DTYPE}
        )
        module = relay.transform.InferType()(module)
        main = module["main"]
        relay_input = (
            tuple(int(dimension) for dimension in main.params[0].checked_type.shape),
            main.params[0].checked_type.dtype,
        )
        if tuple(int(x) for x in main.ret_type.shape) != OUTPUT_SHAPE or main.ret_type.dtype != OUTPUT_DTYPE:
            raise ValueError("imported Relay output differs from the supported tensor contract")
        if relay_input != (INPUT_SHAPE, INPUT_DTYPE):
            raise ValueError(f"imported Relay input differs from the supported tensor contract: {relay_input}")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("model is not a valid supported TFLite FlatBuffer") from error
    return ImportedModel(module, dict(params), digest, input_scale, input_zero,
                         output_scale, output_zero, operator_names)


def prepare_model(model_path, use_vta=False):
    """Canonicalize imported QNN ops without changing their fixed-point math."""
    imported = import_model(model_path)
    # This is the CPU graph as well as the only candidate graph for partitioning.
    # Its output must match imported QNN directly; no scalar-shift rewriting.
    canonical = relay.transform.InferType()(
        relay.qnn.transform.CanonicalizeOps()(imported.module)
    )
    if not use_vta:
        return PreparedModel(imported, canonical, None, RoutingSummary(()), None)

    import vta

    mixed = vta.relay.partition_for_vta(
        canonical.clone(), params=imported.params, mod_name=VTA_MODULE_NAME
    )
    regions = []
    for function in mixed.functions.values():
        if isinstance(function, relay.Function) and function.attrs is not None and "Compiler" in function.attrs:
            if function.attrs.get_str("Compiler") == "vta":
                regions.append(function.attrs.get_str("global_symbol"))
    symbols = tuple(sorted(regions))
    return PreparedModel(imported, canonical, mixed if symbols else None, RoutingSummary(symbols),
                         None if symbols else "model has no real VTA partitions after exact QNN canonicalization")
