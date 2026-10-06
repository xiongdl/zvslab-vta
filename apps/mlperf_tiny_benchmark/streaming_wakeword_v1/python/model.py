"""Streaming wakeword float-feature preparation and quantized Relay import."""

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
INPUT_SHAPE = (1, 30, 1, 40)
INPUT_NAME = "serving_default_input_1:0"
INPUT_DTYPE = "float32"
OUTPUT_NAME = "StatefulPartitionedCall:0"
OUTPUT_SHAPE = (1, 3)
OUTPUT_DTYPE = "float32"
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
    input_name: str
    input_shape: tuple
    input_dtype: str
    output_shape: tuple
    output_dtype: str
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


def load_sample(sample_path):
    """Return the existing normalized log-mel features as float32."""
    features = _log_mel_features(_read_wav(sample_path))
    if features.shape != (30, 40) or features.dtype != np.float32:
        raise ValueError(f"unexpected log-mel feature matrix: {features.shape}, {features.dtype}")
    if not np.isfinite(features).all() or float(features.min()) < 0.0 or float(features.max()) > 1.0:
        raise ValueError("log-mel features must be finite and clipped to [0, 1]")
    return features[None, :, None, :]


def _tensor_shape(tensor):
    return tuple(int(dimension) for dimension in tensor.ShapeAsNumpy())


def _operator_name_map():
    return {value: name for name, value in vars(tflite.BuiltinOperator).items()
            if not name.startswith("_") and isinstance(value, int)}


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
        if input_contract != (INPUT_NAME, INPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
            raise ValueError(f"unexpected model input contract: {input_contract}")
        if output_contract != (OUTPUT_NAME, OUTPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
            raise ValueError(f"unexpected model output contract: {output_contract}")
        tensor_types = {int(graph.Tensors(i).Type()) for i in range(graph.TensorsLength())}
        if any(int(dtype) in tensor_types for dtype in (
            tflite.TensorType.INT8, tflite.TensorType.UINT8, tflite.TensorType.INT16
        )):
            raise ValueError("model must contain float32 tensors, not an integer-quantized model")
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
    return ImportedModel(
        module, dict(params), digest, INPUT_NAME, INPUT_SHAPE, INPUT_DTYPE,
        OUTPUT_SHAPE, OUTPUT_DTYPE, operator_names,
    )


def prepare_model(model_path, use_vta=False):
    """Quantize float Relay and optionally partition the resulting graph for VTA."""
    imported = import_model(model_path)
    with relay.quantize.qconfig(
        calibrate_mode="global_scale", global_scale=8.0, skip_conv_layers=[0]
    ):
        missing = object()
        previous_math = getattr(np, "math", missing)
        np.math = math
        try:
            canonical = relay.quantize.quantize(imported.module, params=imported.params)
        finally:
            if previous_math is missing:
                delattr(np, "math")
            else:
                np.math = previous_math
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
                         None if symbols else "global-scale quantized graph has no VTA partitions")
