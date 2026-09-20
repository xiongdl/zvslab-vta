"""Deterministic audio preprocessing and Relay deployment for anomaly detection."""

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import wave

import numpy as np
import tflite
import tvm
import vta
from tvm import relay


MODEL_SHA256 = "c66636f4d7f8af8b10518e7be750a22c9d8d46ec97326b40b0d94c097e0aad9b"
INPUT_NAME = "input_1"
INPUT_SHAPE = (1, 640)
INPUT_DTYPE = "float32"
OUTPUT_NAME = "Identity"
OUTPUT_SHAPE = (1, 640)
OUTPUT_DTYPE = "float32"
SAMPLE_RATE = 16000
N_MELS = 128
FRAMES = 5
N_FFT = 1024
HOP_LENGTH = 512
POWER = 2.0
CENTRAL_MEL_START = 50
CENTRAL_MEL_END = 250
VTA_BLOCK_IN = int(vta.get_env().BLOCK_IN)
VTA_BLOCK_OUT = int(vta.get_env().BLOCK_OUT)
EXPECTED_VTA_SYMBOLS = tuple(
    f"tvmgen_mlperf_anomaly_vta_main_{index}" for index in range(7)
)
EXPECTED_TFLITE_OPERATORS = ("FULLY_CONNECTED",) * 10
REQUIRED_HOST_OPERATORS = frozenset(
    {"nn.bias_add", "nn.dense", "nn.relu", "nn.conv2d", "reshape"}
)


@dataclass(frozen=True)
class ImportedModel:
    """Validated floating model and its Relay import."""

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
    dense_channel_pairs: tuple


@dataclass(frozen=True)
class RoutingSummary:
    """Structural summary of the fixed VTA/LLVM partition boundary."""

    symbols: tuple
    convolutions_per_partition: tuple
    host_convolution_count: int
    host_dense_count: int
    host_bottleneck_shapes: tuple
    host_operator_names: tuple
    composite_names: tuple


@dataclass(frozen=True)
class PreparedModel:
    """The one quantized reference graph and its mixed VTA partition."""

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
    if input_contract != (INPUT_NAME, INPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
        raise ValueError(f"unexpected model input contract: {input_contract}")
    if output_contract != (OUTPUT_NAME, OUTPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
        raise ValueError(f"unexpected model output contract: {output_contract}")

    names_by_code = _operator_name_map()
    operator_names = []
    dense_channel_pairs = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())
        operator_names.append(names_by_code.get(code, f"UNKNOWN_{code}"))
        if code == int(tflite.BuiltinOperator.FULLY_CONNECTED):
            data = graph.Tensors(operator.Inputs(0))
            weight = graph.Tensors(operator.Inputs(1))
            dense_channel_pairs.append((_tensor_shape(data)[-1], _tensor_shape(weight)[0]))

    operator_names = tuple(operator_names)
    dense_channel_pairs = tuple(dense_channel_pairs)
    if operator_names != EXPECTED_TFLITE_OPERATORS:
        raise ValueError(f"unexpected TFLite operator topology: {operator_names}")
    if len(dense_channel_pairs) != 10:
        raise ValueError(f"expected ten fully-connected layers, found {len(dense_channel_pairs)}")
    return model, operator_names, dense_channel_pairs


def _relay_operator_names(function):
    operator_names = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            operator_names.append(node.op.name)

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(operator_names)


def import_float_model(model_path):
    """Verify and import the exact committed floating TFLite artifact."""
    model_path = Path(model_path)
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if model_sha256 != MODEL_SHA256:
        raise ValueError(
            f"model SHA-256 mismatch: expected {MODEL_SHA256}, received {model_sha256}"
        )

    model, operator_names, dense_channel_pairs = _flatbuffer_contract(model_bytes)
    module, params = relay.frontend.from_tflite(
        model,
        shape_dict={INPUT_NAME: INPUT_SHAPE},
        dtype_dict={INPUT_NAME: INPUT_DTYPE},
    )
    module = relay.transform.InferType()(module)
    main = module["main"]
    relay_input = (_shape(main.params[0]), main.params[0].checked_type.dtype)
    relay_output = (
        tuple(int(dimension) for dimension in main.ret_type.shape),
        main.ret_type.dtype,
    )
    if relay_input != (INPUT_SHAPE, INPUT_DTYPE):
        raise ValueError(f"unexpected Relay input contract: {relay_input}")
    if relay_output != (OUTPUT_SHAPE, OUTPUT_DTYPE):
        raise ValueError(f"unexpected Relay output contract: {relay_output}")
    relay_operators = _relay_operator_names(main)
    if relay_operators.count("nn.dense") != 10:
        raise ValueError("imported Relay model must contain exactly ten dense layers")
    if not REQUIRED_HOST_OPERATORS - {"nn.conv2d"} <= set(relay_operators):
        missing = sorted((REQUIRED_HOST_OPERATORS - {"nn.conv2d"}) - set(relay_operators))
        raise ValueError(f"imported Relay model is missing required operators: {missing}")

    return ImportedModel(
        module=module,
        params=dict(params),
        model_sha256=model_sha256,
        input_name=INPUT_NAME,
        input_shape=INPUT_SHAPE,
        input_dtype=INPUT_DTYPE,
        output_name=OUTPUT_NAME,
        output_shape=OUTPUT_SHAPE,
        output_dtype=OUTPUT_DTYPE,
        tflite_operator_names=operator_names,
        dense_channel_pairs=dense_channel_pairs,
    )


def _dense_to_conv(data, weight, units, *, batch=None, input_channels=None):
    """Rewrite a two-dimensional dense operation as an equivalent NHWC 1x1 conv."""
    if batch is None or input_channels is None:
        batch, input_channels = _shape(data)
    packed_data = relay.reshape(data, (batch, 1, 1, input_channels))
    packed_weight = relay.reshape(
        relay.transpose(weight, axes=(1, 0)),
        (1, 1, input_channels, units),
    )
    conv = relay.nn.conv2d(
        packed_data,
        packed_weight,
        channels=units,
        kernel_size=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_layout="NHWC",
    )
    return relay.reshape(conv, (batch, units))


class _DenseToConvMutator(relay.ExprMutator):
    def visit_call(self, call):
        rewritten = super().visit_call(call)
        if not isinstance(rewritten.op, tvm.ir.Op) or rewritten.op.name != "nn.dense":
            return rewritten

        data, weight = rewritten.args[:2]
        input_shape = _shape(call.args[0])
        weight_shape = _shape(call.args[1])
        units = int(rewritten.attrs.units)
        if (
            len(input_shape) == 2
            and len(weight_shape) == 2
            and input_shape[-1] == weight_shape[1]
            and input_shape[-1] % VTA_BLOCK_IN == 0
            and units % VTA_BLOCK_OUT == 0
        ):
            return _dense_to_conv(
                data,
                weight,
                units,
                batch=input_shape[0],
                input_channels=input_shape[-1],
            )
        return rewritten


def rewrite_dense_layers(module):
    """Convert only block-compatible dense layers before VTA quantization."""
    typed_module = relay.transform.InferType()(module)
    main = typed_module["main"]
    body = _DenseToConvMutator().visit(main.body)
    rewritten_main = relay.Function(
        main.params,
        body,
        type_params=main.type_params,
        attrs=main.attrs,
    )
    rewritten = tvm.IRModule({typed_module.get_global_var("main"): rewritten_main})
    return relay.transform.InferType()(rewritten)


def quantize_model(imported):
    """Quantize exactly once with the fixed global-scale policy."""
    with relay.quantize.qconfig(
        calibrate_mode="global_scale",
        global_scale=8.0,
        skip_conv_layers=[0],
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


def _count_operator(function, name):
    return _relay_operator_names(function).count(name)


def _composite_names(function):
    names = []

    def visit(node):
        if isinstance(node, relay.Function) and node.attrs is not None and "Composite" in node.attrs:
            names.append(node.attrs.get_str("Composite"))

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(names)


def _dense_output_shapes(function):
    shapes = []

    def visit(node):
        if (
            isinstance(node, relay.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "nn.dense"
        ):
            shapes.append(_shape(node))

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(shapes)


def inspect_partitioning(reference_module, mixed_module):
    """Validate and summarize the exact seven-region VTA routing contract."""
    if _count_operator(reference_module["main"], "nn.conv2d") != 8:
        raise ValueError("quantized reference must contain exactly eight convolutions")

    external = []
    for global_var, function in mixed_module.functions.items():
        if (
            isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
            and function.attrs.get_str("Compiler") == "vta"
        ):
            external.append((function.attrs.get_str("global_symbol"), global_var, function))
    external.sort(key=lambda item: int(item[0].rsplit("_", 1)[1]))
    symbols = tuple(item[0] for item in external)
    convolution_counts = tuple(_count_operator(item[2], "nn.conv2d") for item in external)
    main = mixed_module["main"]
    dense_shapes = _dense_output_shapes(main)
    host_operator_names = _relay_operator_names(main)
    summary = RoutingSummary(
        symbols=symbols,
        convolutions_per_partition=convolution_counts,
        host_convolution_count=host_operator_names.count("nn.conv2d"),
        host_dense_count=host_operator_names.count("nn.dense"),
        host_bottleneck_shapes=dense_shapes,
        host_operator_names=tuple(sorted(set(host_operator_names))),
        composite_names=tuple(name for item in external for name in _composite_names(item[2])),
    )
    if summary.symbols != EXPECTED_VTA_SYMBOLS:
        raise ValueError(f"unexpected VTA symbols: {summary.symbols}")
    if summary.convolutions_per_partition != (1,) * 7:
        raise ValueError(
            f"each VTA partition must contain one convolution: {summary.convolutions_per_partition}"
        )
    if summary.host_convolution_count != 1 or summary.host_dense_count != 2:
        raise ValueError(
            "the skipped first convolution and the two bottleneck dense layers must remain on host"
        )
    if not REQUIRED_HOST_OPERATORS <= set(summary.host_operator_names):
        missing = sorted(REQUIRED_HOST_OPERATORS - set(summary.host_operator_names))
        raise ValueError(f"mixed main is missing required host operators: {missing}")
    if len(summary.composite_names) != 7 or not all(
        name.startswith("vta.") for name in summary.composite_names
    ):
        raise ValueError(f"unexpected VTA composites: {summary.composite_names}")
    return summary


def prepare_model(model_path):
    """Import, rewrite, quantize once, and fork reference and mixed graphs."""
    imported = import_float_model(model_path)
    rewritten_module = rewrite_dense_layers(imported.module)
    rewritten_imported = replace(imported, module=rewritten_module)
    quantized_module = quantize_model(rewritten_imported)
    reference_module = quantized_module
    mixed_module = vta.relay.partition_for_vta(
        quantized_module,
        mod_name="mlperf_anomaly",
    )
    routing = inspect_partitioning(reference_module, mixed_module)
    return PreparedModel(
        imported=imported,
        quantized_module=quantized_module,
        reference_module=reference_module,
        mixed_module=mixed_module,
        routing=routing,
    )


def _hz_to_mel(freq):
    freq = np.asarray(freq, dtype=np.float64)
    linear = 3.0 / 200.0 * freq
    logarithmic = 15.0 + np.log(np.maximum(freq, 1000.0) / 1000.0) / np.log(6.4) * 27.0
    return np.where(freq < 1000.0, linear, logarithmic)


def _mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    linear = 200.0 / 3.0 * mel
    logarithmic = 1000.0 * np.exp((mel - 15.0) * np.log(6.4) / 27.0)
    return np.where(mel < 15.0, linear, logarithmic)


def _mel_filterbank():
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    mel_points = np.linspace(
        _hz_to_mel(0.0),
        _hz_to_mel(SAMPLE_RATE / 2.0),
        N_MELS + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    filters = np.zeros((N_MELS, frequencies.size), dtype=np.float64)
    for index in range(N_MELS):
        lower, center, upper = hz_points[index : index + 3]
        rising = (frequencies - lower) / max(center - lower, sys.float_info.epsilon)
        falling = (upper - frequencies) / max(upper - center, sys.float_info.epsilon)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
        filters[index] *= 2.0 / max(upper - lower, sys.float_info.epsilon)
    return filters


def _read_wav(sample_path):
    with wave.open(str(sample_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
            raise ValueError("sample must be mono 16-bit PCM at 16 kHz")
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size == 0:
        raise ValueError("sample must contain audio frames")
    return samples


def load_sample(sample_path):
    """Decode one WAV into deterministic sliding log-mel vectors."""
    samples = _read_wav(sample_path)
    pad = N_FFT // 2
    padded = np.pad(samples, (pad, pad), mode="constant")
    frame_count = 1 + (padded.size - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    frames = frames[:frame_count]
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    power_spectrogram = np.abs(spectrum) ** POWER
    mel_spectrogram = _mel_filterbank() @ power_spectrogram.T
    log_mel = 20.0 / POWER * np.log10(
        np.maximum(mel_spectrogram, sys.float_info.epsilon)
    )
    central = log_mel[:, CENTRAL_MEL_START:CENTRAL_MEL_END]
    vector_count = central.shape[1] - FRAMES + 1
    if vector_count < 1:
        raise ValueError("sample does not contain enough mel frames")
    vectors = np.empty((vector_count, N_MELS * FRAMES), dtype=np.float64)
    for frame in range(FRAMES):
        vectors[:, frame * N_MELS : (frame + 1) * N_MELS] = central[:, frame : frame + vector_count].T
    vectors = vectors.astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("audio preprocessing produced non-finite features")
    return vectors
