# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Import, prepare, and partition the supported Visual Wake Words model."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tflite
import tvm
from PIL import Image
from tvm import relay


MODEL_ID = "visual_wake_words_v1"
MODEL_SHA256 = "115bbc094d2119561320a21f01b6500a18bea8cc8589282ab007097bec8af38c"
INPUT_SHAPE = (1, 96, 96, 3)
INPUT_DTYPE = "float32"
OUTPUT_SHAPE = (1, 2)
OUTPUT_DTYPE = "float32"
CLASS_NAMES = ("non_person", "person")
QUANTIZATION = {"calibrate_mode": "global_scale", "global_scale": 8.0, "skip_conv_layers": [0]}
PREPROCESSING_POLICY = "RGB float32 divided by 255"
EXPECTED_TFLITE_OPERATORS = tuple(
    name for _ in range(13) for name in ("CONV_2D", "DEPTHWISE_CONV_2D")
) + ("CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX")
EXPECTED_CONV_CHANNELS = (8, 16, 32, 32, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256)
REQUIRED_RELAY_OPERATORS = frozenset(
    {"nn.avg_pool2d", "nn.conv2d", "nn.dense", "nn.softmax", "reshape"}
)


@dataclass(frozen=True)
class ImportedModel:
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
    convolution_output_channels: tuple


@dataclass(frozen=True)
class RoutingSummary:
    symbols: tuple
    convolutions_per_partition: tuple
    host_convolution_count: int
    host_depthwise_count: int
    host_operator_names: tuple
    composite_names: tuple


@dataclass(frozen=True)
class PreparedModel:
    imported: ImportedModel
    quantized_module: tvm.IRModule
    mixed_module: tvm.IRModule | None
    routing: RoutingSummary | None


def _shape(tensor):
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
    input_contract = (_tensor_name(input_tensor), _shape(input_tensor), int(input_tensor.Type()))
    output_contract = (_tensor_name(output_tensor), _shape(output_tensor), int(output_tensor.Type()))
    if input_contract[1:] != (INPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
        raise ValueError(f"unexpected model input contract: {input_contract}")
    if output_contract[1:] != (OUTPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
        raise ValueError(f"unexpected model output contract: {output_contract}")

    operator_names = []
    convolution_channels = []
    names_by_code = _operator_name_map()
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())
        operator_names.append(names_by_code.get(code, f"UNKNOWN_{code}"))
        if code == int(tflite.BuiltinOperator.CONV_2D):
            weights = graph.Tensors(operator.Inputs(1))
            convolution_channels.append(int(weights.Shape(0)))
    operator_names = tuple(operator_names)
    convolution_channels = tuple(convolution_channels)
    if operator_names != EXPECTED_TFLITE_OPERATORS:
        raise ValueError(f"unexpected TFLite operator topology: {operator_names}")
    if convolution_channels != EXPECTED_CONV_CHANNELS:
        raise ValueError(f"unexpected convolution channel topology: {convolution_channels}")
    return model, input_contract[0], output_contract[0], operator_names, convolution_channels


def _relay_operator_names(function):
    names = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            names.append(node.op.name)

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(names)


def import_float_model(model_path):
    """Import a model only when its tensor and operation contracts match VWW."""
    model_path = Path(model_path).expanduser().resolve(strict=True)
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    try:
        model, input_name, output_name, operators, channels = _flatbuffer_contract(model_bytes)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("model is not a valid TFLite FlatBuffer") from error

    module, params = relay.frontend.from_tflite(
        model, shape_dict={input_name: INPUT_SHAPE}, dtype_dict={input_name: INPUT_DTYPE}
    )
    module = relay.transform.InferType()(module)
    main = module["main"]
    input_type = main.params[0].checked_type
    output_type = main.ret_type
    relay_input = (tuple(int(dim) for dim in input_type.shape), str(input_type.dtype))
    relay_output = (tuple(int(dim) for dim in output_type.shape), str(output_type.dtype))
    if relay_input != (INPUT_SHAPE, INPUT_DTYPE) or relay_output != (OUTPUT_SHAPE, OUTPUT_DTYPE):
        raise ValueError(f"unexpected imported Relay tensor contract: {relay_input} -> {relay_output}")
    relay_operators = _relay_operator_names(main)
    if relay_operators.count("nn.conv2d") != 27:
        raise ValueError("imported VWW model must contain exactly twenty-seven convolutions")
    if not REQUIRED_RELAY_OPERATORS <= set(relay_operators):
        raise ValueError(f"imported model is missing Relay operators: {sorted(REQUIRED_RELAY_OPERATORS - set(relay_operators))}")
    return ImportedModel(
        module, dict(params), model_sha256, input_name, INPUT_SHAPE, INPUT_DTYPE,
        output_name, OUTPUT_SHAPE, OUTPUT_DTYPE, operators, channels,
    )


def quantize_model(imported):
    """Apply the fixed VWW global-scale policy once, without dataset calibration."""
    with relay.quantize.qconfig(**QUANTIZATION):
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


def _count_depthwise(function):
    count = 0

    def visit(node):
        nonlocal count
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "nn.conv2d":
            groups = int(node.attrs.groups)
            if groups > 1 and groups == int(node.attrs.channels):
                count += 1

    relay.analysis.post_order_visit(function.body, visit)
    return count


def _composite_names(function):
    names = []

    def visit(node):
        if isinstance(node, relay.Function) and node.attrs is not None and "Composite" in node.attrs:
            names.append(node.attrs.get_str("Composite"))

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(names)


def inspect_partitioning(quantized_module, mixed_module):
    """Summarize the partitions actually produced from model computation."""
    if _count_operator(quantized_module["main"], "nn.conv2d") != 27:
        raise ValueError("quantized reference must contain twenty-seven convolutions")
    external = []
    for global_var, function in mixed_module.functions.items():
        if (isinstance(function, relay.Function) and function.attrs is not None
                and "Compiler" in function.attrs and function.attrs.get_str("Compiler") == "vta"):
            external.append((function.attrs.get_str("global_symbol"), global_var, function))
    external.sort(key=lambda row: row[0])
    symbols = tuple(row[0] for row in external)
    counts = tuple(_count_operator(row[2], "nn.conv2d") for row in external)
    composites = tuple(name for row in external for name in _composite_names(row[2]))
    host_names = _relay_operator_names(mixed_module["main"])
    summary = RoutingSummary(
        symbols, counts, host_names.count("nn.conv2d"), _count_depthwise(mixed_module["main"]),
        tuple(sorted(set(host_names))), composites,
    )
    if len(symbols) != len(composites) or any(count < 1 for count in counts):
        raise ValueError("each VTA partition must contain real model convolutions")
    if not REQUIRED_RELAY_OPERATORS <= set(summary.host_operator_names):
        raise ValueError("mixed graph is missing required CPU operators")
    if any(not name.startswith("vta.") for name in composites):
        raise ValueError(f"unexpected VTA composite: {composites}")
    return summary


def prepare_model(model_path, *, use_vta=True):
    imported = import_float_model(model_path)
    quantized_module = quantize_model(imported)
    if use_vta:
        import vta

        mixed_module = vta.relay.partition_for_vta(quantized_module, mod_name="mlperf_vww")
        routing = inspect_partitioning(quantized_module, mixed_module)
    else:
        mixed_module = routing = None
    return PreparedModel(imported, quantized_module, mixed_module, routing)


def load_sample(sample_path):
    """Decode one 96x96 RGB image into VWW's normalized float32 NHWC tensor."""
    with Image.open(Path(sample_path).expanduser().resolve(strict=True)) as image:
        if image.size != (96, 96):
            raise ValueError(f"image must be 96x96 pixels, received {image.size}")
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    if pixels.shape != (96, 96, 3):
        raise ValueError(f"image must decode to RGB pixels, received {pixels.shape}")
    return pixels[None, ...] / np.float32(255.0)


def __getattr__(name):
    if name == "vta":
        import vta

        return vta
    raise AttributeError(name)
