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

"""Deterministic import, quantization, and routing for MLPerf Tiny ResNet-8 Large."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tflite
import tvm
import vta
from PIL import Image
from tvm import relay


MODEL_SHA256 = "fb17ae9c1b6d0e5bd97f0f35024f207556261d7310b249716c87cc0628214b0e"
INPUT_NAME = "serving_default_input_5:0"
INPUT_SHAPE = (1, 32, 32, 3)
INPUT_DTYPE = "float32"
OUTPUT_NAME = "StatefulPartitionedCall:0"
OUTPUT_SHAPE = (1, 10)
OUTPUT_DTYPE = "float32"
EXPECTED_TFLITE_OPERATORS = (
    "CONV_2D",
    "CONV_2D",
    "CONV_2D",
    "ADD",
    "CONV_2D",
    "CONV_2D",
    "CONV_2D",
    "ADD",
    "CONV_2D",
    "CONV_2D",
    "CONV_2D",
    "ADD",
    "AVERAGE_POOL_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
)
EXPECTED_CONV_CHANNELS = (40, 40, 40, 80, 80, 80, 160, 160, 160)
EXPECTED_VTA_SYMBOLS = tuple(
    f"tvmgen_mlperf_resnet_large_vta_main_{index}" for index in range(4)
)
REQUIRED_HOST_OPERATORS = frozenset(
    {"add", "nn.avg_pool2d", "nn.conv2d", "nn.dense", "nn.softmax", "reshape"}
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
    convolution_output_channels: tuple


@dataclass(frozen=True)
class RoutingSummary:
    """Structural summary of the fixed VTA/LLVM partition boundary."""

    symbols: tuple
    convolutions_per_partition: tuple
    host_convolution_count: int
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
    if input_contract != (INPUT_NAME, INPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
        raise ValueError(f"unexpected model input contract: {input_contract}")
    if output_contract != (OUTPUT_NAME, OUTPUT_SHAPE, int(tflite.TensorType.FLOAT32)):
        raise ValueError(f"unexpected model output contract: {output_contract}")

    operator_names = []
    convolution_channels = []
    names_by_code = _operator_name_map()
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = int(model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode())
        operator_names.append(names_by_code.get(code, f"UNKNOWN_{code}"))
        if code == int(tflite.BuiltinOperator.CONV_2D):
            weight = graph.Tensors(operator.Inputs(1))
            convolution_channels.append(int(weight.Shape(0)))

    operator_names = tuple(operator_names)
    convolution_channels = tuple(convolution_channels)
    if operator_names != EXPECTED_TFLITE_OPERATORS:
        raise ValueError(f"unexpected TFLite operator topology: {operator_names}")
    if convolution_channels != EXPECTED_CONV_CHANNELS:
        raise ValueError(f"unexpected convolution channel topology: {convolution_channels}")
    return model, operator_names, convolution_channels


def _relay_operator_names(function):
    operator_names = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            operator_names.append(node.op.name)

    relay.analysis.post_order_visit(function.body, visit)
    return tuple(operator_names)


def import_float_model(model_path):
    """Verify and import the exact committed floating ResNet-8 Large artifact."""
    model_path = Path(model_path)
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if model_sha256 != MODEL_SHA256:
        raise ValueError(
            f"model SHA-256 mismatch: expected {MODEL_SHA256}, received {model_sha256}"
        )

    model, operator_names, convolution_channels = _flatbuffer_contract(model_bytes)
    module, params = relay.frontend.from_tflite(
        model,
        shape_dict={INPUT_NAME: INPUT_SHAPE},
        dtype_dict={INPUT_NAME: INPUT_DTYPE},
    )
    module = relay.transform.InferType()(module)
    main = module["main"]
    parameter_type = main.params[0].checked_type
    result_type = main.ret_type
    relay_input = (
        tuple(int(dimension) for dimension in parameter_type.shape),
        parameter_type.dtype,
    )
    relay_output = (
        tuple(int(dimension) for dimension in result_type.shape),
        result_type.dtype,
    )
    if relay_input != (INPUT_SHAPE, INPUT_DTYPE):
        raise ValueError(f"unexpected Relay input contract: {relay_input}")
    if relay_output != (OUTPUT_SHAPE, OUTPUT_DTYPE):
        raise ValueError(f"unexpected Relay output contract: {relay_output}")
    relay_operators = _relay_operator_names(main)
    if relay_operators.count("nn.conv2d") != 9:
        raise ValueError("imported Relay model must contain exactly nine convolutions")
    if not REQUIRED_HOST_OPERATORS <= set(relay_operators):
        missing = sorted(REQUIRED_HOST_OPERATORS - set(relay_operators))
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
        convolution_output_channels=convolution_channels,
    )


def quantize_model(imported):
    """Quantize once with the fixed TVM policy and no calibration dataset."""
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


def inspect_partitioning(reference_module, mixed_module):
    """Validate and summarize the exact four-region VTA routing contract."""
    if _count_operator(reference_module["main"], "nn.conv2d") != 9:
        raise ValueError("quantized reference must contain exactly nine convolutions")

    external = []
    for global_var, function in mixed_module.functions.items():
        if (
            isinstance(function, relay.Function)
            and function.attrs is not None
            and "Compiler" in function.attrs
            and function.attrs.get_str("Compiler") == "vta"
        ):
            external.append((function.attrs.get_str("global_symbol"), global_var, function))
    external.sort(key=lambda item: item[0])

    symbols = tuple(item[0] for item in external)
    convolution_counts = tuple(_count_operator(item[2], "nn.conv2d") for item in external)
    composite_names = tuple(name for item in external for name in _composite_names(item[2]))
    host_operator_names = _relay_operator_names(mixed_module["main"])
    summary = RoutingSummary(
        symbols=symbols,
        convolutions_per_partition=convolution_counts,
        host_convolution_count=host_operator_names.count("nn.conv2d"),
        host_operator_names=tuple(sorted(set(host_operator_names))),
        composite_names=composite_names,
    )
    if summary.symbols != EXPECTED_VTA_SYMBOLS:
        raise ValueError(f"unexpected VTA symbols: {summary.symbols}")
    if summary.convolutions_per_partition != (1,) * 4:
        raise ValueError(
            f"each VTA partition must contain one convolution: {summary.convolutions_per_partition}"
        )
    if summary.host_convolution_count != 5:
        raise ValueError("the five non-partitioned convolutions must remain on the host")
    if not REQUIRED_HOST_OPERATORS <= set(summary.host_operator_names):
        missing = sorted(REQUIRED_HOST_OPERATORS - set(summary.host_operator_names))
        raise ValueError(f"mixed main is missing required host operators: {missing}")
    if len(summary.composite_names) != 4 or not all(
        name.startswith("vta.") for name in summary.composite_names
    ):
        raise ValueError(f"unexpected VTA composites: {summary.composite_names}")
    return summary


def prepare_model(model_path):
    """Import, quantize once, and fork reference and mixed graphs."""
    imported = import_float_model(model_path)
    quantized_module = quantize_model(imported)
    reference_module = quantized_module
    mixed_module = vta.relay.partition_for_vta(
        quantized_module, mod_name="mlperf_resnet_large"
    )
    routing = inspect_partitioning(reference_module, mixed_module)
    return PreparedModel(
        imported=imported,
        quantized_module=quantized_module,
        reference_module=reference_module,
        mixed_module=mixed_module,
        routing=routing,
    )


def load_sample(sample_path):
    """Decode one RGB sample as unchanged float32 NHWC pixels."""
    with Image.open(sample_path) as image:
        if image.size != (32, 32):
            raise ValueError(f"sample must be 32x32 pixels, received {image.size}")
        pixels = np.asarray(image.convert("RGB"), dtype="uint8")
    return pixels.astype("float32")[None, ...]
