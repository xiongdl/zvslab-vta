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

"""Provenance and byte-level contracts for the MLPerf ResNet Large assets."""

import hashlib
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"
MODEL_README_PATH = APP_ROOT / "model" / "README.md"
MLPERF_LICENSE_PATH = APP_ROOT / "LICENSE.mlperf-tiny"

MODEL_SHA256 = "fb17ae9c1b6d0e5bd97f0f35024f207556261d7310b249716c87cc0628214b0e"
MLPERF_LICENSE_SHA256 = "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594"
EXPECTED_OPERATOR_CODES = [3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 1, 22, 9, 25]
EXPECTED_CONV_CHANNELS = [40, 40, 40, 80, 80, 80, 160, 160, 160]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_is_the_exact_unmodified_mlperf_tiny_v14_large_float_artifact():
    assert _sha256(MODEL_PATH) == MODEL_SHA256
    assert _sha256(MLPERF_LICENSE_PATH) == MLPERF_LICENSE_SHA256

    provenance = MODEL_README_PATH.read_text(encoding="utf-8")
    required_provenance = [
        "MLPerf Tiny v1.4",
        "benchmark/training/image_classification/trained_models/pretrainedResnet_large_float.tflite",
        MODEL_SHA256,
        "byte-for-byte",
        "Apache-2.0",
        "LICENSE.mlperf-tiny",
        'calibrate_mode="global_scale"',
        "global_scale=8.0",
        "skip_conv_layers=[0]",
    ]
    for expected in required_provenance:
        assert expected in provenance


def test_model_flatbuffer_has_the_approved_resnet8_large_topology():
    import tflite

    model = tflite.Model.GetRootAsModel(MODEL_PATH.read_bytes(), 0)
    assert model.Version() == 3
    assert model.SubgraphsLength() == 1

    graph = model.Subgraphs(0)
    assert graph.InputsLength() == 1
    assert graph.OutputsLength() == 1
    input_tensor = graph.Tensors(graph.Inputs(0))
    output_tensor = graph.Tensors(graph.Outputs(0))
    assert input_tensor.Name().decode("utf-8") == "serving_default_input_5:0"
    assert output_tensor.Name().decode("utf-8") == "StatefulPartitionedCall:0"
    assert list(input_tensor.ShapeAsNumpy()) == [1, 32, 32, 3]
    assert input_tensor.Type() == 0  # TensorType.FLOAT32
    assert list(output_tensor.ShapeAsNumpy()) == [1, 10]
    assert output_tensor.Type() == 0

    operator_codes = []
    conv_channels = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        code = model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode()
        operator_codes.append(code)
        if code == 3:  # BuiltinOperator.CONV_2D
            weight = graph.Tensors(operator.Inputs(1))
            conv_channels.append(int(weight.Shape(0)))

    assert operator_codes == EXPECTED_OPERATOR_CODES
    assert conv_channels == EXPECTED_CONV_CHANNELS
