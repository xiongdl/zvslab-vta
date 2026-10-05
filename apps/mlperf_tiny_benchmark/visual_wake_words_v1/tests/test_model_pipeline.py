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

"""Contracts for importing, quantizing, and partitioning the VWW model."""

import ast
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PIPELINE_PATH = APP_ROOT / "python" / "model.py"
MODEL_PATH = APP_ROOT / "model" / "vww_96_float.tflite"
SAMPLE_PATH = APP_ROOT / "samples" / "00-non-person-000000000009.jpg"
MODEL_SHA256 = "115bbc094d2119561320a21f01b6500a18bea8cc8589282ab007097bec8af38c"

EXPECTED_TFLITE_OPERATORS = tuple(
    name for _ in range(13) for name in ("CONV_2D", "DEPTHWISE_CONV_2D")
) + ("CONV_2D", "AVERAGE_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX")
EXPECTED_CONV_CHANNELS = (8, 16, 32, 32, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256)
REQUIRED_HOST_OPERATORS = {
    "nn.avg_pool2d",
    "nn.conv2d",
    "nn.dense",
    "nn.softmax",
    "reshape",
}


@pytest.fixture(scope="module")
def model_pipeline():
    assert MODEL_PIPELINE_PATH.is_file(), f"missing local model implementation: {MODEL_PIPELINE_PATH}"
    spec = importlib.util.spec_from_file_location("mlperf_vww_model", MODEL_PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_custom_model_path_accepts_same_supported_float_vww(model_pipeline, tmp_path):
    custom_model = tmp_path / "custom-vww.tflite"
    model = MODEL_PATH.read_bytes()
    original_name = b"input_1"
    custom_name = b"input_2"
    assert len(original_name) == len(custom_name)
    assert original_name in model
    custom_model.write_bytes(model.replace(original_name, custom_name, 1))
    imported = model_pipeline.import_float_model(custom_model)
    assert imported.model_sha256 != MODEL_SHA256
    assert imported.input_name == custom_name.decode()


def test_import_rejects_malformed_flatbuffer(model_pipeline, tmp_path):
    invalid_model = tmp_path / "invalid.tflite"
    invalid_model.write_bytes(b"not a flatbuffer")
    with pytest.raises(ValueError, match="valid TFLite FlatBuffer"):
        model_pipeline.import_float_model(invalid_model)


@pytest.mark.parametrize(
    "input_shape,input_type",
    [((1, 95, 96, 3), 0), ((1, 96, 96, 3), 1)],
    ids=["wrong-shape", "wrong-dtype"],
)
def test_flatbuffer_boundary_rejects_unsupported_input_before_relay(
    model_pipeline, input_shape, input_type, monkeypatch
):
    class Tensor:
        def __init__(self, name, shape, tensor_type):
            self.name, self.shape, self.tensor_type = name, shape, tensor_type

        def Name(self):
            return self.name

        def ShapeAsNumpy(self):
            return np.asarray(self.shape)

        def Type(self):
            return self.tensor_type

    class Graph:
        tensors = (
            Tensor(b"input_1", input_shape, input_type),
            Tensor(b"Identity", (1, 2), int(model_pipeline.tflite.TensorType.FLOAT32)),
        )

        def InputsLength(self): return 1
        def OutputsLength(self): return 1
        def Inputs(self, _index): return 0
        def Outputs(self, _index): return 1
        def Tensors(self, index): return self.tensors[index]

    class Model:
        def Version(self): return 3
        def SubgraphsLength(self): return 1
        def Subgraphs(self, _index): return Graph()

    monkeypatch.setattr(model_pipeline.tflite.Model, "GetRootAsModel", lambda *_: Model())
    with pytest.raises(ValueError, match="unexpected model input contract"):
        model_pipeline._flatbuffer_contract(b"test bytes")


def test_import_asserts_exact_float_vww_flatbuffer_and_relay_contract(model_pipeline):
    import tvm
    from tvm import relay

    imported = model_pipeline.import_float_model(MODEL_PATH)

    assert isinstance(imported.module, tvm.IRModule)
    assert isinstance(imported.params, dict)
    assert imported.model_sha256 == MODEL_SHA256
    assert imported.input_name == "input_1"
    assert imported.input_shape == (1, 96, 96, 3)
    assert imported.input_dtype == "float32"
    assert imported.output_name == "Identity"
    assert imported.output_shape == (1, 2)
    assert imported.output_dtype == "float32"
    assert imported.tflite_operator_names == EXPECTED_TFLITE_OPERATORS
    assert imported.convolution_output_channels == EXPECTED_CONV_CHANNELS

    main = imported.module["main"]
    assert main.checked_type is not None
    assert tuple(int(dimension) for dimension in main.params[0].checked_type.shape) == (1, 96, 96, 3)
    assert main.params[0].checked_type.dtype == "float32"
    assert tuple(int(dimension) for dimension in main.ret_type.shape) == (1, 2)
    assert main.ret_type.dtype == "float32"

    relay_operators = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            relay_operators.append(node.op.name)

    relay.analysis.post_order_visit(main.body, visit)
    assert relay_operators.count("nn.conv2d") == 27
    for operator_name in REQUIRED_HOST_OPERATORS - {"nn.conv2d"}:
        assert operator_name in relay_operators


def test_quantization_runs_once_with_only_the_approved_qconfig(model_pipeline, monkeypatch):
    source_module = object()
    source_params = {"weight": object()}
    imported = SimpleNamespace(module=source_module, params=source_params)
    quantized_module = object()
    events = []

    @contextmanager
    def fake_qconfig(**kwargs):
        events.append(("qconfig", kwargs))
        yield

    def fake_quantize(mod, params):
        events.append(("quantize", mod, params))
        return quantized_module

    monkeypatch.setattr(model_pipeline.relay.quantize, "qconfig", fake_qconfig)
    monkeypatch.setattr(model_pipeline.relay.quantize, "quantize", fake_quantize)

    actual = model_pipeline.quantize_model(imported)

    assert actual is quantized_module
    assert events == [
        (
            "qconfig",
            {
                "calibrate_mode": "global_scale",
                "global_scale": 8.0,
                "skip_conv_layers": [0],
            },
        ),
        ("quantize", source_module, source_params),
    ]


def test_quantization_restores_numpy_math_when_relay_raises(model_pipeline, monkeypatch):
    imported = SimpleNamespace(module=object(), params={})

    @contextmanager
    def fake_qconfig(**kwargs):
        yield

    def fake_quantize(mod, params):
        assert np.math is __import__("math")
        raise RuntimeError("quantization failed")

    monkeypatch.delattr(np, "math", raising=False)
    monkeypatch.setattr(model_pipeline.relay.quantize, "qconfig", fake_qconfig)
    monkeypatch.setattr(model_pipeline.relay.quantize, "quantize", fake_quantize)

    with pytest.raises(RuntimeError, match="quantization failed"):
        model_pipeline.quantize_model(imported)

    assert not hasattr(np, "math")


def test_prepare_partitions_the_single_quantized_module(model_pipeline, monkeypatch):
    imported = object()
    quantized_module = object()
    mixed_module = object()
    routing = object()
    calls = []

    def fake_import(path):
        calls.append(("import", path))
        return imported

    def fake_quantize(model):
        calls.append(("quantize", model))
        return quantized_module

    def fake_partition(mod, mod_name):
        calls.append(("partition", mod, mod_name))
        return mixed_module

    def fake_inspect(reference, mixed):
        calls.append(("inspect", reference, mixed))
        return routing

    monkeypatch.setattr(model_pipeline, "import_float_model", fake_import)
    monkeypatch.setattr(model_pipeline, "quantize_model", fake_quantize)
    monkeypatch.setattr(model_pipeline.vta.relay, "partition_for_vta", fake_partition)
    monkeypatch.setattr(model_pipeline, "inspect_partitioning", fake_inspect)

    prepared = model_pipeline.prepare_model(MODEL_PATH)

    assert prepared.imported is imported
    assert prepared.quantized_module is quantized_module
    assert prepared.mixed_module is mixed_module
    assert prepared.routing is routing
    assert calls == [
        ("import", MODEL_PATH),
        ("quantize", imported),
        ("partition", quantized_module, "mlperf_vww"),
        ("inspect", quantized_module, mixed_module),
    ]


def test_real_quantized_partition_reports_only_actual_vta_regions(model_pipeline):
    import tvm

    first = model_pipeline.prepare_model(MODEL_PATH)
    second = model_pipeline.prepare_model(MODEL_PATH)

    assert tvm.ir.structural_equal(first.quantized_module, second.quantized_module)
    assert tvm.ir.structural_equal(first.mixed_module, second.mixed_module)
    assert first.routing == second.routing
    assert first.routing.symbols
    assert len(first.routing.convolutions_per_partition) == len(first.routing.symbols)
    assert all(count > 0 for count in first.routing.convolutions_per_partition)
    assert REQUIRED_HOST_OPERATORS <= set(first.routing.host_operator_names)
    assert all(name.startswith("vta.") for name in first.routing.composite_names)
    assert len(first.routing.composite_names) == len(first.routing.symbols)


def test_partition_inspection_accepts_truthful_zero_coverage(model_pipeline):
    imported = model_pipeline.import_float_model(MODEL_PATH)
    quantized = model_pipeline.quantize_model(imported)
    summary = model_pipeline.inspect_partitioning(quantized, quantized)

    assert summary.symbols == ()
    assert summary.convolutions_per_partition == ()
    assert summary.host_convolution_count == 27


def test_sample_preprocessing_is_exact_normalized_float32_nhwc(model_pipeline):
    from PIL import Image

    actual = model_pipeline.load_sample(SAMPLE_PATH)
    with Image.open(SAMPLE_PATH) as image:
        expected = np.asarray(image.convert("RGB"), dtype="float32")[None, ...] / np.float32(255.0)

    assert actual.dtype == np.float32
    assert actual.shape == (1, 96, 96, 3)
    assert float(actual.min()) >= 0.0
    assert float(actual.max()) <= 1.0
    np.testing.assert_array_equal(actual, expected)


def test_sample_loader_rejects_wrong_dimensions_before_compilation(model_pipeline, tmp_path):
    from PIL import Image

    invalid = tmp_path / "wrong-size.jpg"
    Image.new("RGB", (95, 96)).save(invalid)
    with pytest.raises(ValueError, match="96x96"):
        model_pipeline.load_sample(invalid)


def test_model_pipeline_has_no_forbidden_runtime_dependency_or_legacy_flow():
    source = MODEL_PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert "tensorflow" not in imported_roots
    assert "tflite_runtime" not in imported_roots
    lowered = source.lower()
    for forbidden in [
        "autotvm",
        "graphpack",
        "relay.ext." + "vta",
        "tiny-v1.4",
        "cifar-10-batches-py",
        "download_testdata",
    ]:
        assert forbidden not in lowered
