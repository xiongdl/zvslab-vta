"""Contracts for KWS preprocessing, import, quantization, and VTA routing."""

import ast
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PIPELINE_PATH = APP_ROOT / "model_pipeline.py"
MODEL_PATH = APP_ROOT / "model" / "kws_ref_model.tflite"
SAMPLE_PATH = APP_ROOT / "samples" / "yes-004ae714_nohash_0.wav"
MODEL_SHA256 = "aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae"
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


@pytest.fixture(scope="module")
def model_pipeline():
    assert MODEL_PIPELINE_PATH.is_file(), f"missing Task 2 implementation: {MODEL_PIPELINE_PATH}"
    spec = importlib.util.spec_from_file_location("mlperf_kws_model_pipeline", MODEL_PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mfcc_preprocessing_is_deterministic_and_matches_int8_contract(model_pipeline):
    first = model_pipeline.load_sample(SAMPLE_PATH)
    second = model_pipeline.load_sample(SAMPLE_PATH)
    assert first.dtype == np.int8
    assert first.shape == (1, 49, 10, 1)
    assert np.isfinite(first.astype(np.float32)).all()
    np.testing.assert_array_equal(first, second)


def test_import_asserts_exact_tflite_and_relay_contract(model_pipeline):
    import tvm

    imported = model_pipeline.import_model(MODEL_PATH)
    assert isinstance(imported.module, tvm.IRModule)
    assert imported.model_sha256 == MODEL_SHA256
    assert imported.input_name == "input_1"
    assert imported.input_shape == (1, 49, 10, 1)
    assert imported.input_dtype == "int8"
    assert imported.output_name == "Identity"
    assert imported.output_shape == (1, 12)
    assert imported.output_dtype == "int8"
    assert imported.tflite_operator_names == EXPECTED_TFLITE_OPERATORS


def test_quantization_runs_once_with_the_fixed_policy(model_pipeline, monkeypatch):
    source_module = object()
    source_params = {"weight": object()}
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
    imported = SimpleNamespace(module=source_module, params=source_params)
    assert model_pipeline.quantize_model(imported) is quantized_module
    assert events == [
        ("qconfig", {"calibrate_mode": "global_scale", "global_scale": 8.0, "skip_conv_layers": [0]}),
        ("quantize", source_module, source_params),
    ]


def test_prepare_calls_partition_once_and_forks_one_quantized_module(model_pipeline, monkeypatch):
    imported = SimpleNamespace(module=object(), params={})
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

    monkeypatch.setattr(model_pipeline, "import_model", fake_import)
    monkeypatch.setattr(model_pipeline, "quantize_model", fake_quantize)
    monkeypatch.setattr(model_pipeline.vta.relay, "partition_for_vta", fake_partition)
    monkeypatch.setattr(model_pipeline, "inspect_partitioning", fake_inspect)
    prepared = model_pipeline.prepare_model(MODEL_PATH)

    assert prepared.imported is imported
    assert prepared.quantized_module is quantized_module
    assert prepared.reference_module is quantized_module
    assert prepared.mixed_module is mixed_module
    assert prepared.routing is routing
    assert calls == [
        ("import", MODEL_PATH),
        ("quantize", imported),
        ("partition", quantized_module, "mlperf_kws"),
        ("inspect", quantized_module, mixed_module),
    ]


def test_real_quantized_pipeline_has_nonempty_vta_routing(model_pipeline):
    first = model_pipeline.prepare_model(MODEL_PATH)
    second = model_pipeline.prepare_model(MODEL_PATH)
    assert first.routing == second.routing
    assert first.routing.symbols
    assert first.routing.symbols == tuple(
        f"tvmgen_mlperf_kws_vta_main_{index}" for index in range(len(first.routing.symbols))
    )
    assert first.routing.convolutions_per_partition == (1,) * len(first.routing.symbols)
    assert set(first.routing.host_operator_names) >= {
        "nn.avg_pool2d",
        "nn.dense",
        "nn.softmax",
        "reshape",
    }


def test_model_pipeline_has_no_local_environment_dependency(model_pipeline):
    source = MODEL_PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert "librosa" not in imported_roots
    assert ".envs" not in source
