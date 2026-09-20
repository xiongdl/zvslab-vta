"""Contracts for anomaly feature extraction, import, rewriting, and routing."""

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
MODEL_PATH = APP_ROOT / "model" / "ad01_fp32.tflite"
SAMPLE_PATH = APP_ROOT / "samples" / "normal_id_01_00000000.wav"
MODEL_SHA256 = "c66636f4d7f8af8b10518e7be750a22c9d8d46ec97326b40b0d94c097e0aad9b"
EXPECTED_TFLITE_OPERATORS = ("FULLY_CONNECTED",) * 10
EXPECTED_VTA_SYMBOLS = tuple(
    f"tvmgen_mlperf_anomaly_vta_main_{index}" for index in range(7)
)


@pytest.fixture(scope="module")
def model_pipeline():
    assert MODEL_PIPELINE_PATH.is_file(), f"missing Task 5 implementation: {MODEL_PIPELINE_PATH}"
    spec = importlib.util.spec_from_file_location("mlperf_anomaly_model_pipeline", MODEL_PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audio_preprocessing_is_finite_float32_deterministic_and_fixed_width(model_pipeline):
    first = model_pipeline.load_sample(SAMPLE_PATH)
    second = model_pipeline.load_sample(SAMPLE_PATH)
    assert first.dtype == np.float32
    assert first.ndim == 2
    assert first.shape[1] == 640
    assert first.shape[0] > 0
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_import_rejects_any_change_to_the_committed_model_bytes(model_pipeline, tmp_path):
    changed_model = tmp_path / "changed.tflite"
    contents = bytearray(MODEL_PATH.read_bytes())
    contents[-1] ^= 1
    changed_model.write_bytes(contents)
    with pytest.raises(ValueError, match="model SHA-256"):
        model_pipeline.import_float_model(changed_model)


def test_import_asserts_exact_tflite_and_relay_contract(model_pipeline):
    import tvm

    imported = model_pipeline.import_float_model(MODEL_PATH)
    assert isinstance(imported.module, tvm.IRModule)
    assert imported.model_sha256 == MODEL_SHA256
    assert imported.input_name == "input_1"
    assert imported.input_shape == (1, 640)
    assert imported.input_dtype == "float32"
    assert imported.output_name == "Identity"
    assert imported.output_shape == (1, 640)
    assert imported.output_dtype == "float32"
    assert imported.tflite_operator_names == EXPECTED_TFLITE_OPERATORS
    assert imported.dense_channel_pairs == (
        (640, 128),
        (128, 128),
        (128, 128),
        (128, 128),
        (128, 8),
        (8, 128),
        (128, 128),
        (128, 128),
        (128, 128),
        (128, 640),
    )


def test_rewrite_converts_only_block_compatible_dense_layers(model_pipeline):
    imported = model_pipeline.import_float_model(MODEL_PATH)
    rewritten = model_pipeline.rewrite_dense_layers(imported.module)
    main = rewritten["main"]
    assert model_pipeline._count_operator(main, "nn.dense") == 2
    assert model_pipeline._count_operator(main, "nn.conv2d") == 8
    assert model_pipeline._count_operator(main, "nn.bias_add") == 10
    assert model_pipeline._count_operator(main, "nn.relu") == 9


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
    assert model_pipeline.quantize_model(imported) is quantized_module
    assert events == [
        ("qconfig", {"calibrate_mode": "global_scale", "global_scale": 8.0, "skip_conv_layers": [0]}),
        ("quantize", source_module, source_params),
    ]


def test_prepare_forks_reference_and_mixed_from_one_quantized_module(model_pipeline, monkeypatch):
    imported, rewritten, quantized_module, mixed_module, routing = object(), object(), object(), object(), object()
    calls = []

    def fake_import(path):
        calls.append(("import", path))
        return model_pipeline.ImportedModel(
            module=rewritten,
            params={},
            model_sha256="hash",
            input_name="input_1",
            input_shape=(1, 640),
            input_dtype="float32",
            output_name="Identity",
            output_shape=(1, 640),
            output_dtype="float32",
            tflite_operator_names=(),
            dense_channel_pairs=(),
        )

    def fake_rewrite(module):
        calls.append(("rewrite", module))
        return module

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
    monkeypatch.setattr(model_pipeline, "rewrite_dense_layers", fake_rewrite)
    monkeypatch.setattr(model_pipeline, "quantize_model", fake_quantize)
    monkeypatch.setattr(model_pipeline.vta.relay, "partition_for_vta", fake_partition)
    monkeypatch.setattr(model_pipeline, "inspect_partitioning", fake_inspect)
    prepared = model_pipeline.prepare_model(MODEL_PATH)
    assert prepared.imported.model_sha256 == "hash"
    assert prepared.quantized_module is quantized_module
    assert prepared.reference_module is quantized_module
    assert prepared.mixed_module is mixed_module
    assert prepared.routing is routing
    assert calls[:2] == [("import", MODEL_PATH), ("rewrite", rewritten)]
    assert calls[2][0] == "quantize"
    assert calls[2][1].module is rewritten
    assert calls[2][1].params == {}
    assert calls[3:] == [
        ("partition", quantized_module, "mlperf_anomaly"),
        ("inspect", quantized_module, mixed_module),
    ]


def test_real_quantized_partition_has_exact_seven_region_routing(model_pipeline):
    first = model_pipeline.prepare_model(MODEL_PATH)
    second = model_pipeline.prepare_model(MODEL_PATH)
    assert first.routing == second.routing
    assert first.routing.symbols == EXPECTED_VTA_SYMBOLS
    assert first.routing.convolutions_per_partition == (1,) * 7
    assert first.routing.host_convolution_count == 1
    assert first.routing.host_dense_count == 2
    assert first.routing.host_bottleneck_shapes == ((1, 8), (1, 128))
    assert all(name.startswith("vta.") for name in first.routing.composite_names)
    assert len(first.routing.composite_names) == 7


def test_model_pipeline_has_no_librosa_or_local_environment_dependency():
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
