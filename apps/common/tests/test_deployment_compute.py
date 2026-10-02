"""Capture checks against the real IC V2 Relay deployment functions."""

import importlib.util
import sys
from pathlib import Path

import pytest
import tvm


APP_ROOT = Path(__file__).resolve().parents[2] / "mlperf_tiny_benchmark" / "image_classification_v2"
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"


@pytest.fixture(scope="module")
def captured():
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    if getattr(sys.modules.get("vta"), "__file__", None) is None:
        sys.modules.pop("vta", None)
    importlib.import_module("vta.relay")
    spec = importlib.util.spec_from_file_location(
        "deployment_compute_model_pipeline", APP_ROOT / "model_pipeline.py"
    )
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    prepared = pipeline.prepare_model(MODEL_PATH)

    from common.deployment_compute import capture_deployment_compute

    return prepared, capture_deployment_compute(
        prepared.mixed_module,
        model_id="image_classification_v2",
        model_sha256=prepared.imported.model_sha256,
    )


def test_capture_preserves_actual_ordered_functions_and_compute_metadata(captured):
    prepared, deployment = captured

    assert deployment.model_id == "image_classification_v2"
    assert deployment.model_sha256 == prepared.imported.model_sha256
    assert len(deployment.layers) == 8
    assert [layer.occurrence for layer in deployment.layers] == list(range(8))
    assert [layer.symbol for layer in deployment.layers] == list(prepared.routing.symbols)

    for layer in deployment.layers:
        assert layer.function.attrs.get_str("Compiler") == "vta"
        assert isinstance(layer.compute, tvm.relay.Function)
        assert len(layer.compute.params) == len(layer.function.params) + len(layer.compute_constants)
        assert layer.compute_sha256
        assert layer.config_space_size > 0
        assert layer.template == "conv2d_packed.vta"
        assert {template for template, _, _, _ in layer.config_spaces} == {
            "conv2d_packed.vta",
            "add.vta",
        }
        assert all(len(space) > 0 for _, _, _, space in layer.config_spaces)
        assert next(
            len(space) for template, _, _, space in layer.config_spaces
            if template == "conv2d_packed.vta"
        ) > 1
        assert layer.inputs[0].shape
        assert layer.inputs[0].dtype == "int8"
        assert layer.output.shape
        assert layer.output.dtype == "int8"
        assert (layer.input_layout, layer.kernel_layout, layer.output_layout) == (
            "NHWC",
            "HWIO",
            "NHWC",
        )

    assert len({layer.compute_sha256 for layer in deployment.layers}) == 8


def test_capture_uses_outlined_constants_and_fused_postprocessing(captured):
    _, deployment = captured

    # The bias, shift and clip values come from each actual outlined Relay body.
    assert [layer.constants["bias"] for layer in deployment.layers] == [64, 128, 64, 128, 64, 128, 256, 32]
    assert [layer.constants["shift"] for layer in deployment.layers] == [7, 8, 7, 8, 7, 8, 9, 6]
    assert all(layer.constants["clip_min"] == -127 for layer in deployment.layers)
    assert all(layer.constants["clip_max"] == 127 for layer in deployment.layers)
    assert len({layer.constants["weight_sha256"] for layer in deployment.layers}) == 8
