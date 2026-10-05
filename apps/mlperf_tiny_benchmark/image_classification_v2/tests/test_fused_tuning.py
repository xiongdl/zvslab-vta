"""Image classification V2 tuning coverage against actual outlined layers."""

import importlib.util
import sys
from pathlib import Path

import pytest
import tvm


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"


@pytest.fixture(scope="module")
def captured():
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    import vta.relay  # Register the VTA compiler hooks before preparing the graph.

    spec = importlib.util.spec_from_file_location(
        "ic_v1_actual_compute_pipeline", APP_ROOT / "python" / "model.py"
    )
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    prepared = pipeline.prepare_model(MODEL_PATH)
    from python.vta_workload import capture_deployment_compute

    compute = capture_deployment_compute(
        prepared.mixed_module,
        "image_classification_v2",
        prepared.imported.model_sha256,
    )
    return prepared, compute


def test_actual_v1_occurrences_expose_real_compute_and_schedule_spaces(captured):
    prepared, deployment = captured

    assert deployment.model_id == "image_classification_v2"
    assert deployment.model_sha256 == prepared.imported.model_sha256
    assert len(deployment.layers) == len(prepared.routing.symbols)
    assert [layer.occurrence for layer in deployment.layers] == list(range(len(deployment.layers)))
    assert [layer.symbol for layer in deployment.layers] == list(prepared.routing.symbols)

    for layer in deployment.layers:
        assert layer.function.attrs.get_str("Compiler") == "vta"
        assert isinstance(layer.compute, tvm.relay.Function)
        assert layer.compute_sha256
        assert layer.config_space_size > 1
        assert layer.template == "conv2d_packed.vta"
        assert layer.inputs[0].shape
        assert layer.inputs[0].dtype == "int8"
        assert layer.output.shape
        assert layer.output.dtype == "int8"
        assert {template for template, _, _, _ in layer.config_spaces} == {
            "conv2d_packed.vta", "add.vta"
        }
        assert all(len(space) > 0 for _, _, _, space in layer.config_spaces)

    assert len({layer.compute_sha256 for layer in deployment.layers}) == len(deployment.layers)


def test_capture_reads_v1_fusion_constants_from_the_outlined_deployment(captured):
    _, deployment = captured

    assert [layer.constants["bias"] for layer in deployment.layers] == [
        64, 128, 64, 128, 64, 128, 256, 32
    ]
    assert [layer.constants["shift"] for layer in deployment.layers] == [
        7, 8, 7, 8, 7, 8, 9, 6
    ]
    assert all(layer.constants["clip_min"] == -127 for layer in deployment.layers)
    assert all(layer.constants["clip_max"] == 127 for layer in deployment.layers)
    assert all(layer.constants["output_dtype"] == "int8" for layer in deployment.layers)
    assert len({layer.constants["weight_sha256"] for layer in deployment.layers}) == len(deployment.layers)
