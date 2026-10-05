"""Anomaly tuning coverage against actual outlined VTA layers."""

import importlib.util
import sys
import os
from pathlib import Path

import pytest
import tvm


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "ad01_fp32.tflite"


@pytest.fixture(scope="module")
def captured():
    root = APP_ROOT.parents[3]
    os.environ.setdefault("VTA_CONFIG_FILE", str(root / "vta/config/vta_64mac.json"))
    os.environ.setdefault("VTA_BACKEND", "fsim")
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    import vta.relay  # Register the VTA compiler hooks before preparing the graph.

    spec = importlib.util.spec_from_file_location(
        "anomaly_actual_compute_pipeline", APP_ROOT / "python" / "model.py"
    )
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    prepared = pipeline.prepare_model(MODEL_PATH, use_vta=True)
    from python.vta_workload import capture_deployment_compute

    compute = capture_deployment_compute(
        prepared.mixed_module,
        "anomaly_detection_v1",
        prepared.imported.model_sha256,
    )
    return prepared, compute


def test_actual_v1_occurrences_expose_real_compute_and_schedule_spaces(captured):
    prepared, deployment = captured

    assert deployment.model_id == "anomaly_detection_v1"
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


def test_anomaly_routing_is_derived_from_real_outlined_convolutions(captured):
    prepared, deployment = captured
    assert prepared.routing.symbols
    assert prepared.routing.convolutions_per_partition == (1,) * len(deployment.layers)
    assert all(layer.constants["weight_sha256"] for layer in deployment.layers)
