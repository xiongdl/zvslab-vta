"""Deployment reports state one-window results and actual measurements."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_report_records_window_count_mse_and_truthful_cpu_metrics(tmp_path):
    import sys

    app = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app))
    try:
        from python.deployment import write_deployment_report
    finally:
        sys.path.remove(str(app))
    result = SimpleNamespace(
        target="llvm", simulator=None, model_path=Path("model.tflite"),
        input_path=Path("sample.wav"), model_sha256="a" * 64,
        input_sha256="b" * 64, preprocessing_policy="test log mel",
        quantization_policy="global_scale=8.0",
        available_windows=196, executed_windows=1, reconstruction_mse=0.125,
        reconstruction=np.zeros((1, 640), dtype="float32"), schedule=None,
        schedule_coverage=(), layers=(), whole_cycles=None, profiler_stats=None,
        fallback_reason=None,
    )
    report = write_deployment_report(result, tmp_path / "report.md").read_text()
    assert "Available windows: 196" in report
    assert "Executed windows: 1" in report
    assert "One-window reconstruction MSE: 0.125" in report
    assert "Quantization: global_scale=8.0" in report
    assert "threshold" not in report.lower()
    assert "Whole-model cycles: N/A (CPU target)" in report


def test_dense_layers_contribute_logical_macs_to_layer_inventory():
    import sys

    app = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app))
    try:
        import tvm
        from tvm import relay
        from python.deployment import _dense_macs
    finally:
        sys.path.remove(str(app))
    data = relay.var("data", shape=(1, 640), dtype="int8")
    weight = relay.var("weight", shape=(32, 640), dtype="int8")
    function = relay.transform.InferType()(
        tvm.IRModule.from_expr(relay.Function([data, weight], relay.nn.dense(data, weight)))
    )["main"]
    call = function.body
    assert _dense_macs(call) == 640 * 32
