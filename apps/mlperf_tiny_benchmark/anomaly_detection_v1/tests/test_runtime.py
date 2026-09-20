"""HOST/FSIM runtime contracts for the fixed anomaly sample set."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime_module():
    spec = importlib.util.spec_from_file_location("mlperf_anomaly_runtime", APP_ROOT / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class FakeGraph:
    def __init__(self, output):
        self.output = output

    def load_params(self, params):
        self.params = params

    def set_input(self, name, value):
        assert name == "input_1"
        assert value.shape == (1, 640)
        assert value.dtype == np.float32

    def run(self):
        pass

    def get_output(self, index):
        assert index == 0
        return SimpleNamespace(numpy=lambda: self.output)


def _records(runtime_module):
    return runtime_module.committed_sample_records()


def test_manifest_order_and_balanced_labels_are_fixed(runtime_module):
    records = _records(runtime_module)
    assert len(records) == 10
    assert [item.label for item in records] == [0] * 5 + [1] * 5
    assert [item.order for item in records] == list(range(10))
    assert all(".envs" not in str(item.path) for item in records)


def test_host_execution_returns_ten_reconstruction_scores_without_fsim(
    runtime_module, monkeypatch
):
    records = _records(runtime_module)
    artifact = SimpleNamespace(params=b"params", graph_json="{}", module=object(), device="cpu")
    artifacts = SimpleNamespace(reference=artifact, mixed=None, host_codegen="llvm")
    features = np.arange(640, dtype=np.float32).reshape(1, 640)
    monkeypatch.setattr(runtime_module, "load_sample", lambda path: features)
    monkeypatch.setattr(
        runtime_module,
        "_run_graph",
        lambda current, value: value.copy(),
    )

    def fail_fsim():
        raise AssertionError("HOST execution must not initialize FSIM")

    monkeypatch.setattr(runtime_module, "_load_fsim", fail_fsim)
    result = runtime_module.execute_host(artifacts, records)

    assert len(result.samples) == 10
    assert [item.label for item in result.samples] == [0] * 5 + [1] * 5
    assert all(item.predicted_label == 0 for item in result.samples)
    assert all(item.score == 0.0 for item in result.samples)
    assert result.summary["sample_count"] == 10
    assert result.summary["normal_count"] == 5
    assert result.summary["anomaly_count"] == 5


def test_reconstruction_shape_and_dtype_are_checked(runtime_module):
    artifact = SimpleNamespace(params=b"params", graph_json="{}", module=object(), device="cpu")
    with pytest.raises(RuntimeError, match="output shape"):
        runtime_module._run_graph(artifact, np.zeros((1, 640), dtype=np.float32), output=np.zeros((640,), dtype=np.float32))
    with pytest.raises(RuntimeError, match="output dtype"):
        runtime_module._run_graph(artifact, np.zeros((1, 640), dtype=np.float32), output=np.zeros((1, 640), dtype=np.float16))


def test_fsim_execution_requires_matching_reference_and_mixed_scores(runtime_module, monkeypatch):
    records = _records(runtime_module)
    reference = SimpleNamespace(params=b"r", graph_json="r", module=object(), device="cpu")
    mixed = SimpleNamespace(params=b"m", graph_json="m", module=object(), device="ext_dev")
    artifacts = SimpleNamespace(reference=reference, mixed=mixed, host_codegen="llvm", vta_symbols=("symbol",))
    monkeypatch.setattr(runtime_module, "load_sample", lambda path: np.zeros((1, 640), dtype=np.float32))
    monkeypatch.setattr(runtime_module, "_run_graph", lambda artifact, value, output=None: np.zeros((1, 640), dtype=np.float32))
    monkeypatch.setattr(runtime_module, "validate_mixed_symbols", lambda *args: None)
    simulator = SimpleNamespace(clear_stats=lambda: None, stats=lambda: {"gemm_counter": 1, "wgt_load_nbytes": 1, "out_store_nbytes": 1})
    monkeypatch.setattr(runtime_module, "_load_fsim", lambda: simulator)

    result = runtime_module.execute_fsim(artifacts, records)
    assert len(result.samples) == 10
    assert result.profiler_stats["gemm_counter"] == 1
