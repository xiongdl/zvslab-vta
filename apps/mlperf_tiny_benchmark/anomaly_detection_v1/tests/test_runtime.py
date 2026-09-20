"""HOST/FSIM runtime contracts for the fixed anomaly sample set."""

import importlib.util
import hashlib
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


def _write_manifest(tmp_path, mutate=None):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    samples = []
    for order, (label, class_name) in enumerate(
        [(0, "normal")] * 5 + [(1, "anomaly")] * 5
    ):
        filename = f"{class_name}_{order}.wav"
        payload = f"sample-{order}".encode("ascii")
        (samples_dir / filename).write_bytes(payload)
        samples.append(
            {
                "order": order,
                "filename": filename,
                "source_relative_path": f"test/{filename}",
                "class_name": class_name,
                "label": label,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_length": len(payload),
            }
        )
    if mutate is not None:
        mutate(samples, samples_dir, tmp_path)
    manifest_path = samples_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    return manifest_path


def test_manifest_order_and_balanced_labels_are_fixed(runtime_module):
    records = _records(runtime_module)
    assert len(records) == 10
    assert [item.label for item in records] == [0] * 5 + [1] * 5
    assert [item.order for item in records] == list(range(10))
    assert all(".envs" not in str(item.path) for item in records)


def test_custom_manifest_requires_sample_hash(runtime_module, tmp_path):
    def remove_hash(samples, samples_dir, root):
        del samples[0]["sha256"]

    manifest_path = _write_manifest(tmp_path, remove_hash)
    with pytest.raises(ValueError, match="sha256"):
        runtime_module.committed_sample_records(manifest_path)


@pytest.mark.parametrize(
    "source_relative_path",
    (None, "", "/test/sample.wav", "../sample.wav", "dataset/sample.wav", "test/other.wav"),
)
def test_custom_manifest_rejects_invalid_source_relative_path(
    runtime_module, tmp_path, source_relative_path
):
    def change_source_path(samples, samples_dir, root):
        if source_relative_path is None:
            del samples[0]["source_relative_path"]
        else:
            samples[0]["source_relative_path"] = source_relative_path

    manifest_path = _write_manifest(tmp_path, change_source_path)
    with pytest.raises(ValueError, match="source_relative_path"):
        runtime_module.committed_sample_records(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("sha256", "0" * 64), ("byte_length", 0)),
)
def test_custom_manifest_rejects_sample_metadata_mismatch(
    runtime_module, tmp_path, field, value
):
    def change_metadata(samples, samples_dir, root):
        samples[0][field] = value

    manifest_path = _write_manifest(tmp_path, change_metadata)
    with pytest.raises(ValueError, match=field):
        runtime_module.committed_sample_records(manifest_path)


def test_custom_manifest_rejects_duplicate_filenames(runtime_module, tmp_path):
    def duplicate_filename(samples, samples_dir, root):
        samples[1]["filename"] = samples[0]["filename"]
        samples[1]["source_relative_path"] = f"test/{samples[1]['filename']}"

    manifest_path = _write_manifest(tmp_path, duplicate_filename)
    with pytest.raises(ValueError, match="unique"):
        runtime_module.committed_sample_records(manifest_path)


@pytest.mark.parametrize("path_kind", ("parent", "absolute", "symlink"))
def test_custom_manifest_rejects_external_sample_paths(runtime_module, tmp_path, path_kind):
    def external_path(samples, samples_dir, root):
        outside = root / "outside.wav"
        outside.write_bytes(b"outside")
        if path_kind == "parent":
            samples[0]["filename"] = "../outside.wav"
        elif path_kind == "absolute":
            samples[0]["filename"] = str(outside)
        else:
            link = samples_dir / "escaped.wav"
            link.symlink_to(outside)
            samples[0]["filename"] = link.name
            samples[0]["source_relative_path"] = f"test/{link.name}"
            samples[0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
            samples[0]["byte_length"] = outside.stat().st_size

    manifest_path = _write_manifest(tmp_path, external_path)
    with pytest.raises(ValueError, match="unsafe|outside|symlink"):
        runtime_module.committed_sample_records(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("order", 1), ("label", 1), ("class_name", "anomaly")),
)
def test_custom_manifest_rejects_wrong_order_or_class_label(
    runtime_module, tmp_path, field, value
):
    def change_contract(samples, samples_dir, root):
        samples[0][field] = value

    manifest_path = _write_manifest(tmp_path, change_contract)
    with pytest.raises(ValueError, match="order|label|class"):
        runtime_module.committed_sample_records(manifest_path)


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
    profiler = {
        "gemm_counter": 7,
        "wgt_load_nbytes": 11,
        "out_store_nbytes": 13,
    }

    def run_graph(artifact, value, output=None):
        for counter in profiler:
            profiler[counter] += 1
        return np.zeros((1, 640), dtype=np.float32)

    monkeypatch.setattr(runtime_module, "_run_graph", run_graph)
    monkeypatch.setattr(runtime_module, "validate_mixed_symbols", lambda *args: None)

    def clear_stats():
        for counter in profiler:
            profiler[counter] = 0

    simulator = SimpleNamespace(clear_stats=clear_stats, stats=lambda: dict(profiler))
    monkeypatch.setattr(runtime_module, "_load_fsim", lambda: simulator)

    result = runtime_module.execute_fsim(artifacts, records)
    assert len(result.samples) == 10
    assert result.profiler_stats["gemm_counter"] == 10


def test_fsim_execution_rejects_profiler_stats_that_clear_stats_did_not_reset(
    runtime_module, monkeypatch
):
    records = _records(runtime_module)
    reference = SimpleNamespace(params=b"r", graph_json="r", module=object(), device="cpu")
    mixed = SimpleNamespace(params=b"m", graph_json="m", module=object(), device="ext_dev")
    artifacts = SimpleNamespace(reference=reference, mixed=mixed, host_codegen="llvm", vta_symbols=("symbol",))
    monkeypatch.setattr(runtime_module, "load_sample", lambda path: np.zeros((1, 640), dtype=np.float32))
    monkeypatch.setattr(runtime_module, "_run_graph", lambda artifact, value, output=None: np.zeros((1, 640), dtype=np.float32))
    monkeypatch.setattr(runtime_module, "validate_mixed_symbols", lambda *args: None)
    simulator = SimpleNamespace(
        clear_stats=lambda: None,
        stats=lambda: {"gemm_counter": 1, "wgt_load_nbytes": 1, "out_store_nbytes": 1},
    )
    monkeypatch.setattr(runtime_module, "_load_fsim", lambda: simulator)

    with pytest.raises(RuntimeError, match="did not reset|required counters|zero"):
        runtime_module.execute_fsim(artifacts, records)


def test_fsim_execution_rejects_nonzero_profiler_stats_after_clear(
    runtime_module, monkeypatch
):
    records = _records(runtime_module)
    reference = SimpleNamespace(params=b"r", graph_json="r", module=object(), device="cpu")
    mixed = SimpleNamespace(params=b"m", graph_json="m", module=object(), device="ext_dev")
    artifacts = SimpleNamespace(reference=reference, mixed=mixed, host_codegen="llvm", vta_symbols=("symbol",))
    monkeypatch.setattr(runtime_module, "load_sample", lambda path: np.zeros((1, 640), dtype=np.float32))
    monkeypatch.setattr(runtime_module, "_run_graph", lambda artifact, value, output=None: np.zeros((1, 640), dtype=np.float32))
    monkeypatch.setattr(runtime_module, "validate_mixed_symbols", lambda *args: None)
    profiler = {"gemm_counter": 0, "wgt_load_nbytes": 1, "out_store_nbytes": 0}
    simulator = SimpleNamespace(clear_stats=lambda: None, stats=lambda: dict(profiler))
    monkeypatch.setattr(runtime_module, "_load_fsim", lambda: simulator)

    with pytest.raises(RuntimeError, match="did not reset|required counters|zero"):
        runtime_module.execute_fsim(artifacts, records)


@pytest.mark.parametrize("value", (True, "1", 0, -1, float("nan"), float("inf"), float("-inf")))
def test_fsim_profiler_rejects_invalid_counter_values(runtime_module, value):
    stats = {counter: 1 for counter in runtime_module.REQUIRED_PROFILER_COUNTERS}
    stats["gemm_counter"] = value

    with pytest.raises(RuntimeError, match="must be positive"):
        runtime_module._validate_profiler_stats(stats)
