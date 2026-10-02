"""Persistent search ledger identity and candidate failure contracts."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tuning.py"
spec = importlib.util.spec_from_file_location("common_tuning_contract", MODULE_PATH)
tuning = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tuning
spec.loader.exec_module(tuning)


def _identity():
    return {
        "model_id": "image_classification_v2",
        "model_sha256": "m" * 64,
        "geometry_sha256": "g" * 64,
        "compute_sha256": "c" * 64,
        "occurrence": 2,
        "symbol": "vta_2",
        "options": {"backend": "fsim", "trial_batch": 2, "min_successful": 1,
                    "fsim_timeout": 60, "tsim_timeout": 120},
    }


def test_ledger_roundtrip_is_bound_to_compute_and_options(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = tuning.create_ledger(_identity(), path)
    tuning.save_ledger(ledger)
    assert tuning.load_ledger(path, _identity())["identity"] == _identity()

    changed = _identity()
    changed["options"] = {**changed["options"], "trial_batch": 3}
    with pytest.raises(ValueError, match="identity does not match"):
        tuning.load_ledger(path, changed)


def test_ledger_records_candidate_success_and_failures_explicitly(tmp_path):
    ledger = tuning.create_ledger(_identity(), tmp_path / "ledger.json")
    tuning.record_candidate(ledger, [4], {"conv": {"index": 4}}, backend="fsim",
                            result={"cycles": None, "protocol": None})
    tuning.record_candidate(ledger, [5], {"conv": {"index": 5}}, backend="fsim",
                            error=RuntimeError("buffer allocation failed"))
    tuning.save_ledger(ledger)

    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert saved["candidates"][0]["status"] == "measured"
    assert saved["candidates"][1]["status"] == "failed"
    assert saved["candidates"][1]["failures"] == [{
        "stage": "fsim", "type": "RuntimeError", "message": "buffer allocation failed"
    }]


def test_ledger_rejects_duplicate_configurations(tmp_path):
    ledger = tuning.create_ledger(_identity(), tmp_path / "ledger.json")
    config = {"conv": {"index": 4}}
    tuning.record_candidate(ledger, [4], config, backend="fsim", result={"cycles": None})
    with pytest.raises(ValueError, match="already recorded"):
        tuning.record_candidate(ledger, [4], config, backend="fsim", result={"cycles": None})


class _Config:
    def __init__(self, index):
        self.index = index

    def valid(self):
        return True

    def to_json_dict(self):
        return {"index": self.index}


class _Space:
    def __len__(self):
        return 2

    def get(self, index):
        return _Config(index)


def test_search_measures_actual_layer_on_both_backends_and_resumes(tmp_path):
    class Layer:
        occurrence = 2
        config_spaces = [("conv2d_packed.vta", ("workload",), "vta", _Space())]

    calls = []

    def measure(layer, activation, indices, backend, timeout):
        calls.append((layer, backend, indices, timeout))
        return {
            "config_identity": f"identity-{backend}-{indices[0]}",
            "worker_pid": 123,
            "cycles": None if backend == "fsim" else 321,
            "protocol": None if backend == "fsim" else {"name": "tsim_single_call", "version": 1},
        }

    identity = _identity()
    ledger_path = tmp_path / "layer.json"
    ledger = tuning.search_layer(
        Layer(), object(), identity, ledger_path, trial_batch=1, min_successful=1,
        fsim_timeout=60, tsim_timeout=120, measure=measure,
    )
    assert ledger["status"] == "complete"
    assert [call[1] for call in calls] == ["fsim", "tsim"]
    assert calls[0][2] == [0]
    assert ledger["candidates"][0]["measurements"]["tsim"]["cycles"] == 321

    resumed = tuning.search_layer(
        Layer(), object(), identity, ledger_path, trial_batch=1, min_successful=1,
        fsim_timeout=60, tsim_timeout=120, resume=True, measure=measure,
    )
    assert resumed["status"] == "complete"
    assert len(calls) == 2


def test_search_never_claims_success_when_all_tsim_candidates_fail(tmp_path):
    class Layer:
        occurrence = 2
        config_spaces = [("conv2d_packed.vta", ("workload",), "vta", _Space())]

    def measure(_layer, _activation, _indices, backend, timeout):
        if backend == "tsim":
            raise TimeoutError("TSIM worker timed out")
        return {"config_identity": "fsim-config", "worker_pid": 1}

    ledger = tuning.search_layer(
        Layer(), object(), _identity(), tmp_path / "layer.json", trial_batch=1,
        min_successful=1, fsim_timeout=60, tsim_timeout=120, measure=measure,
    )
    assert ledger["status"] == "failed"
    assert ledger["stop_reason"] == "no_successful_tsim_measurements"
    assert any(failure["stage"] == "tsim" for failure in ledger["failures"])


def test_candidate_indices_ignore_backend_plumbing_spaces():
    from common.measurement import _selection

    class Layer:
        config_spaces = [
            ("add.vta", ("plumbing",), "vta", _Space()),
            ("conv2d_packed.vta", ("compute",), "vta -model=fsim_64x32", _Space()),
        ]

    selected, _ = _selection(Layer(), [1])
    assert len(selected) == 1
    assert selected[0][0] == "conv2d_packed.vta"
    assert selected[0][3].to_json_dict() == {"index": 1}


def test_alignment_gate_requires_ten_outputs_one_performance_sample_and_cycles(tmp_path):
    schedule = tmp_path / "seed.log"
    schedule.write_bytes(b"native record")
    report_path = tmp_path / "report.json"
    import hashlib

    report = {
        "status": "passed", "model": "image_classification_v2",
        "model_sha256": "m" * 64, "geometry_sha256": "g" * 64,
        "measurement_protocol": "tsim_single_call_v1", "sample_count": 10,
        "outputs_passed": 10, "performance_sample_count": 1,
        "performance_stats": {"cycle_count": 500},
        "schedule_log_sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
        "occurrences": [{"occurrence": 0, "symbol": "vta_0",
                         "deployment_cycles": 99, "autotvm_cycles": 100, "passed": True}],
    }
    report_path.write_text(json.dumps(report))

    class Compute:
        layers = [type("Layer", (), {"occurrence": 0, "symbol": "vta_0"})()]

    assert tuning.validate_alignment_report(
        report_path, model_id="image_classification_v2", model_sha256="m" * 64,
        geometry_sha256="g" * 64, compute=Compute(), schedule_path=schedule,
    ) == report

    report["outputs_passed"] = 9
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="all ten"):
        tuning.validate_alignment_report(
            report_path, model_id="image_classification_v2", model_sha256="m" * 64,
            geometry_sha256="g" * 64, compute=Compute(), schedule_path=schedule,
        )
