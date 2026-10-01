"""Behavioral tests for shared seed gates, resumable search and replay checks."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "tuning_controller.py"
SPEC = importlib.util.spec_from_file_location("mlperf_tiny_tuning_controller", CONTROLLER_PATH)
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)


def _digest(char="a"):
    return char * 64


def _identity(**overrides):
    identity = {
        "model": "anomaly_detection_v1",
        "model_sha256": _digest("a"),
        "geometry_sha256": _digest("b"),
        "workload_sha256": _digest("c"),
        "fusion_sha256": _digest("d"),
        "symbol": "tvmgen_mlperf_anomaly_vta_main_0",
        "occurrence": 0,
        "config_space_size": 205,
        "valid_config_space_size": 205,
        "trial_batch": 100,
        "min_successful": 20,
        "fsim_timeout_seconds": 60,
        "tsim_timeout_seconds": 120,
    }
    identity.update(overrides)
    return identity


def _seed_identity():
    return {
        "model": "anomaly_detection_v1",
        "model_sha256": _digest("a"),
        "geometry_sha256": _digest("b"),
        "occurrences": [0],
    }


def _seed_files(tmp_path, *, deployment_cycles=110):
    identity = _seed_identity()
    config = {"tile_h": 1}
    config_hash = controller.sha256_json(config)
    entry = {
        "occurrence": 0,
        "symbol": "tvmgen_mlperf_anomaly_vta_main_0",
        "fusion_sha256": _digest("d"),
        "workload_sha256": _digest("c"),
        "config": config,
        "config_sha256": config_hash,
        "config_index": 0,
        "fsim_success": True,
        "autotvm_cycles": 100,
    }
    manifest_path = tmp_path / "seed" / "seed-manifest.json"
    manifest_hash = controller.write_seed_manifest(manifest_path, identity, [entry])
    report = {
        "schema_version": 1,
        "artifact_kind": "vta_deployment_profile_v1",
        "phase": "seed",
        "status": "passed",
        "sample_count": 1,
        "measurement_protocol": controller.TSIM_PROTOCOL,
        "model": identity["model"],
        "model_sha256": identity["model_sha256"],
        "geometry_sha256": identity["geometry_sha256"],
        "selected_manifest_sha256": manifest_hash,
        "occurrences": [{
            "occurrence": 0,
            "symbol": entry["symbol"],
            "fusion_sha256": entry["fusion_sha256"],
            "workload_sha256": entry["workload_sha256"],
            "config_sha256": config_hash,
            "autotvm_cycles": 100,
            "deployment_cycles": deployment_cycles,
            "passed": True,
        }],
    }
    report_path = tmp_path / "seed" / "deployment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return identity, manifest_path, report_path


def _seed_gate(tmp_path):
    identity, manifest, report = _seed_files(tmp_path)
    return controller.validate_seed_gate(
        report, manifest, identity,
        [{"occurrence": 0, "symbol": "tvmgen_mlperf_anomaly_vta_main_0",
          "fusion_sha256": _digest("d"), "workload_sha256": _digest("c")}],
    )


def test_seed_gate_binds_manifest_report_and_exact_ten_percent_passes(tmp_path):
    identity, manifest_path, report_path = _seed_files(tmp_path, deployment_cycles=110)

    binding = controller.validate_seed_gate(
        report_path, manifest_path, identity,
        [{"occurrence": 0, "symbol": "tvmgen_mlperf_anomaly_vta_main_0",
          "fusion_sha256": _digest("d"), "workload_sha256": _digest("c")}],
    )

    assert binding.seed_manifest_sha256 == controller.sha256_file(manifest_path)
    assert binding.seed_gate_report_sha256 == controller.sha256_file(report_path)


def test_seed_gate_rejects_above_ten_percent_and_foreign_manifest(tmp_path):
    identity, manifest_path, report_path = _seed_files(tmp_path, deployment_cycles=111)
    expected = [{"occurrence": 0, "symbol": "tvmgen_mlperf_anomaly_vta_main_0",
                 "fusion_sha256": _digest("d"), "workload_sha256": _digest("c")}]
    with pytest.raises(ValueError, match="exceeds 10%"):
        controller.validate_seed_gate(report_path, manifest_path, identity, expected)

    identity["model"] = "keyword_spotting_v1"
    with pytest.raises(ValueError, match="another model"):
        controller.validate_seed_gate(report_path, manifest_path, identity, expected)


def test_search_batches_unique_trials_to_quota_and_stops_at_exhaustion(tmp_path):
    identity = _identity(valid_config_space_size=205, min_successful=2)
    gate = _seed_gate(tmp_path)
    state = controller.create_state(identity, tmp_path / "state.json", seed_gate=gate)
    assert controller.next_batch_size(state) == 100
    controller.record_fsim_result(state, 0, {"tile": 0}, success=False, error="invalid")
    controller.record_fsim_result(state, 1, {"tile": 1}, success=True)
    assert controller.next_batch_size(state) == 100
    controller.record_fsim_result(state, 2, {"tile": 2}, success=True)
    assert controller.successful_fsim_count(state) == 2
    assert controller.next_batch_size(state) == 0
    assert state["stop_reason"] == "successful_schedule_quota"

    exhausted = controller.create_state(
        _identity(valid_config_space_size=5, min_successful=2),
        tmp_path / "exhausted.json", seed_gate=gate,
    )
    assert controller.next_batch_size(exhausted) == 5
    for index in range(5):
        controller.record_fsim_result(
            exhausted, index, {"tile": index}, success=False, error="schedule rejected"
        )
    assert controller.next_batch_size(exhausted) == 0
    assert exhausted["stop_reason"] == "configuration_space_exhausted"


def test_default_search_uses_100_new_trials_per_batch_until_20_successes(tmp_path):
    state = controller.create_state(_identity(), tmp_path / "default.json",
                                    seed_gate=_seed_gate(tmp_path))
    assert controller.next_batch_size(state) == 100
    for index in range(100):
        controller.record_fsim_result(
            state, index, {"tile": index}, success=index < 10,
            error=None if index < 10 else "compile failed",
        )
    assert controller.successful_fsim_count(state) == 10
    assert controller.next_batch_size(state) == 100
    for index in range(100, 110):
        controller.record_fsim_result(state, index, {"tile": index}, success=True)
    assert controller.successful_fsim_count(state) == 20
    assert state["stop_reason"] == "successful_schedule_quota"
    assert controller.next_batch_size(state) == 0


def test_duplicate_fsim_indices_and_configs_are_rejected(tmp_path):
    state = controller.create_state(_identity(), tmp_path / "state.json",
                                    seed_gate=_seed_gate(tmp_path))
    controller.record_fsim_result(state, 0, {"tile": 1}, success=True)
    with pytest.raises(ValueError, match="already visited"):
        controller.record_fsim_result(state, 0, {"tile": 2}, success=True)
    with pytest.raises(ValueError, match="duplicate configuration"):
        controller.record_fsim_result(state, 1, {"tile": 1}, success=True)


def test_every_fsim_success_gets_one_tsim_attempt_and_best_is_deployable(tmp_path):
    state = controller.create_state(_identity(), tmp_path / "state.json",
                                    seed_gate=_seed_gate(tmp_path))
    controller.record_fsim_result(state, 0, {"tile": 0}, success=True)
    controller.record_fsim_result(state, 1, {"tile": 1}, success=True)
    first, second = state["fsim_results"]
    controller.record_tsim_result(state, first["config_sha256"], cycles=None, error="RPC timeout")
    controller.record_tsim_result(state, second["config_sha256"], cycles=51,
                                   deployment_lowerable=True)
    assert controller.pending_tsim_configs(state) == []
    assert controller.select_best_tsim(state)["config_sha256"] == second["config_sha256"]

    other = controller.create_state(_identity(), tmp_path / "other.json",
                                    seed_gate=_seed_gate(tmp_path))
    controller.record_fsim_result(other, 0, {"tile": 0}, success=True)
    with pytest.raises(ValueError, match="still need TSIM"):
        controller.select_best_tsim(other)


def test_resume_requires_bound_seed_identity_and_untampered_atomic_state(tmp_path):
    identity = _identity()
    path = tmp_path / "state.json"
    with pytest.raises(ValueError, match="validated SeedGateBinding"):
        controller.create_state(identity, path, seed_gate={
            "seed_manifest_sha256": _digest("e"),
            "seed_gate_report_sha256": _digest("f"),
        })
    state = controller.create_state(identity, path, seed_gate=_seed_gate(tmp_path))
    controller.record_fsim_result(state, 0, {"tile": 0}, success=True)
    controller.save_state(state)
    loaded = controller.load_state(path, state["identity"])
    assert loaded["visited_indices"] == [0]

    foreign = dict(loaded["identity"], seed_gate_report_sha256=_digest("9"))
    with pytest.raises(ValueError, match="identity does not match"):
        controller.load_state(path, foreign)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["fsim_results"][0]["config"]["tile"] = 99
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity hash"):
        controller.load_state(path, state["identity"])


def test_replay_rejects_tampered_artifacts_and_accepts_bound_exports(tmp_path):
    identity = _identity(seed_manifest_sha256=_digest("e"), seed_gate_report_sha256=_digest("f"))
    result = tmp_path / "result.json"
    native = tmp_path / "best.tsim.log"
    result_value = {
        "schema_version": 1,
        "model": identity["model"],
        "model_sha256": identity["model_sha256"],
        "geometry_sha256": identity["geometry_sha256"],
        "occurrence": 0,
        "symbol": identity["symbol"],
        "fusion_sha256": identity["fusion_sha256"],
        "workload_sha256": identity["workload_sha256"],
        "config_sha256": _digest("9"),
        "tsim_cycles": 25,
    }
    result.write_text(json.dumps(result_value), encoding="utf-8")
    native.write_text("native\n", encoding="utf-8")
    entry = {
        "occurrence": 0,
        "result_json": result.name,
        "result_sha256": controller.sha256_file(result),
        "symbol": identity["symbol"],
        "fusion_sha256": identity["fusion_sha256"],
        "workload_sha256": identity["workload_sha256"],
        "config_sha256": _digest("9"),
        "native_record": native.name,
        "native_record_sha256": controller.sha256_file(native),
        "tsim_cycles": 25,
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "vta_selected_schedules_v1",
        "identity": identity,
        "identity_sha256": controller.sha256_json(identity),
        "measurement_protocol": controller.TSIM_PROTOCOL,
        "entries": [entry],
    }
    path = tmp_path / "best-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    expected = [{"occurrence": 0, "symbol": identity["symbol"],
                 "fusion_sha256": identity["fusion_sha256"],
                 "workload_sha256": identity["workload_sha256"]}]
    assert controller.validate_replay_manifest(path, identity, expected) == manifest
    native.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched hash"):
        controller.validate_replay_manifest(path, identity, expected)


def test_worker_environment_selects_isolated_backend_and_geometry(tmp_path):
    geometry = tmp_path / "vta.json"
    geometry.write_text("{}", encoding="utf-8")
    environment = controller.worker_environment(
        "tsim", {"PYTHONPATH": "existing", "VTA_BACKEND": "fsim"},
        geometry, ["/repo/tvm/python", "/repo/vta/python"],
    )
    assert environment["VTA_BACKEND"] == "tsim"
    assert environment["VTA_CONFIG_FILE"] == str(geometry.resolve())
    assert environment["PYTHONPATH"].split(":") == [
        "/repo/tvm/python", "/repo/vta/python", "existing"
    ]
    with pytest.raises(ValueError, match="backend"):
        controller.worker_environment("auto", {}, geometry)
