"""Tests for exact real-deployment identities, counters and cycle evidence."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "deployment_evidence.py"
CALCULATOR_PATH = Path(__file__).resolve().parents[4] / "scripts" / "mac_utilization.py"
SPEC = importlib.util.spec_from_file_location("mlperf_tiny_deployment_evidence", EVIDENCE_PATH)
evidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evidence
SPEC.loader.exec_module(evidence)


def _hash(char):
    return char * 64


def _identity():
    return {
        "occurrence": 0,
        "symbol": "tvmgen_remaining_vta_main_0",
        "fusion_sha256": _hash("a"),
        "workload_sha256": _hash("b"),
        "config_sha256": _hash("c"),
    }


@pytest.mark.parametrize("deployment,expected", [(90, True), (110, True), (111, False), (89, False)])
def test_inclusive_ten_percent_uses_exact_integer_gate(deployment, expected):
    assert evidence.cycles_within_ten_percent(deployment, 100) is expected


@pytest.mark.parametrize("deployment", [0, -1, True, 100.0])
def test_cycle_gate_rejects_invalid_cycle_counts(deployment):
    with pytest.raises(ValueError, match="positive integer"):
        evidence.cycles_within_ten_percent(deployment, 100)


def test_selected_config_requires_exact_symbol_fusion_and_config():
    identity = _identity()
    config = {"tile_h": 1}
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "entries": [{**identity, "config": config,
                     "config_sha256": evidence.sha256_json(config)}],
    }
    assert evidence.validate_selected_configs(manifest, [identity], phase="seed") == manifest["entries"]

    foreign = json.loads(json.dumps(manifest))
    foreign["entries"][0]["symbol"] = "foreign_symbol"
    with pytest.raises(ValueError, match="symbol mismatch"):
        evidence.validate_selected_configs(foreign, [identity], phase="selected")

    lowered_calls = []
    lowered = evidence.lower_selected_configs(
        manifest,
        [identity],
        lambda selected_identity, selected_config: lowered_calls.append(
            (selected_identity["symbol"], selected_config)
        ) or SimpleNamespace(schedule=object()),
        phase="selected",
    )
    assert lowered_calls == [(identity["symbol"], config)]
    assert lowered[0].schedule is not None


def test_occurrence_gate_requires_exact_identity_coverage_and_single_call():
    identity = _identity()
    row = {**identity, "autotvm_cycles": 100, "deployment_cycles": 110,
           "counted_invocations": 1}
    assert evidence.validate_occurrence_rows([row], [identity])[0]["passed"] is True
    with pytest.raises(ValueError, match="exceeds 10%"):
        evidence.validate_occurrence_rows(
            [{**row, "deployment_cycles": 111}], [identity]
        )
    with pytest.raises(ValueError, match="one counted"):
        evidence.validate_occurrence_rows(
            [{**row, "counted_invocations": 2}], [identity]
        )
    with pytest.raises(ValueError, match="coverage"):
        evidence.validate_occurrence_rows([], [identity])


def test_counter_agreement_requires_exact_stats_and_positive_cycles():
    stats = {"cycle_count": 10, "alu_count": 2}
    assert evidence.assert_counter_agreement(stats, dict(stats)) == 10
    with pytest.raises(ValueError, match="disagree"):
        evidence.assert_counter_agreement(stats, {"cycle_count": 11, "alu_count": 2})
    with pytest.raises(ValueError, match="positive integer"):
        evidence.assert_counter_agreement({"cycle_count": 0}, {"cycle_count": 0})


def test_graph_node_mapping_and_resident_cycle_profile():
    identity = _identity()
    graph = {"nodes": [
        {"op": "null", "name": "input"},
        {"op": "tvm_op", "name": "compiled_vta_0",
         "attrs": {"func_name": identity["symbol"]}},
    ]}

    class Session:
        cleared = 0

        def clear_and_validate(self, simulator):
            assert simulator == "tsim"
            self.cleared += 1

        def read_stats(self, simulator):
            return {"cycle_count": 27}

        @staticmethod
        def validate_activity(stats):
            assert stats["cycle_count"] > 0

    class DebugGraph:
        nodes = []

        def _execute_node(self, index):
            self.nodes.append(index)

    session, graph_executor = Session(), DebugGraph()
    profile = evidence.profile_graph_resident_nodes(
        graph, [{"occurrence": 0, "symbol": identity["symbol"]}],
        graph_executor, session, "tsim",
    )
    assert profile[0]["deployment_cycles"] == 27
    assert profile[0]["counted_invocations"] == 1
    assert graph_executor.nodes == [1]
    assert session.cleared == 1


def test_graph_symbol_mapping_does_not_confuse_occurrence_1_with_10():
    graph = {"nodes": [
        {"op": "tvm_op", "name": "tvmgen_vww_vta_main_1",
         "attrs": {"func_name": "tvmgen_vww_vta_main_1"}},
        {"op": "tvm_op", "name": "tvmgen_vww_vta_main_10",
         "attrs": {"func_name": "tvmgen_vww_vta_main_10"}},
    ]}
    expected = [
        {"occurrence": 1, "symbol": "tvmgen_vww_vta_main_1"},
        {"occurrence": 10, "symbol": "tvmgen_vww_vta_main_10"},
    ]

    assert evidence.resolve_graph_nodes(graph, expected) == {1: 0, 10: 1}


def test_one_sample_reference_checker_is_called_once():
    calls = []
    sample = evidence.validate_one_sample(
        "sample-0", _hash("e"), [1], [1],
        lambda reference, output: calls.append((reference, output)),
    )
    assert sample["sample_count"] == 1
    assert calls == [([1], [1])]


def test_build_report_has_calculator_compatible_zero_based_occurrence(tmp_path):
    identity = _identity()
    config = {"tile_h": 1}
    identity["config_sha256"] = evidence.sha256_json(config)
    geometry = tmp_path / "geometry.json"
    geometry.write_text('{"LOG_BATCH": 0, "LOG_BLOCK": 3}\n', encoding="utf-8")
    native = tmp_path / "best.log"
    native.write_text("native record\n", encoding="utf-8")
    result_path = tmp_path / "result.json"
    result = {
        "schema_version": 1,
        "model_sha256": _hash("d"),
        "workload_index": 0,
        "occurrence": 0,
        "symbol": identity["symbol"],
        "fusion_sha256": identity["fusion_sha256"],
        "workload_sha256": identity["workload_sha256"],
        "tsim_cycles": 100,
        "conv_config": config,
        "mac_count": 100,
        "best_native_record": native.name,
        "best_native_record_sha256": evidence.sha256_file(native),
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    geometry_hash = evidence.sha256_file(geometry)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "model_sha256": _hash("d"),
        "geometry_sha256": geometry_hash,
        "measurement_protocol": {**evidence.PROTOCOL},
        "workload_count": 1,
        "entries": [{**identity, "config": config, "mac_count": 100,
                     "workload_index": 0, "tsim_cycles": 100,
                     "result_json": result_path.name}],
    }
    manifest_path = tmp_path / "selected.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sample = {"sample_id": "sample-0", "sha256": _hash("e"), "sample_count": 1}
    report = evidence.build_deployment_report(
        phase="selected",
        model_id="remaining_model",
        model_sha256=_hash("d"),
        geometry_path=geometry,
        sample=sample,
        selected_manifest_path=manifest_path,
        full_model={"baseline_cycles": 500, "tuned_cycles": 450},
        occurrences=[{**identity, "autotvm_cycles": 100,
                      "deployment_cycles": 110, "counted_invocations": 1}],
        expected_occurrences=[identity],
        peak_macs_per_cycle=64,
    )
    assert report["phase"] == "selected"
    assert report["completion_label"] == "OPTIMAL_DEPLOYMENT"
    assert report["sample_count"] == 1
    assert report["occurrence_base"] == 0
    assert report["occurrences"][0]["logical_macs_per_invocation"] == 100
    assert report["selected_manifest_sha256"] == evidence.sha256_file(manifest_path)

    spec = importlib.util.spec_from_file_location(
        "mlperf_tiny_deployment_report_calculator", CALCULATOR_PATH
    )
    calculator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = calculator
    spec.loader.exec_module(calculator)
    calculated = calculator.calculate_deployment_report(report)
    assert calculated["occurrences"][0]["occurrence"] == 0
    assert calculated["occurrences"][0]["relative_cycle_difference"] == 0.1


def test_failure_report_keeps_stage_diagnostics(tmp_path):
    path = evidence.write_failure_report(
        tmp_path / "deployment.failure.json", phase="seed", model_id="remaining_model",
        stage="reference_check", error=RuntimeError("bad output"), details={"sample_id": "s0"},
    )
    failure = json.loads(path.read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["error"] == "RuntimeError: bad output"
    assert failure["details"]["sample_id"] == "s0"


def _write_seed_gate(tmp_path, deployment_cycles=110, autotvm_cycles=100):
    geometry = tmp_path / "geometry.json"
    geometry.write_text('{"LOG_BATCH": 0, "LOG_BLOCK": 3}', encoding="utf-8")
    geometry_hash = evidence.sha256_file(geometry)
    model_id = "remaining_model"
    model_hash = _hash("d")
    identity = _identity()
    config = {"tile": 1}
    config_hash = evidence.sha256_json(config)
    native_path = tmp_path / "seed.tsim.log"
    native_path.write_text("native record\n", encoding="utf-8")
    native_hash = evidence.sha256_file(native_path)
    result = {
        "model": model_id, "model_sha256": model_hash,
        "geometry_sha256": geometry_hash, "workload_index": 0,
        "symbol": identity["symbol"], "fusion_sha256": identity["fusion_sha256"],
        "workload_sha256": identity["workload_sha256"], "tsim_cycles": autotvm_cycles,
        "best_native_record": native_path.name,
        "best_native_record_sha256": native_hash, "conv_config": config,
    }
    result_path = tmp_path / "seed-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "schema_version": 1, "status": "complete", "model": model_id,
        "model_id": model_id, "model_sha256": model_hash,
        "geometry_sha256": geometry_hash, "phase": "seed", "bounded": False,
        "completion_label": "SEED_COMPLETE", "workload_count": 1,
        "selected_workload_indices": [0], "failures": [],
        "measurement_protocol": {"name": "tsim_single_call", "version": 1,
                                  "warmup_excluded": True, "counted_invocations": 1},
        "entries": [{**identity, "config": config, "config_sha256": config_hash,
                     "workload_index": 0, "tsim_cycles": autotvm_cycles,
                     "result_json": result_path.name,
                     "native_record": native_path.name,
                     "native_record_sha256": native_hash}],
    }
    manifest_path = tmp_path / "seed-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = {
        "schema_version": 1, "artifact_kind": evidence.REPORT_KIND,
        "phase": "seed", "status": "passed", "completion_label": "SEED_ALIGNMENT",
        "model_id": model_id, "model": model_id, "model_sha256": model_hash,
        "backend": "tsim", "sample_count": 1,
        "measurement_protocol": {**evidence.PROTOCOL,
                                  "operator_counted_invocations": 1,
                                  "full_model_counted_invocations": 1},
        "geometry": {"path": str(geometry.resolve()), "sha256": geometry_hash},
        "selected_manifest": str(manifest_path),
        "selected_manifest_sha256": evidence.sha256_file(manifest_path),
        "occurrences": [{**identity, "config_sha256": config_hash,
                         "deployment_cycles": deployment_cycles,
                         "autotvm_cycles": autotvm_cycles,
                         "relative_cycle_difference": (
                             abs(deployment_cycles - autotvm_cycles) / autotvm_cycles
                             if isinstance(autotvm_cycles, int) and autotvm_cycles > 0 else None
                         ),
                         "counted_invocations": 1, "passed": True}],
    }
    report_path = tmp_path / "seed-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, geometry, geometry_hash, model_hash, model_id, [identity]


def test_seed_alignment_gate_binds_cycles_and_complete_seed_manifest(tmp_path):
    report_path, geometry, geometry_hash, model_hash, model_id, expected = _write_seed_gate(tmp_path)
    assert evidence.validate_seed_alignment_report(
        report_path, model_id=model_id, model_sha256=model_hash,
        geometry_path=geometry, geometry_sha256=geometry_hash,
        expected_occurrences=expected,
    ) == report_path.resolve()


@pytest.mark.parametrize("deployment,autotvm", [(111, 100), (0, 100), (True, 100), (110, 0)])
def test_seed_alignment_gate_rejects_invalid_or_over_ten_percent_cycles(
    tmp_path, deployment, autotvm
):
    report_path, geometry, geometry_hash, model_hash, model_id, expected = _write_seed_gate(
        tmp_path, deployment, autotvm
    )
    with pytest.raises(ValueError):
        evidence.validate_seed_alignment_report(
            report_path, model_id=model_id, model_sha256=model_hash,
            geometry_path=geometry, geometry_sha256=geometry_hash,
            expected_occurrences=expected,
        )


def test_seed_alignment_gate_rejects_tampered_manifest_binding(tmp_path):
    report_path, geometry, geometry_hash, model_hash, model_id, expected = _write_seed_gate(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selected_manifest_sha256"] = _hash("f")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        evidence.validate_seed_alignment_report(
            report_path, model_id=model_id, model_sha256=model_hash,
            geometry_path=geometry, geometry_sha256=geometry_hash,
            expected_occurrences=expected,
        )


@pytest.mark.parametrize("field,value", [
    ("model_id", "foreign_model"),
    ("geometry.sha256", _hash("f")),
])
def test_seed_alignment_gate_rejects_foreign_report_identity(tmp_path, field, value):
    report_path, geometry, geometry_hash, model_hash, model_id, expected = _write_seed_gate(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if field == "geometry.sha256":
        report["geometry"]["sha256"] = value
    else:
        report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        evidence.validate_seed_alignment_report(
            report_path, model_id=model_id, model_sha256=model_hash,
            geometry_path=geometry, geometry_sha256=geometry_hash,
            expected_occurrences=expected,
        )


def test_seed_alignment_gate_rejects_incomplete_bound_manifest(tmp_path):
    report_path, geometry, geometry_hash, model_hash, model_id, expected = _write_seed_gate(tmp_path)
    manifest_path = tmp_path / "seed-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "incomplete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selected_manifest_sha256"] = evidence.sha256_file(manifest_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="complete, matching seed manifest"):
        evidence.validate_seed_alignment_report(
            report_path, model_id=model_id, model_sha256=model_hash,
            geometry_path=geometry, geometry_sha256=geometry_hash,
            expected_occurrences=expected,
        )


@pytest.mark.parametrize("mutate,match", [
    (lambda manifest: manifest.update(status="incomplete"), "incomplete"),
    (lambda manifest: manifest.update(entries=[]), "cover every"),
    (lambda manifest: manifest.update(selected_workload_indices=[]), "cover every"),
])
def test_replay_manifest_requires_complete_occurrence_coverage(tmp_path, mutate, match):
    _, geometry, geometry_hash, model_hash, model_id, expected = _write_seed_gate(tmp_path)
    manifest = json.loads((tmp_path / "seed-manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    with pytest.raises(ValueError, match=match):
        evidence.validate_replay_manifest(
            manifest, model_id=model_id, model_sha256=model_hash,
            geometry_sha256=geometry_hash, expected_occurrences=expected,
        )
