"""Behavioral tests for the adaptive IC V2 FSIM-to-TSIM search."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TUNE_DIR = Path(__file__).resolve().parents[1] / "tune"


def _load_search():
    spec = importlib.util.spec_from_file_location("ic_v2_search", TUNE_DIR / "search.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_entry():
    path = TUNE_DIR / "tune.py"
    spec = importlib.util.spec_from_file_location("ic_v2_two_stage_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def search():
    return _load_search()


def _identity(space=250):
    return {
        "model": "image_classification_v2",
        "model_sha256": "m" * 64,
        "geometry_sha256": "g" * 64,
        "workload_sha256": "w" * 64,
        "occurrence": 2,
        "config_space_size": space,
        "trial_batch": 100,
        "min_successful": 20,
        "fsim_timeout_seconds": 60,
        "tsim_timeout_seconds": 120,
    }


def test_batching_stops_after_success_quota_and_keeps_distinct_visits(search, tmp_path):
    path = tmp_path / "state.json"
    state = search.create_state(_identity(), path)
    assert search.next_batch_size(state) == 100
    for index in range(100):
        search.record_trial(
            state, index, {"index": index}, successful=index < 20,
            error=None if index < 20 else "invalid schedule",
        )
    search.save_state(state)

    assert search.success_count(state) == 20
    assert search.should_continue(state) is False
    assert search.stop_reason(state) == "successful_schedule_quota"
    assert len(json.loads(path.read_text())["visited_indices"]) == 100


def test_resume_preserves_visited_state_and_last_batch_is_short(search, tmp_path):
    path = tmp_path / "state.json"
    state = search.create_state(_identity(space=135), path)
    for index in range(100):
        search.record_trial(state, index, {"index": index}, successful=index < 19)
    search.save_state(state)

    resumed = search.load_state(path, _identity(space=135))

    assert resumed["visited_indices"] == list(range(100))
    assert search.next_batch_size(resumed) == 35
    for index in range(100, 135):
        search.record_trial(resumed, index, {"index": index}, successful=False)
    assert search.should_continue(resumed) is False
    assert search.stop_reason(resumed) == "configuration_space_exhausted"


def test_resume_rejects_mismatched_identity_and_duplicate_indices(search, tmp_path):
    path = tmp_path / "state.json"
    state = search.create_state(_identity(), path)
    search.record_trial(state, 3, {"index": 3}, successful=True)
    search.save_state(state)

    changed = {**_identity(), "geometry_sha256": "x" * 64}
    with pytest.raises(ValueError, match="identity does not match"):
        search.load_state(path, changed)
    with pytest.raises(ValueError, match="already visited"):
        search.record_trial(state, 3, {"index": 3}, successful=True)


def test_successes_are_deduplicated_by_configuration(search, tmp_path):
    state = search.create_state(_identity(), tmp_path / "state.json")
    search.record_trial(state, 0, {"entity": [1, 2]}, successful=True)
    search.record_trial(state, 1, {"entity": [1, 2]}, successful=True)

    assert search.success_count(state) == 1
    assert len(search.successful_configurations(state)) == 1


def test_no_success_still_records_exhaustion_and_rejects_empty_tsims(search, tmp_path):
    state = search.create_state(_identity(space=2), tmp_path / "state.json")
    search.record_trial(state, 0, {"index": 0}, successful=False, error="build failed")
    search.record_trial(state, 1, {"index": 1}, successful=False, error="timeout")

    assert search.stop_reason(state) == "configuration_space_exhausted"
    assert state["failures"] == [
        {"config_index": 0, "error": "build failed"},
        {"config_index": 1, "error": "timeout"},
    ]
    with pytest.raises(ValueError, match="no successful TSIM"):
        search.select_best_tsim([])


def test_best_candidate_uses_minimum_positive_tsim_cycles(search):
    candidates = [
        {"config_index": 0, "tsim_cycles": 900},
        {"config_index": 1, "tsim_cycles": 800},
        {"config_index": 2, "tsim_cycles": 0},
        {"config_index": 3, "tsim_cycles": 850, "error": "measurement error"},
    ]

    assert search.select_best_tsim(candidates) == candidates[1]


def test_worker_selection_keeps_occurrence_index_separate_from_task():
    entry = _load_entry()
    task = object()
    entry.legacy.select_workload = lambda tasks, index: tasks[index]

    index, selected = entry._select_indexed_workload([object(), task], 1)

    assert index == 1
    assert selected is task


def test_exported_native_record_is_hashed_and_independently_loadable(tmp_path):
    artifacts_spec = importlib.util.spec_from_file_location(
        "ic_v2_artifact_helpers", TUNE_DIR / "artifacts.py"
    )
    artifacts = importlib.util.module_from_spec(artifacts_spec)
    sys.modules[artifacts_spec.name] = artifacts
    artifacts_spec.loader.exec_module(artifacts)
    config = {"index": 7, "entity": [["tile", "sp", [1, 2]]]}
    measure_input = type("Input", (), {"config": type(
        "Config", (), {"to_json_dict": lambda self: config}
    )()})()
    result = type("Result", (), {"error_no": 0, "costs": (321,)})()

    class RecordModule:
        @staticmethod
        def encode(actual_input, actual_result):
            return json.dumps({"config": actual_input.config.to_json_dict(), "cycles": actual_result.costs[0]})

        @staticmethod
        def load_from_file(path):
            item = json.loads(Path(path).read_text().strip())
            assert item == {"config": config, "cycles": 321}
            return iter([(measure_input, result)])

    destination = tmp_path / "best-native.log"
    exported = artifacts.export_selected_record(
        [(measure_input, result)], config, 321, destination, RecordModule
    )

    loaded_input, loaded_result = artifacts.load_validated_record(
        destination, exported["sha256"], RecordModule
    )

    assert loaded_input is measure_input
    assert loaded_result is result


def test_resume_manifest_rejects_foreign_model_and_changed_fusion_identity():
    entry = _load_entry()
    identity = {
        "model_sha256": "m" * 64,
        "geometry_path": "/geometry.json",
        "geometry_sha256": "g" * 64,
        "trial_batch": 100,
        "min_successful": 20,
        "fsim_timeout_seconds": 60,
        "tsim_timeout_seconds": 120,
        "workload_indices": [0],
    }
    workloads = [{"workload_index": 0, "fusion_sha256": "f" * 64}]
    manifest = {
        "schema_version": 1,
        "model": "image_classification_v2",
        "workload_count": 1,
        "workloads": workloads,
        "selected_workload_indices": [0],
        "run_identity": identity,
    }

    entry._validate_resume_manifest(manifest, identity, workloads, [0], 1)
    foreign_model = {**manifest, "model": "image_classification_v1"}
    with pytest.raises(ValueError, match="another model"):
        entry._validate_resume_manifest(foreign_model, identity, workloads, [0], 1)
    changed_workload = [{"workload_index": 0, "fusion_sha256": "x" * 64}]
    with pytest.raises(ValueError, match="identities"):
        entry._validate_resume_manifest(manifest, identity, changed_workload, [0], 1)
    changed_geometry = {**identity, "geometry_sha256": "x" * 64}
    with pytest.raises(ValueError, match="identity does not match"):
        entry._validate_resume_manifest(manifest, changed_geometry, workloads, [0], 1)


def test_best_manifest_requires_exact_selected_coverage_and_full_run_claim():
    entry = _load_entry()
    bounded = {
        "selected_workload_indices": [0, 2],
        "entries": [{"workload_index": 0}, {"workload_index": 2}],
        "failures": [],
        "bounded": True,
        "completion_label": "BOUNDED_SMOKE_INCOMPLETE",
    }
    assert entry._validate_best_manifest_coverage(bounded, 8)[0] == [0, 2]

    missing = {**bounded, "entries": [{"workload_index": 0}]}
    with pytest.raises(ValueError, match="cover the selected workloads"):
        entry._validate_best_manifest_coverage(missing, 8)
    bad_full_claim = {
        **bounded,
        "bounded": False,
        "completion_label": "FULL_SEARCH",
        "status": "complete",
    }
    with pytest.raises(ValueError, match="complete occurrence coverage"):
        entry._validate_best_manifest_coverage(bad_full_claim, 8)
