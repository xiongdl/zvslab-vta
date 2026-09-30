"""Behavioral tests for the adaptive IC V1 FSIM-to-TSIM search."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TUNE_DIR = Path(__file__).resolve().parents[1] / "tune"


def _load_search():
    spec = importlib.util.spec_from_file_location("ic_v1_search", TUNE_DIR / "search.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_entry():
    path = TUNE_DIR / "tune.py"
    spec = importlib.util.spec_from_file_location("ic_v1_two_stage_entry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def search():
    return _load_search()


def _identity(space=250):
    return {
        "model": "image_classification_v1",
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
