"""Focused tests for the IC V1 single-workload AutoTVM command."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TUNE_PATH = Path(__file__).resolve().parents[1] / "tune.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("ic_v1_tune", TUNE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tune():
    return _load_tune()


def test_select_workload_uses_zero_based_extraction_order(tune):
    tasks = [SimpleNamespace(name=f"task-{index}") for index in range(3)]

    assert tune.select_workload(tasks, 0) is tasks[0]
    assert tune.select_workload(tasks, 2) is tasks[2]


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_invalid_workload_index_reports_valid_range(tune, index):
    tasks = [SimpleNamespace(name="first"), SimpleNamespace(name="second")]

    with pytest.raises(ValueError, match=r"valid workload indices are 0\.\.1"):
        tune.select_workload(tasks, index)


def test_tuning_defaults_are_random_fsim_local_32_trials_and_120_seconds(tune):
    options = tune.build_tuning_options()

    assert options == {
        "tuner": "random",
        "backend": "fsim",
        "runner": "local",
        "trials": 32,
        "timeout": 120,
    }


def test_invalid_index_fails_before_runner_creation_or_output_write(tune, monkeypatch, tmp_path):
    tasks = [SimpleNamespace(name="only-task")]
    monkeypatch.setattr(tune, "prepare_v1_tasks", lambda: tasks)
    monkeypatch.setattr(tune.shared, "_config_identity", lambda _path: None)
    monkeypatch.setattr(tune.shared, "create_runner", lambda *_a, **_k: pytest.fail("runner created"))
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="valid workload indices"):
        tune.run_tuning(1, output_dir=output_dir)

    assert not output_dir.exists()


def test_tuning_validates_active_alternate_geometry_before_workload_selection(
    tune, monkeypatch, tmp_path
):
    config_path = tmp_path / "alternate-vta-geometry.json"
    config_path.write_text(
        '{"LOG_BATCH": 0, "LOG_BLOCK": 3, "LOG_OUT_WIDTH": 3, "LOG_OUT_HEIGHT": 3}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config_path))
    monkeypatch.setattr(tune, "prepare_v1_tasks", lambda: [SimpleNamespace(name="only-task")])

    with pytest.raises(ValueError, match="valid workload indices"):
        tune.run_tuning(1, output_dir=tmp_path / "out")


def test_tsim_measures_best_fsim_record_config(tune, monkeypatch, tmp_path):
    task = SimpleNamespace(name="conv2d_packed.vta", config_space=[0, 1, 2], flop=2048)
    selected_config = object()
    measure_input = SimpleNamespace(config=selected_config, task=task, target="vta")
    fsim_result = SimpleNamespace(error_no=0, costs=(0.001,))
    tsim_result = SimpleNamespace(error_no=0, costs=(7654,))
    monkeypatch.setattr(tune, "prepare_v1_tasks", lambda: [task])
    monkeypatch.setattr(tune.shared, "_config_identity", lambda _path: None)
    monkeypatch.setattr(tune.shared, "_task_workload_id", lambda _task: "a" * 64)
    monkeypatch.setattr(tune.shared.autotvm.record, "pick_best", lambda _src, _dst: None)
    monkeypatch.setattr(
        tune.shared.autotvm.record,
        "load_from_file",
        lambda _path: iter([(measure_input, fsim_result)]),
    )
    calls = []
    fsim_runners = []

    class FakeTuner:
        def __init__(self, actual_task):
            assert actual_task is task

        def tune(self, **kwargs):
            calls.append(("fsim", kwargs["n_trial"], kwargs["measure_option"]["runner"].timeout))

    class FakeBuilder:
        def set_task(self, actual_task, build_kwargs):
            assert actual_task is task
            assert build_kwargs == {}

        def build(self, inputs):
            assert inputs == [measure_input]
            return [SimpleNamespace(error=None, filename="best.so")]

        def __del__(self):
            pass

    class FakeRunner:
        def __init__(self):
            self.timeout = 120
            self.server = None
            self.tracker = None

        def set_task(self, actual_task):
            assert actual_task is task

        def get_build_kwargs(self):
            return {}

        def run(self, inputs, build_results):
            assert inputs == [measure_input]
            assert build_results[0].filename == "best.so"
            calls.append(("tsim", inputs[0].config))
            return [tsim_result]

    monkeypatch.setattr(tune.shared.autotvm.tuner, "RandomTuner", FakeTuner)
    def make_measure_option(_backend, **_opts):
        runner = SimpleNamespace(timeout=120, server=None, tracker=None)
        fsim_runners.append(runner)
        return {"runner": runner}

    monkeypatch.setattr(tune.shared, "measure_option", make_measure_option)
    monkeypatch.setattr(tune.shared.autotvm, "LocalBuilder", lambda **_opts: FakeBuilder())
    monkeypatch.setattr(tune.shared, "create_runner", lambda backend, **_opts: FakeRunner())
    monkeypatch.setattr(tune.shared, "validate_backend", lambda _backend: None)
    monkeypatch.setattr(tune.shared, "load_simulator_backend", lambda _backend: None)
    monkeypatch.setattr(tune.shared.autotvm, "callback", SimpleNamespace(log_to_file=lambda _path: object()))

    result = tune.run_tuning(0, output_dir=tmp_path / "out", trials=32, timeout=120)

    assert calls[:3] == [("fsim", 1, 120)] * 3
    assert len({id(runner) for runner in fsim_runners}) == 3
    assert calls[3] == ("tsim", selected_config)
    assert result["tsim_cycles"] == 7654
    assert result["mac_count"] == 1024
    assert result["workload_sha256"] == "a" * 64
