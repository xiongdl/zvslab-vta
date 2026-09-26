"""Focused tests for simulator-aware AutoTVM tuning."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


TUNER_PATH = Path(__file__).resolve().parents[1] / "autotvm_tuner.py"


def _load_tuner():
    spec = importlib.util.spec_from_file_location("mlperf_tiny_autotvm_tuner", TUNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tuner():
    return _load_tuner()


def test_backend_validation_requires_explicit_matching_selector(tuner, monkeypatch):
    monkeypatch.setenv("VTA_BACKEND", "tsim")
    assert tuner.validate_backend("tsim") == "tsim"

    with pytest.raises(ValueError, match="backend mismatch"):
        tuner.validate_backend("fsim")

    monkeypatch.delenv("VTA_BACKEND")
    with pytest.raises(ValueError, match="VTA_BACKEND"):
        tuner.validate_backend("tsim")


def test_tsim_profiler_module_loader_resets_and_collects_each_candidate(tuner, monkeypatch):
    events = []
    monkeypatch.setattr(tuner, "tsim_hardware_library", lambda: "/tmp/libvta_hw.dylib")

    class Remote:
        def upload(self, _path):
            events.append("upload")

        def remove(self, _path):
            events.append("remove")

        def get_function(self, name):
            if name.endswith("loadfile_vta-tsim"):
                return lambda _name: events.append("load-hardware") or object()
            if name == "vta.tsim.init":
                return lambda _module: events.append("init-hardware")
            if name.endswith("profiler_clear"):
                return lambda: events.append("clear")
            assert name.endswith("profiler_status")
            readings = iter(({"cycle_count": 0}, {"cycle_count": 37}))
            return lambda: json.dumps(next(readings))

    class BaseLoader:
        @contextmanager
        def __call__(self, remote_kwargs, build_result):
            yield Remote(), object()

    collected = {}
    loader = tuner.ProfilerModuleLoader("tsim", BaseLoader(), collected)
    with loader({}, type("Build", (), {"filename": "candidate.tar"})()):
        events.append("run")

    assert events == [
        "upload",
        "load-hardware",
        "init-hardware",
        "remove",
        "clear",
        "run",
    ]
    assert collected["candidate.tar"] == {"cycle_count": 37}


def test_tsim_trial_cost_requires_positive_integer_cycles(tuner):
    assert tuner.tsim_cycle_cost({"cycle_count": 17}) == 17
    for value in (0, -1, True, 1.5, "17"):
        with pytest.raises(RuntimeError, match="positive integer"):
            tuner.tsim_cycle_cost({"cycle_count": value})


def test_fsim_module_loader_uses_only_fsim_profiler(tuner):
    names = []

    class Remote:
        def get_function(self, name):
            names.append(name)
            if name.endswith("profiler_clear"):
                return lambda: None
            return lambda: json.dumps({"gemm_counter": 0, "wgt_load_nbytes": 0})

    class BaseLoader:
        @contextmanager
        def __call__(self, remote_kwargs, build_result):
            yield Remote(), object()

    loader = tuner.ProfilerModuleLoader("fsim", BaseLoader(), {})
    with loader({}, type("Build", (), {"filename": "candidate.tar"})()):
        pass
    assert names == ["vta.simulator.profiler_clear", "vta.simulator.profiler_status"]


def test_tsim_runner_replaces_wall_clock_result_with_integer_cycles(tuner, monkeypatch):
    runner = tuner.TSIMLocalRunner()
    runner.cycle_stats["candidate.tar"] = {"cycle_count": 41}
    measure_result = tuner.MeasureResult((0.0001,), tuner.MeasureErrorNo.NO_ERROR, 0.1, 12.0)
    monkeypatch.setattr(tuner.LocalRunner, "run", lambda self, _inputs, _builds: [measure_result])
    build_result = type("Build", (), {"filename": "candidate.tar"})()

    measure_input = type("Input", (), {"task": object(), "config": object()})()
    result = runner.run([measure_input], [build_result])[0]

    assert result.costs == (41,)
    assert isinstance(result.costs[0], int)
