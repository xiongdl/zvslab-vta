"""Focused checks for IC V1 timeout selection and candidate isolation."""

import importlib.util
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_PATH = APP_ROOT / "tune" / "measurement.py"


def _load_measurement():
    if str(APP_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(APP_ROOT.parent))
    spec = importlib.util.spec_from_file_location("ic_v1_measurement", MEASUREMENT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def measurement():
    return _load_measurement()


def test_backend_timeouts_are_independent_and_overrideable(measurement):
    assert measurement.backend_timeout("fsim") == 60
    assert measurement.backend_timeout("tsim") == 120
    assert measurement.backend_timeout("fsim", 7) == 7
    assert measurement.backend_timeout("tsim", 9) == 9
    for backend, timeout in (("host", None), ("fsim", 0), ("tsim", True)):
        with pytest.raises(ValueError):
            measurement.backend_timeout(backend, timeout)


def test_candidate_runner_is_fresh_and_closed_after_success_or_failure(
    measurement, monkeypatch
):
    runners = []
    options_seen = []
    batch_calls = []

    class Runner:
        def __init__(self, backend, timeout):
            self.backend = backend
            self.timeout = timeout
            self.closed = False

        def close(self):
            self.closed = True

    class Builder:
        executor = None
        tmp_dir = None

    def fake_measure_option(backend, **kwargs):
        runner = Runner(backend, kwargs["timeout"])
        runners.append(runner)
        options_seen.append((backend, kwargs))
        return {"runner": runner, "builder": Builder()}

    def fake_create_measure_batch(task, option):
        def measure_batch(inputs):
            batch_calls.append((task, option["runner"], inputs))
            if inputs[0].config == "candidate-that-aborts":
                raise RuntimeError("simulated candidate abort")
            return ["measured"]

        return measure_batch

    monkeypatch.setattr(measurement.shared, "measure_option", fake_measure_option)
    monkeypatch.setattr(
        measurement.shared.autotvm.measure, "create_measure_batch", fake_create_measure_batch
    )
    task = type("Task", (), {"target": "vta"})()

    assert measurement.measure_candidate(task, "good", "fsim") == "measured"
    with pytest.raises(RuntimeError, match="candidate abort"):
        measurement.measure_candidate(task, "candidate-that-aborts", "fsim")

    assert [runner.timeout for runner in runners] == [60, 60]
    assert all(runner.closed for runner in runners)
    assert runners[0] is not runners[1]
    assert all(item[1] is runner for item, runner in zip(batch_calls, runners))
    assert [options[0] for options in options_seen] == ["fsim", "fsim"]
