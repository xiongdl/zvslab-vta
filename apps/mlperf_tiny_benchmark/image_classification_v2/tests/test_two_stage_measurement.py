"""Focused checks for IC V2 timeout selection and candidate isolation."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_PATH = APP_ROOT / "tune" / "measurement.py"


def _load_measurement():
    if str(APP_ROOT.parent) not in sys.path:
        sys.path.insert(0, str(APP_ROOT.parent))
    spec = importlib.util.spec_from_file_location("ic_v2_measurement", MEASUREMENT_PATH)
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


@pytest.mark.skipif(
    os.environ.get("IC_V2_SIMULATOR_SMOKE") != "1",
    reason="set IC_V2_SIMULATOR_SMOKE=1 for a real single-candidate simulator run",
)
def test_real_backend_measures_one_bounded_candidate():
    import autotvm_tuner
    import vta

    pipeline = autotvm_tuner._load_model_pipeline("image_classification_v2")
    model_path = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"
    prepared = pipeline.prepare_model(model_path)
    sys.path.insert(0, str(APP_ROOT))
    import fused_tasks

    identity = fused_tasks.extract_fused_identities(prepared)[0]
    task = fused_tasks.create_task(identity, vta.get_env().target)
    config = task.config_space.get(0)
    result = _load_measurement().measure_candidate(task, config, os.environ["VTA_BACKEND"])

    assert result.error_no == 0, f"candidate failed: {result.costs!r}"
    assert result.costs and all(float(cost) > 0 for cost in result.costs)
