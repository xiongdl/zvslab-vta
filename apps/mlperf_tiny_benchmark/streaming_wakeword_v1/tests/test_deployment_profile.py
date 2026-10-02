"""Deployment report, HOST comparison, and schedule evidence contracts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = APP_ROOT / "runtime.py"
RUN = APP_ROOT / "run.py"


def _load_runtime():
    spec = importlib.util.spec_from_file_location("streaming_ww_v1_report_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_cycle_gate_is_inclusive_at_ten_percent():
    runtime = _load_runtime()

    assert runtime.cycles_within_ten_percent(110, 100) is True
    assert runtime.cycles_within_ten_percent(90, 100) is True
    assert runtime.cycles_within_ten_percent(111, 100) is False
    with pytest.raises(ValueError, match="positive integer"):
        runtime.cycles_within_ten_percent(True, 100)


def test_deployment_sample_is_one_fixed_stateless_audio_window(monkeypatch):
    runtime = _load_runtime()
    sample = SimpleNamespace(path=Path("marvin.wav"), filename="marvin.wav", order=0)
    monkeypatch.setattr(runtime, "committed_sample_records", lambda: (sample, object(), object()))
    monkeypatch.setattr(runtime, "load_sample", lambda _path: np.zeros(runtime.INPUT_SHAPE, dtype=np.int8))

    selected, input_data, evidence = runtime.select_deployment_sample()

    assert selected is sample
    assert input_data.shape == runtime.INPUT_SHAPE
    assert input_data.dtype == np.int8
    assert evidence == {
        "sample_count": 1,
        "audio_window_count": 1,
        "audio_window_samples": runtime.CLIP_FRAMES,
        "feature_frame_count": runtime.INPUT_SHAPE[1],
        "model_invocations": 1,
        "state_policy": "stateless_single_invocation",
    }


def test_run_parser_has_one_schedule_and_evidence_interface():
    run_spec = importlib.util.spec_from_file_location("streaming_ww_v1_run_contract", RUN)
    run_module = importlib.util.module_from_spec(run_spec)
    sys.modules[run_spec.name] = run_module
    sys.path.insert(0, str(APP_ROOT))
    try:
        run_spec.loader.exec_module(run_module)
    finally:
        sys.path.pop(0)

    defaults = run_module._parser().parse_args([])
    assert defaults.schedule is None
    assert defaults.deployment_report is None
    assert defaults.validate_schedule_evidence is False
    assert run_module._parser().parse_args(["--schedule", "none"]).schedule == "none"
    assert run_module._parser().parse_args(["--schedule", "candidate.log"]).schedule == "candidate.log"
    assert not hasattr(defaults, "autotvm_log")


def test_schedule_evidence_requires_complete_measured_occurrences():
    runtime = _load_runtime()
    layers = (SimpleNamespace(occurrence=0),)
    selected = SimpleNamespace(measured=True)
    snapshot = SimpleNamespace(selected={0: selected})
    runtime._validate_schedule_evidence(SimpleNamespace(layers=layers), snapshot)

    with pytest.raises(ValueError, match="complete occurrence coverage"):
        runtime._validate_schedule_evidence(
            SimpleNamespace(layers=layers), SimpleNamespace(selected={})
        )
