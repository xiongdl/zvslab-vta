"""Focused AD V1 schedule and deployment-evidence contracts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_runtime():
    spec = importlib.util.spec_from_file_location("ad_v1_profile_runtime", APP_ROOT / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_ad_cycle_gate_keeps_inclusive_ten_percent_boundary():
    runtime = _load_runtime()

    assert runtime.cycles_within_ten_percent(110, 100) is True
    assert runtime.cycles_within_ten_percent(90, 100) is True
    assert runtime.cycles_within_ten_percent(111, 100) is False
    with pytest.raises(ValueError, match="positive integer"):
        runtime.cycles_within_ten_percent(True, 100)


def test_sample_selection_preserves_first_representative_window():
    runtime = _load_runtime()
    features = np.arange(5 * 640, dtype=np.float32).reshape(5, 640)

    selected, sampled, scope = runtime._select_tsim_windows(features, 1)

    assert selected.shape == (1, 640)
    assert np.array_equal(selected[0], features[0])
    assert sampled is True
    assert scope == "representative_windows"


def test_schedule_measurement_requires_tsim_single_call_cycles():
    runtime = _load_runtime()
    from types import SimpleNamespace

    layer = SimpleNamespace(
        occurrence=3,
        config_spaces=(
            ("add.vta", (), "target", range(1)),
            ("conv2d_packed.vta", (), "target", range(2)),
        ),
    )
    selected = SimpleNamespace(
        configs=(None, object()),
        measurement={
            "backend": "tsim",
            "protocol": "tsim_single_call_v1",
            "units": "cycles",
            "results": [{"costs": [17]}],
        },
    )
    assert runtime._schedule_measurement_cycles(selected, layer) == 17

    selected.measurement["protocol"] = "legacy"
    with pytest.raises(ValueError, match="one-call TSIM"):
        runtime._schedule_measurement_cycles(selected, layer)


def test_profiler_adapter_preserves_ad_session_contract():
    runtime = _load_runtime()
    calls = []
    runtime_session = type("RuntimeSession", (), {
        "clear_and_validate": lambda self, simulator: calls.append(("clear", simulator)),
        "read_stats": lambda self, status: status(),
        "validate_activity": lambda self, stats: calls.append(("validate", stats)),
    })()
    adapter = runtime._EvidenceProfileSession(runtime_session)
    simulator = type("Simulator", (), {"stats": lambda self: {"cycle_count": 5}})()

    adapter.clear_and_validate(simulator)
    stats = adapter.read_stats(simulator)
    adapter.validate_activity(stats)

    assert stats == {"cycle_count": 5}
    assert calls == [("clear", simulator), ("validate", stats)]
