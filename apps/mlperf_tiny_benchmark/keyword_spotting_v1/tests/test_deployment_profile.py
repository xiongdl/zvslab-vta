"""KWS V1 unified schedule deployment evidence checks."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = APP_ROOT / "runtime.py"


def _load_runtime():
    spec = importlib.util.spec_from_file_location("kws_v1_profile_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def test_cycle_gate_uses_inclusive_ten_percent_boundary():
    runtime = _load_runtime()

    assert runtime.cycles_within_ten_percent(110, 100)
    assert runtime.cycles_within_ten_percent(90, 100)
    assert not runtime.cycles_within_ten_percent(111, 100)
    with pytest.raises(ValueError, match="positive integer"):
        runtime.cycles_within_ten_percent(True, 100)


def test_partial_schedule_coverage_reports_default_occurrences():
    runtime = _load_runtime()
    artifacts = SimpleNamespace(schedule_coverage=((0, "vta_0", True), (1, "vta_1", False)))

    assert runtime.schedule_coverage_rows(artifacts) == [
        {"occurrence": 0, "symbol": "vta_0", "selected": True},
        {"occurrence": 1, "symbol": "vta_1", "selected": False},
    ]


def test_schedule_evidence_requires_tsim_and_complete_measured_coverage():
    runtime = _load_runtime()
    deployment = SimpleNamespace(layers=(SimpleNamespace(occurrence=0),))
    snapshot = SimpleNamespace(selected={})

    with pytest.raises(ValueError, match="complete occurrence coverage"):
        runtime.validate_schedule_evidence_coverage(deployment, snapshot)


def test_schedule_evidence_rejects_unmeasured_selected_occurrence():
    runtime = _load_runtime()
    layer = SimpleNamespace(occurrence=0)
    deployment = SimpleNamespace(layers=(layer,))
    snapshot = SimpleNamespace(selected={0: SimpleNamespace(measured=False)})

    with pytest.raises(ValueError, match="measured config for occurrence 0"):
        runtime.validate_schedule_evidence_coverage(deployment, snapshot)


def test_schedule_evidence_accepts_complete_measured_occurrences():
    runtime = _load_runtime()
    deployment = SimpleNamespace(layers=(SimpleNamespace(occurrence=0),))
    snapshot = SimpleNamespace(selected={0: SimpleNamespace(measured=True)})

    assert runtime.validate_schedule_evidence_coverage(deployment, snapshot) is None
