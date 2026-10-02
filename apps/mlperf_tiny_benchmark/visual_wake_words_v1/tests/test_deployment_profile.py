"""VWW unified runtime schedule and deployment evidence checks."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_runtime():
    spec = importlib.util.spec_from_file_location("vww_unified_runtime_profile", APP_ROOT / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_cycle_gate_uses_inclusive_ten_percent():
    runtime = _load_runtime()

    assert runtime.cycles_within_ten_percent(110, 100) is True
    assert runtime.cycles_within_ten_percent(90, 100) is True
    assert runtime.cycles_within_ten_percent(111, 100) is False
    assert runtime.cycles_within_ten_percent(89, 100) is False
    with pytest.raises(ValueError, match="positive integer"):
        runtime.cycles_within_ten_percent(True, 100)


def test_runtime_reports_each_occurrence_snapshot_or_default():
    runtime = _load_runtime()
    artifacts = SimpleNamespace(schedule_coverage=(
        (0, "vta_a", True), (1, "vta_b", False),
    ))

    assert runtime.schedule_coverage_rows(artifacts) == [
        {"occurrence": 0, "symbol": "vta_a", "selected": True},
        {"occurrence": 1, "symbol": "vta_b", "selected": False},
    ]


def test_schedule_evidence_requires_measured_complete_coverage():
    runtime = _load_runtime()
    layers = (SimpleNamespace(occurrence=0), SimpleNamespace(occurrence=1))
    deployment = SimpleNamespace(layers=layers)
    selected = {
        0: SimpleNamespace(measured=True),
        1: SimpleNamespace(measured=True),
    }
    runtime._validate_schedule_evidence(deployment, SimpleNamespace(selected=selected))

    with pytest.raises(ValueError, match="complete occurrence coverage"):
        runtime._validate_schedule_evidence(deployment, SimpleNamespace(selected={0: selected[0]}))
    selected[1] = SimpleNamespace(measured=False)
    with pytest.raises(ValueError, match="measured config"):
        runtime._validate_schedule_evidence(deployment, SimpleNamespace(selected=selected))


def test_run_cli_has_one_schedule_and_report_evidence_interface():
    spec = importlib.util.spec_from_file_location("vww_unified_run_cli", APP_ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    args = module._parser().parse_args([
        "--schedule", "none", "--deployment-report", "report.json",
        "--validate-schedule-evidence",
    ])
    assert args.schedule == "none"
    assert args.deployment_report == Path("report.json")
    assert args.validate_schedule_evidence is True
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--autotvm-log", "old.log"])


def test_runtime_module_does_not_depend_on_second_deployment_command():
    assert not (APP_ROOT / "tune" / "deployment.py").exists()
