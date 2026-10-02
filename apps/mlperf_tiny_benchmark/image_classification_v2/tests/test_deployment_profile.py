"""Strict per-occurrence deployment evidence gates."""

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def deployment():
    return _load("common_deployment_evidence", ROOT.parents[1] / "common" / "deployment.py")


@pytest.mark.parametrize(
    "deployed,best,expected",
    [(109, 100, True), (91, 100, True), (110, 100, False), (90, 100, False),
     (111, 100, False), (89, 100, False), (10**30 + 1, 10**30, True)],
)
def test_strict_cycle_gate_uses_exact_integer_arithmetic(deployment, deployed, best, expected):
    assert deployment.cycles_within_strict_ten_percent(deployed, best) is expected


@pytest.mark.parametrize("deployed,best", [(0, 100), (-1, 100), (100, 0), (100, -1), (True, 100)])
def test_strict_cycle_gate_rejects_invalid_counts(deployment, deployed, best):
    with pytest.raises(ValueError, match="positive integer"):
        deployment.cycles_within_strict_ten_percent(deployed, best)


def test_occurrence_gate_requires_exact_unique_coverage(deployment):
    expected = [{"occurrence": i, "symbol": f"vta_{i}"} for i in range(2)]
    valid = [dict(row, deployment_cycles=95, autotvm_cycles=100) for row in expected]
    assert deployment.validate_occurrence_rows(valid, expected) == valid
    with pytest.raises(ValueError, match="coverage"):
        deployment.validate_occurrence_rows(valid[:1], expected)
    with pytest.raises(ValueError, match="duplicate"):
        deployment.validate_occurrence_rows([valid[0], valid[0]], expected)
    failed = [dict(row, deployment_cycles=110, autotvm_cycles=100) for row in expected]
    with pytest.raises(ValueError, match="strict <10%"):
        deployment.validate_occurrence_rows(failed, expected)


def test_run_cli_accepts_shared_report_and_alignment_options():
    run = _load("ic_v2_run_profile", ROOT / "run.py")
    parsed = run._parser().parse_args([
        "--simulator", "tsim", "--schedule", "selected.log",
        "--deployment-report", "evidence.json", "--validate-schedule-evidence",
    ])
    assert parsed.schedule == Path("selected.log")
    assert parsed.deployment_report == Path("evidence.json")
    assert parsed.validate_schedule_evidence is True
