"""Strict cycle-gate and occurrence coverage tests for selected deployment."""

import importlib.util
import sys
from pathlib import Path

import pytest


TUNE_DIR = Path(__file__).resolve().parents[1] / "tune"


def _load_deployment():
    path = TUNE_DIR / "deployment.py"
    spec = importlib.util.spec_from_file_location("ic_v2_deployment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def deployment():
    return _load_deployment()


@pytest.mark.parametrize(
    "deployed,best,expected",
    [(109, 100, True), (91, 100, True), (110, 100, False), (90, 100, False),
     (111, 100, False), (89, 100, False), (10**30 + 1, 10**30, True)],
)
def test_strict_cycle_gate_uses_exact_integer_arithmetic(deployment, deployed, best, expected):
    assert deployment.cycles_within_strict_ten_percent(deployed, best) is expected


@pytest.mark.parametrize("deployed,best", [(0, 100), (-1, 100), (100, 0), (100, -1), (True, 100)])
def test_cycle_gate_rejects_nonpositive_or_noninteger_counts(deployment, deployed, best):
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
