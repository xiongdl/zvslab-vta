"""Validation tests for versioned IC V1 deployment evidence."""

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tune" / "deployment.py"
SPEC = importlib.util.spec_from_file_location("ic_v1_deployment_profile_test", MODULE_PATH)
deployment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = deployment
SPEC.loader.exec_module(deployment)


def test_occurrence_config_map_preserves_different_configs_for_same_conv_shape():
    identities = [
        {"occurrence": 0, "symbol": "fusion_0"},
        {"occurrence": 1, "symbol": "fusion_1"},
    ]
    entries = [
        {"workload_index": 0, "occurrence": 0, "symbol": "fusion_0", "config": {"tile": 1}},
        {"workload_index": 1, "occurrence": 1, "symbol": "fusion_1", "config": {"tile": 2}},
    ]

    assert deployment.build_occurrence_config_map(identities, entries) == {
        0: {"tile": 1}, 1: {"tile": 2}
    }


def test_occurrence_config_map_rejects_missing_or_mismatched_occurrences():
    identities = [
        {"occurrence": 0, "symbol": "fusion_0"},
        {"occurrence": 1, "symbol": "fusion_1"},
    ]
    entries = [
        {"workload_index": 0, "occurrence": 0, "symbol": "fusion_0", "config": {"tile": 1}},
        {"workload_index": 1, "occurrence": 9, "symbol": "fusion_1", "config": {"tile": 1}},
    ]

    with pytest.raises(ValueError, match="selected schedule identity mismatch"):
        deployment.build_occurrence_config_map(identities, entries)


def test_cycle_comparison_uses_autotvm_cycles_and_rejects_over_ten_percent():
    assert deployment.compare_cycles(110, 100)["relative_cycle_difference"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="exceeds 10%"):
        deployment.compare_cycles(111, 100)


def test_report_validation_rejects_duplicate_occurrence_and_unaligned_counts():
    report = deployment.example_valid_report()
    report["occurrences"].append(dict(report["occurrences"][0]))
    with pytest.raises(ValueError, match="occurrence identity must be unique"):
        deployment.validate_deployment_report(report)

    report = deployment.example_valid_report()
    report["measurement_protocol"]["full_model_counted_invocations"] = 2
    report["full_model"]["invocation_count"] = 1
    with pytest.raises(ValueError, match="invocation counts must align"):
        deployment.validate_deployment_report(report)
