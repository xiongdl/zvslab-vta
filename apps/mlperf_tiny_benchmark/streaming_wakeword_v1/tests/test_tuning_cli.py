"""Tuning command contracts and zero-workload guards."""

import importlib.util
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("streaming_wakeword_tune_cli", APP_ROOT / "tune.py")
tune = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tune)


def test_fsim_and_tsim_cli_defaults_match_the_standalone_contract():
    parser = tune._parser()
    fsim = tune.validate_args(parser.parse_args([
        "--workloads", "build/workloads.json", "--output-logs", "tune/fsim.tmp"
    ]))
    assert fsim.workload == -1
    assert fsim.simulator == "fsim"
    assert fsim.timeout == 60
    assert fsim.trial_batch == 100
    assert fsim.min_successful == 20

    tsim = tune.validate_args(parser.parse_args([
        "--workloads", "build/workloads.json", "--simulator", "tsim",
        "--input-logs", "tune/fsim.tmp", "--output-logs", "tune/best.log",
    ]))
    assert tsim.timeout == 120
    assert tsim.input_logs == Path("tune/fsim.tmp")
    assert tsim.trial_batch is None
    assert tsim.min_successful is None


def test_tsim_requires_fsim_logs_and_rejects_fsim_search_options():
    parser = tune._parser()
    missing = parser.parse_args([
        "--workloads", "workloads.json", "--simulator", "tsim", "--output-logs", "best.log"
    ])
    with pytest.raises(ValueError, match="TSIM requires --input-logs"):
        tune.validate_args(missing)

    invalid = parser.parse_args([
        "--workloads", "workloads.json", "--simulator", "tsim",
        "--input-logs", "fsim.tmp", "--output-logs", "best.log", "--trial-batch", "1",
    ])
    with pytest.raises(ValueError, match="only valid for FSIM"):
        tune.validate_args(invalid)


def test_fsim_rejects_invalid_occurrence_and_empty_search_quota():
    parser = tune._parser()
    invalid_index = parser.parse_args([
        "--workloads", "workloads.json", "--workload", "-2", "--output-logs", "out.log"
    ])
    with pytest.raises(ValueError, match="-1 or a non-negative"):
        tune.validate_args(invalid_index)
    invalid_quota = parser.parse_args([
        "--workloads", "workloads.json", "--output-logs", "out.log", "--min-successful", "0"
    ])
    with pytest.raises(ValueError, match="must be positive"):
        tune.validate_args(invalid_quota)
