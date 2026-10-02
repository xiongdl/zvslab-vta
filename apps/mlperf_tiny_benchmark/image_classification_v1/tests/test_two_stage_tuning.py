"""CLI and bounded actual-compute tuning contracts for IC V1."""

import importlib.util
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
TUNE_PATH = APP_ROOT / "tune.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("ic_v1_actual_tune", TUNE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tune_cli_exposes_approved_seed_search_and_resume_controls():
    tune = _load_tune()
    assert not (APP_ROOT / "tune" / "tune.py").exists()
    args = tune._parser().parse_args([
        "--all", "--alignment-report", "seed-report.json", "--trial-batch", "2",
        "--min-successful", "1", "--max-workloads", "1", "--fsim-timeout", "7",
        "--tsim-timeout", "9", "--resume-manifest", "resume.json",
    ])
    assert args.all is True
    assert args.alignment_report == Path("seed-report.json")
    assert args.trial_batch == 2
    assert args.min_successful == 1
    assert args.max_workloads == 1
    assert args.fsim_timeout == 7
    assert args.tsim_timeout == 9
    assert args.resume_manifest == Path("resume.json")

    candidate = tune._parser().parse_args([
        "--export-candidate", "3", "--workload-index", "2",
        "--resume-manifest", "resume.json", "--output-log", "candidate.log",
    ])
    assert candidate.export_candidate == 3
    assert candidate.workload_index == 2
    assert candidate.output_log == Path("candidate.log")

    best = tune._parser().parse_args([
        "--export-best", "--resume-manifest", "resume.json", "--output-log", "best.log",
    ])
    assert best.export_best is True
    assert best.output_log == Path("best.log")


def test_seed_mode_requires_complete_all_occurrences_and_tsim(monkeypatch):
    tune = _load_tune()
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    with pytest.raises(ValueError, match="requires VTA_BACKEND=tsim"):
        tune.create_seed_snapshot()
    with pytest.raises(ValueError, match="requires --seed --all"):
        tune.main(["--seed", "--workload-index", "0"])


def test_search_options_reject_zero_quotas_and_timeouts():
    tune = _load_tune()
    args = tune._parser().parse_args(["--all", "--trial-batch", "0"])
    with pytest.raises(ValueError, match="trial-batch"):
        tune._validate_positive_options(args)


def test_tune_cli_rejects_conflicting_export_modes(monkeypatch):
    tune = _load_tune()
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    with pytest.raises(ValueError, match="exactly one"):
        tune.main([
            "--export-candidate", "0", "--export-best", "--workload-index", "0",
            "--resume-manifest", "resume.json", "--output-log", "out.log",
        ])


def test_resume_manifest_rejects_foreign_compute(tmp_path):
    from common.tuning import create_ledger, load_ledger, save_ledger

    identity = {
        "model_id": "image_classification_v1", "model_sha256": "m" * 64,
        "geometry_sha256": "g" * 64, "compute_sha256": "c" * 64,
        "occurrence": 0, "symbol": "vta_0", "options": {"trial_batch": 1},
    }
    path = tmp_path / "ledger.json"
    ledger = create_ledger(identity, path)
    save_ledger(ledger)
    foreign = {**identity, "compute_sha256": "x" * 64}
    with pytest.raises(ValueError, match="identity does not match"):
        load_ledger(path, foreign)
