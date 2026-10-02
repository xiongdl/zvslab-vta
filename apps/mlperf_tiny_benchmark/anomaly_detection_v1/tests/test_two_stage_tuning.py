"""AD V1 adapter contracts for common actual-compute tuning."""

import importlib.util
import ast
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
TUNE_PATH = APP_ROOT / "tune.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("ad_v1_actual_tune", TUNE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_one_seed_search_resume_and_snapshot_interface():
    tune = _load_tune()
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


def test_seed_mode_requires_all_and_tsim(monkeypatch):
    tune = _load_tune()
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    with pytest.raises(ValueError, match="requires VTA_BACKEND=tsim"):
        tune.create_seed_snapshot()
    with pytest.raises(ValueError, match="requires --seed --all"):
        tune.main(["--seed", "--workload-index", "0"])


def test_search_requires_a_passing_unified_seed_report(monkeypatch):
    tune = _load_tune()
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    with pytest.raises(ValueError, match="requires --alignment-report"):
        tune.main(["--all"])


def test_export_modes_are_exclusive_and_require_resume_inputs():
    tune = _load_tune()
    with pytest.raises(ValueError, match="exactly one"):
        tune.main([
            "--export-candidate", "0", "--export-best", "--workload-index", "0",
            "--resume-manifest", "resume.json", "--output-log", "out.log",
        ])
    with pytest.raises(ValueError, match="requires --resume-manifest and --output-log"):
        tune.main(["--export-best"])


def test_candidate_export_requires_occurrence_and_rejects_negative_index():
    tune = _load_tune()
    with pytest.raises(ValueError, match="requires --workload-index"):
        tune.main([
            "--export-candidate", "0", "--resume-manifest", "resume.json",
            "--output-log", "candidate.log",
        ])
    with pytest.raises(ValueError, match="non-negative"):
        tune.main([
            "--export-candidate", "-1", "--workload-index", "0",
            "--resume-manifest", "resume.json", "--output-log", "candidate.log",
        ])


def test_legacy_deployment_and_tuning_entries_are_retired():
    assert not (APP_ROOT / "tune" / "tune.py").exists()
    assert not (APP_ROOT / "tune" / "deployment.py").exists()
    source = TUNE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "fused_tasks" not in imported
    assert "autotvm_tuner" not in imported
    options = {option for action in _load_tune()._parser()._actions for option in action.option_strings}
    assert "--autotvm-log" not in options
    assert "--autotvm-sidecar" not in options
