"""Tests for the unified IC V1 deployment report contract."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_runtime():
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    spec = importlib.util.spec_from_file_location("ic_v1_deployment_profile_runtime", APP_ROOT / "runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_report_records_v1_samples_outputs_profiler_and_default_coverage(tmp_path):
    runtime = _load_runtime()
    output = SimpleNamespace(sample_path=Path("00-airplane.png"))
    result = SimpleNamespace(
        prepared=SimpleNamespace(imported=SimpleNamespace(model_sha256="model-hash")),
        artifacts=SimpleNamespace(
            simulator="fsim", host_codegen="llvm",
            schedule_coverage=((0, "fusion_0", False), (1, "fusion_1", True)),
            schedule_config_identities=((1, "selected-config-sha"),),
        ),
        execution=SimpleNamespace(comparisons=(output,) * 10, profiler_stats={"gemm_counter": 1}),
    )

    report = runtime.write_deployment_report(result, tmp_path / "report.json", schedule="candidate.log")

    assert report["model"] == "image_classification_v1"
    assert report["sample_count"] == report["outputs_passed"] == 10
    assert report["schedule_coverage"] == [
        {"occurrence": 0, "symbol": "fusion_0", "selected": False},
        {"occurrence": 1, "symbol": "fusion_1", "selected": True},
    ]
    assert report["profiler_stats"] == {"gemm_counter": 1}
    assert report["selected_config_identities"] == [
        {"occurrence": 1, "sha256": "selected-config-sha"}
    ]
    assert (tmp_path / "report.json").is_file()


def test_run_parser_exposes_one_schedule_option_and_rejects_old_log_pair():
    runtime = _load_runtime()
    sys.modules["runtime"] = runtime
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("ic_v1_schedule_run", APP_ROOT / "run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.pop(0)
    args = runner._parser().parse_args(["--schedule", "none"])
    assert args.schedule == "none"
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["--autotvm-log", "candidate.log"])


def test_schedule_evidence_requires_tsim_and_a_report_path():
    runtime = _load_runtime()
    sys.modules["runtime"] = runtime
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec = importlib.util.spec_from_file_location("ic_v1_run_profile", APP_ROOT / "run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.pop(0)
    with pytest.raises(ValueError, match="requires --deployment-report"):
        runner.main(["--validate-schedule-evidence"])


def test_report_rejects_schedule_evidence_for_fsim(tmp_path):
    runtime = _load_runtime()
    result = SimpleNamespace(artifacts=SimpleNamespace(simulator="fsim"))
    with pytest.raises(ValueError, match="requires --simulator tsim"):
        runtime.write_deployment_report(
            result, tmp_path / "report.json", schedule="candidate.log",
            validate_schedule_evidence=True,
        )
