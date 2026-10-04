"""Tests for the single-target deployment command and report contract."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(name, APP_ROOT / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_run_parser_accepts_only_selected_targets_and_single_image_options():
    runner = _load("run.py", "ic_v1_selected_run_contract")
    args = runner._parser().parse_args([])
    assert args.target == "vta,llvm"
    assert args.simulator == "fsim"
    assert args.model == runner.DEFAULT_MODEL
    assert args.input == runner.DEFAULT_INPUT
    assert args.output_dir == runner.DEFAULT_OUTPUT_DIR
    assert runner._parser().parse_args(["--target", "c"]).target == "c"
    assert runner._parser().parse_args(["--target", "llvm"]).target == "llvm"
    assert runner._parser().parse_args(["--target", "vta,c"]).target == "vta,c"
    for retired in ("--host-codegen", "--validate-schedule-evidence"):
        with pytest.raises(SystemExit):
            runner._parser().parse_args([retired, "all"] if retired == "--host-codegen" else [retired])
    assert runner._parser().parse_args(["--export-workloads", "build/workloads.json"]).export_workloads.name == "workloads.json"


def test_cpu_run_does_not_import_vta_or_require_backend(monkeypatch, tmp_path):
    runner = _load("run.py", "ic_v1_cpu_run_contract")
    monkeypatch.delenv("VTA_BACKEND", raising=False)
    loaded_vta_modules = {name for name in sys.modules if name == "vta" or name.startswith("vta.")}
    calls = []
    fake = SimpleNamespace(
        DEFAULT_OUTPUT_DIR=tmp_path,
        run_selected=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
            target="llvm", output=np.array([[0.1, 0.9]] + [[0.0] * 2] * 4, dtype="float32"),
            cycles=None, layers=(), profiler_stats=None,
        ),
        write_deployment_report=lambda *args, **kwargs: "report",
    )
    monkeypatch.setitem(sys.modules, "runtime", fake)
    runner.main(["--target", "llvm"])
    assert calls[0]["target"] == "llvm"
    assert {name for name in sys.modules if name == "vta" or name.startswith("vta.")} == loaded_vta_modules


def test_cpu_rejects_workload_export_before_runtime_import(monkeypatch):
    runner = _load("run.py", "ic_v1_cpu_export_rejected")
    monkeypatch.setitem(sys.modules, "runtime", None)
    with pytest.raises(ValueError, match="requires a target that includes VTA"):
        runner.main(["--target", "c", "--export-workloads", "out.json"])


def test_importing_cpu_runtime_does_not_load_vta_backend():
    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(APP_ROOT), str(APP_ROOT.parents[2] / "python"), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    )
    code = "import sys, runtime; assert not any(name == 'vta' or name.startswith('vta.') for name in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=APP_ROOT, env=env, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_markdown_report_uses_n_a_for_cpu_and_fsim_cycles(tmp_path):
    runtime = _load("runtime.py", "ic_v1_markdown_report_runtime")
    result = SimpleNamespace(
        target="vta,c", simulator="fsim", model_path=Path("model.tflite"),
        input_path=Path("image.png"), model_sha256="a" * 64, input_sha256="b" * 64,
        schedule=None, schedule_coverage=(), predicted_class=2,
        scores=np.array([[0.1, 0.2, 0.7]], dtype="float32"),
        layers=(
            runtime.LayerMetrics("conv0", "vta", "nn.conv2d", 1024, None, 16384, None),
            runtime.LayerMetrics("conv1", "cpu", "nn.conv2d", 2048, None, None, None),
        ),
        whole_cycles=None, profiler_stats=None,
    )
    report_path = tmp_path / "report.md"
    runtime.write_deployment_report(result, report_path)
    report = report_path.read_text(encoding="utf-8")
    assert "| conv0 | vta | nn.conv2d | 1,024 | N/A | 16,384 | N/A |" in report
    assert "| conv1 | cpu | nn.conv2d | 2,048 | N/A | N/A | N/A |" in report
    assert "unavailable" in report.lower()
