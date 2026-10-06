"""Selected deployment, zero-coverage fallback, and Make contracts."""

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_deploy():
    spec = importlib.util.spec_from_file_location(
        "keyword_spotting_v1_deploy_cli", APP_ROOT / "deploy.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_and_selected_target_surface():
    deploy = _load_deploy()
    args = deploy._parser().parse_args([])
    assert args.model == APP_ROOT / "model" / "kws_ref_model_float32.tflite"
    assert args.input == APP_ROOT / "samples" / "down-00176480_nohash_0.wav"
    assert args.target == "vta,llvm"
    assert args.simulator == "fsim"
    for target in ("c", "llvm", "vta,c", "vta,llvm"):
        assert deploy._parser().parse_args(["--target", target]).target == target
    with pytest.raises(SystemExit):
        deploy._parser().parse_args(["--host-codegen", "all"])


def test_cpu_cli_rejects_export_before_importing_runtime(monkeypatch):
    deploy = _load_deploy()
    import types
    fake_python = types.ModuleType("python")
    fake_python.deployment = None
    monkeypatch.setitem(sys.modules, "python", fake_python)
    with pytest.raises(ValueError, match="requires a target that includes VTA"):
        deploy.main(["--target", "llvm", "--export-workloads", "out.json"])


def test_make_clean_is_idempotent_and_preserves_tune_assets(tmp_path):
    app = tmp_path / "app"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(APP_ROOT / "Makefile", app / "Makefile")
    shutil.copy2(APP_ROOT / "scripts/make_tasks.sh", app / "scripts/make_tasks.sh")
    (app / "build").mkdir()
    (app / "build/tmp.txt").write_text("generated")
    persistent = app / "tune/vta_64mac/best.log"
    persistent.parent.mkdir(parents=True)
    persistent.write_text("schedule")
    for _ in range(2):
        result = subprocess.run(["make", "clean"], cwd=app, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert not (app / "build").exists()
        assert persistent.read_text() == "schedule"


def test_cpu_runtime_import_has_no_vta_or_shared_application_dependency():
    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env.pop("VTA_CONFIG_FILE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(APP_ROOT), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    )
    check = "import sys; from python import deployment; assert not any(x == 'vta' or x.startswith('vta.') for x in sys.modules)"
    result = subprocess.run([sys.executable, "-c", check], cwd=APP_ROOT, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (APP_ROOT / "runtime.py").exists()
    assert not (APP_ROOT / "model_pipeline.py").exists()
    assert not (APP_ROOT / "graph_artifacts.py").exists()


def test_deployment_report_states_zero_coverage_and_unavailable_cycles(tmp_path, monkeypatch):
    package_name = "keyword_spotting_v1_test_app"
    if package_name not in sys.modules:
        package_path = APP_ROOT / "python" / "__init__.py"
        package_spec = importlib.util.spec_from_file_location(
            package_name, package_path,
            submodule_search_locations=[str(package_path.parent)],
        )
        package = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = package
        package_spec.loader.exec_module(package)
    deployment = importlib.import_module(f"{package_name}.deployment")
    config = APP_ROOT.parents[3] / "vta/config/vta_64mac.json"
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config))
    report = tmp_path / "deployment.md"
    result = SimpleNamespace(
        target="vta,llvm", simulator=None, model_path=APP_ROOT / "model/kws_ref_model_float32.tflite",
        model_sha256="a" * 64, input_path=APP_ROOT / "samples/down-00176480_nohash_0.wav",
        input_sha256="b" * 64, schedule=None, schedule_coverage=(), predicted_class=0,
        scores=np.array([[0.1] * 12], dtype=np.float32), layers=(), whole_cycles=None,
        profiler_stats=None, fallback_reason="no real VTA partitions; executed on CPU",
    )
    deployment.write_deployment_report(result, report)
    rendered = report.read_text(encoding="utf-8")
    assert "VTA partition coverage: 0 real occurrences" in rendered
    assert "Schedule coverage: 0/0" in rendered
    assert "Whole-model cycles: N/A (CPU target)" in rendered
    assert "VTA fallback: no real VTA partitions" in rendered
