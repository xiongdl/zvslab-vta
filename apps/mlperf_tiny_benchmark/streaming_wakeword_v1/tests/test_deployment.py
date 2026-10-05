"""Selected deployment and truthful zero-coverage behavior."""

from pathlib import Path
import sys

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[3]
MODEL = APP_ROOT / "model/str_ww_ref_model.tflite"
SAMPLE = APP_ROOT / "samples/marvin-00176480_nohash_0.wav"


def _deployment():
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    from python import deployment

    return deployment


@pytest.mark.parametrize("target", ["vta,c", "vta,llvm"])
def test_zero_coverage_runs_cpu_fallback_without_loading_simulator(
    target, tmp_path, monkeypatch
):
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(REPO_ROOT / "vta/config/vta_64mac.json"))
    deployment = _deployment()
    monkeypatch.setattr(
        deployment, "_load_simulator",
        lambda _simulator: pytest.fail("zero-coverage fallback must not load a simulator"),
    )
    host = target.split(",")[1]
    result = deployment.run_selected(
        target=target, simulator="fsim", model_path=MODEL, input_path=SAMPLE,
        output_dir=tmp_path / f"{host}-bundle",
    )

    assert result.target == target
    assert result.execution_target == host
    assert result.simulator is None
    assert result.fallback_reason and "no real VTA partitions" in result.fallback_reason
    assert result.schedule_coverage == ()
    assert result.whole_cycles is None
    assert result.profiler_stats is None
    assert result.scores.dtype == np.int8
    assert result.scores.shape == (1, 3)
    np.testing.assert_array_equal(result.scores, np.array([[127, -128, -128]], dtype=np.int8))
    assert any(layer.device == "cpu" for layer in result.layers)
    assert not any(layer.device == "vta" for layer in result.layers)
    assert any(
        layer.operation == "nn.dense" and layer.logical_macs > 0
        for layer in result.layers
    )
    assert result.predicted_class in (0, 1, 2)

    report = deployment.write_deployment_report(result, tmp_path / f"{host}.md")
    text = report.read_text(encoding="utf-8")
    assert f"Execution target: `{host}`" in text
    assert "VTA coverage: 0 real VTA layers" in text
    assert "Whole-model cycles: N/A (CPU target)." in text
    assert any(line.startswith("| cpu.dense") and "nn.dense" in line for line in text.splitlines())


@pytest.mark.parametrize("option", ["export", "schedule"])
def test_zero_coverage_rejects_tuning_artifacts_before_publication(
    option, tmp_path, monkeypatch
):
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(REPO_ROOT / "vta/config/vta_64mac.json"))
    deployment = _deployment()
    output_dir = tmp_path / "must-not-exist"
    kwargs = {"export_workloads": tmp_path / "workloads.json"} if option == "export" else {
        "schedule": tmp_path / "missing.log"
    }
    with pytest.raises(ValueError, match="no real VTA workloads"):
        deployment.run_selected(
            target="vta,llvm", simulator="fsim", model_path=MODEL,
            input_path=SAMPLE, output_dir=output_dir, **kwargs,
        )
    assert not output_dir.exists()
    assert not (tmp_path / "workloads.json").exists()


def test_cpu_cli_does_not_require_vta_environment():
    import os
    import subprocess

    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env.pop("VTA_CONFIG_FILE", None)
    env["PYTHONPATH"] = ":".join(
        (str(REPO_ROOT / "tvm/python"), str(REPO_ROOT / "vta/python"), str(APP_ROOT))
    )
    command = [
        sys.executable, str(APP_ROOT / "deploy.py"),
        "--target", "c", "--model", str(MODEL), "--input", str(SAMPLE),
        "--output-dir", str(APP_ROOT / "build" / "test-cpu-cli"),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "Wakeword result:" in completed.stdout


def test_zero_coverage_cli_reports_export_rejection_without_artifact(tmp_path):
    import os
    import subprocess

    env = os.environ.copy()
    env["VTA_BACKEND"] = "fsim"
    env["VTA_CONFIG_FILE"] = str(REPO_ROOT / "vta/config/vta_64mac.json")
    env["PYTHONPATH"] = ":".join(
        (str(REPO_ROOT / "tvm/python"), str(REPO_ROOT / "vta/python"), str(APP_ROOT))
    )
    export_path = tmp_path / "workloads.json"
    completed = subprocess.run(
        [
            sys.executable, str(APP_ROOT / "deploy.py"), "--target", "vta,llvm",
            "--simulator", "fsim", "--model", str(MODEL), "--input", str(SAMPLE),
            "--export-workloads", str(export_path), "--output-dir", str(tmp_path / "bundle"),
        ], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "deploy.py: no real VTA workloads" in completed.stderr
    assert not export_path.exists()
    assert not (tmp_path / "bundle").exists()
