"""Float32 CPU and mixed deployment behavior."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[3]
MODEL = APP_ROOT / "model/str_ww_ref_model_floag32.tflite"
SAMPLE = APP_ROOT / "samples/marvin-00176480_nohash_0.wav"


def _deployment():
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    from python import deployment

    return deployment


def test_cpu_and_fsim_mixed_execution_match_and_export_real_workloads(tmp_path, monkeypatch):
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(REPO_ROOT / "vta/config/vta_64mac.json"))
    deployment = _deployment()
    cpu = deployment.run_selected(
        target="llvm", simulator="fsim", model_path=MODEL, input_path=SAMPLE,
        output_dir=tmp_path / "cpu",
    )
    workload_path = tmp_path / "workloads.json"
    mixed = deployment.run_selected(
        target="vta,llvm", simulator="fsim", model_path=MODEL, input_path=SAMPLE,
        output_dir=tmp_path / "mixed", export_workloads=workload_path,
    )

    assert cpu.scores.dtype == mixed.scores.dtype == np.float32
    assert cpu.scores.shape == mixed.scores.shape == (1, 3)
    np.testing.assert_allclose(mixed.scores, cpu.scores, rtol=0, atol=0)
    assert mixed.fallback_reason is None
    assert mixed.simulator == "fsim"
    assert mixed.profiler_stats
    assert sum(layer.device == "vta" for layer in mixed.layers) >= 4
    assert any(layer.device == "cpu" for layer in mixed.layers)

    snapshot = json.loads(workload_path.read_text(encoding="utf-8"))
    assert snapshot["model"]["sha256"] == "c735ab47248df7648d9cb4397c0e7d161fe2e88ede17ad900f34a4163d89b267"
    assert snapshot["input"]["decoded_dtype"] == "float32"
    assert snapshot["model"]["quantization"] == {
        "policy": "Relay global_scale quantization",
        "global_scale": 8.0,
        "skip_conv_layers": [0],
    }
    assert len(snapshot["workloads"]) == 4
    assert all(item["activation"]["dtype"] == "|i1" for item in snapshot["workloads"])


def test_cpu_cli_does_not_require_vta_environment(tmp_path):
    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env.pop("VTA_CONFIG_FILE", None)
    env["PYTHONPATH"] = ":".join(
        (str(REPO_ROOT / "tvm/python"), str(REPO_ROOT / "vta/python"), str(APP_ROOT))
    )
    command = [
        sys.executable, str(APP_ROOT / "deploy.py"),
        "--target", "c", "--model", str(MODEL), "--input", str(SAMPLE),
        "--output-dir", str(tmp_path / "cpu-cli"),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "Wakeword result:" in completed.stdout
    assert "Raw float32 scores:" in completed.stdout
