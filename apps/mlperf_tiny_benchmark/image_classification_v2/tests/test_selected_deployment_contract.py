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
        import importlib
        module_name = "python.deployment" if path in ("python/deployment.py",) else "deploy"
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)



def test_run_parser_accepts_only_selected_targets_and_single_image_options():
    runner = _load("deploy.py", "ic_v1_selected_run_contract")
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
    runner = _load("deploy.py", "ic_v1_cpu_run_contract")
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
    monkeypatch.syspath_prepend(str(APP_ROOT))
    import python
    monkeypatch.setattr(python, "deployment", fake, raising=False)
    runner.main(["--target", "llvm"])
    assert calls[0]["target"] == "llvm"
    assert {name for name in sys.modules if name == "vta" or name.startswith("vta.")} == loaded_vta_modules


def test_cpu_rejects_workload_export_before_runtime_import(monkeypatch):
    runner = _load("deploy.py", "ic_v1_cpu_export_rejected")
    monkeypatch.syspath_prepend(str(APP_ROOT))
    import python
    monkeypatch.setattr(python, "deployment", None, raising=False)
    with pytest.raises(ValueError, match="requires a target that includes VTA"):
        runner.main(["--target", "c", "--export-workloads", "out.json"])


def test_importing_cpu_runtime_does_not_load_vta_backend():
    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(APP_ROOT), str(APP_ROOT.parents[2] / "python"), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    )
    code = "import sys; from python import deployment; assert not any(name == 'vta' or name.startswith('vta.') for name in sys.modules)"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=APP_ROOT, env=env, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_cpu_deployment_accepts_supported_model_with_different_input_tensor_name(
    tmp_path, monkeypatch
):
    runtime = _load("python/deployment.py", "ic_v2_renamed_input_runtime")
    model_path = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"
    renamed_model = tmp_path / "renamed-input.tflite"
    renamed_model.write_bytes(
        model_path.read_bytes().replace(
            b"serving_default_input_5:0", b"testing_default_input_5:0", 1
        )
    )
    monkeypatch.delenv("VTA_BACKEND", raising=False)
    monkeypatch.delenv("VTA_CONFIG_FILE", raising=False)

    result = runtime.run_selected(
        target="llvm", model_path=renamed_model,
        input_path=APP_ROOT / "samples" / "00-airplane.png",
        output_dir=tmp_path / "bundle",
    )

    assert result.scores.shape == (1, 10)
    assert 0 <= result.predicted_class < 10


def test_markdown_report_uses_n_a_for_cpu_and_fsim_cycles(tmp_path, monkeypatch):
    runtime = _load("python/deployment.py", "ic_v1_markdown_report_runtime")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(APP_ROOT.parents[2] / "config" / "vta_64mac.json"))
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


@pytest.mark.parametrize("target", ["vta,c", "vta,llvm"])
@pytest.mark.parametrize("simulator", ["fsim", "tsim"])
def test_zero_coverage_vta_request_falls_back_to_cpu_without_simulator(
    tmp_path, monkeypatch, target, simulator
):
    runtime = _load("python/deployment.py", "ic_v2_zero_coverage_runtime")
    model_path = tmp_path / "model.tflite"
    input_path = tmp_path / "input.png"
    model_path.write_bytes(b"model")
    input_path.write_bytes(b"image")
    prepared = SimpleNamespace(
        imported=SimpleNamespace(model_sha256="a" * 64, input_name="test_input"),
        quantized_module=object(), mixed_module=None,
        routing=SimpleNamespace(symbols=()),
    )
    monkeypatch.setattr(runtime, "load_sample", lambda _path: np.zeros((1, 32, 32, 3), "float32"))
    monkeypatch.setattr(runtime, "prepare_model", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(runtime, "_build_selected_factory", lambda *_args: object())
    monkeypatch.setattr(runtime, "export_graph_bundle", lambda *_args, **_kwargs: SimpleNamespace(
        graph_json="graph", module=object(), params=b"params"
    ))
    monkeypatch.setattr(runtime.graph_executor, "create", lambda *_args: SimpleNamespace(
        load_params=lambda *_args: None,
        set_input=lambda *_args, **_kwargs: None,
        run=lambda: None,
        get_output=lambda _index: SimpleNamespace(numpy=lambda: np.zeros((1, 10), "float32")),
    ))
    monkeypatch.setattr(runtime, "_selected_layer_metrics", lambda *_args: ())
    monkeypatch.setattr(runtime, "_load_simulator", lambda *_args: pytest.fail("simulator loaded"))

    result = runtime.run_selected(
        target=target, simulator=simulator, output_dir=tmp_path / "output",
        model_path=model_path, input_path=input_path,
    )

    assert result.target == target
    assert result.simulator is None
    assert result.whole_cycles is None
    assert result.layers == ()
    assert "no real VTA partitions" in result.fallback_reason


@pytest.mark.parametrize("option", ["schedule", "export_workloads"])
def test_zero_coverage_rejects_schedule_and_export_before_writing(tmp_path, monkeypatch, option):
    runtime = _load("python/deployment.py", f"ic_v2_zero_coverage_{option}")
    model_path = tmp_path / "model.tflite"
    input_path = tmp_path / "input.png"
    model_path.write_bytes(b"model")
    input_path.write_bytes(b"image")
    prepared = SimpleNamespace(
        imported=SimpleNamespace(model_sha256="a" * 64),
        quantized_module=object(), mixed_module=None,
        routing=SimpleNamespace(symbols=()),
    )
    monkeypatch.setattr(runtime, "load_sample", lambda _path: np.zeros((1, 32, 32, 3), "float32"))
    monkeypatch.setattr(runtime, "prepare_model", lambda *_args, **_kwargs: prepared)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="no real VTA workloads"):
        runtime.run_selected(
            target="vta,c", simulator="fsim", output_dir=output_dir,
            model_path=model_path, input_path=input_path,
            **{option: tmp_path / "selected.log"},
        )

    assert not output_dir.exists()
