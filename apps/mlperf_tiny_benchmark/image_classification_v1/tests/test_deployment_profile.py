"""Markdown deployment report and target CLI contracts."""

import importlib.util
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


def test_markdown_report_documents_cpu_and_fsim_unavailable_measurements(tmp_path):
    runtime = _load("runtime.py", "ic_v1_profile_runtime")
    result = SimpleNamespace(
        target="vta,c", simulator="fsim", model_path=Path("resnet.tflite"),
        input_path=Path("image.png"), model_sha256="a" * 64, input_sha256="b" * 64,
        schedule=None, schedule_coverage=(), predicted_class=2,
        scores=np.array([[0.1, 0.2, 0.7]], dtype="float32"),
        layers=(
            runtime.LayerMetrics("vta.conv0", "vta", "nn.conv2d", 1024, None, 64, None),
            runtime.LayerMetrics("cpu.conv0", "cpu", "nn.conv2d", 2048, None, None, None),
        ),
        whole_cycles=None, profiler_stats={"gemm_counter": 12},
    )
    path = tmp_path / "report.md"
    runtime.write_deployment_report(result, path)
    report = path.read_text(encoding="utf-8")
    assert "# ResNet-8 deployment report" in report
    assert "| vta.conv0 | vta | nn.conv2d | 1,024 | N/A | 64 | N/A |" in report
    assert "| cpu.conv0 | cpu | nn.conv2d | 2,048 | N/A | N/A | N/A |" in report
    assert "FSIM does not provide cycle counts" in report
    assert "gemm_counter" in report


def test_tsim_report_calculates_layer_and_whole_model_utilization(tmp_path, monkeypatch):
    runtime = _load("runtime.py", "ic_v1_tsim_profile_runtime")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(APP_ROOT.parents[2] / "config" / "vta_64mac.json"))
    result = SimpleNamespace(
        target="vta,llvm", simulator="tsim", model_path=Path("resnet.tflite"),
        input_path=Path("image.png"), model_sha256="a" * 64, input_sha256="b" * 64,
        schedule=Path("best.log"), schedule_coverage=((0, "layer0", True),),
        predicted_class=1, scores=np.array([[0.2, 0.8]], dtype="float32"),
        layers=(runtime.LayerMetrics("layer0", "vta", "nn.conv2d", 640, 5, 8, 16.0),),
        whole_cycles=10, profiler_stats={"cycle_count": 10},
    )
    path = tmp_path / "report.md"
    runtime.write_deployment_report(result, path)
    report = path.read_text(encoding="utf-8")
    assert "1600.00%" in report
    assert "- Whole-model VTA MAC utilization: 800.00%" in report
    assert "Sum of measured VTA layer cycles: 5; residual against whole-model cycles: 5." in report


def test_conv_mac_count_uses_per_group_kernel_input_extent():
    runtime = _load("runtime.py", "ic_v1_mac_arithmetic_runtime")
    call = SimpleNamespace(
        checked_type=SimpleNamespace(shape=(1, 8, 8, 16)),
        args=(None, SimpleNamespace(checked_type=SimpleNamespace(shape=(3, 3, 4, 16)))),
        attrs=SimpleNamespace(kernel_layout="HWIO", groups=2),
    )
    assert runtime._conv_macs(call) == 36_864


def test_parser_accepts_four_targets_and_rejects_retired_flags():
    runner = _load("run.py", "ic_v1_cli_contract")
    parser = runner._parser()
    assert parser.parse_args([]).target == "vta,llvm"
    for target in ("c", "llvm", "vta,c", "vta,llvm"):
        assert parser.parse_args(["--target", target]).target == target
    for removed in ("--host-codegen", "--validate-schedule-evidence"):
        with pytest.raises(SystemExit):
            parser.parse_args([removed, "all"] if removed == "--host-codegen" else [removed])


def test_cpu_entrypoint_needs_no_vta_backend():
    runner = _load("run.py", "ic_v1_cpu_entrypoint")
    result = SimpleNamespace()
    calls = []
    runtime = SimpleNamespace(
        run_selected=lambda **kwargs: calls.append(kwargs) or result,
        write_deployment_report=lambda *_: None,
    )
    sys.modules["runtime"] = runtime
    try:
        assert runner.main(["--target", "llvm"]) == 0
    finally:
        sys.modules.pop("runtime", None)
    assert calls[0]["target"] == "llvm"
