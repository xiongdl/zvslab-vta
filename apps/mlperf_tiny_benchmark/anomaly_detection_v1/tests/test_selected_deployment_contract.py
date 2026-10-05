"""Selected target CLI and CPU startup contract."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_cli():
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    return importlib.import_module("deploy")


def test_parser_exposes_one_input_and_four_literal_targets():
    cli = _load_cli()
    args = cli._parser().parse_args([])
    assert args.model.name == "ad01_fp32.tflite"
    assert args.input.name == "normal_id_01_00000000.wav"
    assert args.target == "vta,llvm"
    for target in ("c", "llvm", "vta,c", "vta,llvm"):
        assert cli._parser().parse_args(["--target", target]).target == target
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["--host-codegen", "llvm"])


def test_cpu_workload_export_is_rejected_before_startup(monkeypatch):
    cli = _load_cli()
    monkeypatch.delenv("VTA_BACKEND", raising=False)
    monkeypatch.setattr(importlib.import_module("python"), "deployment", None, raising=False)
    with pytest.raises(ValueError, match="requires a target that includes VTA"):
        cli.main(["--target", "c", "--export-workloads", "out.json"])


def test_importing_deployment_module_without_backend_does_not_import_vta():
    env = os.environ.copy()
    env.pop("VTA_BACKEND", None)
    env.pop("VTA_CONFIG_FILE", None)
    env["PYTHONPATH"] = os.pathsep.join((str(APP_ROOT), str(APP_ROOT.parents[3] / "tvm/python")))
    python = APP_ROOT.parents[3] / ".envs/tvm-vta-env/bin/python"
    code = (
        "import sys; import python.deployment; "
        "assert not any(name == 'vta' or name.startswith('vta.') for name in sys.modules)"
    )
    completed = subprocess.run([str(python), "-c", code], cwd=APP_ROOT, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
