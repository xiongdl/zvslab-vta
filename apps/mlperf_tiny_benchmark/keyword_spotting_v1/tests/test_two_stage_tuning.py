"""Independent workloads-driven FSIM and TSIM tuning contracts."""

import importlib.util
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    package_name = "keyword_spotting_v1_test_app"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, APP_ROOT / "python" / "__init__.py",
            submodule_search_locations=[str(APP_ROOT / "python")],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    if name == "tune":
        spec = importlib.util.spec_from_file_location("keyword_spotting_v1_tune_cli", APP_ROOT / "tune.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    module_name = {
        "deployment": "deployment", "runtime": "deployment",
        "model_pipeline": "model", "measurement": "measurement",
        "tuning": "tuning", "workflow": "tuning",
        "schedule": "schedule_io", "publication": "tuning_storage",
        "workloads": "vta_workload", "dispatch": "autotvm_dispatch",
    }[name]
    return importlib.import_module(f"{package_name}.{module_name}")




def test_empty_workload_snapshot_rejects_tuning_without_replacing_prior_files(
    monkeypatch, tmp_path
):
    tuning = _load("workflow")
    tune = _load("tune")
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    monkeypatch.setattr(
        tuning, "load_workloads", lambda _path: type("Snapshot", (), {"layers": ()})()
    )
    workload_file = tmp_path / "empty-workloads.json"
    workload_file.write_text("authenticated empty snapshot fixture", encoding="utf-8")
    output = tmp_path / "existing.tmp"
    output.write_bytes(b"prior validated candidates")
    args = tune._parser().parse_args([
        "--workloads", str(workload_file), "--simulator", "fsim",
        "--trial-batch", "1", "--min-successful", "1", "--output-logs", str(output),
    ])

    with pytest.raises(ValueError, match="no real VTA workloads"):
        tuning.run(args)
    assert output.read_bytes() == b"prior validated candidates"
