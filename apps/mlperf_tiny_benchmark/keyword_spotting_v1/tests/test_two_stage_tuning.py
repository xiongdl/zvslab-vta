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
        tuning, "load_workloads", lambda _path: type("Snapshot", (), {
            "model_sha256": "738a9f29d175aaa3928db9c8281265be5ec3406598fd3d30018b26084a3d5536",
            "layers": (),
        })()
    )
    workload_file = tmp_path / "empty-workloads.json"
    workload_file.write_text("authenticated empty snapshot fixture", encoding="utf-8")
    output = tmp_path / "existing.tmp"
    output.write_bytes(b"prior validated candidates")
    args = tune._parser().parse_args([
        "--model", str(APP_ROOT / "model/kws_ref_model_float32.tflite"),
        "--workloads", str(workload_file), "--simulator", "fsim",
        "--trial-batch", "1", "--min-successful", "1", "--output-logs", str(output),
    ])

    with pytest.raises(ValueError, match="no real VTA workloads"):
        tuning.run(args)
    assert output.read_bytes() == b"prior validated candidates"


def test_workload_snapshot_must_match_float_model_hash(monkeypatch, tmp_path):
    tuning = _load("workflow")
    tune = _load("tune")
    monkeypatch.setenv("VTA_BACKEND", "fsim")
    monkeypatch.setattr(
        tuning, "load_workloads",
        lambda _path: type("Snapshot", (), {"model_sha256": "0" * 64, "layers": ()})(),
    )
    args = tune._parser().parse_args([
        "--model", str(APP_ROOT / "model/kws_ref_model_float32.tflite"),
        "--workloads", str(tmp_path / "snapshot.json"), "--simulator", "fsim",
        "--trial-batch", "1", "--min-successful", "1",
        "--output-logs", str(tmp_path / "candidates.tmp"),
    ])
    with pytest.raises(ValueError, match="workloads model hash does not match --model"):
        tuning.run(args)
