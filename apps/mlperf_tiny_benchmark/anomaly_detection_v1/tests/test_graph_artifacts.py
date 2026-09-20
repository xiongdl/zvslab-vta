"""Tests for authenticated anomaly Graph Executor bundles."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "mlperf_anomaly_graph_artifacts", APP_ROOT / "graph_artifacts.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def artifacts_module():
    return _load_module()


class FakeModule:
    type_key = "llvm"
    imported_modules = ()

    def __init__(self, symbols=()):
        self.symbols = set(symbols)

    def get_source(self, fmt=""):
        return "define @main() { ret void }\n"

    def implements_function(self, symbol, query_imports=False):
        return symbol in self.symbols


class FakeFactory:
    def __init__(self, module=None, library=b"library"):
        self.module = module or FakeModule()
        self.library = library

    def get_graph_json(self):
        return '{"nodes":[],"arg_nodes":[],"heads":[]}'

    def get_params(self):
        return b"params"

    def get_lib(self):
        return self.module

    def export_library(self, path):
        Path(path).write_bytes(self.library)


def _export(artifacts_module, monkeypatch, tmp_path, factory=None):
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda value: value)
    loaded = (factory or FakeFactory()).module
    monkeypatch.setattr(artifacts_module.tvm.runtime, "load_module", lambda path: loaded)
    return artifacts_module.export_graph_bundle(
        factory or FakeFactory(),
        tmp_path,
        "llvm-fsim/reference",
        artifact_name="mlperf_anomaly_llvm_reference",
        artifact_role="reference",
        model_sha256="a" * 64,
        host_codegen="llvm",
        simulator="fsim",
        forbidden_vta_symbols=("tvmgen_mlperf_anomaly_vta_main_0",),
    )


def test_export_has_authenticated_metadata_and_reload_contract(
    artifacts_module, monkeypatch, tmp_path
):
    bundle = _export(artifacts_module, monkeypatch, tmp_path)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == {
        "name": "mlperf_anomaly_llvm_reference",
        "role": "reference",
    }
    assert manifest["host_codegen"] == "llvm"
    assert manifest["simulator"] == "fsim"
    assert manifest["model_sha256"] == "a" * 64
    assert manifest["symbols"]["forbidden"] == ["tvmgen_mlperf_anomaly_vta_main_0"]
    assert manifest["files"]["graph"]["sha256"] == hashlib.sha256(
        bundle.graph_path.read_bytes()
    ).hexdigest()

    reloaded = artifacts_module.load_graph_bundle(tmp_path, "llvm-fsim/reference")
    assert reloaded.graph_json == bundle.graph_json
    assert reloaded.params == b"params"
    assert reloaded.manifest["artifact"]["role"] == "reference"


def test_export_failure_removes_staging_and_preserves_no_partial_bundle(
    artifacts_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda value: value)

    def fail_export(path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("synthetic export failure")

    factory = FakeFactory()
    factory.export_library = fail_export
    with pytest.raises(RuntimeError, match="synthetic export failure"):
        artifacts_module.export_graph_bundle(
            factory,
            tmp_path,
            "llvm-fsim/mixed",
            artifact_name="mlperf_anomaly_llvm_mixed",
            artifact_role="mixed",
            model_sha256="a" * 64,
            host_codegen="llvm",
            simulator="fsim",
        )

    assert not (tmp_path / "llvm-fsim" / "mixed").exists()
    assert not list((tmp_path / "llvm-fsim").glob(".mixed.staging-*"))


def test_metadata_rejects_unsafe_artifact_path_before_mutation(artifacts_module, tmp_path):
    with pytest.raises(ValueError):
        artifacts_module.export_graph_bundle(
            FakeFactory(),
            tmp_path,
            "../escape",
            artifact_name="bad",
            artifact_role="reference",
            model_sha256="a" * 64,
            host_codegen="llvm",
            simulator="fsim",
        )
    assert not list(tmp_path.iterdir())
