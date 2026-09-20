"""Authenticated Graph Executor artifact contracts for streaming wakeword."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts_module():
    path = APP_ROOT / "graph_artifacts.py"
    assert path.is_file(), f"missing Task 3.1 implementation: {path}"
    spec = importlib.util.spec_from_file_location("mlperf_streaming_graph_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeModule:
    def __init__(self, type_key="llvm", source="source", imports=(), symbols=(), error=None):
        self.type_key = type_key
        self._source = source
        self.imported_modules = list(imports)
        self._symbols = set(symbols)
        self._error = error

    def get_source(self, fmt=""):
        if self._error:
            raise RuntimeError(self._error)
        if self._source is None:
            raise RuntimeError("source unavailable")
        return self._source

    def implements_function(self, symbol, query_imports=False):
        return symbol in self._symbols or (
            query_imports
            and any(child.implements_function(symbol, True) for child in self.imported_modules)
        )


class FakeFactory:
    def __init__(self, module, graph='{"nodes":[]}', params=b"params", library=b"library"):
        self.module = module
        self.graph = graph
        self.params = params
        self.library = library

    def get_graph_json(self):
        return self.graph

    def get_params(self):
        return self.params

    def get_lib(self):
        return self.module

    def export_library(self, path):
        Path(path).write_bytes(self.library)


def _export(module, artifacts_module, monkeypatch, tmp_path, **kwargs):
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda params: params)
    monkeypatch.setattr(artifacts_module.tvm.runtime, "load_module", lambda path: module)
    options = {
        "artifact_name": "mlperf_streaming_llvm_reference",
        "artifact_role": "reference",
        "model_sha256": "a" * 64,
        "host_codegen": "llvm",
        "simulator": "host",
    }
    options.update(kwargs)
    return artifacts_module.export_graph_bundle(
        FakeFactory(module), tmp_path, "llvm-host/reference", **options
    )


def test_export_reload_authenticates_metadata_sources_and_symbols(
    artifacts_module, monkeypatch, tmp_path
):
    symbol = "tvmgen_mlperf_streaming_wakeword_vta_main_0"
    bundle = _export(
        FakeModule(symbols=(symbol,)),
        artifacts_module,
        monkeypatch,
        tmp_path,
        artifact_role="mixed",
        artifact_name="mlperf_streaming_llvm_mixed",
        simulator="fsim",
        expected_vta_symbols=(symbol,),
        metadata={
            "model_sha256": "a" * 64,
            "input": {"shape": [1, 30, 1, 40], "dtype": "int8"},
            "output": {"shape": [1, 3], "dtype": "int8"},
        },
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == {
        "name": "mlperf_streaming_llvm_mixed",
        "role": "mixed",
    }
    assert manifest["metadata"]["input"]["shape"] == [1, 30, 1, 40]
    assert manifest["symbols"]["expected"] == [symbol]
    assert manifest["files"]["params"]["sha256"] == hashlib.sha256(
        bundle.params_path.read_bytes()
    ).hexdigest()
    assert bundle.source_paths[0].read_text(encoding="utf-8") == "source"

    reloaded = artifacts_module.load_graph_bundle(tmp_path, "llvm-host/reference")
    assert reloaded.params == b"params"
    assert reloaded.manifest["simulator"] == "fsim"


def test_source_capture_is_recursive_and_does_not_persist_error_details(
    artifacts_module, monkeypatch, tmp_path
):
    leaf = FakeModule(type_key="c", source="int leaf(void) { return 1; }")
    root = FakeModule(imports=(leaf, FakeModule(type_key="runtime", error="private path")))
    bundle = _export(root, artifacts_module, monkeypatch, tmp_path)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert [entry["module_type"] for entry in manifest["sources"]] == [
        "llvm",
        "c",
        "runtime",
    ]
    assert [entry["available"] for entry in manifest["sources"]] == [True, True, False]
    assert manifest["sources"][2]["reason"] == "get_source_failed"
    assert "private path" not in bundle.manifest_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("relative", ("", ".", "../escape", "nested/../../escape", "/absolute"))
def test_unsafe_artifact_paths_are_rejected_before_mutation(
    artifacts_module, tmp_path, relative
):
    with pytest.raises(ValueError):
        artifacts_module.export_graph_bundle(
            FakeFactory(FakeModule()),
            tmp_path,
            relative,
            artifact_name="bad",
            artifact_role="reference",
            model_sha256="a" * 64,
            host_codegen="llvm",
            simulator="host",
        )
    assert list(tmp_path.iterdir()) == []


def test_tampered_file_is_rejected_before_loading_library(
    artifacts_module, monkeypatch, tmp_path
):
    bundle = _export(FakeModule(), artifacts_module, monkeypatch, tmp_path)
    bundle.graph_path.write_text("tampered", encoding="utf-8")
    load_calls = []
    monkeypatch.setattr(
        artifacts_module.tvm.runtime, "load_module", lambda path: load_calls.append(path)
    )
    with pytest.raises(RuntimeError, match="graph hash mismatch"):
        artifacts_module.load_graph_bundle(tmp_path, "llvm-host/reference")
    assert load_calls == []


def test_failed_replacement_preserves_previous_bundle_and_cleans_stage(
    artifacts_module, monkeypatch, tmp_path
):
    module = FakeModule()
    _export(module, artifacts_module, monkeypatch, tmp_path)
    old_manifest = (tmp_path / "llvm-host" / "reference" / "manifest.json").read_bytes()

    monkeypatch.setattr(
        artifacts_module.tvm.runtime,
        "load_module",
        lambda path: (_ for _ in ()).throw(RuntimeError("synthetic reload failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic reload failure"):
        artifacts_module.export_graph_bundle(
            FakeFactory(module, graph="new", params=b"new", library=b"new"),
            tmp_path,
            "llvm-host/reference",
            artifact_name="mlperf_streaming_llvm_reference",
            artifact_role="reference",
            model_sha256="a" * 64,
            host_codegen="llvm",
            simulator="host",
        )
    assert (
        tmp_path / "llvm-host" / "reference" / "manifest.json"
    ).read_bytes() == old_manifest
    assert not list((tmp_path / "llvm-host").glob(".reference.staging-*"))


def test_reference_forbids_vta_symbols_and_mixed_requires_them(
    artifacts_module, monkeypatch, tmp_path
):
    symbol = "tvmgen_mlperf_streaming_wakeword_vta_main_0"
    with pytest.raises(RuntimeError, match=symbol):
        _export(
            FakeModule(),
            artifacts_module,
            monkeypatch,
            tmp_path,
            artifact_role="mixed",
            expected_vta_symbols=(symbol,),
        )

    with pytest.raises(RuntimeError, match="forbidden"):
        _export(
            FakeModule(symbols=(symbol,)),
            artifacts_module,
            monkeypatch,
            tmp_path,
            forbidden_vta_symbols=(symbol,),
        )
