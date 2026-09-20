"""Authenticated Graph Executor artifact contracts for KWS."""

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
    assert path.is_file(), f"missing Task 3 implementation: {path}"
    spec = importlib.util.spec_from_file_location("mlperf_kws_graph_artifacts", path)
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
    def __init__(self, module, graph="{\"nodes\":[]}", params=b"params", library=b"library"):
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
        "artifact_name": "mlperf_kws_llvm_reference",
        "artifact_role": "reference",
        "model_sha256": "a" * 64,
        "host_codegen": "llvm",
        "simulator": "host",
    }
    options.update(kwargs)
    return artifacts_module.export_graph_bundle(
        FakeFactory(module), tmp_path, "llvm-host/reference", **options
    )


def test_export_reload_contains_hashes_sources_metadata_and_symbols(
    artifacts_module, monkeypatch, tmp_path
):
    module = FakeModule(symbols=("tvmgen_mlperf_kws_vta_main_0",))
    bundle = _export(
        module,
        artifacts_module,
        monkeypatch,
        tmp_path,
        artifact_role="mixed",
        artifact_name="mlperf_kws_llvm_mixed",
        simulator="fsim",
        expected_vta_symbols=("tvmgen_mlperf_kws_vta_main_0",),
        metadata={"input": {"shape": [1, 49, 10, 1]}},
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == {
        "name": "mlperf_kws_llvm_mixed",
        "role": "mixed",
    }
    assert manifest["host_codegen"] == "llvm"
    assert manifest["simulator"] == "fsim"
    assert manifest["metadata"]["input"]["shape"] == [1, 49, 10, 1]
    assert manifest["symbols"]["expected"] == ["tvmgen_mlperf_kws_vta_main_0"]
    assert manifest["files"]["params"]["sha256"] == hashlib.sha256(
        bundle.params_path.read_bytes()
    ).hexdigest()
    assert [entry["source_format"] for entry in manifest["sources"]] == ["ll"]
    assert bundle.source_paths[0].read_text(encoding="utf-8") == "source"

    reloaded = artifacts_module.load_graph_bundle(tmp_path, "llvm-host/reference")
    assert reloaded.params == b"params"
    assert reloaded.manifest["simulator"] == "fsim"


def test_source_discovery_is_recursive_and_records_unavailable_source(
    artifacts_module, monkeypatch, tmp_path
):
    leaf = FakeModule(type_key="c", source="int leaf(void) { return 1; }")
    root = FakeModule(imports=(leaf, FakeModule(type_key="runtime", error="/tmp/private")))
    bundle = _export(root, artifacts_module, monkeypatch, tmp_path)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert [entry["module_type"] for entry in manifest["sources"]] == [
        "llvm",
        "c",
        "runtime",
    ]
    assert [entry["available"] for entry in manifest["sources"]] == [True, True, False]
    assert manifest["sources"][2]["reason"] == "get_source_failed"
    assert "/tmp/private" not in bundle.manifest_path.read_text(encoding="utf-8")
    assert [path.name for path in bundle.source_paths] == ["00-module.ll", "01-module.c"]


@pytest.mark.parametrize("relative", ("", ".", "../escape", "nested/../../escape", "/absolute"))
def test_unsafe_paths_are_rejected_before_mutation(artifacts_module, tmp_path, relative):
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


def test_tampered_core_file_or_source_is_rejected_before_module_load(
    artifacts_module, monkeypatch, tmp_path
):
    bundle = _export(FakeModule(), artifacts_module, monkeypatch, tmp_path)
    bundle.graph_path.write_text("tampered", encoding="utf-8")
    load_calls = []
    monkeypatch.setattr(
        artifacts_module.tvm.runtime,
        "load_module",
        lambda path: load_calls.append(path),
    )
    with pytest.raises(RuntimeError, match="artifact graph hash mismatch"):
        artifacts_module.load_graph_bundle(tmp_path, "llvm-host/reference")
    assert load_calls == []


def test_failed_replacement_preserves_previous_bundle_and_cleans_staging(
    artifacts_module, monkeypatch, tmp_path
):
    module = FakeModule()
    _export(module, artifacts_module, monkeypatch, tmp_path)
    old_manifest = (tmp_path / "llvm-host" / "reference" / "manifest.json").read_bytes()

    def fail_reload(path):
        raise RuntimeError("synthetic reload failure")

    monkeypatch.setattr(artifacts_module.tvm.runtime, "load_module", fail_reload)
    with pytest.raises(RuntimeError, match="synthetic reload failure"):
        artifacts_module.export_graph_bundle(
            FakeFactory(module, graph="new", params=b"new", library=b"new"),
            tmp_path,
            "llvm-host/reference",
            artifact_name="mlperf_kws_llvm_reference",
            artifact_role="reference",
            model_sha256="a" * 64,
            host_codegen="llvm",
            simulator="host",
        )
    assert (
        tmp_path / "llvm-host" / "reference" / "manifest.json"
    ).read_bytes() == old_manifest
    assert not list((tmp_path / "llvm-host").glob(".reference.staging-*"))


def test_symbol_contract_rejects_missing_and_forbidden_symbols(
    artifacts_module, monkeypatch, tmp_path
):
    missing = "tvmgen_mlperf_kws_vta_main_0"
    with pytest.raises(RuntimeError, match=missing):
        _export(
            FakeModule(),
            artifacts_module,
            monkeypatch,
            tmp_path,
            artifact_role="mixed",
            expected_vta_symbols=(missing,),
        )

    with pytest.raises(RuntimeError, match="forbidden"):
        _export(
            FakeModule(symbols=(missing,)),
            artifacts_module,
            monkeypatch,
            tmp_path,
            forbidden_vta_symbols=(missing,),
        )
