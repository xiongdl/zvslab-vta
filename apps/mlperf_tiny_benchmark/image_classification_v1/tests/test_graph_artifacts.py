# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Focused tests for the Graph Executor artifact bundle."""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_PATH = APP_ROOT / "graph_artifacts.py"


@pytest.fixture(scope="module")
def artifacts_module():
    try:
        import tvm  # noqa: F401
    except RuntimeError:
        for name in tuple(sys.modules):
            if name == "tvm" or name.startswith("tvm."):
                del sys.modules[name]
        fake_runtime = types.ModuleType("tvm.runtime")
        fake_runtime.Module = object
        fake_runtime.load_module = lambda path: None
        fake_tvm = types.ModuleType("tvm")
        fake_tvm.runtime = fake_runtime
        fake_relay = types.ModuleType("tvm.relay")
        fake_relay.save_param_dict = lambda params: params
        fake_tvm.relay = fake_relay
        sys.modules["tvm"] = fake_tvm
        sys.modules["tvm.runtime"] = fake_runtime
        sys.modules["tvm.relay"] = fake_relay
    spec = importlib.util.spec_from_file_location("resnet_graph_artifacts", ARTIFACTS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeModule:
    def __init__(self, type_key, source=None, imports=(), symbols=()):
        self.type_key = type_key
        self._source = source
        self.imported_modules = list(imports)
        self._symbols = set(symbols)

    def get_source(self, fmt=""):
        if self._source is None:
            raise RuntimeError("source is unavailable")
        return self._source

    def implements_function(self, name, query_imports=False):
        return name in self._symbols or (
            query_imports and any(module.implements_function(name, True) for module in self.imported_modules)
        )


class FakeFactory:
    def __init__(self, module, graph='{"nodes":[]}', params=b"params", library=b"library"):
        self._module = module
        self._graph = graph
        self._params = params
        self._library = library

    def get_graph_json(self):
        return self._graph

    def get_params(self):
        return self._params

    def get_lib(self):
        return self._module

    def export_library(self, path):
        Path(path).write_bytes(self._library)


def _patch_loader(monkeypatch, artifacts_module):
    monkeypatch.setattr(
        artifacts_module.tvm.runtime,
        "load_module",
        lambda path: FakeModule("loaded", source="loaded-source"),
    )


def _export(artifacts_module, monkeypatch, tmp_path, module, **kwargs):
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda params: params)
    _patch_loader(monkeypatch, artifacts_module)
    return artifacts_module.export_graph_bundle(
        FakeFactory(module),
        tmp_path,
        "bundle",
        artifact_name="resnet8",
        artifact_role="reference",
        model_sha256="a" * 64,
        host_codegen="test",
        simulator="fsim",
        **kwargs,
    )


def test_export_writes_exact_files_and_schema_one_manifest(artifacts_module, monkeypatch, tmp_path):
    root = FakeModule("llvm", source="define @main() { ret void }\n")
    result = _export(artifacts_module, monkeypatch, tmp_path, root)

    bundle = tmp_path / "bundle"
    assert result.artifact_dir == bundle
    assert result.graph_json == '{"nodes":[]}'
    assert result.params == b"params"
    assert result.library_path == bundle / ("model" + artifacts_module.shared_library_suffix())
    assert result.graph_path.read_text(encoding="utf-8") == result.graph_json
    assert result.params_path.read_bytes() == result.params
    assert result.module.type_key == "loaded"

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["artifact"] == {"name": "resnet8", "role": "reference"}
    assert manifest["host_codegen"] == "test"
    assert manifest["simulator"] == "fsim"
    assert manifest["files"]["graph"]["sha256"] == hashlib.sha256(b'{"nodes":[]}').hexdigest()
    assert manifest["files"]["params"]["sha256"] == hashlib.sha256(b"params").hexdigest()
    assert all(not str(value).startswith("/") for value in manifest.values() if isinstance(value, str))


def test_source_discovery_is_recursive_preorder_and_records_no_source(artifacts_module, monkeypatch, tmp_path):
    leaf = FakeModule("c", source="int leaf(void) { return 1; }\n")
    no_source = FakeModule("runtime")
    root = FakeModule("llvm", source="define @root() { ret void }\n", imports=(leaf, no_source))
    result = _export(artifacts_module, monkeypatch, tmp_path, root)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [entry["module_type"] for entry in manifest["sources"]] == ["llvm", "c", "runtime"]
    assert [entry["available"] for entry in manifest["sources"]] == [True, True, False]
    assert [path.name for path in result.source_paths] == ["00-llvm.ll", "01-c.c"]
    assert result.source_paths[0].read_text(encoding="utf-8").startswith("define")
    assert result.source_paths[1].read_text(encoding="utf-8").startswith("int leaf")


@pytest.mark.parametrize("relative", ["", ".", "../escape", "nested/../../escape", "/absolute"])
def test_unsafe_artifact_paths_are_rejected_before_output_mutation(
    artifacts_module, monkeypatch, tmp_path, relative
):
    module = FakeModule("llvm", source="source")
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda params: params)
    with pytest.raises(ValueError):
        artifacts_module.export_graph_bundle(
            FakeFactory(module),
            tmp_path,
            relative,
            artifact_name="resnet8",
            artifact_role="reference",
            model_sha256="a" * 64,
            host_codegen="test",
            simulator="fsim",
        )
    assert list(tmp_path.iterdir()) == []


def test_symlink_escape_is_rejected(artifacts_module, monkeypatch, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    module = FakeModule("llvm", source="source")
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda params: params)
    with pytest.raises(ValueError, match="output root"):
        artifacts_module.export_graph_bundle(
            FakeFactory(module),
            tmp_path,
            "link/bundle",
            artifact_name="resnet8",
            artifact_role="reference",
            model_sha256="a" * 64,
            host_codegen="test",
            simulator="fsim",
        )
    assert list(outside.iterdir()) == []


def test_expected_and_forbidden_symbols_are_checked_after_reload(artifacts_module, monkeypatch, tmp_path):
    symbols = ("vta_main_0", "vta_main_1")
    loaded = FakeModule("loaded", source="source", symbols=symbols)
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda params: params)
    monkeypatch.setattr(artifacts_module.tvm.runtime, "load_module", lambda path: loaded)
    result = artifacts_module.export_graph_bundle(
        FakeFactory(FakeModule("llvm", source="source")),
        tmp_path,
        "mixed",
        artifact_name="resnet8",
        artifact_role="mixed",
        model_sha256="a" * 64,
        host_codegen="llvm",
        simulator="fsim",
        expected_vta_symbols=symbols,
        forbidden_vta_symbols=("host_only",),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["symbols"] == {
        "expected": list(symbols),
        "forbidden": ["host_only"],
        "expected_implemented": True,
        "forbidden_absent": True,
    }


def test_failed_replacement_preserves_previous_bundle_and_cleans_staging(
    artifacts_module, monkeypatch, tmp_path
):
    module = FakeModule("llvm", source="source")
    monkeypatch.setattr(artifacts_module.relay, "save_param_dict", lambda params: params)
    _patch_loader(monkeypatch, artifacts_module)
    factory = FakeFactory(module, graph="old", params=b"old", library=b"old")
    artifacts_module.export_graph_bundle(
        factory,
        tmp_path,
        "bundle",
        artifact_name="resnet8",
        artifact_role="reference",
        model_sha256="a" * 64,
        host_codegen="test",
        simulator="fsim",
    )
    old_manifest = (tmp_path / "bundle" / "manifest.json").read_bytes()

    def fail_loader(path):
        raise RuntimeError("synthetic reload failure")

    monkeypatch.setattr(artifacts_module.tvm.runtime, "load_module", fail_loader)
    with pytest.raises(RuntimeError, match="synthetic reload failure"):
        artifacts_module.export_graph_bundle(
            FakeFactory(module, graph="new", params=b"new", library=b"new"),
            tmp_path,
            "bundle",
            artifact_name="resnet8",
            artifact_role="reference",
            model_sha256="a" * 64,
            host_codegen="test",
            simulator="fsim",
        )
    assert (tmp_path / "bundle" / "manifest.json").read_bytes() == old_manifest
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_load_graph_bundle_validates_final_files(artifacts_module, monkeypatch, tmp_path):
    module = FakeModule("llvm", source="source")
    result = _export(artifacts_module, monkeypatch, tmp_path, module)
    loaded = FakeModule("loaded", source="loaded")
    monkeypatch.setattr(artifacts_module.tvm.runtime, "load_module", lambda path: loaded)
    reloaded = artifacts_module.load_graph_bundle(tmp_path, "bundle")
    assert reloaded.artifact_dir == result.artifact_dir
    assert reloaded.graph_json == result.graph_json
    assert reloaded.params == result.params
    assert reloaded.module is loaded


def test_load_graph_bundle_rejects_tampered_source(artifacts_module, monkeypatch, tmp_path):
    result = _export(
        artifacts_module,
        monkeypatch,
        tmp_path,
        FakeModule("llvm", source="define @main() { ret void }\n"),
    )
    result.source_paths[0].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source hash mismatch"):
        artifacts_module.load_graph_bundle(tmp_path, "bundle")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda entry: entry.update(path="../escape.ll"), "unsafe relative"),
        (lambda entry: entry.update(path="source/missing.ll"), "source file is missing"),
        (lambda entry: entry.update(path=None), "invalid source path"),
    ],
)
def test_load_graph_bundle_rejects_malformed_source_entries(
    artifacts_module, monkeypatch, tmp_path, mutation, message
):
    _export(
        artifacts_module,
        monkeypatch,
        tmp_path,
        FakeModule("llvm", source="define @main() { ret void }\n"),
    )
    manifest_path = tmp_path / "bundle" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest["sources"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        artifacts_module.load_graph_bundle(tmp_path, "bundle")
