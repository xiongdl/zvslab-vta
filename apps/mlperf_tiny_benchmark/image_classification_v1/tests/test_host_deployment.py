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

"""HOST export, reload, execution, and FSIM contracts for MLPerf ResNet-8."""

import ast
import importlib.util
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = APP_ROOT / "runtime.py"
RUN_PATH = APP_ROOT / "run.py"
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet.tflite"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
EXPECTED_VTA_SYMBOLS = tuple(f"tvmgen_mlperf_resnet_vta_main_{index}" for index in range(8))
EXPECTED_ARTIFACT_DIRS = ("reference", "mixed")
REQUIRED_PROFILER_COUNTERS = ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes")


def _load_module(path, name):
    assert path.is_file(), f"missing Task 12 implementation: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def deployment_runtime():
    return _load_module(RUNTIME_PATH, "mlperf_resnet_host_runtime")


def _expected_sample_paths():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return tuple(MANIFEST_PATH.parent / sample["filename"] for sample in manifest["samples"])


def test_committed_sample_order_is_the_fixed_manifest_order(deployment_runtime):
    expected = _expected_sample_paths()
    assert len(expected) == 10
    assert len(set(expected)) == 10
    assert deployment_runtime.committed_sample_paths() == expected


def test_build_exports_and_reloads_two_standard_dsos_without_loading_fsim(
    deployment_runtime, monkeypatch, tmp_path
):
    quantized = object()
    mixed = object()
    prepared = SimpleNamespace(
        quantized_module=quantized,
        reference_module=quantized,
        mixed_module=mixed,
        routing=SimpleNamespace(symbols=EXPECTED_VTA_SYMBOLS),
    )
    events = []
    loaded_modules = {}
    params_by_kind = {
        "reference": {"reference_weight": object()},
        "mixed": {"mixed_weight": object()},
    }
    serialized_params_by_kind = {
        "reference": b"serialized-reference-params",
        "mixed": b"serialized-mixed-params",
    }

    class FakeFactory:
        def __init__(self, kind):
            self.kind = kind

        def get_graph_json(self):
            return f'{{"kind":"{self.kind}"}}'

        def get_params(self):
            events.append(("get_params", self.kind))
            return params_by_kind[self.kind]

        def get_lib(self):
            return SimpleNamespace(type_key=self.kind, imported_modules=(), get_source=lambda fmt="": "source")

        def export_library(self, path):
            path = Path(path)
            events.append(("export", self.kind, path))
            path.write_bytes(self.kind.encode("ascii"))

    def fake_build(module, target):
        kind = "reference" if module is quantized else "mixed"
        events.append(("build", kind, module, target))
        return FakeFactory(kind)

    def fake_load(path):
        path = Path(path)
        events.append(("reload", path))
        symbols = EXPECTED_VTA_SYMBOLS if path.parent.name == "mixed" or path.parent.name.startswith(".mixed") else ()
        loaded = SimpleNamespace(path=path, implements_function=lambda symbol, query_imports: symbol in symbols)
        loaded_modules[path] = loaded
        return loaded

    def fake_save_param_dict(params):
        kind = next(kind for kind, expected in params_by_kind.items() if params is expected)
        events.append(("serialize_params", kind))
        return serialized_params_by_kind[kind]

    @contextmanager
    def fake_build_config():
        events.append(("build_config_enter",))
        yield
        events.append(("build_config_exit",))

    def forbidden_fsim_load():
        raise AssertionError("FSIM must not load during build, export, or reload")

    mixed_target = object()
    monkeypatch.setattr(deployment_runtime.relay, "build", fake_build)
    monkeypatch.setattr(deployment_runtime.relay, "save_param_dict", fake_save_param_dict)
    monkeypatch.setattr(deployment_runtime.tvm.runtime, "load_module", fake_load)
    monkeypatch.setattr(deployment_runtime.vta, "build_config", fake_build_config)
    monkeypatch.setattr(deployment_runtime, "_mixed_target", lambda: mixed_target)
    monkeypatch.setattr(deployment_runtime, "_load_fsim", forbidden_fsim_load)

    artifacts = deployment_runtime.build_host_artifacts(prepared, tmp_path)

    suffix = deployment_runtime.shared_library_suffix()
    assert artifacts.reference.path == tmp_path / "reference" / ("model" + suffix)
    assert artifacts.mixed.path == tmp_path / "mixed" / ("model" + suffix)
    assert artifacts.reference.artifact_dir == tmp_path / "reference"
    assert artifacts.mixed.artifact_dir == tmp_path / "mixed"
    assert artifacts.reference.graph_json == '{"kind":"reference"}'
    assert artifacts.mixed.graph_json == '{"kind":"mixed"}'
    assert artifacts.reference.module is loaded_modules[artifacts.reference.path]
    assert artifacts.mixed.module is loaded_modules[artifacts.mixed.path]
    assert artifacts.reference.params == serialized_params_by_kind["reference"]
    assert artifacts.mixed.params == serialized_params_by_kind["mixed"]
    assert artifacts.reference.params is not artifacts.mixed.params
    assert artifacts.vta_symbols == EXPECTED_VTA_SYMBOLS
    assert artifacts.reference.path.read_bytes() == b"reference"
    assert artifacts.mixed.path.read_bytes() == b"mixed"

    build_events = [event for event in events if event[0] == "build"]
    assert build_events == [
        ("build", "reference", quantized, "llvm"),
        ("build", "mixed", mixed, mixed_target),
    ]
    assert events.index(("build_config_enter",)) < events.index(build_events[1])
    assert events.index(build_events[1]) < events.index(("build_config_exit",))
    assert [event[0] for event in events].count("export") == 2
    assert [event[0] for event in events].count("reload") == 4
    param_events = [event for event in events if event[0] in {"get_params", "serialize_params"}]
    assert len(param_events) == 4
    assert set(param_events) == {
        ("get_params", "reference"),
        ("serialize_params", "reference"),
        ("get_params", "mixed"),
        ("serialize_params", "mixed"),
    }


def test_each_reloaded_graph_loads_only_its_own_params_before_execution(
    deployment_runtime, monkeypatch
):
    input_data = np.zeros((1, 32, 32, 3), dtype="float32")
    reference_module = object()
    mixed_module = object()
    reference_params = b"reference-param-blob"
    mixed_params = b"mixed-param-blob"
    expected_params = {"reference": reference_params, "mixed": mixed_params}
    outputs = {
        "reference": np.arange(10, dtype="float32").reshape(1, 10),
        "mixed": np.arange(10, dtype="float32").reshape(1, 10),
    }
    module_kinds = {reference_module: "reference", mixed_module: "mixed"}
    events = []

    class FakeGraphModule:
        def __init__(self, kind):
            self.kind = kind

        def load_params(self, params):
            assert params is expected_params[self.kind]
            events.append(("load_params", self.kind, params))

        def set_input(self, name, value):
            events.append(("set_input", self.kind, name, value))

        def run(self):
            events.append(("run", self.kind))

        def get_output(self, index):
            assert index == 0
            events.append(("get_output", self.kind))
            return SimpleNamespace(numpy=lambda: outputs[self.kind])

    def fake_create(graph_json, module, device):
        kind = module_kinds[module]
        events.append(("create", kind, graph_json, device))
        return FakeGraphModule(kind)

    monkeypatch.setattr(deployment_runtime.graph_executor, "create", fake_create)
    reference_artifact = SimpleNamespace(
        graph_json="reference-graph",
        module=reference_module,
        device="cpu",
        params=reference_params,
    )
    mixed_artifact = SimpleNamespace(
        graph_json="mixed-graph",
        module=mixed_module,
        device="ext_dev",
        params=mixed_params,
    )

    reference_output = deployment_runtime._run_graph(reference_artifact, input_data)
    mixed_output = deployment_runtime._run_graph(mixed_artifact, input_data)

    np.testing.assert_array_equal(reference_output, outputs["reference"])
    np.testing.assert_array_equal(mixed_output, outputs["mixed"])
    for kind, params in (("reference", reference_params), ("mixed", mixed_params)):
        per_graph_events = [event for event in events if len(event) > 1 and event[1] == kind]
        assert [event[0] for event in per_graph_events] == [
            "create",
            "load_params",
            "set_input",
            "run",
            "get_output",
        ]
        assert per_graph_events[1] == ("load_params", kind, params)


def test_partial_artifacts_are_removed_when_export_fails(deployment_runtime, monkeypatch, tmp_path):
    prepared = SimpleNamespace(
        quantized_module=object(),
        reference_module=object(),
        mixed_module=object(),
        routing=SimpleNamespace(symbols=EXPECTED_VTA_SYMBOLS),
    )
    prepared.reference_module = prepared.quantized_module
    exports = 0

    class FailingFactory:
        def __init__(self, kind):
            self.kind = kind

        def get_graph_json(self):
            return "{}"

        def get_params(self):
            return {f"{self.kind}_weight": object()}

        def get_lib(self):
            return SimpleNamespace(type_key="llvm", imported_modules=(), get_source=lambda fmt="": "source")

        def export_library(self, path):
            nonlocal exports
            exports += 1
            Path(path).write_bytes(b"partial")
            if exports == 2:
                raise RuntimeError("synthetic export failure")

    @contextmanager
    def fake_build_config():
        yield

    factories = iter((FailingFactory("reference"), FailingFactory("mixed")))
    monkeypatch.setattr(deployment_runtime.relay, "build", lambda *args, **kwargs: next(factories))
    monkeypatch.setattr(
        deployment_runtime.relay,
        "save_param_dict",
        lambda params: b"serialized-" + next(iter(params)).encode("ascii"),
    )
    monkeypatch.setattr(deployment_runtime.vta, "build_config", fake_build_config)
    monkeypatch.setattr(deployment_runtime, "_mixed_target", object)
    monkeypatch.setattr(
        deployment_runtime.tvm.runtime,
        "load_module",
        lambda path: SimpleNamespace(
            implements_function=lambda symbol, query_imports: Path(path).parent.name == "mixed"
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic export failure"):
        deployment_runtime.build_host_artifacts(prepared, tmp_path)

    suffix = deployment_runtime.shared_library_suffix()
    assert (tmp_path / "reference" / "manifest.json").is_file()
    assert not (tmp_path / "mixed").exists()
    assert not list(tmp_path.glob(".reference.staging-*"))
    assert not list(tmp_path.glob(".mixed.staging-*"))


def test_reloaded_mixed_artifact_requires_every_deterministic_vta_symbol(deployment_runtime):
    missing = EXPECTED_VTA_SYMBOLS[-1]
    loaded = SimpleNamespace(
        implements_function=lambda symbol, query_imports: symbol != missing,
    )
    with pytest.raises(RuntimeError, match=missing):
        deployment_runtime.validate_mixed_symbols(loaded, EXPECTED_VTA_SYMBOLS)


@pytest.mark.parametrize(
    ("reference", "mixed", "message"),
    [
        (np.zeros((1, 10), dtype="float32"), np.zeros((10,), dtype="float32"), "shape"),
        (np.zeros((1, 10), dtype="float32"), np.zeros((1, 10), dtype="int8"), "dtype"),
        (
            np.zeros((1, 10), dtype="float32"),
            np.ones((1, 10), dtype="float32"),
            "elementwise",
        ),
    ],
)
def test_output_comparison_fails_loudly_for_any_difference(
    deployment_runtime, reference, mixed, message
):
    with pytest.raises(RuntimeError, match=rf"00-airplane\.png.*{message}"):
        deployment_runtime.compare_outputs(Path("00-airplane.png"), reference, mixed)


def test_profiler_validation_requires_all_three_accelerator_counters(deployment_runtime):
    valid = {name: 1 for name in REQUIRED_PROFILER_COUNTERS}
    deployment_runtime.validate_profiler_stats(valid)

    for missing_or_zero in REQUIRED_PROFILER_COUNTERS:
        invalid = dict(valid)
        invalid[missing_or_zero] = 0
        with pytest.raises(RuntimeError, match=missing_or_zero):
            deployment_runtime.validate_profiler_stats(invalid)


def test_all_reference_runs_finish_before_fsim_load_and_mixed_execution(
    deployment_runtime, monkeypatch
):
    sample_paths = _expected_sample_paths()
    reference_artifact = SimpleNamespace(module=object())
    mixed_artifact = SimpleNamespace(module=object())
    artifacts = SimpleNamespace(
        reference=reference_artifact,
        mixed=mixed_artifact,
        vta_symbols=EXPECTED_VTA_SYMBOLS,
    )
    events = []
    output_by_sample = {
        path: np.eye(10, dtype="float32")[[index]] for index, path in enumerate(sample_paths)
    }

    def fake_load_sample(path):
        events.append(("load_sample", path))
        return np.full((1, 32, 32, 3), sample_paths.index(path), dtype="float32")

    def fake_run(artifact, input_data):
        index = int(input_data[0, 0, 0, 0])
        path = sample_paths[index]
        kind = "reference" if artifact is reference_artifact else "mixed"
        events.append(("run", kind, path))
        return output_by_sample[path].copy()

    class FakeSimulator:
        def __init__(self):
            self.stats_calls = 0

        def clear_stats(self):
            events.append(("profiler_clear",))

        def stats(self):
            self.stats_calls += 1
            events.append(("profiler_stats", self.stats_calls))
            value = 0 if self.stats_calls == 1 else 1
            return {name: value for name in REQUIRED_PROFILER_COUNTERS}

    simulator = FakeSimulator()

    def fake_load_fsim():
        events.append(("load_fsim",))
        return simulator

    monkeypatch.setattr(deployment_runtime, "load_sample", fake_load_sample)
    monkeypatch.setattr(deployment_runtime, "_run_graph", fake_run)
    monkeypatch.setattr(deployment_runtime, "_load_fsim", fake_load_fsim)
    monkeypatch.setattr(deployment_runtime, "validate_mixed_symbols", lambda *args: None)

    summary = deployment_runtime.execute_samples(artifacts, sample_paths)

    reference_events = [event for event in events if event[:2] == ("run", "reference")]
    mixed_events = [event for event in events if event[:2] == ("run", "mixed")]
    assert [event[2] for event in reference_events] == list(sample_paths)
    assert [event[2] for event in mixed_events] == list(sample_paths)
    assert max(events.index(event) for event in reference_events) < events.index(("load_fsim",))
    assert events.index(("load_fsim",)) < events.index(("profiler_clear",))
    assert events.index(("profiler_clear",)) < min(events.index(event) for event in mixed_events)
    assert events.index(("profiler_stats", 1)) < min(events.index(event) for event in mixed_events)
    assert max(events.index(event) for event in mixed_events) < events.index(("profiler_stats", 2))

    assert len(summary.comparisons) == 10
    assert tuple(comparison.sample_path for comparison in summary.comparisons) == sample_paths
    assert tuple(comparison.top1 for comparison in summary.comparisons) == tuple(range(10))
    assert summary.profiler_stats == {name: 1 for name in REQUIRED_PROFILER_COUNTERS}


def test_cli_has_only_the_operational_output_directory_option(
    deployment_runtime, monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "runtime", deployment_runtime)
    run_module = _load_module(RUN_PATH, "mlperf_resnet_run")
    calls = []

    def fake_deploy(output_dir):
        calls.append(Path(output_dir))
        return SimpleNamespace(
            execution=SimpleNamespace(comparisons=tuple(range(10)), profiler_stats={})
        )

    monkeypatch.setattr(run_module, "deploy", fake_deploy)

    assert run_module.main(["--output-dir", str(tmp_path)]) == 0
    assert calls == [tmp_path]
    with pytest.raises(SystemExit):
        run_module.main(["--target", "c"])


@pytest.mark.parametrize("host_codegens", [(), ("c", "llvm"), ("llvm", "llvm"), ("llvm", "cuda")])
def test_fsim_matrix_rejects_any_host_order_before_model_preparation(
    deployment_runtime, monkeypatch, tmp_path, host_codegens
):
    monkeypatch.setattr(
        deployment_runtime,
        "prepare_model",
        lambda *_: pytest.fail("model preparation must not start for an invalid matrix"),
    )
    with pytest.raises(ValueError, match=r"exactly \('llvm', 'c'\)"):
        deployment_runtime.deploy_fsim_matrix(tmp_path, host_codegens=host_codegens)


def test_matrix_identities_and_bundle_layout_are_host_specific(deployment_runtime):
    assert deployment_runtime._artifact_identity("llvm", "reference") == "mlperf_resnet_llvm"
    assert deployment_runtime._artifact_identity("llvm", "mixed") == "mlperf_resnet_vta_llvm"
    assert deployment_runtime._artifact_identity("c", "reference") == "mlperf_resnet_c"
    assert deployment_runtime._artifact_identity("c", "mixed") == "mlperf_resnet_vta_c"
    assert deployment_runtime._matrix_artifact_root(Path("out"), "llvm") == Path("out/llvm-fsim")
    assert deployment_runtime._matrix_artifact_root(Path("out"), "c") == Path("out/c-fsim")


def test_fsim_matrix_result_records_simulator_and_is_frozen(deployment_runtime, monkeypatch, tmp_path):
    prepared = SimpleNamespace(
        routing=SimpleNamespace(symbols=(), composite_names=(), host_operator_names=())
    )
    monkeypatch.setattr(deployment_runtime, "prepare_model", lambda *_: prepared)
    monkeypatch.setattr(deployment_runtime, "committed_sample_paths", lambda: ())
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="sim"))
    monkeypatch.setattr(deployment_runtime, "build_host_artifacts", lambda *args, **kwargs: ())
    monkeypatch.setattr(deployment_runtime, "_execute_matrix", lambda *args: ())

    result = deployment_runtime.deploy_fsim_matrix(tmp_path)

    assert result.simulator == "fsim"
    with pytest.raises(FrozenInstanceError):
        result.simulator = "tsim"


def test_cli_exposes_llvm_c_and_all_matrix_modes(deployment_runtime, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "runtime", deployment_runtime)
    run_module = _load_module(RUN_PATH, "mlperf_resnet_run_matrix")
    assert run_module._parser().parse_args([]).host_codegen == "llvm"
    assert run_module._parser().parse_args(["--host-codegen", "c"]).host_codegen == "c"
    assert run_module._parser().parse_args(["--host-codegen", "all"]).host_codegen == "all"


def test_application_sources_use_only_the_approved_host_flow():
    assert RUNTIME_PATH.is_file(), f"missing Task 12 implementation: {RUNTIME_PATH}"
    assert RUN_PATH.is_file(), f"missing Task 12 implementation: {RUN_PATH}"
    artifact_path = APP_ROOT / "graph_artifacts.py"
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (RUNTIME_PATH, RUN_PATH, artifact_path)
    }
    combined = "\n".join(sources.values())
    lowered = combined.lower()
    for forbidden in [
        "tensorflow",
        "tflite_runtime",
        "autotvm",
        "graphpack",
        "relay.ext." + "vta",
        "tiny-v1.4",
        "cifar-10-batches-py",
        "download_testdata",
        "cmsis",
        "fvp",
    ]:
        assert forbidden not in lowered

    runtime_tree = ast.parse(sources["runtime.py"])
    top_level_imports = [
        node.module
        for node in runtime_tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert "vta.testing" not in top_level_imports
    assert "relay.build" in combined
    assert ".export_library" in combined
    assert "tvm.runtime.load_module" in combined
    assert "graph_executor.create" in combined
    assert 'target="llvm"' in combined


def test_end_to_end_host_fsim_deployment(deployment_runtime, tmp_path):
    result = deployment_runtime.deploy(tmp_path)

    assert result.artifacts.reference.path.is_file()
    assert result.artifacts.mixed.path.is_file()
    assert result.artifacts.vta_symbols == EXPECTED_VTA_SYMBOLS
    assert len(result.execution.comparisons) == 10
    assert tuple(item.sample_path for item in result.execution.comparisons) == _expected_sample_paths()
    for comparison in result.execution.comparisons:
        assert comparison.reference.shape == comparison.mixed.shape
        assert comparison.reference.dtype == comparison.mixed.dtype
        np.testing.assert_array_equal(comparison.reference, comparison.mixed)
        assert comparison.top1 == int(np.argmax(comparison.reference, axis=1)[0])
    for counter in REQUIRED_PROFILER_COUNTERS:
        assert result.execution.profiler_stats[counter] > 0

    for artifact in (result.artifacts.reference, result.artifacts.mixed):
        manifest = json.loads(
            (artifact.artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        llvm_sources = [
            entry
            for entry in manifest["sources"]
            if entry["source_format"] == "ll" and entry["path"]
        ]
        assert llvm_sources, f"{artifact.artifact_dir} has no LLVM IR source entry"
        assert all(
            (artifact.artifact_dir / entry["path"]).is_file()
            and (artifact.artifact_dir / entry["path"]).stat().st_size > 0
            for entry in llvm_sources
        )


def test_real_c_mixed_source_has_static_uop_and_safe_constant_contract(
    deployment_runtime, tmp_path
):
    result = deployment_runtime.deploy_fsim_matrix(tmp_path)
    c_artifacts = result.artifacts[1]
    source_paths = sorted((c_artifacts.mixed.artifact_dir / "source").glob("*.c"))
    assert source_paths
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)

    assert "VTAPushGEMMOp" in source
    assert "VTAPushALUOp" in source
    callbacks = set(re.findall(r"static int32_t (__tvm_static_init_lambda(?:_\d+)?)\(", source))
    handles = set(re.findall(r"static void\* (__tvm_static_handle(?:_\d+)?) = NULL;", source))
    assert callbacks
    assert len(callbacks) == len(handles)
    assert all(
        re.search(r"VTAPush(?:GEMM|ALU)Op\(&" + re.escape(handle), source)
        for handle in handles
    )
    assert set(re.findall(r"tvmgen_mlperf_resnet_vta_main_\d+", source)) == set(
        EXPECTED_VTA_SYMBOLS
    )
    assert re.search(r"static const (?:int8_t|int32_t).*vta_const_\d+_host", source)

    # Ext-dev allocations are opaque VTA handles.  Constants must be copied
    # through VTABufferCPUPtr; direct C stores through the handle are unsafe.
    assert not re.search(r"\(\([^)]*\*\)vta_const_\d+\)\[", source)
    assert "coproc_uop_scope" not in source
    assert len(result.executions) == 2
    assert all(len(execution.comparisons) == 10 for execution in result.executions)
    assert all(
        all(execution.profiler_stats[counter] > 0 for counter in REQUIRED_PROFILER_COUNTERS)
        for execution in result.executions
    )
