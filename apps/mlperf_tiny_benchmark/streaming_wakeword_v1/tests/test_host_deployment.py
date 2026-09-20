"""HOST and FSIM deployment contracts for streaming wakeword v1."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = APP_ROOT / "runtime.py"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
EXPECTED_VTA_SYMBOLS = ("tvmgen_mlperf_streaming_wakeword_vta_main_0",)
REQUIRED_PROFILER_COUNTERS = ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes")


@pytest.fixture(scope="module")
def deployment_runtime():
    assert RUNTIME_PATH.is_file(), f"missing Task 3.2 implementation: {RUNTIME_PATH}"
    spec = importlib.util.spec_from_file_location("mlperf_streaming_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _expected_sample_paths():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return tuple(MANIFEST_PATH.parent / item["filename"] for item in manifest["samples"])


def test_committed_sample_order_is_the_fixed_three_class_order(deployment_runtime):
    paths = _expected_sample_paths()
    assert len(paths) == 3
    assert deployment_runtime.committed_sample_paths() == paths
    assert [record.label_name for record in deployment_runtime.committed_sample_records()] == [
        "Marvin",
        "Silence",
        "Unknown",
    ]


def test_build_exports_reference_and_mixed_without_loading_fsim(
    deployment_runtime, monkeypatch, tmp_path
):
    reference_module = object()
    mixed_module = object()
    prepared = SimpleNamespace(
        reference_module=reference_module,
        mixed_module=mixed_module,
        imported=SimpleNamespace(model_sha256="a" * 64),
        routing=SimpleNamespace(symbols=EXPECTED_VTA_SYMBOLS),
    )
    events = []

    class Factory:
        def __init__(self, kind):
            self.kind = kind

    def fake_build(module, target):
        kind = "reference" if module is reference_module else "mixed"
        events.append(("build", kind, target))
        return Factory(kind)

    def fake_export(factory, root, relative, **kwargs):
        events.append(("export", factory.kind, relative, kwargs))
        return SimpleNamespace(
            library_path=Path(root) / relative / "model.so",
            artifact_dir=Path(root) / relative,
            graph_json="{}",
            params=factory.kind.encode(),
            module=object(),
        )

    monkeypatch.setattr(deployment_runtime.relay, "build", fake_build)
    monkeypatch.setattr(deployment_runtime, "export_graph_bundle", fake_export)
    monkeypatch.setattr(
        deployment_runtime,
        "_mixed_build_plan",
        lambda module, codegen: SimpleNamespace(module=module, targets=("vta", codegen)),
    )
    monkeypatch.setattr(deployment_runtime.tvm, "cpu", lambda index: "cpu")
    monkeypatch.setattr(deployment_runtime.tvm, "ext_dev", lambda index: "ext")

    @contextmanager
    def fake_build_config(**kwargs):
        yield

    monkeypatch.setattr(deployment_runtime.vta, "build_config", fake_build_config)
    monkeypatch.setattr(
        deployment_runtime,
        "_load_fsim",
        lambda: pytest.fail("artifact build must not load FSIM"),
    )

    artifacts = deployment_runtime.build_host_artifacts(prepared, tmp_path, "llvm", "host")
    assert artifacts.host_codegen == "llvm"
    assert artifacts.vta_symbols == EXPECTED_VTA_SYMBOLS
    exports = [event for event in events if event[0] == "export"]
    assert [event[2] for event in exports] == ["reference", "mixed"]
    assert exports[0][3]["forbidden_vta_symbols"] == EXPECTED_VTA_SYMBOLS
    assert exports[1][3]["expected_vta_symbols"] == EXPECTED_VTA_SYMBOLS


@pytest.mark.parametrize("relative_root", (".envs", ".envs/nested"))
def test_build_rejects_environment_output_root_before_compile(
    deployment_runtime, monkeypatch, tmp_path, relative_root
):
    prepared = SimpleNamespace(
        reference_module=object(),
        mixed_module=object(),
        routing=SimpleNamespace(symbols=EXPECTED_VTA_SYMBOLS),
    )
    compile_calls = []

    monkeypatch.setattr(
        deployment_runtime.relay,
        "build",
        lambda *args, **kwargs: compile_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=r"\.envs"):
        deployment_runtime.build_host_artifacts(
            prepared, tmp_path / relative_root, "llvm", "host"
        )

    assert compile_calls == []
    assert not (tmp_path / relative_root).exists()


def test_run_graph_rejects_input_and_output_contract_mismatches(deployment_runtime, monkeypatch):
    class Graph:
        def load_params(self, params):
            pass

        def set_input(self, name, value):
            pass

        def run(self):
            pass

        def get_output(self, index):
            return SimpleNamespace(numpy=lambda: np.zeros((1, 3), dtype=np.int8))

    monkeypatch.setattr(
        deployment_runtime.graph_executor,
        "create",
        lambda graph, module, device: Graph(),
    )
    artifact = SimpleNamespace(graph_json="{}", module=object(), device="cpu", params=b"params")
    with pytest.raises(RuntimeError, match="input must have shape"):
        deployment_runtime._run_graph(
            artifact, np.zeros((1, 30, 1, 40), dtype=np.uint8)
        )


@pytest.mark.parametrize(
    ("reference", "mixed", "message"),
    [
        (np.zeros((1, 3), dtype=np.int8), np.zeros((3,), dtype=np.int8), "shape"),
        (np.zeros((1, 3), dtype=np.int8), np.zeros((1, 3), dtype=np.float32), "dtype"),
        (np.zeros((1, 3), dtype=np.int8), np.ones((1, 3), dtype=np.int8), "elementwise"),
    ],
)
def test_output_comparison_rejects_shape_dtype_and_value_mismatch(
    deployment_runtime, reference, mixed, message
):
    with pytest.raises(RuntimeError, match=rf"sample.wav.*{message}"):
        deployment_runtime.compare_outputs(Path("sample.wav"), reference, mixed)


def test_host_executes_exactly_three_reference_samples_without_fsim(
    deployment_runtime, monkeypatch
):
    paths = _expected_sample_paths()
    reference = object()
    artifacts = SimpleNamespace(reference=reference, host_codegen="llvm")
    events = []

    monkeypatch.setattr(
        deployment_runtime,
        "load_sample",
        lambda path: np.full(deployment_runtime.INPUT_SHAPE, paths.index(path), dtype=np.int8),
    )

    def fake_run(artifact, value):
        assert artifact is reference
        index = int(value.flat[0])
        events.append(index)
        result = np.zeros(deployment_runtime.OUTPUT_SHAPE, dtype=np.int8)
        result[0, index] = 10
        return result

    monkeypatch.setattr(deployment_runtime, "_run_graph", fake_run)
    monkeypatch.setattr(
        deployment_runtime,
        "_load_fsim",
        lambda: pytest.fail("HOST execution must not load FSIM"),
    )
    result = deployment_runtime.execute_host(artifacts, paths)
    assert len(result.comparisons) == 3
    assert [item.top1 for item in result.comparisons] == [0, 1, 2]
    assert [item.mixed for item in result.comparisons] == [None] * 3
    assert events == [0, 1, 2]
    assert result.profiler_stats == {}


def test_fsim_runs_reference_first_then_three_mixed_comparisons_with_activity(
    deployment_runtime, monkeypatch
):
    paths = _expected_sample_paths()
    reference = object()
    mixed = object()
    artifacts = SimpleNamespace(
        reference=SimpleNamespace(module=reference),
        mixed=SimpleNamespace(module=mixed),
        vta_symbols=EXPECTED_VTA_SYMBOLS,
        host_codegen="llvm",
    )
    events = []

    monkeypatch.setattr(
        deployment_runtime,
        "load_sample",
        lambda path: np.full(deployment_runtime.INPUT_SHAPE, paths.index(path), dtype=np.int8),
    )
    monkeypatch.setattr(deployment_runtime, "validate_mixed_symbols", lambda *args: None)

    def fake_run(artifact, value):
        index = int(value.flat[0])
        kind = "reference" if artifact is artifacts.reference else "mixed"
        events.append(("run", kind, index))
        output = np.zeros(deployment_runtime.OUTPUT_SHAPE, dtype=np.int8)
        output[0, index] = 10
        return output

    monkeypatch.setattr(deployment_runtime, "_run_graph", fake_run)
    profiler = {counter: 0 for counter in REQUIRED_PROFILER_COUNTERS}

    class Simulator:
        def clear_stats(self):
            events.append(("clear",))
            for counter in profiler:
                profiler[counter] = 0

        def stats(self):
            events.append(("stats",))
            if any(event[:2] == ("run", "mixed") for event in events):
                return {counter: index + 1 for index, counter in enumerate(profiler)}
            return dict(profiler)

    simulator = Simulator()
    monkeypatch.setattr(deployment_runtime, "_load_fsim", lambda: events.append(("load",)) or simulator)
    result = deployment_runtime.execute_fsim(artifacts, paths)

    assert len(result.comparisons) == 3
    assert result.profiler_stats == {counter: index + 1 for index, counter in enumerate(profiler)}
    assert events.index(("load",)) > max(
        index for index, event in enumerate(events) if event[:2] == ("run", "reference")
    )
    assert events.index(("clear",)) < min(
        index for index, event in enumerate(events) if event[:2] == ("run", "mixed")
    )


def test_fsim_rejects_missing_or_nonpositive_activity(deployment_runtime):
    valid = {counter: 1 for counter in REQUIRED_PROFILER_COUNTERS}
    deployment_runtime.validate_profiler_stats(valid)
    for counter in REQUIRED_PROFILER_COUNTERS:
        invalid = dict(valid)
        invalid[counter] = 0
        with pytest.raises(RuntimeError, match=counter):
            deployment_runtime.validate_profiler_stats(invalid)


def test_mixed_symbol_contract_is_validated_before_execution(deployment_runtime):
    class Module:
        def implements_function(self, symbol, query_imports):
            return False

    with pytest.raises(RuntimeError, match="missing VTA symbols"):
        deployment_runtime.validate_mixed_symbols(Module(), EXPECTED_VTA_SYMBOLS)


def test_matrix_requires_ordered_llvm_c_before_model_preparation(deployment_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(
        deployment_runtime,
        "prepare_model",
        lambda *_: pytest.fail("invalid matrix must fail before model preparation"),
    )
    with pytest.raises(ValueError, match=r"exactly \('llvm', 'c'\)"):
        deployment_runtime.deploy_fsim_matrix(tmp_path, host_codegens=("c", "llvm"))
