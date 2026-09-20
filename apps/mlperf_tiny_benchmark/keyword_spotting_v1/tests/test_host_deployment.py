"""HOST and FSIM deployment contracts for KWS."""

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
EXPECTED_VTA_SYMBOLS = tuple(f"tvmgen_mlperf_kws_vta_main_{i}" for i in range(4))
REQUIRED_PROFILER_COUNTERS = ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes")


@pytest.fixture(scope="module")
def deployment_runtime():
    assert RUNTIME_PATH.is_file(), f"missing Task 4 implementation: {RUNTIME_PATH}"
    spec = importlib.util.spec_from_file_location("mlperf_kws_host_runtime", RUNTIME_PATH)
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


def test_committed_sample_order_is_the_fixed_twelve_label_order(deployment_runtime):
    paths = _expected_sample_paths()
    assert len(paths) == 12
    assert len(set(paths)) == 12
    assert deployment_runtime.committed_sample_paths() == paths
    assert [record.label for record in deployment_runtime.committed_sample_records()] == list(range(12))


def test_build_exports_two_bundles_with_own_params_without_loading_fsim(
    deployment_runtime, monkeypatch, tmp_path
):
    quantized = object()
    mixed = object()
    prepared = SimpleNamespace(
        quantized_module=quantized,
        reference_module=quantized,
        mixed_module=mixed,
        imported=SimpleNamespace(model_sha256="a" * 64),
        routing=SimpleNamespace(symbols=EXPECTED_VTA_SYMBOLS),
    )
    events = []
    params = {"reference": b"reference", "mixed": b"mixed"}

    class Factory:
        def __init__(self, kind):
            self.kind = kind

        def get_graph_json(self):
            return json.dumps({"kind": self.kind})

        def get_params(self):
            events.append(("params", self.kind))
            return self.kind

        def get_lib(self):
            return SimpleNamespace(type_key="llvm", imported_modules=(), get_source=lambda fmt="": "source")

        def export_library(self, path):
            events.append(("export", self.kind))
            Path(path).write_bytes(self.kind.encode("ascii"))

    def fake_build(module, target):
        kind = "reference" if module is quantized else "mixed"
        events.append(("build", kind, target))
        return Factory(kind)

    def fake_load(path):
        kind = Path(path).parent.name
        return SimpleNamespace(
            implements_function=lambda symbol, query_imports: (
                (kind == "mixed" or kind.startswith(".mixed"))
                and symbol in EXPECTED_VTA_SYMBOLS
            )
        )

    monkeypatch.setattr(deployment_runtime.relay, "build", fake_build)
    monkeypatch.setattr(
        deployment_runtime.relay,
        "save_param_dict",
        lambda value: params["reference" if value == "reference" else "mixed"],
    )
    monkeypatch.setattr(deployment_runtime.tvm.runtime, "load_module", fake_load)
    monkeypatch.setattr(deployment_runtime, "_mixed_target", lambda codegen: ("vta", codegen))

    @contextmanager
    def fake_build_config(**kwargs):
        yield

    monkeypatch.setattr(deployment_runtime.vta, "build_config", fake_build_config)
    monkeypatch.setattr(
        deployment_runtime,
        "_load_fsim",
        lambda: pytest.fail("HOST artifact build must not load FSIM"),
    )

    artifacts = deployment_runtime.build_host_artifacts(prepared, tmp_path, "llvm", "host")
    assert artifacts.reference.params == b"reference"
    assert artifacts.mixed.params == b"mixed"
    assert artifacts.vta_symbols == EXPECTED_VTA_SYMBOLS
    assert (tmp_path / "reference" / "manifest.json").is_file()
    assert (tmp_path / "mixed" / "manifest.json").is_file()
    assert [item[:2] for item in events if item[0] == "build"] == [
        ("build", "reference"),
        ("build", "mixed"),
    ]


def test_each_graph_executor_loads_only_its_own_params(deployment_runtime, monkeypatch):
    reference_module = object()
    mixed_module = object()
    reference_params = b"reference-params"
    mixed_params = b"mixed-params"
    expected = {reference_module: reference_params, mixed_module: mixed_params}
    events = []

    class Graph:
        def __init__(self, module):
            self.module = module

        def load_params(self, value):
            assert value is expected[self.module]
            events.append(("params", self.module, value))

        def set_input(self, name, value):
            events.append(("input", self.module, name, value))

        def run(self):
            events.append(("run", self.module))

        def get_output(self, index):
            return SimpleNamespace(numpy=lambda: np.zeros((1, 12), dtype=np.int8))

    monkeypatch.setattr(
        deployment_runtime.graph_executor,
        "create",
        lambda graph, module, device: Graph(module),
    )
    input_data = np.zeros(deployment_runtime.INPUT_SHAPE, dtype=np.int8)
    for module, params in ((reference_module, reference_params), (mixed_module, mixed_params)):
        artifact = SimpleNamespace(
            graph_json="{}", module=module, device="device", params=params
        )
        output = deployment_runtime._run_graph(artifact, input_data)
        assert output.shape == deployment_runtime.OUTPUT_SHAPE
    assert [event[0] for event in events] == ["params", "input", "run", "params", "input", "run"]


@pytest.mark.parametrize(
    ("reference", "mixed", "message"),
    [
        (np.zeros((1, 12), dtype=np.int8), np.zeros((12,), dtype=np.int8), "shape"),
        (np.zeros((1, 12), dtype=np.int8), np.zeros((1, 12), dtype=np.float32), "dtype"),
        (np.zeros((1, 12), dtype=np.int8), np.ones((1, 12), dtype=np.int8), "elementwise"),
    ],
)
def test_output_comparison_rejects_contract_mismatch(deployment_runtime, reference, mixed, message):
    with pytest.raises(RuntimeError, match=rf"sample.wav.*{message}"):
        deployment_runtime.compare_outputs(Path("sample.wav"), reference, mixed)


def test_host_executes_all_samples_without_fsim(deployment_runtime, monkeypatch):
    paths = _expected_sample_paths()
    reference = SimpleNamespace()
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
    assert len(result.comparisons) == 12
    assert [item.top1 for item in result.comparisons] == list(range(12))
    assert events == list(range(12))
    assert result.profiler_stats == {}


def test_fsim_runs_reference_before_lazy_load_and_compares_twelve_samples(
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
        events.append(("run", "reference" if artifact is artifacts.reference else "mixed", index))
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
            if any(event[0] == "run" and event[1] == "mixed" for event in events):
                return {counter: index + 1 for index, counter in enumerate(profiler)}
            return dict(profiler)

    simulator = Simulator()
    monkeypatch.setattr(deployment_runtime, "_load_fsim", lambda: events.append(("load",)) or simulator)
    result = deployment_runtime.execute_fsim(artifacts, paths)

    assert len(result.comparisons) == 12
    assert result.profiler_stats == {counter: index + 1 for index, counter in enumerate(profiler)}
    assert events.index(("load",)) > max(
        index for index, event in enumerate(events) if event[:2] == ("run", "reference")
    )
    assert events.index(("clear",)) < min(
        index for index, event in enumerate(events) if event[:2] == ("run", "mixed")
    )


def test_profiler_requires_positive_gemm_weight_and_store_activity(deployment_runtime):
    valid = {counter: 1 for counter in REQUIRED_PROFILER_COUNTERS}
    deployment_runtime.validate_profiler_stats(valid)
    for counter in REQUIRED_PROFILER_COUNTERS:
        invalid = dict(valid)
        invalid[counter] = 0
        with pytest.raises(RuntimeError, match=counter):
            deployment_runtime.validate_profiler_stats(invalid)


def test_matrix_requires_ordered_llvm_c_before_preparation(deployment_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(
        deployment_runtime,
        "prepare_model",
        lambda *_: pytest.fail("invalid matrix must fail before model preparation"),
    )
    with pytest.raises(ValueError, match=r"exactly \('llvm', 'c'\)"):
        deployment_runtime.deploy_fsim_matrix(tmp_path, host_codegens=("c", "llvm"))


def test_real_prepared_graph_has_buildable_nonnegative_shift_amounts(deployment_runtime):
    from tvm import relay

    prepared = deployment_runtime.prepare_model(deployment_runtime.MODEL_PATH)
    shifts = []

    def visit(node):
        if isinstance(node, relay.Call) and getattr(node.op, "name", None) == "right_shift":
            shifts.append(int(node.args[1].data.numpy()))

    relay.analysis.post_order_visit(prepared.reference_module["main"].body, visit)
    assert shifts
    assert all(0 <= shift < 32 for shift in shifts)
