"""Focused TSIM registry, reset, and matrix contracts for KWS."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = APP_ROOT / "runtime.py"
RUN_PATH = APP_ROOT / "run.py"


@pytest.fixture(scope="module")
def deployment_runtime():
    spec = importlib.util.spec_from_file_location("mlperf_kws_tsim_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_tsim_mapping_is_explicit_and_requires_hardware_library(deployment_runtime):
    session = deployment_runtime._simulator_session("tsim")
    assert session.environment_target == "tsim"
    assert session.clear_registry == "vta.tsim.profiler_clear"
    assert session.status_registry == "vta.tsim.profiler_status"
    assert session.required_registries == (
        "vta.tsim.init",
        "vta.tsim.profiler_clear",
        "vta.tsim.profiler_status",
        "runtime.module.loadfile_vta-tsim",
    )
    assert session.activity_counter == "cycle_count"
    assert "--target libvta_hw" in session.diagnostic


def test_cli_exposes_host_codegen_simulator_and_output_directory_options(deployment_runtime):
    assert RUN_PATH.is_file(), f"missing Task 6 CLI: {RUN_PATH}"
    spec = importlib.util.spec_from_file_location("mlperf_kws_run", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    defaults = module._parser().parse_args([])
    assert defaults.output_dir == str(deployment_runtime.DEFAULT_OUTPUT_DIR)
    assert defaults.host_codegen == "llvm"
    assert defaults.simulator == "fsim"
    assert module._parser().parse_args(["--output-dir", "out"]).output_dir == "out"
    assert module._parser().parse_args(["--host-codegen", "c"]).host_codegen == "c"
    assert module._parser().parse_args(["--host-codegen", "all"]).host_codegen == "all"
    assert module._parser().parse_args(["--simulator", "host"]).simulator == "host"
    assert module._parser().parse_args(["--simulator", "tsim"]).simulator == "tsim"


@pytest.mark.parametrize(
    ("prefix", "mixed", "mixed_top1"),
    [
        ("llvm-host", None, "not-run"),
        ("llvm-fsim", np.array([[0, 0, 0, 5]], dtype=np.int8), "3"),
        ("llvm-tsim", np.array([[0, 0, 0, 5]], dtype=np.int8), "3"),
    ],
)
def test_cli_prints_deterministic_per_sample_top1_results(
    deployment_runtime, capsys, prefix, mixed, mixed_top1
):
    spec = importlib.util.spec_from_file_location("mlperf_kws_run_output", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    comparison = SimpleNamespace(
        sample_path=Path("samples/go-004ae714_nohash_0.wav"),
        top1=1,
        mixed=mixed,
    )
    execution = SimpleNamespace(comparisons=(comparison,), profiler_stats={"cycles": 7})

    module._print_execution(prefix, execution)

    assert capsys.readouterr().out.splitlines() == [
        f"{prefix} compared samples: 1",
        f"{prefix} sample: go-004ae714_nohash_0.wav reference top-1: 1 mixed top-1: {mixed_top1}",
        f"{prefix} profiler: {{'cycles': 7}}",
    ]


def test_tsim_rejects_wrong_target_before_model_preparation(deployment_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="sim"))
    monkeypatch.setattr(
        deployment_runtime,
        "prepare_model",
        lambda *_: pytest.fail("model preparation must not start for a target mismatch"),
    )
    with pytest.raises(RuntimeError, match="requires VTA target 'tsim'"):
        deployment_runtime.deploy_tsim_matrix(tmp_path)


def test_tsim_missing_registry_reports_libvta_hw_requirement(deployment_runtime, monkeypatch):
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim"))
    monkeypatch.setattr(
        deployment_runtime.tvm,
        "get_global_func",
        lambda name, allow_missing=False: None if name == "vta.tsim.init" else object(),
    )
    with pytest.raises(RuntimeError, match=r"vta\.tsim\.init.*libvta_hw"):
        deployment_runtime._simulator_session("tsim").load()


def test_tsim_requires_exact_zero_reset_and_positive_integer_cycles(deployment_runtime):
    session = deployment_runtime._simulator_session("tsim")
    assert session.read_stats(lambda: '{"cycle_count": 0}') == {"cycle_count": 0}
    session.validate_activity({"cycle_count": 1})

    with pytest.raises(RuntimeError, match="cycle_count"):
        session.clear_and_validate(
            SimpleNamespace(clear_stats=lambda: None, stats=lambda: {"cycle_count": 1})
        )
    for value in (0, -1, True, 1.5, "1"):
        with pytest.raises(RuntimeError, match="positive integer"):
            session.validate_activity({"cycle_count": value})


def test_tsim_matrix_loads_once_after_both_host_artifacts_and_runs_twelve_samples(
    deployment_runtime, monkeypatch, tmp_path
):
    sample_paths = tuple(Path(f"sample-{index}.wav") for index in range(12))
    prepared = SimpleNamespace(
        routing=SimpleNamespace(symbols=(), composite_names=(), host_operator_names=())
    )
    events = []
    artifacts = []

    monkeypatch.setattr(deployment_runtime, "prepare_model", lambda *_: prepared)
    monkeypatch.setattr(deployment_runtime, "committed_sample_paths", lambda: sample_paths)
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim"))

    def fake_build(*args, **kwargs):
        host = kwargs["host_codegen"]
        events.extend((("build", host), ("export", host), ("reload", host)))
        artifact = SimpleNamespace(
            host_codegen=host,
            mixed=SimpleNamespace(module=(host, "mixed")),
            reference=SimpleNamespace(module=(host, "reference")),
            vta_symbols=prepared.routing.symbols,
        )
        artifacts.append(artifact)
        return artifact

    monkeypatch.setattr(deployment_runtime, "build_host_artifacts", fake_build)
    monkeypatch.setattr(deployment_runtime, "load_sample", lambda _path: object())

    class FakeSimulator:
        def __init__(self):
            self.reset = False

        def clear_stats(self):
            host = ("llvm", "c")[len([event for event in events if event[0] == "clear"])]
            events.append(("clear", host))
            self.reset = True

        def stats(self):
            if self.reset:
                self.reset = False
                events.append(("stats", "zero"))
                return {"cycle_count": 0}
            events.append(("stats", "positive"))
            return {"cycle_count": 1}

    simulator = FakeSimulator()
    monkeypatch.setattr(
        deployment_runtime,
        "_load_simulator",
        lambda label: (events.append(("load", label)) or deployment_runtime._simulator_session(label), simulator),
    )
    monkeypatch.setattr(deployment_runtime, "validate_mixed_symbols", lambda *_: None)

    def fake_run(artifact, _input):
        host, role = artifact.module
        if role == "mixed":
            events.append(("mixed", host))
        return np.zeros(deployment_runtime.OUTPUT_SHAPE, dtype=deployment_runtime.OUTPUT_DTYPE)

    monkeypatch.setattr(deployment_runtime, "_run_graph", fake_run)

    result = deployment_runtime.deploy_tsim_matrix(tmp_path)

    assert [event for event in events if event[0] == "build"] == [("build", "llvm"), ("build", "c")]
    assert [event for event in events if event[0] == "load"] == [("load", "tsim")]
    load_index = events.index(("load", "tsim"))
    assert all(events.index(event) < load_index for event in events if event[0] in {"build", "export", "reload"})
    assert len(result.artifacts) == 2
    assert all(len(execution.comparisons) == 12 for execution in result.executions)
    assert all(execution.profiler_stats == {"cycle_count": 1} for execution in result.executions)
