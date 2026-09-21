"""Focused TSIM target, registry, process-order, and three-sample contracts."""

import importlib.util
import os
import subprocess
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
    spec = importlib.util.spec_from_file_location("mlperf_streaming_tsim_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_tsim_mapping_requires_hardware_simulator_registries(deployment_runtime):
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
    assert "--backend tsim" in session.diagnostic


def test_cli_parser_exposes_the_approved_defaults_and_options(deployment_runtime):
    spec = importlib.util.spec_from_file_location("mlperf_streaming_run", RUN_PATH)
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


def test_tsim_rejects_wrong_target_before_model_preparation(
    deployment_runtime, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="fsim")
    )
    monkeypatch.setattr(
        deployment_runtime,
        "prepare_model",
        lambda *_: pytest.fail("model preparation must not start for a target mismatch"),
    )

    monkeypatch.setenv("VTA_BACKEND", "fsim")
    with pytest.raises(ValueError, match="backend mismatch"):
        deployment_runtime.deploy_tsim_matrix(tmp_path)


def test_tsim_missing_registry_reports_libvta_hw_requirement(deployment_runtime, monkeypatch):
    monkeypatch.setattr(
        deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim")
    )
    monkeypatch.setattr(
        deployment_runtime.tvm,
        "get_global_func",
        lambda name, allow_missing=False: None if name == "vta.tsim.init" else object(),
    )

    with pytest.raises(RuntimeError, match=r"vta\.tsim\.init.*backend tsim"):
        deployment_runtime._simulator_session("tsim").load()


def test_tsim_simulator_is_not_initialized_when_runtime_is_imported_in_a_fresh_process():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(APP_ROOT),
            str(APP_ROOT.parents[3] / "tvm" / "python"),
            str(APP_ROOT.parents[3] / "vta" / "python"),
        )
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import runtime; "
            "assert 'vta.testing.simulator' not in sys.modules",
        ],
        cwd=APP_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr


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


def test_tsim_matrix_loads_once_after_both_host_variants_and_runs_three_samples(
    deployment_runtime, monkeypatch, tmp_path
):
    sample_paths = deployment_runtime.committed_sample_paths()
    prepared = SimpleNamespace(routing=SimpleNamespace(symbols=()))
    events = []
    artifacts = []

    monkeypatch.setattr(deployment_runtime, "prepare_model", lambda *_: prepared)
    monkeypatch.setattr(
        deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim")
    )

    def fake_build(*args, **kwargs):
        host_codegen = args[2]
        events.extend((("build", host_codegen), ("export", host_codegen), ("reload", host_codegen)))
        artifact = SimpleNamespace(
            host_codegen=host_codegen,
            mixed=SimpleNamespace(module=(host_codegen, "mixed")),
            reference=SimpleNamespace(module=(host_codegen, "reference")),
            vta_symbols=(),
        )
        artifacts.append(artifact)
        return artifact

    monkeypatch.setattr(deployment_runtime, "build_host_artifacts", fake_build)
    monkeypatch.setattr(deployment_runtime, "load_sample", lambda _path: np.zeros(
        deployment_runtime.INPUT_SHAPE, dtype=np.int8
    ))
    monkeypatch.setattr(deployment_runtime, "validate_mixed_symbols", lambda *_: None)

    class FakeSimulator:
        def __init__(self):
            self.after_clear = False
            self.mixed_runs = 0

        def clear_stats(self):
            events.append(("clear",))
            self.after_clear = True
            self.mixed_runs = 0

        def stats(self):
            if self.after_clear and self.mixed_runs == 0:
                events.append(("stats", "zero"))
                return {"cycle_count": 0}
            events.append(("stats", "positive"))
            return {"cycle_count": 1}

    simulator = FakeSimulator()
    session = deployment_runtime._simulator_session("tsim")
    monkeypatch.setattr(
        deployment_runtime,
        "_load_simulator",
        lambda label: (events.append(("load", label)) or (session, simulator)),
    )

    def fake_run(artifact, _input):
        host_codegen, role = artifact.module
        if role == "mixed":
            simulator.mixed_runs += 1
            events.append(("mixed", host_codegen))
        return np.zeros(deployment_runtime.OUTPUT_SHAPE, dtype=np.int8)

    monkeypatch.setattr(deployment_runtime, "_run_graph", fake_run)

    result = deployment_runtime.deploy_tsim_matrix(tmp_path)

    assert [event for event in events if event[0] == "build"] == [
        ("build", "llvm"),
        ("build", "c"),
    ]
    assert [event for event in events if event[0] == "load"] == [("load", "tsim")]
    load_index = events.index(("load", "tsim"))
    assert all(
        events.index(event) < load_index
        for event in events
        if event[0] in {"build", "export", "reload"}
    )
    assert len(result.artifacts) == 2
    assert all(len(execution.comparisons) == 3 for execution in result.executions)
    assert all(execution.profiler_stats == {"cycle_count": 1} for execution in result.executions)
    assert [event for event in events if event[0] == "mixed"] == [
        ("mixed", "llvm"),
        ("mixed", "llvm"),
        ("mixed", "llvm"),
        ("mixed", "c"),
        ("mixed", "c"),
        ("mixed", "c"),
    ]
