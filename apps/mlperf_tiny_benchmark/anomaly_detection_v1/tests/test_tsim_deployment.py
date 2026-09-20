"""TSIM session and ten-sample matrix contracts for anomaly detection."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime_module():
    spec = importlib.util.spec_from_file_location(
        "mlperf_anomaly_tsim_runtime", APP_ROOT / "runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_tsim_mapping_is_explicit_and_uses_hardware_simulator(runtime_module):
    session = runtime_module._simulator_session("tsim")

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


def test_tsim_rejects_wrong_environment_before_model_preparation(
    runtime_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(runtime_module.vta, "get_env", lambda: SimpleNamespace(TARGET="sim"))
    monkeypatch.setattr(
        runtime_module,
        "prepare_model",
        lambda *_: pytest.fail("model preparation must not start for a target mismatch"),
    )

    with pytest.raises(RuntimeError, match="requires VTA target 'tsim'"):
        runtime_module.deploy_tsim_matrix(tmp_path)


def test_tsim_stats_require_zero_reset_and_positive_integer_cycles(runtime_module):
    session = runtime_module._simulator_session("tsim")

    assert session.read_stats(lambda: '{"cycle_count": 0}') == {"cycle_count": 0}
    session.validate_activity({"cycle_count": 1})
    with pytest.raises(RuntimeError, match="cycle_count"):
        session.clear_and_validate(
            SimpleNamespace(clear_stats=lambda: None, stats=lambda: {"cycle_count": 1})
        )
    for value in (0, -1, True, 1.5, "1"):
        with pytest.raises(RuntimeError, match="positive integer"):
            session.validate_activity({"cycle_count": value})


def test_tsim_matrix_builds_both_hosts_before_lazy_load(runtime_module, monkeypatch, tmp_path):
    prepared = SimpleNamespace(routing=SimpleNamespace(symbols=()))
    events = []

    monkeypatch.setattr(runtime_module.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim"))
    monkeypatch.setattr(runtime_module, "prepare_model", lambda *_: prepared)
    monkeypatch.setattr(runtime_module, "committed_sample_records", lambda *_: ())

    def fake_build(*args, **kwargs):
        events.append(("build", args[2], args[3]))
        return SimpleNamespace(
            host_codegen=args[2],
            reference=SimpleNamespace(),
            mixed=SimpleNamespace(module=object()),
            vta_symbols=(),
        )

    monkeypatch.setattr(runtime_module, "build_host_artifacts", fake_build)
    monkeypatch.setattr(runtime_module, "_reference_raw", lambda *_: ())
    monkeypatch.setattr(
        runtime_module,
        "_load_simulator",
        lambda label: (events.append(("load", label)) or (object(), object())),
    )
    monkeypatch.setattr(runtime_module, "_execute_tsim_matrix", lambda *args: ())

    runtime_module.deploy_tsim_matrix(tmp_path)

    assert events == [("build", "llvm", "tsim"), ("build", "c", "tsim"), ("load", "tsim")]


def test_tsim_cli_accepts_all_host_codegen(runtime_module):
    run_spec = importlib.util.spec_from_file_location("mlperf_anomaly_run", APP_ROOT / "run.py")
    run_module = importlib.util.module_from_spec(run_spec)
    sys.modules[run_spec.name] = run_module
    sys.path.insert(0, str(APP_ROOT))
    try:
        run_spec.loader.exec_module(run_module)
    finally:
        sys.path.pop(0)

    args = run_module._parser().parse_args(["--simulator", "tsim", "--host-codegen", "all"])
    assert args.simulator == "tsim"
    assert args.host_codegen == "all"


def test_tsim_cli_dispatches_matrix(runtime_module, monkeypatch, capsys):
    run_spec = importlib.util.spec_from_file_location("mlperf_anomaly_run_dispatch", APP_ROOT / "run.py")
    run_module = importlib.util.module_from_spec(run_spec)
    sys.modules[run_spec.name] = run_module
    sys.path.insert(0, str(APP_ROOT))
    try:
        run_spec.loader.exec_module(run_module)
    finally:
        sys.path.pop(0)

    calls = []
    monkeypatch.setattr(
        run_module.runtime,
        "deploy_matrix",
        lambda *args: calls.append(args) or (SimpleNamespace(), (), ()),
    )

    assert run_module.main(["--simulator", "tsim", "--host-codegen", "all"]) == 0
    assert calls and calls[0][1] == "tsim"
    assert "reserved" not in capsys.readouterr().err


def test_end_to_end_tsim_matrix_contract_has_ten_five_five_results(runtime_module):
    result = runtime_module.deploy_tsim_matrix(APP_ROOT / "build" / "test-tsim")

    assert len(result[1]) == 2
    assert len(result[2]) == 2
    for execution in result[2]:
        assert len(execution.samples) == 10
        assert [item.label for item in execution.samples] == [0] * 5 + [1] * 5
        assert all(item.input_shape == (1, 640) for item in execution.samples)
        assert all(item.output_shape == (1, 640) for item in execution.samples)
        assert all(item.output_dtype == "float32" for item in execution.samples)
        assert all(item.mixed_score is not None for item in execution.samples)
        assert execution.summary["normal_count"] == 5
        assert execution.summary["anomaly_count"] == 5
        assert isinstance(execution.profiler_stats["cycle_count"], int)
        assert execution.profiler_stats["cycle_count"] > 0
