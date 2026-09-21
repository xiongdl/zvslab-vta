"""TSIM session and ten-sample matrix contracts for anomaly detection."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


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
    assert "--backend tsim" in session.diagnostic


def test_tsim_rejects_wrong_environment_before_model_preparation(
    runtime_module, monkeypatch, tmp_path
):
    monkeypatch.setattr(runtime_module.vta, "get_env", lambda: SimpleNamespace(TARGET="sim"))
    monkeypatch.setattr(
        runtime_module,
        "prepare_model",
        lambda *_: pytest.fail("model preparation must not start for a target mismatch"),
    )

    monkeypatch.setenv("VTA_BACKEND", "fsim")
    with pytest.raises(ValueError, match="backend mismatch"):
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


def test_tsim_window_budget_defaults_to_one_and_rejects_invalid_values(runtime_module, monkeypatch):
    monkeypatch.delenv(runtime_module.TSIM_WINDOW_BUDGET_ENV, raising=False)
    assert runtime_module.resolve_tsim_window_budget() == 1
    assert runtime_module.resolve_tsim_window_budget(3) == 3

    monkeypatch.setenv(runtime_module.TSIM_WINDOW_BUDGET_ENV, "4")
    assert runtime_module.resolve_tsim_window_budget() == 4

    for value in (0, -1, True, 1.5, "not-an-int"):
        with pytest.raises(ValueError, match="positive integer"):
            runtime_module.resolve_tsim_window_budget(value)


def test_tsim_reference_records_total_and_executed_window_contract(
    runtime_module, monkeypatch
):
    records = runtime_module.committed_sample_records()[:1]
    features = np.arange(4 * 640, dtype="float32").reshape(4, 640)
    monkeypatch.setattr(runtime_module, "load_sample", lambda path: features)
    monkeypatch.setattr(
        runtime_module,
        "_score_feature_matrix",
        lambda artifact, selected, reuse_executor=False: (0.25, np.dtype("float32")),
    )

    raw = runtime_module._reference_raw(SimpleNamespace(reference=object()), records, window_budget=1)

    assert raw[0]["feature_shape"] == (4, 640)
    assert raw[0]["executed_feature_shape"] == (1, 640)
    assert raw[0]["total_window_count"] == 4
    assert raw[0]["executed_window_count"] == 1
    assert raw[0]["sampled"] is True
    assert raw[0]["score_scope"] == "representative_windows"
    assert raw[0]["_executed_features"].shape == (1, 640)
    samples, threshold = runtime_module._with_predictions(raw)
    summary = runtime_module._summary(samples, threshold, tsim_window_budget=1)
    assert samples[0].feature_shape == (4, 640)
    assert samples[0].executed_feature_shape == (1, 640)
    assert samples[0].total_window_count == 4
    assert samples[0].executed_window_count == 1
    assert samples[0].sampled is True
    assert summary["score_scope"] == "representative_windows"
    assert summary["sampled"] is True


def test_tsim_budget_can_cover_all_windows_without_claiming_sampling(runtime_module, monkeypatch):
    records = runtime_module.committed_sample_records()[:1]
    features = np.zeros((2, 640), dtype="float32")
    monkeypatch.setattr(runtime_module, "load_sample", lambda path: features)
    monkeypatch.setattr(
        runtime_module,
        "_score_feature_matrix",
        lambda artifact, selected, reuse_executor=False: (0.5, np.dtype("float32")),
    )

    raw = runtime_module._reference_raw(SimpleNamespace(reference=object()), records, window_budget=8)

    assert raw[0]["executed_window_count"] == 2
    assert raw[0]["sampled"] is False
    assert raw[0]["score_scope"] == "full_windows"


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
    monkeypatch.setattr(runtime_module, "_reference_raw", lambda *args, **kwargs: ())
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
    assert run_module._parser().parse_args(["--tsim-window-budget", "3"]).tsim_window_budget == 3
    with pytest.raises(SystemExit):
        run_module._parser().parse_args(["--tsim-window-budget", "0"])


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


@pytest.mark.skipif(
    os.environ.get("ANOMALY_TSIM_RUN_INTEGRATION") != "1",
    reason="run the real TSIM matrix explicitly with ANOMALY_TSIM_RUN_INTEGRATION=1",
)
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
        assert execution.summary["tsim_window_budget"] == 1
        assert execution.summary["score_scope"] == "representative_windows"
        assert execution.summary["sampled"] is True
        assert all(item.total_window_count > item.executed_window_count for item in execution.samples)
        assert all(item.sampled is True for item in execution.samples)
        assert isinstance(execution.profiler_stats["cycle_count"], int)
        assert execution.profiler_stats["cycle_count"] > 0
