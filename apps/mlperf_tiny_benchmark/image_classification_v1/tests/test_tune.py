"""Focused tests for the IC V1 single-workload AutoTVM command."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TUNE_PATH = Path(__file__).resolve().parents[1] / "tune.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("ic_v1_tune", TUNE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tune():
    return _load_tune()


def test_select_workload_uses_zero_based_extraction_order(tune):
    tasks = [SimpleNamespace(name=f"task-{index}") for index in range(3)]

    assert tune.select_workload(tasks, 0) is tasks[0]
    assert tune.select_workload(tasks, 2) is tasks[2]


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_invalid_workload_index_reports_valid_range(tune, index):
    tasks = [SimpleNamespace(name="first"), SimpleNamespace(name="second")]

    with pytest.raises(ValueError, match=r"valid workload indices are 0\.\.1"):
        tune.select_workload(tasks, index)


def test_tuning_defaults_are_random_fsim_local_32_trials_and_120_seconds(tune):
    options = tune.build_tuning_options()

    assert options == {
        "tuner": "random",
        "backend": "fsim",
        "runner": "local",
        "trials": 32,
        "timeout": 120,
    }


def test_invalid_index_fails_before_runner_creation_or_output_write(tune, monkeypatch, tmp_path):
    tasks = [SimpleNamespace(name="only-task")]
    identity = SimpleNamespace()
    monkeypatch.setattr(
        tune, "prepare_v1_workloads", lambda: (SimpleNamespace(), [identity], tasks)
    )
    monkeypatch.setattr(tune.shared, "_config_identity", lambda _path: (Path("config"), "a" * 64))
    monkeypatch.setattr(tune.shared, "create_runner", lambda *_a, **_k: pytest.fail("runner created"))
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="valid workload indices"):
        tune.run_tuning(1, output_dir=output_dir)

    assert not output_dir.exists()


def test_tuning_validates_active_alternate_geometry_before_workload_selection(
    tune, monkeypatch, tmp_path
):
    config_path = tmp_path / "alternate-vta-geometry.json"
    config_path.write_text(
        '{"LOG_BATCH": 0, "LOG_BLOCK": 3, "LOG_OUT_WIDTH": 3, "LOG_OUT_HEIGHT": 3}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config_path))
    monkeypatch.setattr(
        tune,
        "prepare_v1_workloads",
        lambda: (SimpleNamespace(), [SimpleNamespace()], [SimpleNamespace(name="only-task")]),
    )

    with pytest.raises(ValueError, match="valid workload indices"):
        tune.run_tuning(1, output_dir=tmp_path / "out")


def test_tsim_measures_best_fsim_record_config(tune, monkeypatch, tmp_path):
    identity = tune.fused.FusedConvIdentity(
        ("conv2d_packed.vta", ("TENSOR", (1, 2, 32, 32, 1, 8), "int8"),
         ("TENSOR", (2, 2, 3, 3, 8, 8), "int8"), (1, 1), (1, 1, 1, 1), (1, 1),
         "NCHW1n8c", "int32"),
        64, 7, -127, 127, "int8", "tvmgen_test_vta_main_0", 0,
    )
    workload = (tune.fused.TASK_NAME, *identity.task_args())
    selected_config = SimpleNamespace(to_json_dict=lambda: {"index": 1})
    task = SimpleNamespace(name=tune.fused.TASK_NAME, workload=workload,
                           config_space=[0, 1, 2], flop=2048)
    target = SimpleNamespace(kind=SimpleNamespace(name="vta"))
    measure_input = SimpleNamespace(config=selected_config, task=task, target=target)
    fsim_result = SimpleNamespace(error_no=0, costs=(0.001,))
    tsim_result = SimpleNamespace(error_no=0, costs=(7654,))
    prepared = SimpleNamespace()
    monkeypatch.setattr(tune, "prepare_v1_workloads", lambda: (prepared, [identity], [task]))
    monkeypatch.setattr(tune.shared, "_config_identity", lambda _path: (Path("geometry.json"), "g" * 64))
    monkeypatch.setattr(tune.shared, "_sha256_file", lambda _path: "m" * 64)
    monkeypatch.setattr(tune.shared, "_task_workload_id", lambda _task: "a" * 64)
    monkeypatch.setattr(tune.fused, "lower_with_fused_config", lambda *_: SimpleNamespace(schedule=object()))
    monkeypatch.setattr(tune.shared.autotvm.record, "pick_best", lambda _src, _dst: None)
    monkeypatch.setattr(
        tune.shared.autotvm.record,
        "load_from_file",
        lambda _path: iter([(measure_input, fsim_result)]),
    )
    calls = []
    fsim_runners = []

    class FakeTuner:
        def __init__(self, actual_task):
            assert actual_task is task

        def tune(self, **kwargs):
            calls.append(("fsim", kwargs["n_trial"], kwargs["measure_option"]["runner"].timeout))

    class FakeBuilder:
        def set_task(self, actual_task, build_kwargs):
            assert actual_task is task
            assert build_kwargs == {}

        def build(self, inputs):
            assert inputs == [measure_input]
            return [SimpleNamespace(error=None, filename="best.so")]

        def __del__(self):
            pass

    class FakeRunner:
        def __init__(self):
            self.timeout = 120
            self.server = None
            self.tracker = None

        def set_task(self, actual_task):
            assert actual_task is task

        def get_build_kwargs(self):
            return {}

        def run(self, inputs, build_results):
            assert inputs == [measure_input]
            assert build_results[0].filename == "best.so"
            calls.append(("tsim", inputs[0].config))
            return [tsim_result]

    monkeypatch.setattr(tune.shared.autotvm.tuner, "RandomTuner", FakeTuner)
    def make_measure_option(_backend, **_opts):
        runner = SimpleNamespace(timeout=120, server=None, tracker=None)
        fsim_runners.append(runner)
        return {"runner": runner}

    monkeypatch.setattr(tune.shared, "measure_option", make_measure_option)
    monkeypatch.setattr(tune.shared.autotvm, "LocalBuilder", lambda **_opts: FakeBuilder())
    monkeypatch.setattr(tune.shared, "create_runner", lambda backend, **_opts: FakeRunner())
    monkeypatch.setattr(tune.shared, "validate_backend", lambda _backend: None)
    monkeypatch.setattr(tune.shared, "load_simulator_backend", lambda _backend: None)
    monkeypatch.setattr(tune.shared.autotvm, "callback", SimpleNamespace(log_to_file=lambda _path: object()))

    result = tune.run_tuning(0, output_dir=tmp_path / "out", trials=32, timeout=120)

    assert calls[:3] == [("fsim", 1, 120)] * 3
    assert len({id(runner) for runner in fsim_runners}) == 3
    assert all(runner.server is None and runner.tracker is None for runner in fsim_runners)
    assert calls[3] == ("tsim", selected_config)
    assert result["tsim_cycles"] == 7654
    assert result["mac_count"] == 1024
    assert result["workload_sha256"] == "a" * 64
    assert result["measurement_scope"] == "isolated_complete_vta_conv_fusion"
    assert result["conv_schedule_key"] == tune.fused.conv_schedule_key(identity)
    assert result["fusion_sha256"] == identity.sha256
    persisted = json.loads(Path(result["result_json"]).read_text(encoding="utf-8"))
    tune.validate_fusion_result(
        persisted, identity, "m" * 64, "g" * 64, expected_workload_sha256="a" * 64
    )


def test_fused_result_validator_rejects_old_bare_conv_record(tune):
    identity = tune.fused.FusedConvIdentity(
        ("conv2d_packed.vta", ("TENSOR", (1, 2, 8, 8, 1, 8), "int8"),
         ("TENSOR", (2, 2, 3, 3, 8, 8), "int8"), (1, 1), (1, 1, 1, 1), (1, 1),
         "NCHW1n8c", "int32"),
        64, 7, -127, 127, "int8", "symbol", 0,
    )
    bare = {"template": "conv2d_packed.vta", "workload_sha256": "a" * 64}

    with pytest.raises(ValueError, match="schema"):
        tune.validate_fusion_result(bare, identity, "m" * 64, "g" * 64)


def test_fused_result_validator_rejects_mismatched_semantic_identity(tune):
    identity = tune.fused.FusedConvIdentity(
        ("conv2d_packed.vta", ("TENSOR", (1, 2, 8, 8, 1, 8), "int8"),
         ("TENSOR", (2, 2, 3, 3, 8, 8), "int8"), (1, 1), (1, 1, 1, 1), (1, 1),
         "NCHW1n8c", "int32"),
        64, 7, -127, 127, "int8", "symbol", 0,
    )
    result = {
        "schema_version": 1,
        "measurement_scope": "isolated_complete_vta_conv_fusion",
        "fusion_sha256": "x" * 64,
        "fusion_identity": {},
        "template": tune.fused.TASK_NAME,
        "real_conv_lowering": True,
    }

    with pytest.raises(ValueError, match="identity"):
        tune.validate_fusion_result(result, identity, "m" * 64, "g" * 64)


def test_recognizes_vta_targets_serialized_by_autotvm_fsim_runner(tune):
    target = SimpleNamespace(
        kind=SimpleNamespace(name="ext_dev"),
        keys=["vta", "cpu"],
        attrs={"device": "vta"},
    )
    assert tune._is_vta_record_target(target)
    assert not tune._is_vta_record_target(
        SimpleNamespace(kind=SimpleNamespace(name="ext_dev"), keys=["cpu"], attrs={"device": "cpu"})
    )


def _replay_fixture(tune, tmp_path, monkeypatch):
    identity = tune.fused.FusedConvIdentity(
        ("conv2d_packed.vta", ("TENSOR", (1, 2, 8, 8, 1, 8), "int8"),
         ("TENSOR", (2, 2, 3, 3, 8, 8), "int8"), (1, 1), (1, 1, 1, 1), (1, 1),
         "NCHW1n8c", "int32"),
        64, 7, -127, 127, "int8", "symbol", 0,
    )
    task = SimpleNamespace(
        name=tune.fused.TASK_NAME,
        workload=(tune.fused.TASK_NAME, *identity.task_args()),
    )
    config_json = {"index": 32, "entity": [["tile_h", "sp", [-1, 4]]]}
    config = SimpleNamespace(to_json_dict=lambda: config_json)
    measure_input = SimpleNamespace(
        task=task,
        target=SimpleNamespace(kind=SimpleNamespace(name="vta")),
        config=config,
    )
    measure_result = SimpleNamespace(error_no=tune.shared.MeasureErrorNo.NO_ERROR, costs=(0.01,))
    fsim_path = tmp_path / "fsim.log"
    best_path = tmp_path / "best.log"
    fsim_path.write_bytes(b"native fsim log bytes")
    best_path.write_bytes(b"native picked-best log bytes")
    records_by_path = {
        str(fsim_path): [(measure_input, measure_result)],
        str(best_path): [(measure_input, measure_result)],
    }
    monkeypatch.setattr(
        tune.shared.autotvm.record,
        "load_from_file",
        lambda path: iter(records_by_path[str(path)]),
    )
    model_path = tune.APP_ROOT / "model" / "pretrainedResnet.tflite"
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text("{}\n", encoding="utf-8")
    geometry_sha = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    result_path = tmp_path / "result.json"
    result = {
        "schema_version": 1,
        "measurement_scope": "isolated_complete_vta_conv_fusion",
        "workload_index": 0,
        "occurrence": 0,
        "symbol": identity.symbol,
        "template": task.name,
        "workload_sha256": "w" * 64,
        "fusion_identity": json.loads(identity.canonical_json()),
        "fusion_sha256": identity.sha256,
        "model": "image_classification_v1",
        "model_sha256": model_sha,
        "geometry_path": str(geometry_path),
        "geometry_sha256": geometry_sha,
        "conv_schedule_key": tune.fused.conv_schedule_key(identity),
        "conv_config": config_json,
        "real_conv_lowering": True,
        "fsim_log_sha256": hashlib.sha256(fsim_path.read_bytes()).hexdigest(),
        "best_fsim_log_sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
        "fsim_log": str(fsim_path),
        "best_fsim_log": str(best_path),
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    prepared = SimpleNamespace()
    monkeypatch.setattr(tune, "prepare_v1_workloads", lambda: (prepared, [identity], [task]))
    monkeypatch.setattr(tune.shared, "_config_identity", lambda _path: (geometry_path, geometry_sha))
    monkeypatch.setattr(tune.shared, "_task_workload_id", lambda _task: "w" * 64)
    lowered = SimpleNamespace(schedule=object())
    lower_calls = []

    def lower(actual_prepared, actual_identity, actual_config):
        lower_calls.append((actual_prepared, actual_identity, actual_config))
        return lowered

    monkeypatch.setattr(tune.fused, "lower_with_fused_config", lower)
    return result_path, result, records_by_path, config, lowered, lower_calls, prepared, identity


def test_replay_validates_artifacts_and_applies_best_config_to_real_fusion(
    tune, tmp_path, monkeypatch
):
    result_path, result, _records, config, lowered, calls, prepared, identity = _replay_fixture(
        tune, tmp_path, monkeypatch
    )

    replay = tune.replay_result(result_path)

    assert replay["result"] == json.loads(result_path.read_text(encoding="utf-8"))
    assert replay["config"] is config
    assert replay["lowered"] is lowered
    assert calls == [(prepared, identity, config)]


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_log", "missing or unreadable"),
        ("modified_log", "SHA-256 does not match"),
        ("changed_hash", "SHA-256 does not match"),
        ("changed_config", "Conv config does not match"),
        ("wrong_task", "no successful record"),
        ("best_not_in_fsim", "not a successful matching record"),
    ],
)
def test_replay_rejects_missing_or_mismatched_artifacts(tune, tmp_path, monkeypatch, corruption, message):
    result_path, result, records, _config, _lowered, calls, *_ = _replay_fixture(
        tune, tmp_path, monkeypatch
    )
    if corruption == "missing_log":
        Path(result["fsim_log"]).unlink()
    elif corruption == "modified_log":
        Path(result["fsim_log"]).write_bytes(b"modified native fsim log")
    elif corruption == "changed_hash":
        result["fsim_log_sha256"] = "0" * 64
    elif corruption == "changed_config":
        result["conv_config"] = {"index": 999}
    elif corruption == "wrong_task":
        measure_input, measure_result = records[result["fsim_log"]][0]
        wrong_task = SimpleNamespace(workload=("wrong",))
        records[result["fsim_log"]] = [(SimpleNamespace(
            task=wrong_task, target=measure_input.target, config=measure_input.config,
        ), measure_result)]
    else:
        measure_input, measure_result = records[result["best_fsim_log"]][0]
        wrong_config = SimpleNamespace(to_json_dict=lambda: {"index": 999})
        records[result["best_fsim_log"]] = [(SimpleNamespace(
            task=measure_input.task, target=measure_input.target, config=wrong_config,
        ), measure_result)]
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        tune.replay_result(result_path)

    assert calls == []
