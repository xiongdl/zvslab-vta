"""Focused tests for simulator-aware AutoTVM tuning."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


TUNER_PATH = Path(__file__).resolve().parents[1] / "autotvm_tuner.py"
MODEL_PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "image_classification_v1"
    / "model_pipeline.py"
)
MODEL_PATH = MODEL_PIPELINE_PATH.parent / "model" / "pretrainedResnet.tflite"


def _load_tuner():
    spec = importlib.util.spec_from_file_location("mlperf_tiny_autotvm_tuner", TUNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tuner():
    return _load_tuner()


def test_backend_validation_requires_explicit_matching_selector(tuner, monkeypatch):
    monkeypatch.setenv("VTA_BACKEND", "tsim")
    assert tuner.validate_backend("tsim") == "tsim"

    with pytest.raises(ValueError, match="backend mismatch"):
        tuner.validate_backend("fsim")

    monkeypatch.delenv("VTA_BACKEND")
    with pytest.raises(ValueError, match="VTA_BACKEND"):
        tuner.validate_backend("tsim")


@pytest.mark.parametrize(
    "model_id,model_filename",
    [
        ("keyword_spotting_v1", "kws_ref_model.tflite"),
        ("streaming_wakeword_v1", "str_ww_ref_model.tflite"),
    ],
)
def test_audio_model_pipelines_are_registered(tuner, model_id, model_filename):
    assert tuner.MODEL_PIPELINES[model_id] == (model_id, "model", model_filename)
    pipeline = tuner._load_model_pipeline(model_id)
    model_path = Path(tuner.__file__).resolve().parent / model_id / "model" / model_filename
    prepared = pipeline.prepare_model(model_path)
    assert prepared.imported.model_sha256 == pipeline.MODEL_SHA256
    tasks, report = tuner.extract_model_tasks(prepared)
    assert tasks
    assert report["supported"]
    assert all(item["template"] in tuner.SUPPORTED_TEMPLATES for item in report["supported"])
    assert all(item["template"] not in tuner.SUPPORTED_TEMPLATES for item in report["unsupported"])


def test_vww_model_pipeline_reports_supported_and_unsupported_vta_tasks(tuner):
    model_id = "visual_wake_words_v1"
    pipeline = tuner._load_model_pipeline(model_id)
    model_path = Path(__file__).resolve().parents[1] / model_id / "model" / "vww_96_float.tflite"
    prepared = pipeline.prepare_model(model_path)

    tasks, report = tuner.extract_model_tasks(prepared)

    assert tasks
    assert {entry["template"] for entry in report["supported"]} == {
        task.name for task in tasks
    }
    assert report["unsupported"]
    assert all(
        entry["template"] not in tuner.SUPPORTED_TEMPLATES
        and len(entry["workload_sha256"]) == 64
        for entry in report["unsupported"]
    )
    assert {entry["workload_sha256"] for entry in report["supported"]} == {
        tuner._task_workload_id(task) for task in tasks
    }


def test_backend_loading_preserves_missing_library_diagnostic(tuner, monkeypatch):
    from vta.testing import simulator

    monkeypatch.setenv("VTA_BACKEND", "fsim")

    def missing_library(_backend):
        raise RuntimeError("FSIM backend requires libvta_fsim; missing library libvta_fsim")

    monkeypatch.setattr(simulator, "load_backend", missing_library)

    with pytest.raises(RuntimeError, match="missing library libvta_fsim"):
        tuner.load_simulator_backend("fsim")


def test_backend_loading_reports_missing_profiler_registries(tuner, monkeypatch):
    from vta.testing import simulator

    monkeypatch.setenv("VTA_BACKEND", "tsim")
    monkeypatch.setattr(simulator, "load_backend", lambda _backend: None)
    monkeypatch.setattr(tuner.tvm, "get_global_func", lambda _name, allow_missing: None)

    with pytest.raises(RuntimeError, match="TSIM profiler/runtime registries are unavailable"):
        tuner.load_simulator_backend("tsim")


def test_tsim_profiler_module_loader_resets_and_collects_each_candidate(
    tuner, monkeypatch, tmp_path
):
    events = []
    monkeypatch.setattr(tuner, "tsim_hardware_library", lambda: "/tmp/libvta_hw.dylib")

    class Remote:
        def upload(self, _path):
            events.append("upload")

        def remove(self, _path):
            events.append("remove")

        def get_function(self, name):
            if name.endswith("loadfile_vta-tsim"):
                return lambda _name: events.append("load-hardware") or object()
            if name == "vta.tsim.init":
                return lambda _module: events.append("init-hardware")
            if name.endswith("profiler_clear"):
                return lambda: events.append("clear")
            assert name.endswith("profiler_status")
            readings = iter(({"cycle_count": 0}, {"cycle_count": 37}))
            return lambda: json.dumps(next(readings))

    class BaseLoader:
        @contextmanager
        def __call__(self, remote_kwargs, build_result):
            yield Remote(), object()

    collected = {}
    loader = tuner.ProfilerModuleLoader("tsim", BaseLoader(), collected)
    filename = str(tmp_path / "candidate.tar")
    with loader({}, type("Build", (), {"filename": filename})()):
        events.append("run")

    assert events == [
        "upload",
        "load-hardware",
        "init-hardware",
        "remove",
        "clear",
        "run",
    ]
    assert collected[filename] == {"cycle_count": 37}
    assert json.loads(Path(filename + ".tsim.json").read_text(encoding="utf-8")) == {
        "cycle_count": 37
    }


def test_tsim_trial_cost_requires_positive_integer_cycles(tuner):
    assert tuner.tsim_cycle_cost({"cycle_count": 17}) == 17
    for value in (0, -1, True, 1.5, "17"):
        with pytest.raises(RuntimeError, match="positive integer"):
            tuner.tsim_cycle_cost({"cycle_count": value})


def test_fsim_module_loader_uses_only_fsim_profiler(tuner):
    names = []

    class Remote:
        def get_function(self, name):
            names.append(name)
            if name.endswith("profiler_clear"):
                return lambda: None
            return lambda: json.dumps({"gemm_counter": 0, "wgt_load_nbytes": 0})

    class BaseLoader:
        @contextmanager
        def __call__(self, remote_kwargs, build_result):
            yield Remote(), object()

    loader = tuner.ProfilerModuleLoader("fsim", BaseLoader(), {})
    with loader({}, type("Build", (), {"filename": "candidate.tar"})()):
        pass
    assert names == ["vta.simulator.profiler_clear", "vta.simulator.profiler_status"]


def test_tsim_runner_replaces_wall_clock_result_with_integer_cycles(tuner, monkeypatch, tmp_path):
    runner = tuner.TSIMLocalRunner()
    measure_result = tuner.MeasureResult((0.0001,), tuner.MeasureErrorNo.NO_ERROR, 0.1, 12.0)
    monkeypatch.setattr(tuner.LocalRunner, "run", lambda self, _inputs, _builds: [measure_result])
    build_result = type("Build", (), {"filename": str(tmp_path / "candidate.tar")})()
    runner.cycle_stats[build_result.filename] = {"cycle_count": 41}

    measure_input = type("Input", (), {"task": object(), "config": object()})()
    result = runner.run([measure_input], [build_result])[0]

    assert result.costs == (41,)
    assert isinstance(result.costs[0], int)

    runner.cycle_stats.clear()
    result = runner.run([measure_input], [build_result])[0]
    assert result.error_no == tuner.MeasureErrorNo.RUNTIME_DEVICE
    assert len(result.costs) == 2

    (tmp_path / "candidate.tar.tsim.json").write_text(
        json.dumps({"cycle_count": 43}), encoding="utf-8"
    )
    result = runner.run([measure_input], [build_result])[0]
    assert result.costs == (43,)
    assert not (tmp_path / "candidate.tar.tsim.json").exists()


def test_v1_task_extraction_builds_supported_vta_templates(tuner):
    model_spec = importlib.util.spec_from_file_location(
        "mlperf_resnet_autotvm_model_pipeline", MODEL_PIPELINE_PATH
    )
    model_pipeline = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = model_pipeline
    model_spec.loader.exec_module(model_pipeline)

    prepared = model_pipeline.prepare_model(MODEL_PATH)
    tasks = tuner.extract_v1_tasks(prepared)

    assert tasks
    assert {task.name for task in tasks} == {"conv2d_packed.vta"}
    import vta

    for task in tasks:
        with task.target:
            schedule, args = task.instantiate(task.config_space.get(0))
        assert schedule
        vta.build(schedule, args, target=task.target, target_host=task.target_host)


def test_v2_task_extraction_reports_supported_and_unsupported_vta_tasks(tuner):
    model_id = "image_classification_v2"
    pipeline = tuner._load_model_pipeline(model_id)
    model_path = Path(__file__).resolve().parents[1] / model_id / "model" / "pretrainedResnet_large_float.tflite"
    prepared = pipeline.prepare_model(model_path)

    tasks, report = tuner.extract_model_tasks(prepared)

    assert tasks
    assert {entry["template"] for entry in report["supported"]} == {
        task.name for task in tasks
    }
    assert report["unsupported"]
    assert all(
        entry["template"] not in tuner.SUPPORTED_TEMPLATES
        and len(entry["workload_sha256"]) == 64
        for entry in report["unsupported"]
    )
    assert {entry["workload_sha256"] for entry in report["supported"]} == {
        tuner._task_workload_id(task) for task in tasks
    }
    for task in tasks:
        with task.target:
            schedule, args = task.instantiate(task.config_space.get(0))
        assert schedule
        import vta
        vta.build(schedule, args, target=task.target, target_host=task.target_host)


def test_task_extraction_names_unsupported_vta_template(tuner, monkeypatch):
    from types import SimpleNamespace

    supported = SimpleNamespace(
        name="conv2d_packed.vta",
        target=SimpleNamespace(kind=SimpleNamespace(name="vta")),
        workload=("supported",),
    )
    unsupported = SimpleNamespace(
        name="pool_packed.vta",
        target=SimpleNamespace(kind=SimpleNamespace(name="vta")),
        workload=("unsupported",),
    )
    monkeypatch.setattr(
        tuner.autotvm.task,
        "extract_from_program",
        lambda *_args, **_kwargs: [supported, unsupported],
    )

    tasks, report = tuner.extract_model_tasks(SimpleNamespace(mixed_module=object()))

    assert tasks == [supported]
    assert report["supported"][0]["template"] == "conv2d_packed.vta"
    assert report["unsupported"] == [
        {
            "template": "pool_packed.vta",
            "workload_sha256": tuner._workload_id(("unsupported",)),
        }
    ]


def test_anomaly_task_extraction_reports_supported_and_unsupported_vta_tasks(tuner):
    model_id = "anomaly_detection_v1"
    pipeline = tuner._load_model_pipeline(model_id)
    model_path = Path(__file__).resolve().parents[1] / model_id / "model" / "ad01_fp32.tflite"
    prepared = pipeline.prepare_model(model_path)

    tasks, report = tuner.extract_model_tasks(prepared)

    assert tasks
    assert {entry["template"] for entry in report["supported"]} == {
        task.name for task in tasks
    }
    assert report["unsupported"]
    assert all(
        entry["template"] not in tuner.SUPPORTED_TEMPLATES
        and len(entry["workload_sha256"]) == 64
        for entry in report["unsupported"]
    )
    assert {entry["workload_sha256"] for entry in report["supported"]} == {
        tuner._task_workload_id(task) for task in tasks
    }
    for task in tasks:
        with task.target:
            schedule, args = task.instantiate(task.config_space.get(0))
        assert schedule
        import vta
        vta.build(schedule, args, target=task.target, target_host=task.target_host)


def test_dense_autotvm_template_builds_with_vta_target():
    import tvm
    import vta

    env = vta.get_env()
    target = tvm.target.Target("vta", host=env.target_host)
    data = tvm.te.placeholder((1, 2, 1, 8), dtype=env.inp_dtype)
    weight = tvm.te.placeholder((2, 2, 8, 8), dtype=env.wgt_dtype)
    task = tvm.autotvm.task.create(
        "dense_packed.vta",
        args=(data, weight, None, env.acc_dtype),
        target=target,
        target_host=env.target_host,
    )
    with task.target:
        schedule, args = task.instantiate(task.config_space.get(0))

    assert schedule
    vta.build(schedule, args, target=task.target, target_host=task.target_host)


def _dense_autotvm_task():
    import tvm
    import vta

    env = vta.get_env()
    target = tvm.target.Target("vta", host=env.target_host)
    data = tvm.te.placeholder((1, 2, 1, 8), dtype=env.inp_dtype)
    weight = tvm.te.placeholder((2, 2, 8, 8), dtype=env.wgt_dtype)
    return tvm.autotvm.task.create(
        "dense_packed.vta",
        args=(data, weight, None, env.acc_dtype),
        target=target,
        target_host=env.target_host,
    )


def _write_native_record(tuner, path, task):
    measure_input = tuner.MeasureInput(task.target, task, task.config_space.get(0))
    result = tuner.MeasureResult((123,), tuner.MeasureErrorNo.NO_ERROR, 0.01, 1.0)
    path.write_text(tuner.autotvm.record.encode(measure_input, result) + "\n", encoding="utf-8")


def test_tuning_sidecar_validates_pair_and_history_best(tuner, monkeypatch, tmp_path):
    import vta  # noqa: F401 - register VTA AutoTVM templates for log replay

    monkeypatch.setenv("VTA_BACKEND", "fsim")
    config_path = (Path(__file__).resolve().parents[3] / "config" / "vta_64mac.json").resolve()
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config_path))
    task = _dense_autotvm_task()
    log_path = tmp_path / "v1-fsim.log"
    sidecar_path = tmp_path / "v1-fsim.json"
    _write_native_record(tuner, log_path, task)
    options = {"tuner": "grid_search", "trials_per_task": 1, "timeout": 60}

    metadata = tuner.write_tuning_sidecar(
        log_path,
        sidecar_path,
        model_id="image_classification_v1",
        model_sha256="a" * 64,
        backend="fsim",
        config_path=config_path,
        tasks=[task],
        tuning_options=options,
    )

    assert metadata["task_count"] == 1
    assert metadata["trial_count"] == 1
    assert metadata["log_sha256"]
    sidecar_bytes = sidecar_path.read_bytes()
    tuner.write_tuning_sidecar(
        log_path,
        sidecar_path,
        model_id="image_classification_v1",
        model_sha256="a" * 64,
        backend="fsim",
        config_path=config_path,
        tasks=[task],
        tuning_options=options,
    )
    assert sidecar_path.read_bytes() == sidecar_bytes
    validated = tuner.validate_tuning_artifacts(
        log_path,
        sidecar_path,
        model_id="image_classification_v1",
        model_sha256="a" * 64,
        backend="fsim",
        config_path=Path(metadata["config_path"]),
        expected_tuning_options=options,
    )
    assert validated == metadata

    with tuner.history_best(
        log_path,
        sidecar_path,
        model_id="image_classification_v1",
        model_sha256="a" * 64,
        backend="fsim",
        config_path=Path(metadata["config_path"]),
        expected_tuning_options=options,
    ):
        assert tuner.autotvm.task.DispatchContext.current is not None


def test_tuning_sidecar_rejects_incomplete_or_mismatched_replay(tuner, monkeypatch, tmp_path):
    import vta  # noqa: F401 - register VTA AutoTVM templates for log replay

    monkeypatch.setenv("VTA_BACKEND", "fsim")
    config_path = (Path(__file__).resolve().parents[3] / "config" / "vta_64mac.json").resolve()
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config_path))
    task = _dense_autotvm_task()
    log_path = tmp_path / "v1-fsim.log"
    sidecar_path = tmp_path / "v1-fsim.json"
    _write_native_record(tuner, log_path, task)
    options = {"tuner": "grid_search", "trials_per_task": 1, "timeout": 60}
    tuner.write_tuning_sidecar(
        log_path,
        sidecar_path,
        model_id="image_classification_v1",
        model_sha256="a" * 64,
        backend="fsim",
        config_path=config_path,
        tasks=[task],
        tuning_options=options,
    )

    with pytest.raises(ValueError, match="tuning options"):
        tuner.validate_tuning_artifacts(
            log_path,
            sidecar_path,
            model_id="image_classification_v1",
            model_sha256="a" * 64,
            backend="fsim",
            config_path=config_path,
            expected_tuning_options={**options, "timeout": 30},
        )

    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    metadata["trial_count"] += 1
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="trial_count"):
        tuner.validate_tuning_artifacts(
            log_path,
            sidecar_path,
            model_id="image_classification_v1",
            model_sha256="a" * 64,
            backend="fsim",
            config_path=config_path,
        )


def test_tuning_sidecar_rejects_geometry_config_mismatch(tuner, monkeypatch, tmp_path):
    import vta  # noqa: F401 - register VTA AutoTVM templates for log replay

    monkeypatch.setenv("VTA_BACKEND", "fsim")
    config_path = (Path(__file__).resolve().parents[3] / "config" / "vta_64mac.json").resolve()
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config_path))
    task = _dense_autotvm_task()
    log_path = tmp_path / "v1-fsim.log"
    sidecar_path = tmp_path / "v1-fsim.json"
    _write_native_record(tuner, log_path, task)
    options = {"tuner": "grid_search", "trials_per_task": 1, "timeout": 60}
    tuner.write_tuning_sidecar(
        log_path,
        sidecar_path,
        model_id="image_classification_v1",
        model_sha256="a" * 64,
        backend="fsim",
        config_path=config_path,
        tasks=[task],
        tuning_options=options,
    )

    alternate_config = tmp_path / "vta_64mac.json"
    alternate_config.write_bytes(config_path.read_bytes() + b"\n")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(alternate_config))
    with pytest.raises(ValueError, match="config_path mismatch"):
        tuner.validate_tuning_artifacts(
            log_path,
            sidecar_path,
            model_id="image_classification_v1",
            model_sha256="a" * 64,
            backend="fsim",
            config_path=alternate_config,
        )
