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

"""AutoTVM support for simulator-backed MLPerf Tiny schedules."""

import contextlib
import argparse
import hashlib
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import tvm
from tvm import autotvm
from tvm.autotvm.measure import MeasureErrorNo, MeasureInput, MeasureResult
from tvm.autotvm.measure.measure_methods import (
    LocalRunner,
    RPCRunner,
    check_remote,
    default_module_loader,
)


PROFILER_REGISTRIES = {
    "fsim": ("vta.simulator.profiler_clear", "vta.simulator.profiler_status"),
    "tsim": ("vta.tsim.profiler_clear", "vta.tsim.profiler_status"),
}
ARTIFACT_SCHEMA_VERSION = 1
TUNER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = TUNER_ROOT / "config" / "vta_64mac.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "build" / "autotvm"
MODEL_PIPELINES = {
    "image_classification_v1": ("image_classification_v1", "model", "pretrainedResnet.tflite"),
    "image_classification_v2": ("image_classification_v2", "model", "pretrainedResnet_large_float.tflite"),
    "anomaly_detection_v1": ("anomaly_detection_v1", "model", "ad01_fp32.tflite"),
    "keyword_spotting_v1": ("keyword_spotting_v1", "model", "kws_ref_model.tflite"),
    "streaming_wakeword_v1": (
        "streaming_wakeword_v1", "model", "str_ww_ref_model.tflite"
    ),
    "visual_wake_words_v1": ("visual_wake_words_v1", "model", "vww_96_float.tflite"),
}
SUPPORTED_TEMPLATES = {"conv2d_packed.vta", "dense_packed.vta"}


def validate_backend(backend):
    """Require an explicit backend selector matching the requested simulator."""
    if backend not in PROFILER_REGISTRIES:
        raise ValueError(f"unsupported backend {backend!r}; expected fsim or tsim")
    active_backend = os.environ.get("VTA_BACKEND")
    if active_backend is None:
        raise ValueError("VTA_BACKEND must explicitly select fsim or tsim")
    if active_backend != backend:
        raise ValueError(
            f"simulator backend mismatch: requested {backend!r}, "
            f"VTA_BACKEND is {active_backend!r}"
        )
    return backend


def load_simulator_backend(backend):
    """Load the selected simulator and fail with its missing-library diagnostics."""
    validate_backend(backend)
    from vta.testing import simulator

    simulator.load_backend(backend)
    missing = [
        name
        for name in simulator.BACKEND_REGISTRIES[backend]
        if tvm.get_global_func(name, allow_missing=True) is None
    ]
    if missing:
        raise RuntimeError(
            f"{backend.upper()} profiler/runtime registries are unavailable: "
            f"{', '.join(missing)}"
        )


def simulator_server_libraries(backend):
    """Return the runtime libraries to load in AutoTVM's local RPC workers."""
    import vta
    from vta.libinfo import _find_compiler_extension, find_libvta

    names = ("libvta_fsim",) if backend == "fsim" else ("libvta_tsim",)
    return ":".join([_find_compiler_extension()] + [find_libvta(name)[0] for name in names])


def tsim_hardware_library():
    """Return the built TSIM hardware module path."""
    from vta.libinfo import find_libvta

    return find_libvta("libvta_hw")[0]


class SimulatorLocalRunner(LocalRunner):
    """Local RPC runner that initializes VTA only on the loopback server."""

    def __init__(self, backend, **kwargs):
        self.backend = backend
        super().__init__(**kwargs)

    def set_task(self, task):
        from tvm.rpc.server import Server
        from tvm.rpc.tracker import Tracker

        self.task = task
        tracker = Tracker(host="127.0.0.1", port=9000, port_end=10000, silent=True)
        device_key = f"$local$device${tracker.port}"
        server = Server(
            host="127.0.0.1",
            port=9000,
            port_end=10000,
            key=device_key,
            load_library=simulator_server_libraries(self.backend),
            silent=True,
            tracker_addr=("127.0.0.1", tracker.port),
        )
        self.tracker = tracker
        self.server = server
        self.key = device_key
        self.host = "127.0.0.1"
        self.port = tracker.port
        if check_remote(vta_target(), self.key, self.host, self.port):
            return server, tracker
        raise RuntimeError("VTA ext_dev device is unavailable in the local simulator RPC server")

    def run(self, measure_inputs, build_results):
        device_target = vta_target()
        runtime_inputs = [
            MeasureInput(device_target, measure_input.task, measure_input.config)
            for measure_input in measure_inputs
        ]
        return super().run(runtime_inputs, build_results)


def vta_target():
    """Return the configured ext_dev target used to access the VTA runtime."""
    import vta

    return vta.get_env().target


class ProfilerModuleLoader:
    """Wrap AutoTVM's local RPC loader with per-candidate profiler handling."""

    def __init__(self, backend, base_loader=None, collected=None):
        self.backend = backend
        self.base_loader = base_loader or default_module_loader()
        self.collected = collected if collected is not None else {}

    @contextlib.contextmanager
    def __call__(self, remote_kwargs, build_result):
        clear_registry, status_registry = PROFILER_REGISTRIES[self.backend]
        filename = build_result.filename
        with self.base_loader(remote_kwargs, build_result) as (remote, module):
            try:
                if self.backend == "tsim":
                    hardware_path = tsim_hardware_library()
                    hardware_name = os.path.basename(hardware_path)
                    remote.upload(hardware_path)
                    load_hardware = remote.get_function("runtime.module.loadfile_vta-tsim")
                    hardware_module = load_hardware(hardware_name)
                    remote.get_function("vta.tsim.init")(hardware_module)
                    remote.remove(hardware_name)
                clear = remote.get_function(clear_registry)
                status = remote.get_function(status_registry)
            except Exception as error:
                libraries = "libvta_tsim and libvta_hw" if self.backend == "tsim" else "libvta_fsim"
                raise RuntimeError(
                    f"{self.backend.upper()} AutoTVM profiler registries are unavailable "
                    f"in the local runner; build {libraries} for the selected config"
                ) from error
            clear()
            reset_raw = status()
            reset_stats = json.loads(reset_raw) if isinstance(reset_raw, str) else reset_raw
            if self.backend == "tsim":
                if reset_stats != {"cycle_count": 0}:
                    raise RuntimeError(
                        f"TSIM profiler did not reset before candidate measurement: {reset_stats}"
                    )
            elif any(value != 0 for value in reset_stats.values()):
                raise RuntimeError(
                    f"FSIM profiler did not reset before candidate measurement: {reset_stats}"
                )
            try:
                yield remote, module
            finally:
                if self.backend == "tsim":
                    try:
                        raw = status()
                        stats = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception as error:
                        raise RuntimeError("TSIM AutoTVM profiler status could not be read") from error
                    self.collected[filename] = stats
                    with open(filename + ".tsim.json", "w", encoding="utf-8") as stats_file:
                        json.dump(stats, stats_file, sort_keys=True)


def tsim_cycle_cost(stats):
    """Validate and return the TSIM cycle counter as the AutoTVM trial cost."""
    value = stats.get("cycle_count") if isinstance(stats, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"TSIM profiler counter cycle_count must be a positive integer: {stats}"
        )
    return value


class TSIMLocalRunner(SimulatorLocalRunner):
    """Use native TSIM cycle counts instead of local RPC wall-clock durations."""

    def __init__(self, **kwargs):
        self.cycle_stats = {}
        loader = ProfilerModuleLoader("tsim", collected=self.cycle_stats)
        super().__init__("tsim", module_loader=loader, **kwargs)

    def run(self, measure_inputs, build_results):
        results = super().run(measure_inputs, build_results)
        converted = []
        for build_result, result in zip(build_results, results):
            if result.error_no != MeasureErrorNo.NO_ERROR:
                converted.append(result)
                continue
            stats_path = build_result.filename + ".tsim.json"
            try:
                stats = self.cycle_stats.pop(build_result.filename, None)
                if stats is None:
                    with open(stats_path, encoding="utf-8") as stats_file:
                        stats = json.load(stats_file)
                cost = tsim_cycle_cost(stats)
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                converted.append(
                    MeasureResult(
                        (str(error), error),
                        MeasureErrorNo.RUNTIME_DEVICE,
                        result.all_cost,
                        time.time(),
                    )
                )
                continue
            finally:
                try:
                    os.remove(stats_path)
                except OSError:
                    pass
            converted.append(
                MeasureResult((cost,), MeasureErrorNo.NO_ERROR, result.all_cost, result.timestamp)
            )
        return converted


def create_runner(backend, timeout=60, number=1, repeat=1, cooldown_interval=0):
    """Create an isolated backend runner with cycle-based TSIM costs."""
    validate_backend(backend)
    load_simulator_backend(backend)
    runner_options = {
        "timeout": timeout,
        "number": number,
        "repeat": repeat,
        "cooldown_interval": cooldown_interval,
    }
    if backend == "tsim":
        return TSIMLocalRunner(**runner_options)
    return SimulatorLocalRunner("fsim", module_loader=ProfilerModuleLoader("fsim"), **runner_options)


def measure_option(backend, **runner_options):
    """Build local schedules and measure them using the selected simulator."""
    return autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=1),
        runner=create_runner(backend, **runner_options),
    )


def extract_v1_tasks(prepared):
    """Extract supported VTA AutoTVM tasks from the prepared V1 graph."""
    tasks, _ = extract_model_tasks(prepared)
    return tasks


def extract_model_tasks(prepared):
    """Return supported VTA tasks and an explicit supported/unsupported report."""
    import vta

    env = vta.get_env()
    task_target = tvm.target.Target("vta", host=env.target_host)
    extracted = autotvm.task.extract_from_program(
        prepared.mixed_module,
        target=task_target,
        target_host=env.target_host,
        params={},
    )
    vta_tasks = [task for task in extracted if task.target.kind.name == "vta"]
    tasks = [task for task in vta_tasks if task.name in SUPPORTED_TEMPLATES]
    unsupported = [task for task in vta_tasks if task.name not in SUPPORTED_TEMPLATES]
    for task in tasks:
        # AutoTVM's LocalBuilder selects VTA's build_config by device_name;
        # the canonical ext_dev target keeps the measurement device usable.
        task.target = env.target
    report = {
        "supported": [
            {"template": task.name, "workload_sha256": _task_workload_id(task)}
            for task in tasks
        ],
        "unsupported": [
            {"template": task.name, "workload_sha256": _task_workload_id(task)}
            for task in unsupported
        ],
    }
    return tasks, report


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_identity(config_path):
    path = Path(config_path).expanduser().resolve(strict=True)
    active_path = os.environ.get("VTA_CONFIG_FILE")
    if active_path is None:
        raise ValueError("VTA_CONFIG_FILE must explicitly select the geometry config")
    if Path(active_path).expanduser().resolve() != path:
        raise ValueError(
            f"VTA_CONFIG_FILE mismatch: artifact uses {path}, active environment uses "
            f"{Path(active_path).expanduser().resolve()}"
        )
    try:
        with path.open(encoding="utf-8") as source:
            config = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"VTA geometry config is not valid JSON: {path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"VTA geometry config must be a JSON object: {path}")
    return path, _sha256_file(path)


def _workload_id(workload):
    payload = json.dumps(workload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task_workload_id(task):
    return _workload_id(task.workload)


def _load_log_summary(log_path):
    path = Path(log_path).expanduser().resolve(strict=True)
    if path.stat().st_size == 0:
        raise ValueError(f"AutoTVM log is empty: {path}")
    try:
        records = list(autotvm.record.load_from_file(str(path)))
    except Exception as error:
        raise ValueError(f"AutoTVM log cannot be decoded: {path}") from error
    if not records:
        raise ValueError(f"AutoTVM log has no records: {path}")

    trial_counts = {}
    successful = set()
    for measure_input, result in records:
        workload_id = _workload_id(measure_input.task.workload)
        trial_counts[workload_id] = trial_counts.get(workload_id, 0) + 1
        if result.error_no == MeasureErrorNo.NO_ERROR:
            successful.add(workload_id)
    return path, trial_counts, successful, len(records)


def _validate_tuning_options(options):
    if not isinstance(options, dict) or not isinstance(options.get("tuner"), str):
        raise ValueError("tuning options must include a tuner name")
    trials = options.get("trials_per_task")
    if trials is not None and (
        not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0
    ):
        raise ValueError("tuning options trials_per_task must be a positive integer or null")
    for name in ("timeout", "number", "repeat"):
        value = options.get(name)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ValueError(f"tuning options {name} must be a positive integer")
    cooldown = options.get("cooldown_interval")
    if cooldown is not None and (
        not isinstance(cooldown, (int, float)) or isinstance(cooldown, bool) or cooldown < 0
    ):
        raise ValueError("tuning options cooldown_interval must be non-negative")
    try:
        json.dumps(options, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("tuning options must be JSON serializable") from error


def write_tuning_sidecar(
    log_path,
    sidecar_path,
    *,
    model_id,
    model_sha256,
    backend,
    config_path,
    tasks,
    tuning_options,
    task_report=None,
):
    """Write a deterministic JSON sidecar for a complete native AutoTVM log."""
    validate_backend(backend)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ValueError("model_sha256 must be a 64-character SHA-256 hex digest")
    try:
        int(model_sha256, 16)
    except ValueError as error:
        raise ValueError("model_sha256 must be a 64-character SHA-256 hex digest") from error
    _validate_tuning_options(tuning_options)
    config_path, config_sha256 = _config_identity(config_path)
    log_path, trial_counts, successful, trial_count = _load_log_summary(log_path)
    task_workloads = sorted({_task_workload_id(task) for task in tasks})
    if not task_workloads:
        raise ValueError("cannot write tuning metadata without extracted VTA tasks")
    if set(trial_counts) != set(task_workloads):
        raise ValueError("AutoTVM log task coverage does not match the extracted VTA tasks")
    if not set(task_workloads) <= successful:
        missing = sorted(set(task_workloads) - successful)
        raise ValueError(f"AutoTVM log has no successful measurement for tasks: {missing}")

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_id": model_id,
        "model_sha256": model_sha256,
        "backend": backend,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "log_path": str(log_path),
        "log_sha256": _sha256_file(log_path),
        "task_count": len(task_workloads),
        "trial_count": trial_count,
        "task_workloads": task_workloads,
        "task_trials": dict(sorted(trial_counts.items())),
        "task_report": task_report or {
            "supported": [
                {"template": task.name, "workload_sha256": _task_workload_id(task)}
                for task in tasks
            ],
            "unsupported": [],
        },
        "tuning_options": tuning_options,
    }
    sidecar_path = Path(sidecar_path).expanduser().resolve()
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = sidecar_path.with_name(sidecar_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(sidecar_path)
    return metadata


def validate_tuning_artifacts(
    log_path,
    sidecar_path,
    *,
    model_id,
    model_sha256,
    backend,
    config_path,
    expected_tuning_options=None,
):
    """Reject incomplete or mismatched logs before applying history-best."""
    validate_backend(backend)
    config_path, config_sha256 = _config_identity(config_path)
    sidecar_path = Path(sidecar_path).expanduser().resolve(strict=True)
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"tuning sidecar is not valid JSON: {sidecar_path}") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported tuning sidecar schema")
    for name, expected in (
        ("model_id", model_id),
        ("model_sha256", model_sha256),
        ("backend", backend),
        ("config_path", str(config_path)),
        ("config_sha256", config_sha256),
    ):
        if metadata.get(name) != expected:
            raise ValueError(f"tuning sidecar {name} mismatch")

    log_path = Path(log_path).expanduser().resolve(strict=True)
    if metadata.get("log_path") != str(log_path):
        raise ValueError("tuning sidecar log_path mismatch")
    if metadata.get("log_sha256") != _sha256_file(log_path):
        raise ValueError("AutoTVM log SHA-256 does not match its sidecar")
    options = metadata.get("tuning_options")
    _validate_tuning_options(options)
    if expected_tuning_options is not None and options != expected_tuning_options:
        raise ValueError("tuning options do not match the expected replay parameters")

    workloads = metadata.get("task_workloads")
    trials = metadata.get("task_trials")
    task_count = metadata.get("task_count")
    trial_count = metadata.get("trial_count")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("tuning sidecar task_workloads is incomplete")
    if not all(isinstance(item, str) and len(item) == 64 for item in workloads):
        raise ValueError("tuning sidecar task_workloads contains an invalid identity")
    if (
        len(set(workloads)) != len(workloads)
        or not isinstance(task_count, int)
        or isinstance(task_count, bool)
        or task_count != len(workloads)
        or not isinstance(trials, dict)
        or set(trials) != set(workloads)
        or not isinstance(trial_count, int)
        or isinstance(trial_count, bool)
        or trial_count <= 0
    ):
        raise ValueError("tuning sidecar task/trial counts are incomplete")
    report = metadata.get("task_report")
    if not isinstance(report, dict) or set(report) != {"supported", "unsupported"}:
        raise ValueError("tuning sidecar task_report is incomplete")
    for category in ("supported", "unsupported"):
        entries = report.get(category)
        if not isinstance(entries, list) or not all(
            isinstance(entry, dict)
            and isinstance(entry.get("template"), str)
            and isinstance(entry.get("workload_sha256"), str)
            and len(entry["workload_sha256"]) == 64
            for entry in entries
        ):
            raise ValueError(f"tuning sidecar task_report {category} entries are invalid")
    reported_supported = {entry["workload_sha256"] for entry in report["supported"]}
    reported_unsupported = {entry["workload_sha256"] for entry in report["unsupported"]}
    if reported_supported != set(workloads) or reported_supported & reported_unsupported:
        raise ValueError("tuning sidecar task_report does not match supported log coverage")
    log_path, actual_trials, successful, actual_trial_count = _load_log_summary(log_path)
    if actual_trial_count != trial_count:
        raise ValueError("tuning sidecar trial_count does not match the native AutoTVM log")
    if actual_trials != trials or set(actual_trials) != set(workloads):
        raise ValueError("tuning sidecar task coverage does not match the native AutoTVM log")
    if not set(workloads) <= successful:
        raise ValueError("native AutoTVM log has no successful record for every task")
    return metadata


@contextlib.contextmanager
def history_best(
    log_path,
    sidecar_path,
    *,
    model_id,
    model_sha256,
    backend,
    config_path,
    expected_tuning_options=None,
):
    """Validate a paired artifact before entering AutoTVM history-best."""
    validate_tuning_artifacts(
        log_path,
        sidecar_path,
        model_id=model_id,
        model_sha256=model_sha256,
        backend=backend,
        config_path=config_path,
        expected_tuning_options=expected_tuning_options,
    )
    with autotvm.apply_history_best(str(log_path)) as dispatch_context:
        yield dispatch_context


def build_tuning_options(backend, trials_per_task=None, timeout=None):
    """Resolve the reproducible per-backend options written to each sidecar."""
    validate_backend(backend)
    if trials_per_task is not None and (
        not isinstance(trials_per_task, int)
        or isinstance(trials_per_task, bool)
        or trials_per_task <= 0
    ):
        raise ValueError("trials_per_task must be a positive integer or null")
    default_timeout = 120 if backend == "fsim" else 180
    selected_timeout = default_timeout if timeout is None else timeout
    if (
        not isinstance(selected_timeout, int)
        or isinstance(selected_timeout, bool)
        or selected_timeout <= 0
    ):
        raise ValueError("timeout must be a positive integer number of seconds")
    return {
        "tuner": "grid_search",
        "trials_per_task": trials_per_task,
        "timeout": selected_timeout,
        "number": 1,
        "repeat": 1,
        "cooldown_interval": 0.0,
    }


def _load_model_pipeline(model_id):
    if model_id not in MODEL_PIPELINES:
        raise ValueError(f"unsupported MLPerf Tiny model {model_id!r}")
    pipeline_name = MODEL_PIPELINES[model_id][0]
    pipeline_path = Path(__file__).resolve().parent / pipeline_name / "model_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        f"mlperf_{model_id}_autotvm_model_pipeline", pipeline_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {model_id} model pipeline: {pipeline_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tune_model(
    model_id,
    backend,
    *,
    trials_per_task=None,
    timeout=None,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Tune supported VTA tasks for one prepared MLPerf Tiny model."""
    validate_backend(backend)
    if model_id not in MODEL_PIPELINES:
        raise ValueError(f"unsupported MLPerf Tiny model {model_id!r}")
    config_path, _ = _config_identity(DEFAULT_CONFIG_PATH)
    pipeline = _load_model_pipeline(model_id)
    _, model_dir, model_filename = MODEL_PIPELINES[model_id]
    model_path = Path(__file__).resolve().parent / model_id / model_dir / model_filename
    prepared = pipeline.prepare_model(model_path)
    tasks, task_report = extract_model_tasks(prepared)
    if not tasks:
        raise RuntimeError(
            f"{model_id} contains no supported VTA AutoTVM tasks; "
            f"task report: {json.dumps(task_report, sort_keys=True)}"
        )

    options = build_tuning_options(backend, trials_per_task, timeout)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{model_id}-{backend}-{run_id}"
    log_path = output_dir / f"{stem}.log"
    sidecar_path = output_dir / f"{stem}.json"
    callbacks = [autotvm.callback.log_to_file(str(log_path))]

    for task in tasks:
        total_configs = len(task.config_space)
        trial_budget = total_configs if options["trials_per_task"] is None else min(
            options["trials_per_task"], total_configs
        )
        if trial_budget <= 0:
            raise RuntimeError(f"AutoTVM task {task.name} has an empty configuration space")
        measure = measure_option(
            backend,
            timeout=options["timeout"],
            number=options["number"],
            repeat=options["repeat"],
            cooldown_interval=options["cooldown_interval"],
        )
        runner = measure["runner"]
        tuner = autotvm.tuner.GridSearchTuner(task)
        try:
            tuner.tune(
                n_trial=trial_budget,
                measure_option=measure,
                callbacks=callbacks,
            )
        finally:
            if getattr(runner, "server", None) is not None:
                runner.server.terminate()
                runner.server = None
            if getattr(runner, "tracker", None) is not None:
                runner.tracker.terminate()
                runner.tracker = None

    metadata = write_tuning_sidecar(
        log_path,
        sidecar_path,
        model_id=model_id,
        model_sha256=prepared.imported.model_sha256,
        backend=backend,
        config_path=config_path,
        tasks=tasks,
        tuning_options=options,
        task_report=task_report,
    )
    return log_path, sidecar_path, metadata


def tune_v1(backend, **kwargs):
    """Compatibility wrapper for the original V1 tuner API."""
    return tune_model("image_classification_v1", backend, **kwargs)


def _write_json_atomic(path, payload):
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _resume_entry(summary, model_id, *, backend, config_path, tuning_options):
    """Return a prior model result only when its paired artifact still validates."""
    entry = summary.get("results", {}).get(model_id)
    if not isinstance(entry, dict) or entry.get("status") != "succeeded":
        return None
    log_path = entry.get("log_path")
    sidecar_path = entry.get("sidecar_path")
    if not isinstance(log_path, str) or not isinstance(sidecar_path, str):
        return None
    pipeline = _load_model_pipeline(model_id)
    try:
        metadata = validate_tuning_artifacts(
            log_path,
            sidecar_path,
            model_id=model_id,
            model_sha256=pipeline.MODEL_SHA256,
            backend=backend,
            config_path=config_path,
            expected_tuning_options=tuning_options,
        )
    except (OSError, ValueError, RuntimeError):
        return None
    if metadata.get("task_count") != entry.get("task_count"):
        return None
    return {
        "status": "reused",
        "log_path": str(Path(log_path).resolve()),
        "sidecar_path": str(Path(sidecar_path).resolve()),
        "task_count": metadata["task_count"],
        "trial_count": metadata["trial_count"],
        "task_report": metadata["task_report"],
        "tuning_options": metadata["tuning_options"],
    }


def tune_all(
    backend,
    *,
    trials_per_task=None,
    timeout=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    resume_summary=None,
):
    """Tune all registered models sequentially and atomically record progress."""
    validate_backend(backend)
    config_path, config_sha256 = _config_identity(DEFAULT_CONFIG_PATH)
    options = build_tuning_options(backend, trials_per_task, timeout)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_order = list(MODEL_PIPELINES)
    if resume_summary is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        summary_path = output_dir / f"autotvm-all-{backend}-{run_id}.json"
        previous = None
    else:
        summary_path = Path(resume_summary).expanduser().resolve(strict=True)
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"resume summary is not valid JSON: {summary_path}") from error
        if (
            not isinstance(previous, dict)
            or previous.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or previous.get("model_order") != model_order
            or previous.get("backend") != backend
            or previous.get("config_path") != str(config_path)
            or previous.get("config_sha256") != config_sha256
            or previous.get("tuning_options") != options
            or not isinstance(previous.get("results"), dict)
        ):
            raise ValueError("resume summary does not match the current models/backend/config/options")
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_selector": "all",
        "model_order": model_order,
        "backend": backend,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "tuning_options": options,
        "results": {},
    }
    if previous is not None:
        summary["results"].update(previous["results"])
    _write_json_atomic(summary_path, summary)

    for model_id in model_order:
        reused = (
            _resume_entry(
                previous, model_id, backend=backend, config_path=config_path,
                tuning_options=options,
            )
            if previous is not None
            else None
        )
        if reused is not None:
            summary["results"][model_id] = reused
            _write_json_atomic(summary_path, summary)
            continue

        summary["results"][model_id] = {"status": "running"}
        _write_json_atomic(summary_path, summary)
        try:
            log_path, sidecar_path, metadata = tune_model(
                model_id,
                backend,
                trials_per_task=trials_per_task,
                timeout=timeout,
                output_dir=output_dir,
            )
            summary["results"][model_id] = {
                "status": "succeeded",
                "log_path": str(Path(log_path).resolve()),
                "sidecar_path": str(Path(sidecar_path).resolve()),
                "task_count": metadata["task_count"],
                "trial_count": metadata["trial_count"],
                "task_report": metadata["task_report"],
                "tuning_options": metadata["tuning_options"],
            }
        except Exception as error:  # Continue so per-model failures are all reported.
            summary["results"][model_id] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        _write_json_atomic(summary_path, summary)

    summary["summary_path"] = str(summary_path)
    _write_json_atomic(summary_path, summary)
    return summary


def _positive_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune MLPerf Tiny VTA schedules with AutoTVM"
    )
    parser.add_argument("--model", choices=(*MODEL_PIPELINES, "all"), required=True)
    parser.add_argument("--backend", choices=("fsim", "tsim"), required=True)
    parser.add_argument(
        "--trials-per-task",
        type=_positive_integer,
        help="bound grid search per task; default searches each complete task space",
    )
    parser.add_argument(
        "--timeout", type=_positive_integer, help="override backend default timeout in seconds"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume-summary", type=Path,
        help="resume --model all from an earlier aggregate summary with valid completed pairs",
    )
    args = parser.parse_args(argv)
    if args.resume_summary is not None and args.model != "all":
        parser.error("--resume-summary is only valid with --model all")
    if args.model == "all":
        summary = tune_all(
            args.backend,
            trials_per_task=args.trials_per_task,
            timeout=args.timeout,
            output_dir=args.output_dir,
            resume_summary=args.resume_summary,
        )
        print(f"Aggregate summary: {summary['summary_path']}")
        for model_id in summary["model_order"]:
            result = summary["results"][model_id]
            if result["status"] in {"succeeded", "reused"}:
                unsupported = len(result["task_report"]["unsupported"])
                print(
                    f"model={model_id} status={result['status']} backend={args.backend} "
                    f"tasks={result['task_count']} trials={result['trial_count']} "
                    f"unsupported={unsupported} log={result['log_path']} "
                    f"sidecar={result['sidecar_path']}"
                )
            else:
                print(f"model={model_id} status=failed error={result['error']}")
        return int(any(value["status"] == "failed" for value in summary["results"].values()))
    log_path, sidecar_path, metadata = tune_model(
        args.model,
        args.backend,
        trials_per_task=args.trials_per_task,
        timeout=args.timeout,
        output_dir=args.output_dir,
    )
    print(f"AutoTVM log: {log_path}")
    print(f"JSON sidecar: {sidecar_path}")
    print(
        f"model={metadata['model_id']} backend={metadata['backend']} "
        f"tasks={metadata['task_count']} trials={metadata['trial_count']} "
        f"config_sha256={metadata['config_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
