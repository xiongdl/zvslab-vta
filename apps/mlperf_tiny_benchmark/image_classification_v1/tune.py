#!/usr/bin/env python3
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

"""Random-tune one IC V1 VTA workload on FSIM and measure it on TSIM."""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_ROOT.parent / "build" / "autotvm" / "image_classification_v1"
SHARED_TUNER_PATH = APP_ROOT.parent / "autotvm_tuner.py"


def _load_shared_tuner():
    worker_import_path = str(APP_ROOT.parent)
    if worker_import_path not in sys.path:
        sys.path.insert(0, worker_import_path)
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if worker_import_path not in python_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            path for path in (*python_paths, worker_import_path) if path
        )
    spec = importlib.util.spec_from_file_location("autotvm_tuner", SHARED_TUNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared_tuner()


def _load_fused_tasks():
    worker_import_path = str(APP_ROOT)
    if worker_import_path not in sys.path:
        sys.path.insert(0, worker_import_path)
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if worker_import_path not in python_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            path for path in (*python_paths, worker_import_path) if path
        )
    registered = sys.modules.get("fused_tasks")
    if (
        registered is not None
        and Path(registered.__file__).resolve() == (APP_ROOT / "fused_tasks.py").resolve()
    ):
        return registered
    spec = importlib.util.spec_from_file_location("fused_tasks", APP_ROOT / "fused_tasks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fused = _load_fused_tasks()


def select_workload(tasks, workload_index):
    """Select exactly one task by its zero-based outlined fusion occurrence index."""
    if isinstance(workload_index, bool) or not isinstance(workload_index, int):
        raise ValueError(f"workload index must be an integer, got {workload_index!r}")
    if not 0 <= workload_index < len(tasks):
        high = len(tasks) - 1
        raise ValueError(
            f"workload index {workload_index} is out of range; "
            f"valid workload indices are 0..{high} (fused occurrence count: {len(tasks)})"
        )
    return tasks[workload_index]


def build_tuning_options(trials=32, timeout=120):
    """Return the command's effective defaults and selected runtime backends."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("trials must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer in seconds")
    return {
        "tuner": "random",
        "backend": "fsim",
        "runner": "local",
        "trials": trials,
        "timeout": timeout,
    }


def prepare_v1_workloads():
    """Prepare IC V1 and build one complete task per outlined fusion occurrence."""
    import vta

    pipeline = shared._load_model_pipeline("image_classification_v1")
    _, model_dir, model_filename = shared.MODEL_PIPELINES["image_classification_v1"]
    model_path = APP_ROOT / model_dir / model_filename
    prepared = pipeline.prepare_model(model_path)
    identities = fused.extract_fused_identities(prepared)
    tasks = [fused.create_task(identity, vta.get_env().target) for identity in identities]
    if not tasks:
        raise RuntimeError("image_classification_v1 contains no supported VTA AutoTVM workloads")
    return prepared, identities, tasks


def prepare_v1_tasks():
    """Compatibility helper returning complete fused tasks in occurrence order."""
    return prepare_v1_workloads()[2]


def _logical_mac_count(task):
    flop_count = task.flop
    try:
        integer_flops = int(flop_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"AutoTVM task FLOP count is not an integer: {flop_count!r}") from error
    if integer_flops <= 0 or integer_flops != flop_count or integer_flops % 2:
        raise ValueError(f"AutoTVM task FLOP count must be a positive even integer: {flop_count!r}")
    return integer_flops // 2


def _cleanup_runner(runner):
    for name in ("server", "tracker"):
        process = getattr(runner, name, None)
        if process is not None:
            process.terminate()
            setattr(runner, name, None)


def validate_fusion_result(
    result, identity, model_sha256, geometry_sha256, expected_workload_sha256=None
):
    """Reject artifacts whose measurement does not identify this full fusion."""
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise ValueError("unsupported or missing fused tuning result schema")
    if result.get("measurement_scope") != "isolated_complete_vta_conv_fusion":
        raise ValueError("result is not a complete fused-convolution measurement")
    if result.get("fusion_sha256") != identity.sha256:
        raise ValueError("fused tuning result identity does not match this occurrence")
    if result.get("fusion_identity") != json.loads(identity.canonical_json()):
        raise ValueError("fused tuning result semantic attributes do not match")
    if result.get("template") != fused.TASK_NAME:
        raise ValueError("result was not measured with the complete fused task template")
    if result.get("real_conv_lowering") is not True:
        raise ValueError("fused tuning result did not verify real model Conv lowering")
    if result.get("model_sha256") != model_sha256:
        raise ValueError("fused tuning result model hash does not match")
    if result.get("geometry_sha256") != geometry_sha256:
        raise ValueError("fused tuning result geometry hash does not match")
    if tuple(result.get("conv_schedule_key", ())) != fused.conv_schedule_key(identity):
        raise ValueError("fused tuning result Conv schedule key does not match")
    workload_id = result.get("workload_sha256")
    if not isinstance(workload_id, str) or len(workload_id) != 64:
        raise ValueError("fused tuning result workload identity is missing or invalid")
    if expected_workload_sha256 is not None and workload_id != expected_workload_sha256:
        raise ValueError("fused tuning result AutoTVM workload hash does not match")
    for key in ("fsim_log_sha256", "best_fsim_log_sha256"):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"fused tuning result {key} is missing or invalid")
    return result


def run_tuning(workload_index, *, output_dir=DEFAULT_OUTPUT_DIR, trials=32, timeout=120):
    """Tune one selected workload with FSIM, then measure its best record on TSIM."""
    options = build_tuning_options(trials, timeout)
    config_path = os.environ.get("VTA_CONFIG_FILE", shared.DEFAULT_CONFIG_PATH)
    geometry_path, geometry_sha = shared._config_identity(config_path)
    prepared, identities, tasks = prepare_v1_workloads()
    task = select_workload(tasks, workload_index)
    identity = identities[workload_index]
    if task.name != fused.TASK_NAME:
        raise ValueError("selected task is not the complete IC V1 fused task")
    workload_id = shared._task_workload_id(task)
    mac_count = _logical_mac_count(task)

    shared.validate_backend("fsim")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"image_classification_v1-workload-{workload_index}-{run_id}"
    fsim_log = output_dir / f"{stem}-fsim.log"
    best_log = output_dir / f"{stem}-best.log"
    result_path = output_dir / f"{stem}-result.json"

    trial_count = min(options["trials"], len(task.config_space))
    if trial_count <= 0:
        raise RuntimeError(f"AutoTVM workload {workload_index} has an empty configuration space")

    # A candidate can abort the FSIM RPC server process. AutoTVM's LocalRunner
    # keeps that server for its lifetime, so a later candidate would otherwise
    # continue against a dead server (or remain blocked waiting for its RPC
    # future). Keep RandomTuner's visited-config state, but give each trial a
    # fresh local runner so a failed candidate cannot poison the remaining
    # search.
    tuner = shared.autotvm.tuner.RandomTuner(task)
    log_callback = shared.autotvm.callback.log_to_file(str(fsim_log))
    for _ in range(trial_count):
        fsim_measure = shared.measure_option(
            "fsim", timeout=timeout, number=1, repeat=1, cooldown_interval=0
        )
        fsim_runner = fsim_measure["runner"]
        try:
            tuner.tune(
                n_trial=1,
                measure_option=fsim_measure,
                callbacks=[log_callback],
            )
        finally:
            _cleanup_runner(fsim_runner)

    shared.autotvm.record.pick_best(str(fsim_log), str(best_log))
    best_records = list(shared.autotvm.record.load_from_file(str(best_log)))
    successful = [record for record in best_records if record[1].error_no == shared.MeasureErrorNo.NO_ERROR]
    if not successful:
        raise RuntimeError(f"FSIM tuning produced no successful record for workload {workload_index}")
    best_input, _ = min(successful, key=lambda record: sum(record[1].costs))
    if best_input.task.workload != task.workload:
        raise ValueError("FSIM record does not contain the selected complete fusion task")
    if best_input.target.kind.name != "vta":
        raise ValueError("FSIM best record target is not VTA")
    selected_config = best_input.config
    real_lowering = fused.lower_with_fused_config(prepared, identity, selected_config)
    conv_schedule_key = fused.conv_schedule_key(identity)

    # The environment selector is process-local. Each helper validates that it
    # matches the backend it is about to load and measure.
    os.environ["VTA_BACKEND"] = "tsim"
    tsim_runner = shared.create_runner("tsim", timeout=timeout, number=1, repeat=1, cooldown_interval=0)
    tsim_builder = shared.autotvm.LocalBuilder(n_parallel=1)
    try:
        tsim_runner.set_task(task)
        tsim_builder.set_task(task, tsim_runner.get_build_kwargs())
        tsim_builds = tsim_builder.build([best_input])
        build_result = tsim_builds[0] if tsim_builds else None
        build_error = getattr(build_result, "error", None) if build_result is not None else None
        build_failed = build_result is None or build_error is not None
        if hasattr(build_result, "error_no"):
            build_failed = build_result.error_no != shared.MeasureErrorNo.NO_ERROR
            build_error = build_result.costs
        if build_failed:
            failure = build_error if build_result is not None else "builder returned no result"
            raise RuntimeError(f"TSIM could not build the selected FSIM schedule: {failure}")
        tsim_results = tsim_runner.run([best_input], tsim_builds)
    finally:
        _cleanup_runner(tsim_runner)
    if not tsim_results or tsim_results[0].error_no != shared.MeasureErrorNo.NO_ERROR:
        failure = tsim_results[0].costs if tsim_results else "runner returned no result"
        raise RuntimeError(f"TSIM failed to measure the selected FSIM schedule: {failure}")
    cycles = tsim_results[0].costs[0]
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise RuntimeError(f"TSIM cycle_count must be a positive integer, got {cycles!r}")

    _, model_dir, model_filename = shared.MODEL_PIPELINES["image_classification_v1"]
    model_path = APP_ROOT / model_dir / model_filename
    model_sha = shared._sha256_file(model_path)
    fsim_log_sha = shared._sha256_file(fsim_log)
    best_log_sha = shared._sha256_file(best_log)
    result = {
        "schema_version": 1,
        "measurement_scope": "isolated_complete_vta_conv_fusion",
        "workload_index": workload_index,
        "occurrence": identity.occurrence,
        "symbol": identity.symbol,
        "template": task.name,
        "workload_sha256": workload_id,
        "fusion_identity": json.loads(identity.canonical_json()),
        "fusion_sha256": identity.sha256,
        "model": "image_classification_v1",
        "model_sha256": model_sha,
        "geometry_path": str(geometry_path),
        "geometry_sha256": geometry_sha,
        "conv_schedule_key": conv_schedule_key,
        "conv_config": selected_config.to_json_dict(),
        "real_conv_lowering": real_lowering.schedule is not None,
        "fsim_log_sha256": fsim_log_sha,
        "best_fsim_log_sha256": best_log_sha,
        "mac_count": mac_count,
        "tsim_cycles": cycles,
        "fsim_trials": trial_count,
        "fsim_log": str(fsim_log),
        "best_fsim_log": str(best_log),
        "result_json": str(result_path),
    }
    validate_fusion_result(result, identity, model_sha, geometry_sha, workload_id)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload-index",
        type=int,
        required=True,
        help="zero-based outlined fusion occurrence index",
    )
    parser.add_argument("--trials", type=_positive_int, default=32, help="maximum random search trials (default: 32)")
    parser.add_argument("--timeout", type=_positive_int, default=120, help="per-measurement timeout in seconds (default: 120)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for native logs and result JSON")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        result = run_tuning(
            args.workload_index,
            output_dir=args.output_dir,
            trials=args.trials,
            timeout=args.timeout,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Workload index: {result['workload_index']}")
    print(f"Fusion occurrence: {result['occurrence']} ({result['symbol']})")
    print(f"Workload template: {result['template']}")
    print(f"Workload SHA-256: {result['workload_sha256']}")
    print(f"Logical MAC count: {result['mac_count']} MACs")
    print(f"TSIM cycle_count: {result['tsim_cycles']} cycles")
    print(f"FSIM log: {result['fsim_log']}")
    print(f"Best FSIM record: {result['best_fsim_log']}")
    print(f"Result JSON: {result['result_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
