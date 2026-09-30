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


def select_workload(tasks, workload_index):
    """Select exactly one task by its zero-based supported extraction index."""
    if isinstance(workload_index, bool) or not isinstance(workload_index, int):
        raise ValueError(f"workload index must be an integer, got {workload_index!r}")
    if not 0 <= workload_index < len(tasks):
        high = len(tasks) - 1
        raise ValueError(
            f"workload index {workload_index} is out of range; "
            f"valid workload indices are 0..{high} (supported workload count: {len(tasks)})"
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


def prepare_v1_tasks():
    """Prepare the committed IC V1 graph and extract its supported VTA tasks."""
    pipeline = shared._load_model_pipeline("image_classification_v1")
    _, model_dir, model_filename = shared.MODEL_PIPELINES["image_classification_v1"]
    model_path = APP_ROOT / model_dir / model_filename
    prepared = pipeline.prepare_model(model_path)
    tasks, _ = shared.extract_model_tasks(prepared)
    if not tasks:
        raise RuntimeError("image_classification_v1 contains no supported VTA AutoTVM workloads")
    return tasks


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


def run_tuning(workload_index, *, output_dir=DEFAULT_OUTPUT_DIR, trials=32, timeout=120):
    """Tune one selected workload with FSIM, then measure its best record on TSIM."""
    options = build_tuning_options(trials, timeout)
    config_path = os.environ.get("VTA_CONFIG_FILE", shared.DEFAULT_CONFIG_PATH)
    shared._config_identity(config_path)
    tasks = prepare_v1_tasks()
    task = select_workload(tasks, workload_index)
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

    # The environment selector is process-local. Each helper validates that it
    # matches the backend it is about to load and measure.
    os.environ["VTA_BACKEND"] = "tsim"
    tsim_runner = shared.create_runner("tsim", timeout=timeout, number=1, repeat=1, cooldown_interval=0)
    tsim_builder = shared.autotvm.LocalBuilder(n_parallel=1)
    try:
        tsim_runner.set_task(task)
        tsim_builder.set_task(task, tsim_runner.get_build_kwargs())
        tsim_builds = tsim_builder.build([best_input])
        if not tsim_builds or tsim_builds[0].error is not None:
            failure = tsim_builds[0].error if tsim_builds else "builder returned no result"
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

    result = {
        "workload_index": workload_index,
        "template": task.name,
        "workload_sha256": workload_id,
        "mac_count": mac_count,
        "tsim_cycles": cycles,
        "fsim_trials": trial_count,
        "fsim_log": str(fsim_log),
        "best_fsim_log": str(best_log),
        "result_json": str(result_path),
    }
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
    parser.add_argument("--workload-index", type=int, required=True, help="zero-based supported workload extraction index")
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
