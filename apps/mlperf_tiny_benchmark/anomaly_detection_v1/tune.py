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

"""Random-tune one AD V1 VTA workload on FSIM and measure it on TSIM."""

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_ROOT.parent / "build" / "autotvm" / "anomaly_detection_v1"
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
    worker_import_path = str(APP_ROOT.parent)
    if worker_import_path not in sys.path:
        sys.path.insert(0, worker_import_path)
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if worker_import_path not in python_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(
            path for path in (*python_paths, worker_import_path) if path
        )
    module_path = (APP_ROOT.parent / "fused_tasks.py").resolve()
    for name in ("fused_tasks", "mlperf_tiny_fused_tasks"):
        registered = sys.modules.get(name)
        if registered is not None and Path(registered.__file__).resolve() == module_path:
            sys.modules["fused_tasks"] = registered
            return registered
    spec = importlib.util.spec_from_file_location("fused_tasks", APP_ROOT.parent / "fused_tasks.py")
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
    """Prepare AD V1 and build one complete task per outlined fusion occurrence."""
    import vta

    pipeline = shared._load_model_pipeline("anomaly_detection_v1")
    _, model_dir, model_filename = shared.MODEL_PIPELINES["anomaly_detection_v1"]
    model_path = APP_ROOT / model_dir / model_filename
    prepared = pipeline.prepare_model(model_path)
    identities = fused.extract_fused_identities(prepared)
    tasks = [fused.create_task(identity, vta.get_env().target) for identity in identities]
    if not tasks:
        raise RuntimeError("anomaly_detection_v1 contains no supported VTA AutoTVM workloads")
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


def _freeze_json_value(value):
    """Normalize tuple/list structures to one comparable JSON-like form."""
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_json_value(item)) for key, item in value.items()))
    return value


def _config_json(config):
    return json.dumps(config.to_json_dict(), sort_keys=True, separators=(",", ":"), default=str)


def _record_config_matches(left, right):
    return _config_json(left) == _config_json(right)


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
    if result.get("model") != "anomaly_detection_v1":
        raise ValueError("fused tuning result model identity does not match")
    if (
        result.get("occurrence") != identity.occurrence
        or result.get("symbol") != identity.symbol
    ):
        raise ValueError("fused tuning result occurrence does not match this fusion")
    if result.get("template") != fused.TASK_NAME:
        raise ValueError("result was not measured with the complete fused task template")
    if result.get("real_conv_lowering") is not True:
        raise ValueError("fused tuning result did not verify real model Conv lowering")
    if result.get("model_sha256") != model_sha256:
        raise ValueError("fused tuning result model hash does not match")
    if result.get("geometry_sha256") != geometry_sha256:
        raise ValueError("fused tuning result geometry hash does not match")
    if _freeze_json_value(result.get("conv_schedule_key", ())) != _freeze_json_value(
        fused.conv_schedule_key(identity)
    ):
        raise ValueError("fused tuning result Conv schedule key does not match")
    workload_id = result.get("workload_sha256")
    if not isinstance(workload_id, str) or len(workload_id) != 64:
        raise ValueError("fused tuning result workload identity is missing or invalid")
    if expected_workload_sha256 is not None and workload_id != expected_workload_sha256:
        raise ValueError("fused tuning result AutoTVM workload hash does not match")
    if result.get("artifact_kind") == "self_contained_native_best_v1":
        value = result.get("best_native_record_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("fused tuning best native record hash is missing or invalid")
    else:
        for key in ("fsim_log_sha256", "best_fsim_log_sha256"):
            value = result.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"fused tuning result {key} is missing or invalid")
    if result.get("measurement_protocol") != shared.TSIM_MEASUREMENT_PROTOCOL:
        raise ValueError(
            "fused tuning result has missing or incompatible TSIM cycles; "
            "rerun a single-call TSIM measurement"
        )
    cycles = result.get("tsim_cycles")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ValueError(
            "fused tuning result TSIM cycles must be a positive single-call count; "
            "rerun the TSIM measurement"
        )
    return result


def _record_cost(record):
    try:
        return sum(float(cost) for cost in record[1].costs)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("AutoTVM record has invalid measurement costs") from error


def _is_vta_record_target(target):
    return target.kind.name == "vta" or (
        "vta" in target.keys and target.attrs.get("device") == "vta"
    )


def _record_matches_task(measure_input, task):
    return (
        _freeze_json_value(measure_input.task.workload) == _freeze_json_value(task.workload)
        and _is_vta_record_target(measure_input.target)
    )


def _load_result_records(result, task, result_dir=None):
    """Verify referenced AutoTVM bytes and return the selected native record."""
    if result.get("artifact_kind") == "self_contained_native_best_v1":
        raw_path = result.get("best_native_record")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("fused tuning best native record path is missing")
        path = Path(raw_path).expanduser()
        if not path.is_absolute() and result_dir is not None:
            path = Path(result_dir) / path
        try:
            actual_hash = shared._sha256_file(path)
        except OSError as error:
            raise ValueError(f"fused tuning best native record is missing or unreadable: {path}") from error
        if actual_hash != result.get("best_native_record_sha256"):
            raise ValueError("fused tuning best native record SHA-256 does not match")
        try:
            records = list(shared.autotvm.record.load_from_file(str(path)))
        except Exception as error:
            raise ValueError("fused tuning best native record cannot be decoded") from error
        matches = [
            (measure_input, measure_result)
            for measure_input, measure_result in records
            if _record_matches_task(measure_input, task)
            and measure_result.error_no == shared.MeasureErrorNo.NO_ERROR
        ]
        if len(matches) != 1:
            raise ValueError("fused tuning best native record must contain one successful matching task")
        best_input, best_result = matches[0]
        if not isinstance(result.get("conv_config"), dict) or _freeze_json_value(
            result["conv_config"]
        ) != _freeze_json_value(best_input.config.to_json_dict()):
            raise ValueError("fused tuning result Conv config does not match the best native record")
        cycles = best_result.costs[0] if best_result.costs else None
        if cycles != result.get("tsim_cycles"):
            raise ValueError("fused tuning result cycles do not match the best native record")
        return best_input

    paths = {}
    for path_key, hash_key in (
        ("fsim_log", "fsim_log_sha256"),
        ("best_fsim_log", "best_fsim_log_sha256"),
    ):
        raw_path = result.get(path_key)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"fused tuning result {path_key} is missing")
        path = Path(raw_path).expanduser()
        try:
            actual_hash = shared._sha256_file(path)
        except OSError as error:
            raise ValueError(f"fused tuning result {path_key} is missing or unreadable: {path}") from error
        if actual_hash != result.get(hash_key):
            raise ValueError(f"fused tuning result {path_key} SHA-256 does not match")
        paths[path_key] = path

    try:
        fsim_records = list(shared.autotvm.record.load_from_file(str(paths["fsim_log"])))
        best_records = list(shared.autotvm.record.load_from_file(str(paths["best_fsim_log"])))
    except Exception as error:
        raise ValueError("fused tuning AutoTVM logs cannot be decoded") from error
    successful_fsim = [
        (measure_input, measure_result)
        for measure_input, measure_result in fsim_records
        if _record_matches_task(measure_input, task)
        and measure_result.error_no == shared.MeasureErrorNo.NO_ERROR
    ]
    successful_best = [
        (measure_input, measure_result)
        for measure_input, measure_result in best_records
        if _record_matches_task(measure_input, task)
        and measure_result.error_no == shared.MeasureErrorNo.NO_ERROR
    ]
    if not successful_fsim:
        raise ValueError("fused tuning FSIM log has no successful record for the selected task")
    if not successful_best:
        raise ValueError("fused tuning best log has no successful record for the selected task")
    best_input, best_result = min(successful_best, key=lambda record: _record_cost(record))
    if not any(
        _record_config_matches(best_input.config, fsim_input.config)
        and tuple(best_result.costs) == tuple(fsim_result.costs)
        for fsim_input, fsim_result in successful_fsim
    ):
        raise ValueError("selected best record is not a successful matching record in the FSIM log")
    if not isinstance(result.get("conv_config"), dict) or _freeze_json_value(
        result["conv_config"]
    ) != _freeze_json_value(best_input.config.to_json_dict()):
        raise ValueError("fused tuning result Conv config does not match the best FSIM record")
    return best_input


def replay_result(result_path, *, expected_workload_index=None):
    """Validate a saved result, then apply its config to the real outlined fusion."""
    result_path = Path(result_path).expanduser().resolve(strict=True)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"fused tuning result is not readable JSON: {result_path}") from error
    if not isinstance(result, dict):
        raise ValueError("fused tuning result must be a JSON object")
    workload_index = result.get("workload_index")
    if isinstance(workload_index, bool) or not isinstance(workload_index, int):
        raise ValueError("fused tuning result workload index is missing or invalid")
    if expected_workload_index is not None and workload_index != expected_workload_index:
        raise ValueError("fused tuning result workload index does not match the requested index")

    prepared, identities, tasks = prepare_v1_workloads()
    task = select_workload(tasks, workload_index)
    identity = identities[workload_index]
    _, model_dir, model_filename = shared.MODEL_PIPELINES["anomaly_detection_v1"]
    model_sha = shared._sha256_file(APP_ROOT / model_dir / model_filename)
    config_path = os.environ.get("VTA_CONFIG_FILE", shared.DEFAULT_CONFIG_PATH)
    geometry_path, geometry_sha = shared._config_identity(config_path)
    recorded_geometry_path = result.get("geometry_path")
    if not isinstance(recorded_geometry_path, str) or Path(
        recorded_geometry_path
    ).expanduser().resolve() != geometry_path:
        raise ValueError("fused tuning result geometry path does not match the active config")
    workload_id = shared._task_workload_id(task)
    validate_fusion_result(result, identity, model_sha, geometry_sha, workload_id)
    best_input = _load_result_records(result, task, result_path.parent)
    lowered = fused.lower_with_fused_config(prepared, identity, best_input.config)
    if lowered.schedule is None:
        raise ValueError("saved Conv config did not produce a real model fusion schedule")
    return {"result": result, "config": best_input.config, "lowered": lowered}


def run_tuning(workload_index, *, output_dir=DEFAULT_OUTPUT_DIR, trials=32, timeout=120):
    """Tune one selected workload with FSIM, then measure its best record on TSIM."""
    options = build_tuning_options(trials, timeout)
    config_path = os.environ.get("VTA_CONFIG_FILE", shared.DEFAULT_CONFIG_PATH)
    geometry_path, geometry_sha = shared._config_identity(config_path)
    prepared, identities, tasks = prepare_v1_workloads()
    task = select_workload(tasks, workload_index)
    identity = identities[workload_index]
    if task.name != fused.TASK_NAME:
        raise ValueError("selected task is not the complete AD V1 fused task")
    workload_id = shared._task_workload_id(task)
    mac_count = _logical_mac_count(task)

    shared.validate_backend("fsim")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"anomaly_detection_v1-workload-{workload_index}-{run_id}"
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
    if not _is_vta_record_target(best_input.target):
        raise ValueError("FSIM best record target is not VTA")
    selected_config = best_input.config
    real_lowering = fused.lower_with_fused_config(prepared, identity, selected_config)
    conv_schedule_key = fused.conv_schedule_key(identity)

    # The environment selector is process-local. Each helper validates that it
    # matches the backend it is about to load and measure.
    previous_backend = os.environ.get("VTA_BACKEND")
    tsim_runner = None
    try:
        os.environ["VTA_BACKEND"] = "tsim"
        tsim_runner = shared.create_runner(
            "tsim", timeout=timeout, number=1, repeat=1, cooldown_interval=0
        )
        tsim_builder = shared.autotvm.LocalBuilder(n_parallel=1)
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
        try:
            if tsim_runner is not None:
                _cleanup_runner(tsim_runner)
        finally:
            if previous_backend is None:
                os.environ.pop("VTA_BACKEND", None)
            else:
                os.environ["VTA_BACKEND"] = previous_backend
    if not tsim_results or tsim_results[0].error_no != shared.MeasureErrorNo.NO_ERROR:
        failure = tsim_results[0].costs if tsim_results else "runner returned no result"
        raise RuntimeError(f"TSIM failed to measure the selected FSIM schedule: {failure}")
    cycles = tsim_results[0].costs[0]
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise RuntimeError(f"TSIM cycle_count must be a positive integer, got {cycles!r}")

    _, model_dir, model_filename = shared.MODEL_PIPELINES["anomaly_detection_v1"]
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
        "model": "anomaly_detection_v1",
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
        "measurement_protocol": shared.TSIM_MEASUREMENT_PROTOCOL,
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
        help="zero-based outlined fusion occurrence index (required for tuning)",
    )
    parser.add_argument(
        "--replay-result",
        type=Path,
        help="validate a saved result and apply its config to the real outlined fusion",
    )
    parser.add_argument("--trials", type=_positive_int, default=32, help="maximum random search trials (default: 32)")
    parser.add_argument("--timeout", type=_positive_int, default=120, help="per-measurement timeout in seconds (default: 120)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for native logs and result JSON")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.replay_result is not None:
        try:
            replay = replay_result(
                args.replay_result, expected_workload_index=args.workload_index
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"Validated result: {args.replay_result}")
        print(f"Fusion occurrence: {replay['result']['occurrence']} ({replay['result']['symbol']})")
        print(f"Workload SHA-256: {replay['result']['workload_sha256']}")
        print(f"Conv config: {json.dumps(replay['config'].to_json_dict(), sort_keys=True)}")
        print("Real outlined model fusion lowering: verified")
        return 0
    if args.workload_index is None:
        raise SystemExit("--workload-index is required unless --replay-result is supplied")
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
