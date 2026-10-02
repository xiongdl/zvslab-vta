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

"""Adaptive FSIM search and exhaustive TSIM evaluation for Visual Wake Words V1 fusions."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TUNE_DIR = Path(__file__).resolve().parent
APP_ROOT = TUNE_DIR.parent
REPO_ROOT = APP_ROOT.parents[3]
BUILD_ROOT = APP_ROOT / "build" / "two_stage_tuning"


def _load_legacy():
    path = APP_ROOT / "tune.py"
    spec = importlib.util.spec_from_file_location("ic_v1_legacy_tune", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_local_module(name):
    path = (APP_ROOT.parent / "image_classification_v1" / "tune" / f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"ic_v1_two_stage_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy()
search = _load_local_module("search")
measurement = _load_local_module("measurement")
artifacts = _load_local_module("artifacts")


def _sha256_json(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _space_size(task):
    config_space = task.config_space
    raw_size = int(config_space.range_length)
    valid_size = int(config_space.subrange_length(0, raw_size))
    if raw_size <= 0 or valid_size <= 0:
        raise RuntimeError(f"AutoTVM task {task.name} has no valid configurations")
    return raw_size, valid_size


def _select_indexed_workload(tasks, workload_index):
    """Validate an occurrence index and return it together with its task."""
    task = legacy.select_workload(tasks, workload_index)
    return workload_index, task


def _task_run_identity(index, identity, task, geometry_path, geometry_sha256,
                       model_path, trial_batch, min_successful,
                       fsim_timeout, tsim_timeout):
    raw_size, valid_size = _space_size(task)
    return {
        "model": "visual_wake_words_v1",
        "model_sha256": legacy.shared._sha256_file(model_path),
        "geometry_path": str(geometry_path),
        "geometry_sha256": geometry_sha256,
        "workload_sha256": legacy.shared._task_workload_id(task),
        "fusion_sha256": identity.sha256,
        "symbol": identity.symbol,
        "occurrence": identity.occurrence,
        "workload_index": index,
        "config_space_size": raw_size,
        "valid_config_space_size": valid_size,
        "trial_batch": trial_batch,
        "min_successful": min_successful,
        "fsim_timeout_seconds": fsim_timeout,
        "tsim_timeout_seconds": tsim_timeout,
    }


def _record_path(run_dir, index, backend):
    return run_dir / f"workload-{index:03d}-{backend}.log"


def _state_path(run_dir, index, backend):
    return run_dir / f"workload-{index:03d}-{backend}.json"


def _append_record(path, measure_input, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(legacy.shared.autotvm.record.encode(measure_input, result) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _config_identity(config):
    return json.dumps(config.to_json_dict(), sort_keys=True, separators=(",", ":"))


def _select_deployable_tsim_candidate(tsim_state, records, prepared, identity):
    """Choose the fastest measured config that lowers for the real fusion."""
    successful_records = {
        _config_identity(measure_input.config): measure_input
        for measure_input, result in records
        if result.error_no == legacy.shared.MeasureErrorNo.NO_ERROR
    }
    candidates = sorted(
        (
            candidate
            for candidate in tsim_state["candidates"]
            if candidate.get("error") is None
            and candidate.get("tsim_cycles") is not None
            and int(candidate["tsim_cycles"]) > 0
        ),
        key=lambda candidate: (int(candidate["tsim_cycles"]), candidate["config_index"]),
    )
    rejected = []
    for candidate in candidates:
        measure_input = successful_records.get(
            json.dumps(candidate["config"], sort_keys=True, separators=(",", ":"))
        )
        if measure_input is None:
            rejected.append({"config_index": candidate["config_index"], "reason": "missing native TSIM record"})
            continue
        try:
            lowered = legacy.fused.lower_with_fused_config(
                prepared, identity, measure_input.config
            )
        except Exception as exc:  # an AutoTVM-measured schedule may exceed real VTA limits
            error_lines = str(exc).splitlines()
            reason = next(
                (
                    line.strip()
                    for marker in ("Allocation exceed bound", "Check failed:", "InternalError:")
                    for line in error_lines
                    if marker in line
                ),
                error_lines[0] if error_lines else type(exc).__name__,
            )
            rejected.append({
                "config_index": candidate["config_index"],
                "reason": reason[:240],
            })
            continue
        if lowered.schedule is None:
            rejected.append({"config_index": candidate["config_index"], "reason": "lowering returned no schedule"})
            continue
        return candidate, measure_input, lowered, rejected
    raise ValueError(
        f"no measured TSIM candidate lowers for real fusion occurrence {identity.occurrence}"
    )


def _validate_fsim_resume(state, native_log, task):
    """Require durable progress and native AutoTVM records to describe the same trials."""
    attempted = state["attempted_count"]
    if not attempted:
        return
    if not native_log.is_file():
        raise ValueError(f"FSIM native log is missing for persisted progress: {native_log}")
    records = list(legacy.shared.autotvm.record.load_from_file(str(native_log)))
    logged_indices = []
    successful_configs = set()
    for measure_input, result in records:
        if not legacy._record_matches_task(measure_input, task):
            raise ValueError("FSIM log contains a record for another task or target")
        index = int(measure_input.config.index)
        logged_indices.append(index)
        if result.error_no == legacy.shared.MeasureErrorNo.NO_ERROR:
            successful_configs.add(_config_identity(measure_input.config))
    if len(records) != attempted or logged_indices != state["visited_indices"]:
        raise ValueError("FSIM log and persisted visited trial sequence disagree")
    saved_configs = set(state["successful_configurations"])
    if successful_configs != saved_configs:
        raise ValueError("FSIM log and persisted successful schedule set disagree")


def _fsim_worker(args):
    import vta
    from tvm import autotvm
    from tvm.autotvm.measure import MeasureInput

    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared, identities, tasks = legacy.prepare_v1_workloads()
    index, task = _select_indexed_workload(tasks, args.workload_index)
    identity = identities[index]
    config_path, geometry_sha = legacy.shared._config_identity(
        os.environ.get("VTA_CONFIG_FILE", legacy.shared.DEFAULT_CONFIG_PATH)
    )
    _, model_dir, model_filename = legacy.shared.MODEL_PIPELINES["visual_wake_words_v1"]
    model_path = APP_ROOT / model_dir / model_filename
    run_identity = _task_run_identity(
        index, identity, task, config_path, geometry_sha, model_path,
        args.trial_batch, args.min_successful, args.fsim_timeout, args.tsim_timeout,
    )
    state_path = _state_path(run_dir, index, "fsim")
    if args.resume:
        state = search.load_state(state_path, run_identity)
    else:
        if state_path.exists():
            raise ValueError(f"FSIM state already exists; choose --resume or a new run: {state_path}")
        state = search.create_state(run_identity, state_path)
    native_log = _record_path(run_dir, index, "fsim")
    if args.resume:
        _validate_fsim_resume(state, native_log, task)

    while search.should_continue(state):
        tuner = autotvm.tuner.RandomTuner(task)
        tuner.visited = list(state["visited_indices"])
        batch_size = search.next_batch_size(state)
        configs = tuner.next_batch(batch_size)
        if not configs:
            state["stop_reason"] = "configuration_space_exhausted"
            search.save_state(state)
            break
        for config in configs:
            config_index = int(config.index)
            measure_input = MeasureInput(task.target, task, config)
            try:
                result = measurement.measure_candidate(
                    task, config, "fsim", timeout=args.fsim_timeout
                )
                successful = result.error_no == legacy.shared.MeasureErrorNo.NO_ERROR
                failure = None if successful else repr(result.costs)
            except legacy.shared.SimulatorInfrastructureError as error:
                search.record_infrastructure_error(state, "fsim", config_index, error)
                search.save_state(state)
                raise
            except Exception as error:  # retain candidate-local compiler/measurement failures
                result = legacy.shared.MeasureResult(
                    (type(error).__name__, str(error)),
                    legacy.shared.MeasureErrorNo.RUNTIME_DEVICE, 0.0,
                    datetime.now(timezone.utc).timestamp(),
                )
                successful = False
                failure = f"{type(error).__name__}: {error}"
            _append_record(native_log, measure_input, result)
            search.record_trial(
                state, config_index, config.to_json_dict(),
                successful=successful, error=failure,
            )
            search.save_state(state)
            print(
                f"FSIM workload={index} attempt={state['attempted_count']} "
                f"successes={search.success_count(state)} "
                f"space={run_identity['valid_config_space_size']} "
                f"stop={search.stop_reason(state) or 'searching'}",
                flush=True,
            )
    if state.get("stop_reason") is None:
        state["stop_reason"] = search.stop_reason(state)
    search.save_state(state)
    print(json.dumps({
        "backend": "fsim", "workload_index": index,
        "attempted_count": state["attempted_count"],
        "successful_schedule_count": search.success_count(state),
        "valid_config_space_size": run_identity["valid_config_space_size"],
        "stop_reason": state["stop_reason"], "native_log": str(native_log),
        "state": str(state_path),
    }, sort_keys=True), flush=True)
    return 0


def _tsim_worker(args):
    import vta
    from tvm import autotvm

    run_dir = args.run_dir.expanduser().resolve()
    prepared, identities, tasks = legacy.prepare_v1_workloads()
    index, task = _select_indexed_workload(tasks, args.workload_index)
    identity = identities[index]
    config_path, geometry_sha = legacy.shared._config_identity(
        os.environ.get("VTA_CONFIG_FILE", legacy.shared.DEFAULT_CONFIG_PATH)
    )
    _, model_dir, model_filename = legacy.shared.MODEL_PIPELINES["visual_wake_words_v1"]
    model_path = APP_ROOT / model_dir / model_filename
    run_identity = _task_run_identity(
        index, identity, task, config_path, geometry_sha, model_path,
        args.trial_batch, args.min_successful, args.fsim_timeout, args.tsim_timeout,
    )
    fsim_state = search.load_state(_state_path(run_dir, index, "fsim"), run_identity)
    fsim_log = _record_path(run_dir, index, "fsim")
    if not fsim_log.is_file():
        raise ValueError(f"FSIM native log is missing: {fsim_log}")
    successful = {}
    for measure_input, result in autotvm.record.load_from_file(str(fsim_log)):
        if result.error_no != legacy.shared.MeasureErrorNo.NO_ERROR:
            continue
        if not legacy._record_matches_task(measure_input, task):
            continue
        successful.setdefault(_config_identity(measure_input.config), measure_input)
    if len(successful) != search.success_count(fsim_state):
        raise ValueError(
            "FSIM native log and persisted distinct-success count disagree: "
            f"{len(successful)} != {search.success_count(fsim_state)}"
        )

    state_path = _state_path(run_dir, index, "tsim")
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity_sha256") != _sha256_json(run_identity):
            raise ValueError("TSIM resume identity does not match model, geometry, workload, or options")
    else:
        if state_path.exists():
            raise ValueError(f"TSIM state already exists; choose --resume or a new run: {state_path}")
        state = {
            "schema_version": 1,
            "identity_sha256": _sha256_json(run_identity),
            "candidates": [],
            "infrastructure_errors": [],
        }
    tsim_log = _record_path(run_dir, index, "tsim")
    if not args.resume and tsim_log.exists():
        raise ValueError(f"TSIM native log already exists; choose --resume or a new run: {tsim_log}")
    state_keys = [item.get("config_key") for item in state["candidates"]]
    if len(set(state_keys)) != len(state_keys):
        raise ValueError("TSIM state contains duplicate candidate identities")
    if not set(state_keys).issubset(successful):
        raise ValueError("TSIM state contains a config absent from successful FSIM records")
    logged_keys = []
    if args.resume and tsim_log.exists():
        for measure_input, _ in autotvm.record.load_from_file(str(tsim_log)):
            logged_keys.append(_config_identity(measure_input.config))
    elif state_keys:
        raise ValueError("TSIM state records candidates but its native log is missing")
    if args.resume and set(logged_keys) != set(state_keys):
        raise ValueError("TSIM native log and persisted candidate state disagree")
    completed = set(state_keys)

    for config_key, fsim_input in successful.items():
        if config_key in completed:
            continue
        config = fsim_input.config
        try:
            result = measurement.measure_candidate(
                task, config, "tsim", timeout=args.tsim_timeout
            )
            succeeded = result.error_no == legacy.shared.MeasureErrorNo.NO_ERROR
            cycles = result.costs[0] if succeeded and result.costs else None
            candidate_error = None if succeeded else repr(result.costs)
        except legacy.shared.SimulatorInfrastructureError as error:
            state.setdefault("infrastructure_errors", []).append({
                "backend": "tsim", "config_index": int(config.index),
                "error": f"{type(error).__name__}: {error}",
            })
            temporary = state_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, state_path)
            raise
        except Exception as error:
            result = legacy.shared.MeasureResult(
                (type(error).__name__, str(error)),
                legacy.shared.MeasureErrorNo.RUNTIME_DEVICE, 0.0,
                datetime.now(timezone.utc).timestamp(),
            )
            cycles = None
            candidate_error = f"{type(error).__name__}: {error}"
        _append_record(tsim_log, fsim_input, result)
        item = {
            "config_key": config_key,
            "config_index": int(config.index),
            "config": config.to_json_dict(),
            "tsim_cycles": cycles,
            "error": candidate_error,
        }
        state["candidates"].append(item)
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, state_path)
        completed.add(config_key)
        print(
            f"TSIM workload={index} measured={len(completed)}/{len(successful)} "
            f"cycles={cycles if cycles is not None else 'failed'}",
            flush=True,
        )
    state["status"] = "complete" if len(completed) == len(successful) else "incomplete"
    if not successful:
        state["status"] = "no_fsim_successes"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if successful:
        best = search.select_best_tsim(state["candidates"])
        state["best_config_index"] = best["config_index"]
        state["best_tsim_cycles"] = best["tsim_cycles"]
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        best = None
    print(json.dumps({
        "backend": "tsim", "workload_index": index,
        "successful_fsim_configurations": len(successful),
        "tsim_measurement_count": len(state["candidates"]),
        "best_tsim_cycles": best["tsim_cycles"] if best else None,
        "status": state["status"], "native_log": str(tsim_log), "state": str(state_path),
    }, sort_keys=True), flush=True)
    return 0 if best is not None else 2


def _run_dir_from_manifest(path):
    path = Path(path).expanduser().resolve(strict=True)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"run manifest is not readable JSON: {path}") from error
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("run_identity"), dict):
        raise ValueError("unsupported or incomplete two-stage run manifest")
    return path.parent, manifest


def _worker_command(args, index, backend, run_dir):
    return [
        sys.executable, str(TUNE_DIR / "tune.py"), "--worker-backend", backend,
        "--workload-index", str(index), "--run-dir", str(run_dir),
        "--trial-batch", str(args.trial_batch),
        "--min-successful", str(args.min_successful),
        "--fsim-timeout", str(args.fsim_timeout),
        "--tsim-timeout", str(args.tsim_timeout),
    ] + (["--resume"] if args.resume_manifest else [])


def _worker_env(backend):
    env = os.environ.copy()
    config = Path(env.get("VTA_CONFIG_FILE", legacy.shared.DEFAULT_CONFIG_PATH)).expanduser().resolve()
    env["VTA_CONFIG_FILE"] = str(config)
    env["VTA_BACKEND"] = backend
    required_paths = [str(REPO_ROOT / "tvm" / "python"), str(REPO_ROOT / "vta" / "python"),
                      str(REPO_ROOT / "vta" / "apps" / "mlperf_tiny_benchmark"), str(APP_ROOT)]
    existing = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(required_paths + existing))
    return env


def _export_best_artifacts(run_dir, run_manifest, prepared, identities, tasks, artifact_dir):
    """Export selected TSIM records and metadata without build/ references."""
    artifact_dir = artifact_dir.expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _, model_dir, model_filename = legacy.shared.MODEL_PIPELINES["visual_wake_words_v1"]
    model_path = APP_ROOT / model_dir / model_filename
    model_sha = legacy.shared._sha256_file(model_path)
    geometry_path = Path(run_manifest["run_identity"]["geometry_path"])
    geometry_sha = run_manifest["run_identity"]["geometry_sha256"]
    _, active_geometry_sha = legacy.shared._config_identity(str(geometry_path))
    if active_geometry_sha != geometry_sha:
        raise ValueError("run manifest geometry hash does not match the active geometry file")
    entries = []
    failures = []
    for index in run_manifest["selected_workload_indices"]:
        fsim_state_path = _state_path(run_dir, index, "fsim")
        tsim_state_path = _state_path(run_dir, index, "tsim")
        if not fsim_state_path.is_file() or not tsim_state_path.is_file():
            failures.append({"workload_index": index, "reason": "missing backend state"})
            continue
        fsim_state = search.load_state(
            fsim_state_path,
            _task_run_identity(
                index, identities[index], tasks[index], geometry_path, geometry_sha,
                model_path,
                run_manifest["run_identity"]["trial_batch"],
                run_manifest["run_identity"]["min_successful"],
                run_manifest["run_identity"]["fsim_timeout_seconds"],
                run_manifest["run_identity"]["tsim_timeout_seconds"],
            ),
        )
        tsim_state = json.loads(tsim_state_path.read_text(encoding="utf-8"))
        if tsim_state.get("status") != "complete" or "best_tsim_cycles" not in tsim_state:
            failures.append({"workload_index": index, "reason": tsim_state.get("status", "no TSIM best")})
            continue
        native_log = _record_path(run_dir, index, "tsim")
        records = list(legacy.shared.autotvm.record.load_from_file(str(native_log)))
        selected, best_input, lowered, rejected = _select_deployable_tsim_candidate(
            tsim_state, records, prepared, identities[index]
        )
        best_config = selected["config"]
        selected_cycles = int(selected["tsim_cycles"])
        prefix = f"visual_wake_words_v1-workload-{index:03d}"
        native_path = artifact_dir / f"{prefix}-best.tsim.log"
        native_meta = artifacts.export_selected_record(
            records, best_config, selected_cycles, native_path,
            legacy.shared.autotvm.record,
        )
        result = {
            "schema_version": 1,
            "artifact_kind": "self_contained_native_best_v1",
            "measurement_scope": "isolated_complete_vta_conv_fusion",
            "measurement_protocol": legacy.shared.TSIM_MEASUREMENT_PROTOCOL,
            "workload_index": index,
            "occurrence": identities[index].occurrence,
            "symbol": identities[index].symbol,
            "template": tasks[index].name,
            "workload_sha256": legacy.shared._task_workload_id(tasks[index]),
            "fusion_identity": json.loads(identities[index].canonical_json()),
            "fusion_sha256": identities[index].sha256,
            "model": "visual_wake_words_v1",
            "model_sha256": model_sha,
            "geometry_path": str(geometry_path),
            "geometry_sha256": geometry_sha,
            "conv_schedule_key": legacy.fused.conv_schedule_key(identities[index]),
            "conv_config": best_config,
            "real_conv_lowering": True,
            "best_native_record": native_path.name,
            "best_native_record_sha256": native_meta["sha256"],
            "mac_count": legacy._logical_mac_count(tasks[index]),
            "tsim_cycles": selected_cycles,
            "tsim_config_index": int(selected["config_index"]),
            "tsim_best_observed_cycles": int(tsim_state["best_tsim_cycles"]),
            "rejected_unlowerable_tsim_candidates": rejected,
            "fsim_trials": fsim_state["attempted_count"],
            "fsim_success_count": search.success_count(fsim_state),
            "fsim_failures": fsim_state["failures"],
            "fsim_stop_reason": fsim_state["stop_reason"],
            "valid_config_space_size": fsim_state["identity"]["valid_config_space_size"],
            "tsim_candidate_count": len(tsim_state["candidates"]),
            "tsim_failures": [item for item in tsim_state["candidates"] if item.get("error")],
            "bounded": run_manifest["bounded"],
        }
        result_path = artifacts.write_json(artifact_dir / f"{prefix}-best.json", result)
        entries.append({
            "workload_index": index,
            "occurrence": identities[index].occurrence,
            "symbol": identities[index].symbol,
            "fusion_sha256": identities[index].sha256,
            "workload_sha256": legacy.shared._task_workload_id(tasks[index]),
            "config": best_config,
            "config_sha256": _sha256_json(best_config),
            "logical_macs_per_invocation": legacy._logical_mac_count(tasks[index]),
            "result_json": result_path.name,
            "native_record": native_path.name,
            "native_record_sha256": native_meta["sha256"],
            "tsim_cycles": selected_cycles,
            "tsim_config_index": int(selected["config_index"]),
        })
    manifest = {
        "schema_version": 1,
        "status": "complete" if not failures and len(entries) == len(run_manifest["selected_workload_indices"]) else "incomplete",
        "phase": "seed" if run_manifest.get("phase") == "seed" else "selected",
        "model": "visual_wake_words_v1",
        "model_id": "visual_wake_words_v1",
        "model_sha256": model_sha,
        "geometry_path": str(geometry_path),
        "geometry_sha256": geometry_sha,
        "measurement_protocol": legacy.shared.TSIM_MEASUREMENT_PROTOCOL,
        "bounded": run_manifest["bounded"],
        "completion_label": run_manifest["completion_label"],
        "workload_count": len(tasks),
        "selected_workload_indices": run_manifest["selected_workload_indices"],
        "entries": entries,
        "failures": failures,
    }
    manifest_path = artifacts.write_json(artifact_dir / "best-manifest.json", manifest)
    return manifest_path


def _validate_alignment_report(path, prepared, identities, tasks):
    from deployment_evidence import validate_seed_alignment_report

    geometry_path, geometry_sha = legacy.shared._config_identity(
        os.environ.get("VTA_CONFIG_FILE", legacy.shared.DEFAULT_CONFIG_PATH)
    )
    _, model_dir, model_filename = legacy.shared.MODEL_PIPELINES["visual_wake_words_v1"]
    model_sha = legacy.shared._sha256_file(APP_ROOT / model_dir / model_filename)
    expected = [
        {"occurrence": identity.occurrence, "symbol": identity.symbol,
         "fusion_sha256": identity.sha256,
         "workload_sha256": legacy.shared._task_workload_id(task)}
        for identity, task in zip(identities, tasks)
    ]
    return validate_seed_alignment_report(
        path, model_id="visual_wake_words_v1", model_sha256=model_sha,
        geometry_path=geometry_path, geometry_sha256=geometry_sha,
        expected_occurrences=expected,
    )


def _run_controller(args):
    import vta

    del vta
    if args.seed:
        args.min_successful = 1
    prepared, identities, tasks = legacy.prepare_v1_workloads()
    if not args.seed:
        args.alignment_report = _validate_alignment_report(
            args.alignment_report, prepared, identities, tasks
        )
    if args.workload_index is not None:
        legacy.select_workload(tasks, args.workload_index)
        indices = [args.workload_index]
    else:
        indices = list(range(len(tasks)))
        if args.max_workloads is not None:
            indices = indices[:args.max_workloads]
    if not indices:
        raise ValueError("no workloads selected")
    if args.resume_manifest:
        run_dir, manifest = _run_dir_from_manifest(args.resume_manifest)
        run_identity = manifest["run_identity"]
        expected = {
            "trial_batch": args.trial_batch,
            "min_successful": args.min_successful,
            "fsim_timeout_seconds": args.fsim_timeout,
            "tsim_timeout_seconds": args.tsim_timeout,
            "workload_indices": indices,
            "seed_gate_report_sha256": (
                legacy.shared._sha256_file(args.alignment_report) if args.alignment_report else None
            ),
        }
        if manifest.get("phase") != ("seed" if args.seed else "full"):
            raise ValueError("resume manifest phase does not match requested seed/full operation")
        for key, value in expected.items():
            if run_identity.get(key) != value:
                raise ValueError(f"resume manifest {key} does not match requested options")
        if manifest.get("workload_count") != len(tasks):
            raise ValueError("resume manifest workload count does not match prepared model")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = (args.build_dir or BUILD_ROOT) / run_id
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        geometry_path, geometry_sha = legacy.shared._config_identity(
            os.environ.get("VTA_CONFIG_FILE", legacy.shared.DEFAULT_CONFIG_PATH)
        )
        _, model_dir, model_filename = legacy.shared.MODEL_PIPELINES["visual_wake_words_v1"]
        model_path = APP_ROOT / model_dir / model_filename
        run_identity = {
            "model_sha256": legacy.shared._sha256_file(model_path),
            "geometry_path": str(geometry_path),
            "geometry_sha256": geometry_sha,
            "trial_batch": args.trial_batch,
            "min_successful": args.min_successful,
            "fsim_timeout_seconds": args.fsim_timeout,
            "tsim_timeout_seconds": args.tsim_timeout,
            "workload_indices": indices,
            "seed_gate_report_sha256": (
                legacy.shared._sha256_file(args.alignment_report) if args.alignment_report else None
            ),
        }
        manifest = {
            "schema_version": 1,
            "model": "visual_wake_words_v1",
            "workload_count": len(tasks),
            "phase": "seed" if args.seed else "full",
            "alignment_report": str(args.alignment_report) if args.alignment_report else None,
            "workloads": [
                {"workload_index": i, "occurrence": item.occurrence,
                 "symbol": item.symbol, "fusion_sha256": item.sha256,
                 "workload_sha256": legacy.shared._task_workload_id(tasks[i])}
                for i, item in enumerate(identities)
            ],
            "run_identity": run_identity,
            "selected_workload_indices": indices,
            "bounded": (
                not args.seed and (args.workload_index is not None or args.max_workloads is not None
                or args.trial_batch != 100 or args.min_successful != 20)
            ),
            "status": "running",
        }
    manifest_path = run_dir / "manifest.json"
    if not args.resume_manifest:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    results = []
    for index in indices:
        for backend in ("fsim", "tsim"):
            command = _worker_command(args, index, backend, run_dir)
            print(f"Running {backend.upper()} workload {index} in isolated process", flush=True)
            completed = subprocess.run(command, env=_worker_env(backend), check=False)
            entry = {"workload_index": index, "backend": backend, "returncode": completed.returncode}
            results.append(entry)
            if completed.returncode != 0:
                manifest.setdefault("worker_failures", []).append(entry)
                manifest["status"] = "incomplete"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                if backend == "fsim":
                    break
    all_ok = all(result["returncode"] == 0 for result in results)
    manifest["worker_results"] = results
    manifest["status"] = "complete" if all_ok else "incomplete"
    manifest["bounded"] = manifest.get("bounded", False)
    manifest["completion_label"] = (
        "SEED_COMPLETE" if args.seed and all_ok else
        "BOUNDED_SMOKE_INCOMPLETE" if manifest["bounded"] else "FULL_SEARCH"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    default_phase_dir = "seed" if args.seed else "optimal"
    artifact_dir = args.artifact_dir or (APP_ROOT / "tune" / default_phase_dir / run_dir.name)
    artifact_manifest = _export_best_artifacts(
        run_dir, manifest, prepared, identities, tasks, artifact_dir
    )
    manifest["best_manifest"] = str(artifact_manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Run manifest: {manifest_path}")
    print(f"Best artifact manifest: {artifact_manifest}")
    print(f"Run status: {manifest['status']} ({manifest['completion_label']})")
    return 0 if all_ok else 2


def _replay_manifest(path):
    path = Path(path).expanduser().resolve(strict=True)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"best manifest is not readable JSON: {path}") from error
    _, identities, tasks = legacy.prepare_v1_workloads()
    _, geometry_sha = legacy.shared._config_identity(
        os.environ.get("VTA_CONFIG_FILE", legacy.shared.DEFAULT_CONFIG_PATH)
    )
    _, model_dir, model_filename = legacy.shared.MODEL_PIPELINES["visual_wake_words_v1"]
    model_sha = legacy.shared._sha256_file(APP_ROOT / model_dir / model_filename)
    expected = [
        {"occurrence": identity.occurrence, "symbol": identity.symbol,
         "fusion_sha256": identity.sha256,
         "workload_sha256": legacy.shared._task_workload_id(task)}
        for identity, task in zip(identities, tasks)
    ]
    from deployment_evidence import validate_replay_manifest

    entries = validate_replay_manifest(
        manifest, model_id="visual_wake_words_v1", model_sha256=model_sha,
        geometry_sha256=geometry_sha, expected_occurrences=expected,
    )
    for entry in entries:
        for field in ("result_json", "native_record"):
            name = entry.get(field)
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"best manifest {field} must be a local filename")
        native_path = path.parent / entry["native_record"]
        if (not native_path.is_file()
                or legacy.shared._sha256_file(native_path) != entry.get("native_record_sha256")):
            raise ValueError("best manifest native record hash does not match its artifact")
        result_path = path.parent / entry["result_json"]
        replay = legacy.replay_result(
            result_path, expected_workload_index=entry.get("workload_index")
        )
        result = replay["result"]
        if result.get("fusion_sha256") != entry.get("fusion_sha256"):
            raise ValueError("best manifest fusion identity does not match its result")
        result_config = result.get("conv_config", result.get("config"))
        if (result.get("workload_sha256") != entry.get("workload_sha256")
                or _sha256_json(result_config) != entry.get("config_sha256")
                or result.get("best_native_record") != entry.get("native_record")
                or result.get("best_native_record_sha256") != entry.get("native_record_sha256")):
            raise ValueError("best manifest identity does not match its result")
        if result.get("tsim_cycles") != entry.get("tsim_cycles"):
            raise ValueError("best manifest cycles do not match its result")
    print(f"Validated self-contained best manifest: {path}")
    print(f"Replayed artifacts: {len(entries)}")
    print(f"Completion label: {manifest.get('completion_label', 'unknown')}")
    return 0


def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="tune every prepared VTA fusion occurrence")
    parser.add_argument("--seed", action="store_true", help="measure one successful schedule per occurrence and export separate seed artifacts")
    parser.add_argument("--alignment-report", type=Path, help="required passing seed deployment report before full search")
    parser.add_argument("--workload-index", type=_nonnegative_int, help="tune one zero-based fusion occurrence")
    parser.add_argument("--trial-batch", type=_positive_int, default=100)
    parser.add_argument("--min-successful", type=_positive_int, default=20)
    parser.add_argument("--fsim-timeout", type=_positive_int, default=60)
    parser.add_argument("--tsim-timeout", type=_positive_int, default=120)
    parser.add_argument("--build-dir", type=Path, help="intermediate run directory parent (default: Visual Wake Words V1/build/two_stage_tuning)")
    parser.add_argument("--resume-manifest", type=Path, help="resume the run identified by a prior manifest")
    parser.add_argument("--artifact-dir", type=Path, help="self-contained best artifacts output directory (default: tune/optimal/<run-id>)")
    parser.add_argument("--replay-manifest", type=Path, help="validate and apply every result in a self-contained best manifest")
    parser.add_argument("--max-workloads", type=_positive_int, help="bounded smoke: process at most N workloads and label the result incomplete")
    parser.add_argument("--worker-backend", choices=("fsim", "tsim"), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.trial_batch <= 0 or args.min_successful <= 0:
        raise SystemExit("--trial-batch and --min-successful must be positive")
    if args.replay_manifest:
        try:
            return _replay_manifest(args.replay_manifest)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    if args.worker_backend:
        if args.workload_index is None or args.run_dir is None:
            raise SystemExit("worker requires --workload-index and --run-dir")
        return _fsim_worker(args) if args.worker_backend == "fsim" else _tsim_worker(args)
    if args.all == (args.workload_index is not None):
        raise SystemExit("select exactly one of --all or --workload-index")
    if args.seed and (not args.all or args.alignment_report):
        raise SystemExit("--seed requires --all and cannot use --alignment-report")
    if not args.seed and not args.alignment_report:
        raise SystemExit("full search requires --alignment-report from the passing seed deployment")
    if args.resume_manifest and args.build_dir:
        raise SystemExit("--build-dir cannot be combined with --resume-manifest")
    if args.max_workloads is not None and not args.all:
        raise SystemExit("--max-workloads requires --all")
    args.resume_manifest = args.resume_manifest.expanduser().resolve() if args.resume_manifest else None
    return _run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
