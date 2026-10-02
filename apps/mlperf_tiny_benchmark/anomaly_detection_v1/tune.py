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

"""Tune actual AD V1 deployment occurrences through shared VTA lowering."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
APPS_ROOT = APP_ROOT.parent.parent
DEFAULT_BUILD_ROOT = APP_ROOT / "build" / "actual_compute_tuning"


def _load_runtime():
    for path in (str(APPS_ROOT), str(APP_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import runtime

    return runtime


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tune-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _active_backend():
    backend = os.environ.get("VTA_BACKEND")
    if backend not in ("fsim", "tsim"):
        raise ValueError("VTA_BACKEND must explicitly select fsim or tsim")
    return backend


def _prepare_actual_compute(runtime, backend):
    """Capture normal prepared layers and each real graph-resident activation."""
    from tvm.contrib.debugger import debug_executor

    from common.deployment_compute import capture_deployment_compute

    prepared = runtime.prepare_model(runtime.MODEL_PATH)
    model_sha256 = getattr(
        getattr(prepared, "imported", None), "model_sha256", runtime.MODEL_SHA256
    )
    compute = capture_deployment_compute(prepared.mixed_module, runtime.MODEL_ID, model_sha256)
    artifacts = runtime.build_host_artifacts(
        prepared, DEFAULT_BUILD_ROOT / f"activation-{backend}", mode=backend
    )
    session, simulator = runtime._load_simulator(backend)
    graph = debug_executor.create(
        artifacts.mixed.graph_json, artifacts.mixed.module, artifacts.mixed.device
    )
    graph.load_params(artifacts.mixed.params)
    sample = runtime.committed_sample_paths()[0]
    features = runtime.load_sample(sample)
    selected, _, _ = runtime._select_tsim_windows(features, 1)
    graph.set_input(runtime.INPUT_NAME, selected[0][None, :])
    session.clear_and_validate(simulator)
    graph._run_per_layer()
    nodes = json.loads(artifacts.mixed.graph_json)["nodes"]
    outputs = graph.debug_datum.get_output_tensors()
    activations = {}
    for layer in compute.layers:
        matches = [
            index for index, node in enumerate(nodes)
            if node.get("op") == "tvm_op"
            and node.get("attrs", {}).get("func_name") == layer.symbol
        ]
        if len(matches) != 1:
            raise ValueError(f"actual graph does not identify occurrence {layer.occurrence} uniquely")
        node = nodes[matches[0]]
        if not node.get("inputs"):
            raise ValueError(f"actual graph occurrence {layer.occurrence} has no activation input")
        source_index, source_output, _ = node["inputs"][0]
        source = nodes[source_index]["name"]
        key = f"{source}____topo-index:{source_index}____output-num:{source_output}"
        activation = outputs[key].numpy()
        if tuple(activation.shape) != layer.inputs[0].shape:
            raise ValueError(f"captured activation shape changed for occurrence {layer.occurrence}")
        activations[layer.occurrence] = activation
    return prepared, compute, activations


def _config_indices(layer, index):
    values = []
    for template, _, _, space in layer.config_spaces:
        if template == "add.vta" or len(space) == 1:
            continue
        if index >= len(space):
            raise ValueError(
                f"seed configuration index {index} is outside {template} space [0, {len(space)})"
            )
        values.append(index)
    return values


def _snapshot_indices(layer, tunable_indices):
    """Expand portable tunable indices into this backend's captured spaces."""
    values = []
    selected = iter(tunable_indices)
    for template, _, _, space in layer.config_spaces:
        if template != "add.vta" and len(space) > 1:
            values.append(next(selected))
        else:
            values.append(0)
    try:
        next(selected)
    except StopIteration:
        return values
    raise ValueError(f"occurrence {layer.occurrence} has extra portable config indices")


def create_seed_snapshot(*, output_log=None):
    """Measure each actual default candidate once on TSIM and export a seed pair."""
    runtime = _load_runtime()
    if _active_backend() != "tsim":
        raise ValueError("--seed requires VTA_BACKEND=tsim for schedule-alignment evidence")
    _, compute, activations = _prepare_actual_compute(runtime, "tsim")
    from common.measurement import measure_candidate
    from common.schedule import export_schedule_snapshot

    selections = {}
    measurements = {}
    for layer in compute.layers:
        indices = _config_indices(layer, 0)
        measured = measure_candidate(
            layer, activations[layer.occurrence], indices, "tsim"
        )
        if measured["cycles"] is None:
            raise RuntimeError(f"TSIM did not measure occurrence {layer.occurrence}")
        selections[layer.occurrence] = _snapshot_indices(layer, indices)
        measurements[layer.occurrence] = {
            "backend": "tsim",
            "protocol": "tsim_single_call_v1",
            "units": "cycles",
            "results": [
                {"costs": [measured["cycles"]], "error_no": 0, "all_cost": 0.0,
                 "timestamp": measured["timestamp"]}
                for _ in selections[layer.occurrence]
            ],
        }
        print(f"seed occurrence {layer.occurrence}: {measured['cycles']} cycles", flush=True)
    output_log = Path(output_log or DEFAULT_BUILD_ROOT / "seed" / "seed.log").expanduser().resolve()
    snapshot = export_schedule_snapshot(output_log, compute, selections, measurements=measurements)
    print(f"Seed native log: {snapshot.path}")
    print(f"Seed metadata: {snapshot.path.with_suffix('.json')}")
    return snapshot


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true", help="measure defaults and export a full seed snapshot")
    parser.add_argument("--all", action="store_true", help="search all selected actual layer occurrences")
    parser.add_argument("--workload-index", type=int, help="limit search to one zero-based occurrence")
    parser.add_argument("--trial-batch", type=int, default=100)
    parser.add_argument("--min-successful", type=int, default=20)
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--fsim-timeout", type=int, default=60)
    parser.add_argument("--tsim-timeout", type=int, default=120)
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--alignment-report", type=Path)
    parser.add_argument("--export-candidate", type=int, help="export a zero-based ledger candidate")
    parser.add_argument("--export-best", action="store_true", help="export best successful TSIM candidate per selected occurrence")
    parser.add_argument("--output-log", type=Path, help="native schedule log path for seed or snapshot export")
    return parser


def _validate_positive_options(args):
    for name in ("trial_batch", "min_successful", "fsim_timeout", "tsim_timeout"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if args.max_workloads is not None and args.max_workloads <= 0:
        raise ValueError("--max-workloads must be positive")
    if args.workload_index is not None and args.workload_index < 0:
        raise ValueError("--workload-index must be non-negative")


def _search(args):
    runtime = _load_runtime()
    if _active_backend() != "fsim":
        raise ValueError("schedule search requires VTA_BACKEND=fsim; TSIM candidate workers select TSIM independently")
    if args.alignment_report is None:
        raise ValueError("search requires --alignment-report from a passing unified TSIM seed deployment")
    _validate_positive_options(args)
    prepared, compute, activations = _prepare_actual_compute(runtime, "fsim")
    from common import tuning
    from common.schedule import (
        _geometry_identity,
        _portable_config_space_identity,
        load_schedule_snapshot,
    )

    model_sha = getattr(getattr(prepared, "imported", None), "model_sha256", runtime.MODEL_SHA256)
    geometry_sha = _geometry_identity(compute)
    try:
        report = json.loads(args.alignment_report.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("alignment report is not readable JSON") from error
    seed_path = report.get("schedule") if isinstance(report, dict) else None
    if not isinstance(seed_path, str):
        raise ValueError("alignment report does not identify its seed schedule")
    seed_snapshot = load_schedule_snapshot(seed_path, compute)
    if len(seed_snapshot.selected) != len(compute.layers):
        raise ValueError("search seed snapshot must cover all actual layer occurrences")
    tuning.validate_alignment_report(
        args.alignment_report,
        model_id=runtime.MODEL_ID,
        model_sha256=model_sha,
        geometry_sha256=geometry_sha,
        compute=compute,
        schedule_path=seed_path,
    )
    options = {
        "trial_batch": args.trial_batch,
        "min_successful": args.min_successful,
        "fsim_timeout_seconds": args.fsim_timeout,
        "tsim_timeout_seconds": args.tsim_timeout,
        "search_backends": ["fsim", "tsim"],
        "alignment_report_sha256": _sha256_bytes(args.alignment_report.expanduser().resolve().read_bytes()),
        "seed_schedule_sha256": _sha256_bytes(Path(seed_path).read_bytes()),
    }
    selected = list(compute.layers)
    if args.workload_index is not None:
        if args.workload_index >= len(selected):
            raise ValueError(f"workload index must be in 0..{len(selected) - 1}")
        selected = [selected[args.workload_index]]
    if args.max_workloads is not None:
        if args.max_workloads > len(selected):
            raise ValueError("--max-workloads cannot exceed the selected occurrence count")
        selected = selected[:args.max_workloads]
    identity = {
        "model_id": runtime.MODEL_ID,
        "model_sha256": model_sha,
        "geometry_sha256": geometry_sha,
        "options": options,
        "selected_occurrences": [
            {"occurrence": layer.occurrence, "symbol": layer.symbol,
             "compute_sha256": layer.compute_sha256,
             "config_space_sha256": _portable_config_space_identity(layer)}
            for layer in selected
        ],
    }
    if args.resume_manifest is not None:
        manifest_path = args.resume_manifest.expanduser().resolve(strict=True)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("resume manifest is not readable JSON") from error
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or manifest.get("identity") != identity):
            raise ValueError("resume manifest identity does not match model, compute, geometry, or options")
        run_dir = Path(manifest["run_dir"]).expanduser().resolve(strict=True)
        ledgers = manifest.get("ledgers")
        if not isinstance(ledgers, dict):
            raise ValueError("resume manifest ledger map is malformed")
    else:
        run_dir = DEFAULT_BUILD_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "resume-manifest.json"
        ledgers = {
            str(layer.occurrence): str(run_dir / f"occurrence-{layer.occurrence:03d}.ledger.json")
            for layer in selected
        }
        manifest = {
            "schema_version": 1,
            "model": runtime.MODEL_ID,
            "identity": identity,
            "run_dir": str(run_dir),
            "ledgers": ledgers,
            "status": "searching",
        }
        _write_json_atomic(manifest_path, manifest)

    from common.tuning import search_layer

    ledger_statuses = []
    ledger_objects = {}
    for layer in selected:
        layer_identity = {
            "model_id": runtime.MODEL_ID,
            "model_sha256": model_sha,
            "geometry_sha256": geometry_sha,
            "compute_sha256": layer.compute_sha256,
            "config_space_sha256": _portable_config_space_identity(layer),
            "occurrence": layer.occurrence,
            "symbol": layer.symbol,
            "options": options,
        }
        ledger_path = Path(ledgers.get(str(layer.occurrence), ""))
        if not ledger_path.is_absolute():
            ledger_path = run_dir / ledger_path
        ledger = search_layer(
            layer, activations[layer.occurrence], layer_identity, ledger_path,
            trial_batch=args.trial_batch, min_successful=args.min_successful,
            fsim_timeout=args.fsim_timeout, tsim_timeout=args.tsim_timeout,
            seed=layer.occurrence, resume=args.resume_manifest is not None,
        )
        ledger_statuses.append(ledger["status"])
        ledger_objects[layer.occurrence] = ledger
        print(
            f"occurrence {layer.occurrence}: {ledger['status']} "
            f"({len(ledger['candidates'])} candidates, {len(ledger['failures'])} failures)"
        )
    bounded = len(selected) != len(compute.layers)
    manifest["status"] = (
        "failed" if any(status == "failed" for status in ledger_statuses)
        else "bounded_incomplete" if bounded or any(status == "incomplete" for status in ledger_statuses)
        else "complete"
    )
    manifest["completion_label"] = (
        "FAILED_INCOMPLETE" if manifest["status"] == "failed"
        else "BOUNDED_SMOKE_INCOMPLETE" if manifest["status"] == "bounded_incomplete"
        else "FULL_SEARCH"
    )
    best_candidates = {}
    for occurrence, ledger in ledger_objects.items():
        try:
            best_candidates[occurrence] = tuning.best_tsim_candidate(ledger)[0]
        except ValueError as error:
            if "no successful TSIM candidate" not in str(error):
                raise
            best_candidates = {}
            break
    if len(best_candidates) == len(ledger_objects) and best_candidates:
        best_path = run_dir / "best.log"
        exported = tuning.export_best_snapshot(best_path, compute, ledger_objects)
        manifest["best_snapshot"] = {
            "log": str(exported["snapshot"].path),
            "metadata": str(exported["snapshot"].path.with_suffix(".json")),
            "selections": exported["selections"],
        }
        print(f"Best native log: {exported['snapshot'].path}")
        print(f"Best metadata: {exported['snapshot'].path.with_suffix('.json')}")
    else:
        manifest.pop("best_snapshot", None)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, manifest)
    print(f"Search resume manifest: {manifest_path}")
    return manifest


def _export_snapshot(args):
    runtime = _load_runtime()
    _active_backend()
    if args.resume_manifest is None or args.output_log is None:
        raise ValueError("snapshot export requires --resume-manifest and --output-log")
    if (args.export_candidate is None) == (not args.export_best):
        raise ValueError("select exactly one of --export-candidate N or --export-best")
    if args.export_candidate is not None:
        if args.export_candidate < 0:
            raise ValueError("--export-candidate must be a non-negative ledger index")
        if args.workload_index is None:
            raise ValueError("--export-candidate requires --workload-index N")
    elif args.workload_index is not None and args.workload_index < 0:
        raise ValueError("--workload-index must be non-negative")

    manifest_path = args.resume_manifest.expanduser().resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("resume manifest is not readable JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported or missing search resume manifest schema")

    from common.deployment_compute import capture_deployment_compute
    from common.schedule import _geometry_identity, _portable_config_space_identity
    from common.tuning import export_best_snapshot, export_candidate_snapshot, load_ledger

    prepared = runtime.prepare_model(runtime.MODEL_PATH)
    model_sha = getattr(getattr(prepared, "imported", None), "model_sha256", runtime.MODEL_SHA256)
    compute = capture_deployment_compute(prepared.mixed_module, runtime.MODEL_ID, model_sha)
    identity = manifest.get("identity", {})
    if (manifest.get("model") != runtime.MODEL_ID
            or identity.get("model_sha256") != model_sha
            or identity.get("geometry_sha256") != _geometry_identity(compute)):
        raise ValueError("resume manifest does not match the current model or geometry")
    selected_rows = identity.get("selected_occurrences")
    if not isinstance(selected_rows, list):
        raise ValueError("resume manifest selected occurrence list is malformed")
    selected_by_occurrence = {
        row.get("occurrence"): row for row in selected_rows if isinstance(row, dict)
    }
    if len(selected_by_occurrence) != len(selected_rows):
        raise ValueError("resume manifest contains duplicate selected occurrences")
    if args.export_candidate is not None:
        occurrences = [args.workload_index]
    elif args.workload_index is not None:
        occurrences = [args.workload_index]
    else:
        occurrences = sorted(selected_by_occurrence)
    layers = {layer.occurrence: layer for layer in compute.layers}
    ledgers = {}
    ledger_paths = manifest.get("ledgers")
    if not isinstance(ledger_paths, dict):
        raise ValueError("resume manifest ledger map is malformed")
    for occurrence in occurrences:
        row = selected_by_occurrence.get(occurrence)
        layer = layers.get(occurrence)
        if row is None or layer is None:
            raise ValueError(f"occurrence {occurrence} is absent from the selected search")
        if (row.get("symbol") != layer.symbol
                or row.get("compute_sha256") != layer.compute_sha256
                or row.get("config_space_sha256") != _portable_config_space_identity(layer)):
            raise ValueError(f"resume manifest occurrence {occurrence} compute identity changed")
        raw_path = ledger_paths.get(str(occurrence))
        if not isinstance(raw_path, str):
            raise ValueError(f"resume manifest has no ledger for occurrence {occurrence}")
        ledger_path = Path(raw_path).expanduser()
        if not ledger_path.is_absolute():
            ledger_path = Path(manifest["run_dir"]) / ledger_path
        expected_ledger_identity = {
            "model_id": runtime.MODEL_ID,
            "model_sha256": model_sha,
            "geometry_sha256": _geometry_identity(compute),
            "compute_sha256": layer.compute_sha256,
            "config_space_sha256": _portable_config_space_identity(layer),
            "occurrence": occurrence,
            "symbol": layer.symbol,
            "options": identity.get("options"),
        }
        ledgers[occurrence] = load_ledger(ledger_path, expected_ledger_identity)

    output_log = args.output_log.expanduser().resolve()
    if args.export_candidate is not None:
        result = export_candidate_snapshot(
            output_log, compute, ledgers[occurrences[0]], occurrences[0], args.export_candidate
        )
    else:
        result = export_best_snapshot(output_log, compute, ledgers)
    print(f"Schedule native log: {result['snapshot'].path}")
    print(f"Schedule metadata: {result['snapshot'].path.with_suffix('.json')}")
    for selection in result["selections"]:
        print(
            f"occurrence {selection['occurrence']}: "
            f"candidate {selection['candidate_index']} {selection['validation_status']}"
        )
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.export_candidate is not None or args.export_best:
        if args.seed or args.all or args.alignment_report is not None:
            raise ValueError("snapshot export cannot be combined with seed/search options")
        _export_snapshot(args)
        return 0
    if args.seed:
        if not args.all or args.workload_index is not None:
            raise ValueError("seed mode requires --seed --all and cannot select one occurrence")
        if args.resume_manifest is not None or args.alignment_report is not None:
            raise ValueError("seed mode does not accept resume or alignment report options")
        create_seed_snapshot(output_log=args.output_log)
    else:
        if not args.all and args.workload_index is None:
            raise ValueError("search requires --all or --workload-index N")
        _search(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
