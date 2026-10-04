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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tune exported ResNet-8 VTA workloads with separate FSIM and TSIM stages."""

import argparse
import hashlib
import json
import os
import tempfile
from itertools import islice
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--workload", type=int, default=-1,
                        help="-1 selects every exported VTA occurrence; otherwise select one index")
    parser.add_argument("--simulator", choices=("fsim", "tsim"), default="fsim")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--trial-batch", type=int)
    parser.add_argument("--min-successful", type=int)
    parser.add_argument("--input-logs", type=Path)
    parser.add_argument("--output-logs", type=Path, required=True)
    return parser


def validate_args(args):
    if args.workload < -1:
        raise ValueError("--workload must be -1 or a non-negative occurrence index")
    if args.timeout is None:
        args.timeout = 60 if args.simulator == "fsim" else 120
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive seconds")
    if args.simulator == "fsim":
        if args.input_logs is not None:
            raise ValueError("--input-logs is only valid for TSIM")
        args.trial_batch = 100 if args.trial_batch is None else args.trial_batch
        args.min_successful = 20 if args.min_successful is None else args.min_successful
        if args.trial_batch <= 0 or args.min_successful <= 0:
            raise ValueError("--trial-batch and --min-successful must be positive")
    else:
        if args.input_logs is None:
            raise ValueError("TSIM requires --input-logs from a successful FSIM stage")
        if args.trial_batch is not None or args.min_successful is not None:
            raise ValueError("--trial-batch and --min-successful are only valid for FSIM")
    return args


def _backend(args):
    selected = os.environ.get("VTA_BACKEND")
    if selected != args.simulator:
        raise ValueError(
            f"VTA_BACKEND={selected!r} must match --simulator {args.simulator}"
        )


def load_workloads(path):
    """Load authoritative Relay functions and activations without model imports."""
    from workloads import load_workloads as load

    return load(path)


def _identity(snapshot, workload_path):
    raw = Path(workload_path).expanduser().resolve(strict=True).read_bytes()
    return {
        "model_sha256": snapshot.model_sha256,
        "config_sha256": snapshot.config_sha256,
        "geometry_sha256": snapshot.geometry_sha256,
        "workloads_sha256": hashlib.sha256(raw).hexdigest(),
        "occurrences": [
            {"index": layer.index, "symbol": layer.symbol,
             "compute_sha256": layer.compute_sha256,
             "config_space_identity": layer.config_space_identity}
            for layer in snapshot.layers
        ],
    }


def _select_layers(snapshot, workload):
    if workload == -1:
        return list(snapshot.layers)
    if workload >= len(snapshot.layers):
        raise ValueError(f"--workload {workload} is outside exported occurrence range 0..{len(snapshot.layers)-1}")
    return [snapshot.layers[workload]]


def _capture_layers(snapshot):
    """Requery each exported function's AutoTVM space in the active VTA env."""
    import vta
    from vta.relay.transform import VTACompilerConfig
    from deployment_compute import _capture_layer

    config = VTACompilerConfig.from_env(vta.get_env())
    layers = tuple(
        replace(
            _capture_layer(item.index, item.function, config),
            # Relay JSON round-trips can normalize internal IDs. The snapshot
            # hash identifies the exact exported source bytes used by deploy.
            compute_sha256=item.compute_sha256,
        )
        for item in snapshot.layers
    )
    for stored, current in zip(snapshot.layers, layers):
        from workloads import portable_config_space_identity

        if stored.config_space_identity != portable_config_space_identity(current):
            raise ValueError(f"workload {stored.index} config-space identity changed after load")
    return layers, config


def _deployment(snapshot, layers, compiler_config=None):
    if compiler_config is None:
        import vta
        from vta.relay.transform import VTACompilerConfig

        compiler_config = VTACompilerConfig.from_env(vta.get_env())
    from deployment_compute import _geometry

    return SimpleNamespace(
        model_id="image_classification_v1",
        model_sha256=snapshot.model_sha256,
        geometry=_geometry(compiler_config),
        layers=tuple(layers),
    )


def _schedule_module():
    import schedule

    return schedule


def _selected_by_occurrence(path, deployment):
    if not Path(path).exists():
        return {}, {}
    snapshot = _schedule_module().load_schedule_snapshot(path, deployment)
    selections = {}
    measurements = {}
    layers = {layer.occurrence: layer for layer in deployment.layers}
    for occurrence, selected in snapshot.selected.items():
        layer = layers[occurrence]
        selections[occurrence] = _config_indices_for_selection(layer, selected.configs)
        if selected.measured:
            measurements[occurrence] = _measurement_for_export(layer, selected.measurement)
    return selections, measurements


def _config_indices_for_selection(layer, configs):
    if len(configs) != len(layer.config_spaces):
        raise ValueError(f"occurrence {layer.occurrence} has incomplete selected config metadata")
    indices = []
    for (template, _, _, space), config in zip(layer.config_spaces, configs):
        if template == "add.vta" or len(space) == 1:
            indices.append(0)
            continue
        if config is None:
            raise ValueError(f"selected schedule is missing {template}")
        value = config.to_json_dict()
        index = next((i for i in range(len(space))
                      if json.dumps(space.get(i).to_json_dict(), sort_keys=True)
                      == json.dumps(value, sort_keys=True)), None)
        if index is None or not space.get(index).valid():
            raise ValueError(f"stored best config for occurrence {layer.occurrence} is no longer valid")
        indices.append(index)
    return indices


def _measurement_for_export(layer, measurement):
    """Restore fallback slots omitted by a loaded snapshot's native records."""
    results = iter(measurement["results"])
    expanded = []
    for template, _, _, space in layer.config_spaces:
        if template == "add.vta" or len(space) == 1:
            expanded.append(None)
        else:
            expanded.append(next(results))
    try:
        next(results)
    except StopIteration:
        pass
    else:
        raise ValueError(f"occurrence {layer.occurrence} has extra measurement records")
    return {**measurement, "results": expanded}


def _output_paths(path):
    path = Path(path).expanduser().resolve()
    if path.suffix not in (".log", ".tmp"):
        raise ValueError("--output-logs must end in .log or .tmp")
    return path, path.with_suffix(".json")


def _base_identity(snapshot, workload_path):
    return _identity(snapshot, workload_path)


def run_fsim(args, snapshot):
    from measurement import MeasurementInfrastructureError, measure_candidate
    import tuning
    from publication import config_files, publish_file_set, validate_config_snapshot, writer_lock

    all_layers, compiler_config = _capture_layers(snapshot)
    selected = _select_layers(snapshot, args.workload)
    layer_by_index = {layer.occurrence: layer for layer in all_layers}
    output, metadata = _output_paths(args.output_logs)
    config_json, config_sha_file = config_files(output.parent)
    identity = _base_identity(snapshot, args.workloads)
    updates = []
    quotas = {}
    with writer_lock(output.parent):
        has_config = validate_config_snapshot(
            output.parent, snapshot.config_bytes, snapshot.config_sha256,
            allow_mismatch=args.workload == -1,
        )
        existing_groups = []
        replacing_all = args.workload == -1
        if output.exists() or metadata.exists():
            if not output.is_file() or not metadata.is_file():
                raise ValueError("existing FSIM candidate log/metadata pair is incomplete")
            if has_config and not replacing_all:
                old_sidecar = json.loads(metadata.read_text(encoding="utf-8"))
                expected_old = old_sidecar.get("identity")
                if expected_old != identity:
                    raise ValueError("single-layer FSIM merge identity differs from existing candidates")
                existing_groups = list(tuning.decode_candidate_log(
                    output.read_bytes(), metadata.read_bytes(), expected=identity
                ))
            elif not replacing_all:
                raise ValueError("single-layer FSIM merge requires an existing matching config snapshot")

        for item in selected:
            layer = layer_by_index[item.index]
            successful = 0
            tried = 0
            candidates = iter(tuning.candidate_indices(layer, seed=item.index))
            while successful < args.min_successful:
                batch = list(islice(candidates, args.trial_batch))
                if not batch:
                    break
                for indices in batch:
                    tried += 1
                    try:
                        result = measure_candidate(layer, item.activation, indices, "fsim", args.timeout)
                    except MeasurementInfrastructureError:
                        raise
                    except Exception as error:
                        print(f"FSIM occurrence {item.index} candidate failed: {error}", flush=True)
                        continue
                    configs = tuning.configs_for_indices(layer, indices)
                    candidate_digest = tuning.candidate_identity(configs)
                    candidate_id = hashlib.sha256(
                        f"{item.index}:{candidate_digest}".encode("utf-8")
                    ).hexdigest()
                    updates.append({
                        "occurrence": item.index,
                        "symbol": item.symbol,
                        "candidate_id": candidate_id,
                        "config_identity": result["config_identity"],
                        "config_indices": list(indices),
                        "configs": configs,
                        "backend": "fsim",
                        "output_verified": True,
                        "measurement": {
                            "backend": "fsim",
                            "protocol": "relay_cpu_exact_output_v1",
                            "units": "seconds",
                            "cost": result.get("duration_seconds", 1e-9),
                            "timestamp": result["timestamp"],
                        },
                        "timestamp": result["timestamp"],
                        "records": tuning.native_records(
                            layer, indices, result.get("duration_seconds", 1e-9)
                        ),
                    })
                    successful += 1
                    if successful >= args.min_successful:
                        break
                if successful < args.min_successful:
                    print(f"FSIM occurrence {item.index}: checked {tried} candidates", flush=True)
            quotas[item.index] = successful
            if successful == 0:
                raise RuntimeError(f"FSIM found no successful candidate for occurrence {item.index}; output was not changed")
            if successful < args.min_successful:
                print(
                    f"FSIM occurrence {item.index}: configuration space exhausted at "
                    f"{successful}/{args.min_successful} successful candidates",
                    flush=True,
                )

        groups = tuning.merge_candidate_groups(
            existing_groups, updates, selected={item.index for item in selected},
            replace_all=replacing_all,
        )
        encoded = tuning.encode_candidate_log(groups, identity)
        changes = {
            config_json: snapshot.config_bytes,
            config_sha_file: (snapshot.config_sha256 + "\n").encode("ascii"),
            output: encoded.log_bytes,
            metadata: encoded.sidecar_bytes,
        }
        best_log = output.parent / "best.log"
        best_json = output.parent / "best.json"
        if replacing_all:
            changes[best_log] = None
            changes[best_json] = None
        elif best_log.exists() or best_json.exists():
            if not best_log.is_file() or not best_json.is_file():
                raise ValueError("existing best schedule log/metadata pair is incomplete")
            current = _schedule_module().load_schedule_snapshot(
                best_log, _deployment(snapshot, all_layers, compiler_config)
            )
            kept = {
                occurrence: selection for occurrence, selection in current.selected.items()
                if occurrence not in {item.index for item in selected}
            }
            if len(kept) != len(current.selected):
                # Rebuild the surviving partial snapshot before publication.
                keep_indices = {}
                keep_measurements = {}
                for occurrence, selection in kept.items():
                    layer = layer_by_index[occurrence]
                    keep_indices[occurrence] = _config_indices_for_selection(
                        layer, selection.configs
                    )
                    if selection.measured:
                        keep_measurements[occurrence] = _measurement_for_export(
                            layer, selection.measurement
                        )
                with tempfile.TemporaryDirectory(prefix="ic-vta-best-") as staging:
                    staged_path = Path(staging) / "best.log"
                    _schedule_module().export_schedule_snapshot(
                        staged_path, _deployment(snapshot, all_layers, compiler_config), keep_indices,
                        measurements=keep_measurements,
                        provenance={"kind": "tsim_best", "retained_after_fsim_update": True},
                    )
                    changes[best_log] = staged_path.read_bytes()
                    changes[best_json] = staged_path.with_suffix(".json").read_bytes()

        def validate_staged(paths):
            staged_log = paths[output]
            staged_meta = paths[metadata]
            tuning.decode_candidate_log(staged_log.read_bytes(), staged_meta.read_bytes(), expected=identity)
            staged_config = paths[config_json].read_bytes()
            staged_sha = paths[config_sha_file].read_text(encoding="ascii").strip()
            if hashlib.sha256(staged_config).hexdigest() != staged_sha or staged_sha != snapshot.config_sha256:
                raise ValueError("staged VTA config snapshot failed its SHA-256 check")

        publish_file_set(changes, validate=validate_staged)
    print(f"FSIM candidates: {output}")
    print(f"FSIM metadata: {metadata}")
    return quotas


def _candidate_record_indices(group, layer):
    from tvm import autotvm
    from schedule import _portable_target

    indices = []
    decoded = [autotvm.record.decode(row) for row in group["records"]]
    configs = group.get("configs")
    if not isinstance(configs, list):
        raise ValueError("FSIM candidate is missing config entities")
    selected = []
    for entry in configs:
        selected.append(entry.get("config"))
    current = []
    for template, workload, target, space in layer.config_spaces:
        if template == "add.vta" or len(space) == 1:
            indices.append(0)
            continue
        match = next((position for position, candidate in enumerate(configs)
                      if candidate.get("template") == template
                      and candidate.get("workload") == repr(workload)), None)
        if match is None:
            raise ValueError(f"FSIM candidate does not cover {template} for {group['occurrence']}")
        value = selected[match]
        index = next((i for i in range(len(space))
                      if json.dumps(space.get(i).to_json_dict(), sort_keys=True)
                      == json.dumps(value, sort_keys=True)), None)
        if index is None or not space.get(index).valid():
            raise ValueError("FSIM candidate configuration is invalid in the current AutoTVM space")
        indices.append(index)
        current.append((template, workload, target, space, index))
    if len(decoded) != len(current):
        raise ValueError("FSIM candidate native record count differs from tunable spaces")
    for record, (template, workload, target, space, index) in zip(decoded, current):
        measure_input, _ = record
        if (measure_input.task.name != template
                or repr(measure_input.task.workload) != repr(workload)
                or _portable_target(measure_input.target) != _portable_target(target)
                or measure_input.config.to_json_dict() != space.get(index).to_json_dict()):
            raise ValueError("FSIM native record disagrees with candidate occurrence/config identity")
    return indices


def run_tsim(args, snapshot):
    from publication import writer_lock

    output, _ = _output_paths(args.output_logs)
    with writer_lock(output.parent):
        return _run_tsim_locked(args, snapshot)


def _run_tsim_locked(args, snapshot):
    from measurement import MeasurementInfrastructureError, measure_candidate
    import tuning
    from publication import publish_file_set, validate_config_snapshot
    from schedule import export_schedule_snapshot

    output, sidecar = _output_paths(args.output_logs)
    validate_config_snapshot(output.parent, snapshot.config_bytes, snapshot.config_sha256)
    all_layers, compiler_config = _capture_layers(snapshot)
    selected = _select_layers(snapshot, args.workload)
    layers = {layer.occurrence: layer for layer in all_layers}
    candidates_path = args.input_logs.expanduser().resolve(strict=True)
    candidates_sidecar = candidates_path.with_suffix(".json")
    identity = _base_identity(snapshot, args.workloads)
    groups = tuning.decode_candidate_log(
        candidates_path.read_bytes(), candidates_sidecar.read_bytes(), expected=identity
    )
    selected_ids = {layer.index for layer in selected}
    groups_by_occurrence = {index: [] for index in selected_ids}
    for group in groups:
        occurrence = group["occurrence"]
        if occurrence in groups_by_occurrence:
            groups_by_occurrence[occurrence].append(group)

    selections = {}
    measurements = {}
    for item in selected:
        layer = layers[item.index]
        trials = []
        for group in groups_by_occurrence[item.index]:
            if group.get("symbol") != item.symbol:
                raise ValueError(f"FSIM candidate occurrence {item.index} symbol changed")
            indices = _candidate_record_indices(group, layer)
            try:
                result = measure_candidate(layer, item.activation, indices, "tsim", args.timeout)
            except MeasurementInfrastructureError:
                raise
            except Exception as error:
                print(
                    f"TSIM occurrence {item.index} candidate "
                    f"{group['candidate_id']} failed: {error}",
                    flush=True,
                )
                continue
            if result["config_identity"] != group.get("config_identity"):
                raise ValueError(f"TSIM candidate identity differs at occurrence {item.index}")
            trials.append({**result, "indices": indices, "group": group})
        if not trials:
            raise RuntimeError(
                f"TSIM found no successful candidate for occurrence {item.index}; output was not changed"
            )
        winner = tuning.select_minimum_cycles(trials)
        selections[item.index] = winner["indices"]
        measurements[item.index] = {
            "backend": "tsim",
            "protocol": "tsim_single_call_v1",
            "units": "cycles",
            "results": [
                {"costs": [winner["cycles"]], "error_no": 0, "all_cost": 0.0,
                 "timestamp": winner["timestamp"]}
                for _ in selections[item.index]
            ],
        }
        print(
            f"TSIM occurrence {item.index}: {winner['cycles']} cycles "
            f"({winner['group']['candidate_id']})",
            flush=True,
        )

    previous, previous_measurements = _selected_by_occurrence(
        output, _deployment(snapshot, all_layers, compiler_config)
    )
    previous.update(selections)
    previous_measurements.update(measurements)
    if args.workload == -1:
        previous = selections
        previous_measurements = measurements
    with tempfile.TemporaryDirectory(prefix="ic-vta-schedule-") as staging:
        staged = Path(staging) / output.name
        export_schedule_snapshot(
            staged, _deployment(snapshot, all_layers, compiler_config), previous,
            measurements=previous_measurements,
            provenance={
                "kind": "tsim_best",
                "workloads_sha256": identity["workloads_sha256"],
                "candidate_log_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
                "selected_occurrences": sorted(previous),
            },
        )
        changes = {
            output: staged.read_bytes(),
            sidecar: staged.with_suffix(".json").read_bytes(),
        }
        def validate_staged(paths):
            with tempfile.TemporaryDirectory(prefix="ic-vta-validate-") as temporary:
                final_log = Path(temporary) / output.name
                final_sidecar = final_log.with_suffix(".json")
                final_log.write_bytes(paths[output].read_bytes())
                final_sidecar.write_bytes(paths[sidecar].read_bytes())
                _schedule_module().load_schedule_snapshot(
                    final_log, _deployment(snapshot, all_layers, compiler_config)
                )
        publish_file_set(changes, validate=validate_staged)
    print(f"TSIM selected schedules: {output}")
    print(f"TSIM metadata: {sidecar}")
    return selections


def main(argv=None):
    args = _parser().parse_args(argv) if not isinstance(argv, argparse.Namespace) else argv
    validate_args(args)
    _backend(args)
    snapshot = load_workloads(args.workloads)
    if not snapshot.layers:
        raise ValueError("workloads file contains no VTA occurrences")
    return run_fsim(args, snapshot) if args.simulator == "fsim" else run_tsim(args, snapshot)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"tune.py: {error}") from error
