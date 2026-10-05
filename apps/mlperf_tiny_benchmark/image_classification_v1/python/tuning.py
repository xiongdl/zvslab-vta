import hashlib
import itertools
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

from tvm import autotvm

@dataclass(frozen=True)
class CandidateLog:
    log_bytes: bytes
    sidecar_bytes: bytes
    groups: tuple


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def tunable_spaces(layer):
    return tuple(
        entry for entry in layer.config_spaces
        if entry[0] != "add.vta" and len(entry[3]) > 1
    )


def candidate_indices(layer, seed=0):
    """Yield distinct valid configurations, starting with the default."""
    spaces = tunable_spaces(layer)
    if not spaces:
        raise ValueError(f"occurrence {layer.occurrence} has no tunable config space")
    options = []
    for position, (_, _, _, space) in enumerate(spaces):
        indices = list(range(len(space)))
        random.Random(seed + position).shuffle(indices)
        options.append(indices)
    default = [0] * len(spaces)
    if all(space.get(0).valid() for _, _, _, space in spaces):
        yield default
    for values in itertools.product(*options):
        selected = list(values)
        if selected != default and all(
            entry[3].get(selected[index]).valid()
            for index, entry in enumerate(spaces)
        ):
            yield selected


def configs_for_indices(layer, indices):
    spaces = tunable_spaces(layer)
    if len(indices) != len(spaces):
        raise ValueError("candidate configuration count does not match tunable schedule spaces")
    return [
        {"template": template, "workload": repr(workload), "target": str(target),
         "config": space.get(index).to_json_dict()}
        for (template, workload, target, space), index in zip(spaces, indices)
    ]


def expand_candidate_indices(layer, configs):
    """Resolve portable candidate entities against freshly queried spaces."""
    spaces = tunable_spaces(layer)
    by_key = {}
    for row in configs:
        if not isinstance(row, dict):
            raise ValueError("candidate configuration row must be an object")
        key = (row.get("template"), row.get("workload"))
        if key in by_key:
            raise ValueError("candidate contains duplicate schedule spaces")
        by_key[key] = row.get("config")
    if len(by_key) != len(spaces):
        raise ValueError("candidate schedule-space coverage differs from workloads")
    indices = []
    for template, workload, _, space in spaces:
        key = (template, repr(workload))
        config = by_key.pop(key, None)
        if config is None:
            raise ValueError(f"candidate is missing {template} {key[1]}")
        index = next((i for i in range(len(space))
                      if _canonical(space.get(i).to_json_dict()) == _canonical(config)), None)
        if index is None or not space.get(index).valid():
            raise ValueError(f"candidate config is no longer valid for {template}")
        indices.append(index)
    if by_key:
        raise ValueError("candidate contains unknown schedule spaces")
    return indices


def candidate_identity(configs):
    return _sha256(_canonical(configs).encode("utf-8"))


def native_records(layer, indices, cost_seconds):
    """Encode one candidate as normal AutoTVM records for every tunable template."""
    if isinstance(cost_seconds, bool) or not isinstance(cost_seconds, (int, float)) or cost_seconds <= 0:
        cost_seconds = 1e-9
    rows = []
    for (template, workload, target, space), index in zip(tunable_spaces(layer), indices):
        task = autotvm.task.Task(template, tuple(workload[1:]))
        measure_input = autotvm.measure.MeasureInput(target, task, space.get(index))
        result = autotvm.measure.MeasureResult((float(cost_seconds),), 0, 0.0, time.time())
        rows.append(autotvm.record.encode(measure_input, result))
    return rows


def encode_candidate_log(groups, identity):
    """Seal grouped candidate metadata around a flat native AutoTVM log."""
    if not isinstance(identity, dict) or not identity:
        raise ValueError("candidate log identity must be a non-empty object")
    rows = []
    sealed_groups = []
    seen = set()
    for group in groups:
        occurrence = group.get("occurrence")
        candidate_id = group.get("candidate_id")
        records = group.get("records")
        if (isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0
                or not isinstance(candidate_id, str) or len(candidate_id) != 64
                or candidate_id in seen or not isinstance(records, list) or not records):
            raise ValueError("candidate group identity or native records are invalid")
        seen.add(candidate_id)
        refs = []
        for record in records:
            if not isinstance(record, str):
                raise ValueError("candidate records must be native AutoTVM log lines")
            index = len(rows)
            rows.append(record)
            refs.append({"record_index": index, "record_sha256": _sha256(record.encode("utf-8"))})
        sealed_groups.append({
            **{key: value for key, value in group.items() if key != "records"},
            "records": refs,
        })
    log_bytes = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
    sidecar = {
        "schema_version": 1,
        "kind": "fsim_candidate_groups",
        "identity": identity,
        "identity_sha256": _sha256(_canonical(identity).encode("utf-8")),
        "log_sha256": _sha256(log_bytes),
        "groups": sealed_groups,
    }
    return CandidateLog(log_bytes, (_canonical(sidecar) + "\n").encode("utf-8"), tuple(sealed_groups))


def decode_candidate_log(log_bytes, sidecar_bytes, *, expected):
    """Validate integrity, workload identity, native rows, and candidate groups."""
    try:
        sidecar = json.loads(sidecar_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("FSIM candidate metadata is not valid UTF-8 JSON") from error
    if not isinstance(sidecar, dict) or sidecar.get("schema_version") != 1:
        raise ValueError("unsupported FSIM candidate metadata schema")
    if sidecar.get("kind") != "fsim_candidate_groups":
        raise ValueError("input schedule log is not an FSIM candidate set")
    identity = sidecar.get("identity")
    if (not isinstance(identity, dict)
            or sidecar.get("identity_sha256") != _sha256(_canonical(identity).encode("utf-8"))):
        raise ValueError("FSIM candidate identity hash mismatch")
    if expected is not None and identity != expected:
        raise ValueError("FSIM candidate identity does not match workloads/config/geometry")
    if sidecar.get("log_sha256") != _sha256(log_bytes):
        raise ValueError("FSIM candidate log hash mismatch")
    try:
        lines = log_bytes.decode("utf-8").splitlines()
        records = [autotvm.record.decode(line) for line in lines]
    except Exception as error:
        raise ValueError("FSIM candidate log contains invalid native AutoTVM records") from error
    if not lines or any(record is None for record in records):
        raise ValueError("FSIM candidate log has no supported native records")
    groups = sidecar.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("FSIM candidate metadata contains no successful candidates")
    output = []
    seen = set()
    referenced = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("FSIM candidate group must be an object")
        occurrence = group.get("occurrence")
        candidate_id = group.get("candidate_id")
        refs = group.get("records")
        if (isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0
                or not isinstance(candidate_id, str) or len(candidate_id) != 64
                or candidate_id in seen or not isinstance(refs, list) or not refs):
            raise ValueError("FSIM candidate group is malformed or duplicated")
        configs = group.get("configs")
        config_identity = group.get("config_identity")
        measurement = group.get("measurement")
        expected_id = hashlib.sha256(
            f"{occurrence}:{_sha256(_canonical(configs).encode('utf-8'))}".encode("utf-8")
        ).hexdigest() if isinstance(configs, list) else None
        if (group.get("backend") != "fsim" or group.get("output_verified") is not True
                or not isinstance(config_identity, str) or len(config_identity) != 64
                or candidate_id != expected_id or not isinstance(measurement, dict)
                or measurement.get("backend") != "fsim"
                or measurement.get("protocol") != "relay_cpu_exact_output_v1"
                or measurement.get("units") != "seconds"
                or isinstance(measurement.get("cost"), bool)
                or not isinstance(measurement.get("cost"), (int, float))
                or measurement["cost"] <= 0):
            raise ValueError("FSIM candidate group provenance or config identity is invalid")
        seen.add(candidate_id)
        raw_lines = []
        for ref in refs:
            index = ref.get("record_index") if isinstance(ref, dict) else None
            if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                    or index >= len(lines) or index in referenced
                    or ref.get("record_sha256") != _sha256(lines[index].encode("utf-8"))):
                raise ValueError("FSIM candidate native record reference/hash is invalid")
            referenced.add(index)
            raw_lines.append(lines[index])
        output.append({**group, "records": raw_lines})
    if referenced != set(range(len(lines))):
        raise ValueError("FSIM candidate log contains unreferenced native records")
    return tuple(output)


def merge_candidate_groups(existing, updates, *, selected, replace_all):
    """Replace selected occurrence groups while retaining all other candidates."""
    if replace_all:
        return list(updates)
    kept = [group for group in existing if group.get("occurrence") not in selected]
    kept.extend(updates)
    return sorted(kept, key=lambda group: (group["occurrence"], group["candidate_id"]))


def select_minimum_cycles(measurements):
    """Pick cycle minimum with stable config-identity tie breaking."""
    valid = []
    for item in measurements:
        cycles = item.get("cycles")
        identity = item.get("config_identity")
        if (isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0
                and isinstance(identity, str) and len(identity) == 64):
            valid.append(item)
    if not valid:
        raise ValueError("no successful positive-cycle TSIM candidates")
    return min(valid, key=lambda item: (item["cycles"], item["config_identity"]))


#!/usr/bin/env python3


def _backend(args):
    selected = os.environ.get("VTA_BACKEND")
    if selected != args.simulator:
        raise ValueError(
            f"VTA_BACKEND={selected!r} must match --simulator {args.simulator}"
        )


def load_workloads(path):
    """Load authoritative Relay functions and activations without model imports."""
    from .vta_workload import load_workloads as load

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
    from .vta_workload import _capture_layer

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
        from .vta_workload import portable_config_space_identity

        if stored.config_space_identity != portable_config_space_identity(current):
            raise ValueError(f"workload {stored.index} config-space identity changed after load")
    return layers, config


def _deployment(snapshot, layers, compiler_config=None):
    if compiler_config is None:
        import vta
        from vta.relay.transform import VTACompilerConfig

        compiler_config = VTACompilerConfig.from_env(vta.get_env())
    from .vta_workload import _geometry

    return SimpleNamespace(
        model_id="image_classification_v1",
        model_sha256=snapshot.model_sha256,
        geometry=_geometry(compiler_config),
        layers=tuple(layers),
    )


def _schedule_module():
    from . import schedule_io as schedule

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


def _report_fsim_search(occurrence, trials, successes, quota, termination):
    print(
        f"FSIM occurrence {occurrence}: trials={trials} successes={successes} "
        f"quota={quota} termination={termination}",
        flush=True,
    )


def run_fsim(args, snapshot):
    from .measurement import MeasurementInfrastructureError, measure_candidate
    from .tuning_storage import config_files, publish_file_set, validate_config_snapshot, writer_lock

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
                existing_groups = list(decode_candidate_log(
                    output.read_bytes(), metadata.read_bytes(), expected=identity
                ))
            elif not replacing_all:
                raise ValueError("single-layer FSIM merge requires an existing matching config snapshot")

        for item in selected:
            layer = layer_by_index[item.index]
            successful = 0
            tried = 0
            candidates = iter(candidate_indices(layer, seed=item.index))
            while successful < args.min_successful:
                batch = list(islice(candidates, args.trial_batch))
                if not batch:
                    break
                for indices in batch:
                    tried += 1
                    try:
                        result = measure_candidate(layer, item.activation, indices, "fsim", args.timeout)
                    except MeasurementInfrastructureError:
                        _report_fsim_search(
                            item.index, tried, successful, args.min_successful,
                            "infrastructure_failure",
                        )
                        raise
                    except Exception as error:
                        print(f"FSIM occurrence {item.index} candidate failed: {error}", flush=True)
                        continue
                    configs = configs_for_indices(layer, indices)
                    candidate_digest = candidate_identity(configs)
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
                        "records": native_records(
                            layer, indices, result.get("duration_seconds", 1e-9)
                        ),
                    })
                    successful += 1
                    if successful >= args.min_successful:
                        break
                if successful < args.min_successful:
                    print(f"FSIM occurrence {item.index}: checked {tried} candidates", flush=True)
            termination = "quota_reached" if successful >= args.min_successful else "space_exhausted"
            _report_fsim_search(
                item.index, tried, successful, args.min_successful, termination
            )
            quotas[item.index] = successful
            if successful == 0:
                raise RuntimeError(
                    f"FSIM found no successful candidate for occurrence {item.index}; "
                    f"trials={tried}, quota={args.min_successful}, "
                    f"termination={termination}; output was not changed"
                )

        groups = merge_candidate_groups(
            existing_groups, updates, selected={item.index for item in selected},
            replace_all=replacing_all,
        )
        encoded = encode_candidate_log(groups, identity)
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
            decode_candidate_log(staged_log.read_bytes(), staged_meta.read_bytes(), expected=identity)
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
    from .schedule_io import _portable_target

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
    from .tuning_storage import writer_lock

    output, _ = _output_paths(args.output_logs)
    with writer_lock(output.parent):
        return _run_tsim_locked(args, snapshot)


def _run_tsim_locked(args, snapshot):
    from .measurement import MeasurementInfrastructureError, measure_candidate
    from .tuning_storage import publish_file_set, validate_config_snapshot
    from .schedule_io import export_schedule_snapshot

    output, sidecar = _output_paths(args.output_logs)
    validate_config_snapshot(output.parent, snapshot.config_bytes, snapshot.config_sha256)
    all_layers, compiler_config = _capture_layers(snapshot)
    selected = _select_layers(snapshot, args.workload)
    layers = {layer.occurrence: layer for layer in all_layers}
    candidates_path = args.input_logs.expanduser().resolve(strict=True)
    candidates_sidecar = candidates_path.with_suffix(".json")
    identity = _base_identity(snapshot, args.workloads)
    groups = decode_candidate_log(
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
        print(
            f"TSIM occurrence {item.index}: trials={len(groups_by_occurrence[item.index])} "
            f"successes={len(trials)} candidates={len(groups_by_occurrence[item.index])} "
            "termination=all_candidates_measured",
            flush=True,
        )
        if not trials:
            raise RuntimeError(
                f"TSIM found no successful candidate for occurrence {item.index}; output was not changed"
            )
        winner = select_minimum_cycles(trials)
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


def run(args):
    _backend(args)
    snapshot = load_workloads(args.workloads)
    if not snapshot.layers:
        raise ValueError("workloads file contains no VTA occurrences")
    return run_fsim(args, snapshot) if args.simulator == "fsim" else run_tsim(args, snapshot)
