"""Actual-compute search ledgers and resumable per-occurrence tuning."""

import hashlib
import itertools
import json
import os
import random
import tempfile
import time
from pathlib import Path


LEDGER_SCHEMA_VERSION = 1


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def create_ledger(identity, path):
    """Create a ledger bound to one actual layer, geometry, and option set."""
    if not isinstance(identity, dict):
        raise ValueError("tuning identity must be a JSON object")
    for name in ("model_sha256", "geometry_sha256", "compute_sha256"):
        value = identity.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"tuning identity {name} must be a SHA-256 digest")
    occurrence = identity.get("occurrence")
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0:
        raise ValueError("tuning identity occurrence must be a non-negative integer")
    if not isinstance(identity.get("options"), dict):
        raise ValueError("tuning identity options must be an object")
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": _digest(identity),
        "ledger_path": str(Path(path).expanduser().resolve()),
        "candidates": [],
        "failures": [],
        "status": "searching",
    }


def load_ledger(path, expected_identity):
    """Load a ledger only when its computation and options match exactly."""
    path = Path(path).expanduser().resolve(strict=True)
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"tuning ledger is not readable JSON: {path}") from error
    if (not isinstance(ledger, dict)
            or ledger.get("schema_version") != LEDGER_SCHEMA_VERSION
            or ledger.get("identity") != expected_identity
            or ledger.get("identity_sha256") != _digest(expected_identity)):
        raise ValueError("tuning ledger identity does not match model, compute, geometry, or options")
    candidates = ledger.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("tuning ledger candidates must be a list")
    seen = set()
    for candidate in candidates:
        key = _canonical(candidate.get("config_indices"))
        if key in seen:
            raise ValueError("tuning ledger contains duplicate candidate configurations")
        seen.add(key)
    ledger["ledger_path"] = str(path)
    return ledger


def save_ledger(ledger):
    """Atomically persist the candidate ledger after each measurement."""
    path = Path(ledger["ledger_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tuning-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(ledger, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def record_candidate(ledger, config_indices, config, *, backend, result=None, error=None):
    """Record one distinct measured configuration or its backend failure."""
    if backend not in ("fsim", "tsim"):
        raise ValueError("candidate backend must be fsim or tsim")
    indices_key = _canonical(config_indices)
    for candidate in ledger["candidates"]:
        if _canonical(candidate["config_indices"]) == indices_key:
            raise ValueError("candidate configuration was already recorded")
    candidate = {
        "config_indices": list(config_indices),
        "config": config,
        "status": "failed" if error is not None else "measured",
        "measurements": {},
        "failures": [],
    }
    if error is None:
        candidate["measurements"][backend] = result
    else:
        failure = {"stage": backend, "type": type(error).__name__, "message": str(error)}
        candidate["failures"].append(failure)
        ledger["failures"].append({**failure, "config_indices": list(config_indices)})
    ledger["candidates"].append(candidate)
    return candidate


def record_candidate_stage(candidate, *, backend, result=None, error=None):
    """Add the second backend outcome for an already discovered candidate."""
    if backend not in ("fsim", "tsim"):
        raise ValueError("candidate backend must be fsim or tsim")
    if backend in candidate["measurements"] or any(
        failure["stage"] == backend for failure in candidate["failures"]
    ):
        raise ValueError(f"candidate already has a {backend} outcome")
    if error is None:
        candidate["measurements"][backend] = result
    else:
        failure = {"stage": backend, "type": type(error).__name__, "message": str(error)}
        candidate["failures"].append(failure)
    return candidate


def _config_options(layer):
    spaces = [
        entry for entry in layer.config_spaces
        if entry[0] != "add.vta" and len(entry[3]) > 1
    ]
    if not spaces:
        raise ValueError(f"occurrence {layer.occurrence} has no tunable VTA schedule space")
    return spaces


def _candidate_indices(layer, seed):
    spaces = _config_options(layer)
    default = [0] * len(spaces)
    if all(space.get(0).valid() for _, _, _, space in spaces):
        yield default
    options = []
    for position, entry in enumerate(spaces):
        space = entry[3]
        indices = list(range(len(space)))
        random.Random(seed + position).shuffle(indices)
        options.append(indices)
    for selected in itertools.product(*options):
        config_indices = list(selected)
        if config_indices == default:
            continue
        if all(
            entry[3].get(config_indices[position]).valid()
            for position, entry in enumerate(spaces)
        ):
            yield config_indices


def _configs_json(layer, config_indices):
    return [
        {"template": template, "workload": repr(workload), "target": str(target),
         "config": space.get(index).to_json_dict()}
        for (template, workload, target, space), index in zip(_config_options(layer), config_indices)
    ]


def _candidate_key(config_indices):
    return _canonical(config_indices)


def search_layer(layer, activation, identity, ledger_path, *, trial_batch, min_successful,
                 fsim_timeout, tsim_timeout, seed=0, resume=False, measure=None):
    """Search a real captured layer with isolated FSIM then one-call TSIM measurements."""
    from measurement import measure_candidate

    measure = measure_candidate if measure is None else measure
    ledger_path = Path(ledger_path).expanduser().resolve()
    if resume:
        ledger = load_ledger(ledger_path, identity)
    else:
        if ledger_path.exists():
            raise ValueError(f"tuning ledger already exists; use resume or another run: {ledger_path}")
        ledger = create_ledger(identity, ledger_path)

    existing = {_candidate_key(row["config_indices"]): row for row in ledger["candidates"]}
    successes = {
        key for key, row in existing.items()
        if "fsim" in row["measurements"] and row["status"] != "failed"
    }
    visited = set(existing)
    attempted_in_batch = 0
    exhausted = len(successes) < min_successful
    if exhausted:
        for config_indices in _candidate_indices(layer, seed):
            key = _candidate_key(config_indices)
            if key in visited:
                continue
            exhausted = False
            try:
                result = measure(layer, activation, config_indices, "fsim", timeout=fsim_timeout)
                config = _configs_json(layer, config_indices)
                row = record_candidate(
                    ledger, config_indices, config, backend="fsim",
                    result={"config_identity": result["config_identity"], "worker_pid": result["worker_pid"]},
                )
                successes.add(key)
            except Exception as error:
                config = _configs_json(layer, config_indices)
                record_candidate(ledger, config_indices, config, backend="fsim", error=error)
            visited.add(key)
            save_ledger(ledger)
            attempted_in_batch += 1
            if len(successes) >= min_successful:
                break
            if attempted_in_batch >= trial_batch:
                attempted_in_batch = 0
        else:
            exhausted = True

    for row in ledger["candidates"]:
        if "fsim" not in row["measurements"] or row["status"] == "failed":
            continue
        if "tsim" in row["measurements"] or any(failure["stage"] == "tsim" for failure in row["failures"]):
            continue
        try:
            result = measure(layer, activation, row["config_indices"], "tsim", timeout=tsim_timeout)
            record_candidate_stage(row, backend="tsim", result={
                "cycles": result["cycles"], "protocol": result["protocol"],
                "config_identity": result["config_identity"],
                "timestamp": result["timestamp"],
            })
        except Exception as error:
            record_candidate_stage(row, backend="tsim", error=error)
            ledger["failures"].append({
                "stage": "tsim", "type": type(error).__name__, "message": str(error),
                "config_indices": row["config_indices"],
            })
        save_ledger(ledger)
    tsim_successes = [
        row for row in ledger["candidates"]
        if isinstance(row.get("measurements", {}).get("tsim", {}).get("cycles"), int)
        and not isinstance(row["measurements"]["tsim"]["cycles"], bool)
        and row["measurements"]["tsim"]["cycles"] > 0
    ]
    if not tsim_successes:
        ledger["status"] = "failed"
        ledger["stop_reason"] = "no_successful_tsim_measurements"
    else:
        ledger["status"] = "complete" if len(successes) >= min_successful or exhausted else "incomplete"
        ledger["stop_reason"] = (
            "successful_schedule_quota" if len(successes) >= min_successful
            else "configuration_space_exhausted" if exhausted else "bounded_incomplete"
        )
    save_ledger(ledger)
    return ledger


def validate_alignment_report(path, *, model_id, model_sha256, geometry_sha256,
                              compute, schedule_path):
    """Require a passing unified TSIM seed report for the exact captured compute."""
    path = Path(path).expanduser().resolve(strict=True)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"alignment report is not readable JSON: {path}") from error
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise ValueError("alignment report must record passed deployment evidence")
    if report.get("model") != model_id or report.get("model_sha256") != model_sha256:
        raise ValueError("alignment report model identity does not match the current model")
    if report.get("geometry_sha256") != geometry_sha256:
        raise ValueError("alignment report geometry identity does not match the active VTA geometry")
    if report.get("measurement_protocol") != "tsim_single_call_v1":
        raise ValueError("alignment report requires tsim_single_call_v1 cycle evidence")
    if report.get("sample_count") != 10 or report.get("outputs_passed") != 10:
        raise ValueError("alignment report must pass all ten committed correctness samples")
    if report.get("performance_sample_count") != 1:
        raise ValueError("alignment report must contain one performance sample")
    full_cycles = report.get("performance_stats", {}).get("cycle_count")
    if isinstance(full_cycles, bool) or not isinstance(full_cycles, int) or full_cycles <= 0:
        raise ValueError("alignment report must contain positive graph-resident TSIM counters")
    actual_hash = hashlib.sha256(Path(schedule_path).read_bytes()).hexdigest()
    if report.get("schedule_log_sha256") != actual_hash:
        raise ValueError("alignment report schedule log does not match the seed snapshot")
    expected = [{"occurrence": layer.occurrence, "symbol": layer.symbol} for layer in compute.layers]
    validate_evidence_rows(report.get("occurrences"), expected)
    return report


def validate_evidence_rows(rows, expected):
    """Validate complete passing alignment rows without importing deployment code."""
    if not isinstance(rows, list):
        raise ValueError("alignment report occurrences must be a list")
    expected_keys = {(row["occurrence"], row["symbol"]) for row in expected}
    actual_keys = [(row.get("occurrence"), row.get("symbol")) for row in rows if isinstance(row, dict)]
    if len(actual_keys) != len(rows) or len(actual_keys) != len(set(actual_keys)):
        raise ValueError("alignment report occurrence rows are malformed or duplicated")
    if set(actual_keys) != expected_keys:
        raise ValueError("alignment report occurrence coverage is incomplete")
    for row in rows:
        deployed = row.get("deployment_cycles")
        measured = row.get("autotvm_cycles")
        if (isinstance(deployed, bool) or not isinstance(deployed, int) or deployed <= 0
                or isinstance(measured, bool) or not isinstance(measured, int) or measured <= 0):
            raise ValueError("alignment report occurrence cycles must be positive integers")
        if row.get("passed") is not True or 10 * abs(deployed - measured) >= measured:
            raise ValueError("alignment report contains a failed strict <10% cycle gate")


def _successful_tsim_candidate(candidate):
    result = candidate.get("measurements", {}).get("tsim")
    cycles = result.get("cycles") if isinstance(result, dict) else None
    protocol = result.get("protocol") if isinstance(result, dict) else None
    timestamp = result.get("timestamp") if isinstance(result, dict) else None
    return (
        isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0
        and isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and timestamp > 0
        and isinstance(protocol, dict)
        and protocol.get("name") == "tsim_single_call"
        and protocol.get("version") == 1
        and protocol.get("counted_invocations") == 1
        and protocol.get("warmup_excluded") is True
        and not any(failure.get("stage") == "tsim" for failure in candidate.get("failures", []))
    )


def best_tsim_candidate(ledger):
    """Return the lowest-cycle valid TSIM candidate, or fail explicitly."""
    candidates = [
        (index, candidate) for index, candidate in enumerate(ledger.get("candidates", []))
        if _successful_tsim_candidate(candidate)
    ]
    if not candidates:
        occurrence = ledger.get("identity", {}).get("occurrence", "?")
        raise ValueError(f"occurrence {occurrence} has no successful TSIM candidate")
    return min(
        candidates,
        key=lambda pair: (pair[1]["measurements"]["tsim"]["cycles"], pair[0]),
    )


def _candidate_for_export(ledger, candidate_index=None):
    if candidate_index is None:
        return best_tsim_candidate(ledger)
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise ValueError("candidate index must be a non-negative integer")
    candidates = ledger.get("candidates", [])
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise ValueError(
            f"candidate index {candidate_index} is out of range for {len(candidates)} ledger entries"
        )
    return candidate_index, candidates[candidate_index]


def _source_ledger_sha256(ledger):
    path = ledger.get("ledger_path")
    if isinstance(path, str) and Path(path).is_file():
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _digest(ledger)


def _expand_candidate_config(layer, candidate):
    """Map portable candidate configs into this backend's actual captured spaces."""
    from schedule import _portable_config_space_identity

    config_rows = candidate.get("config")
    if not isinstance(config_rows, list):
        raise ValueError("ledger candidate configuration must be a list")
    expected = [
        entry for entry in layer.config_spaces
        if entry[0] != "add.vta" and len(entry[3]) > 1
    ]
    refs = {}
    for row in config_rows:
        if not isinstance(row, dict):
            raise ValueError("ledger candidate configuration entry is malformed")
        key = (row.get("template"), row.get("workload"))
        if key in refs:
            raise ValueError("ledger candidate has duplicate template/workload configurations")
        refs[key] = row.get("config")
    if len(refs) != len(expected):
        raise ValueError("ledger candidate config coverage does not match captured schedule spaces")

    lookup = {}
    for template, workload, _, space in expected:
        key = (template, repr(workload))
        config_json = refs.pop(key, None)
        if config_json is None:
            raise ValueError(f"ledger candidate is missing {template} for occurrence {layer.occurrence}")
        config_index = next(
            (index for index in range(len(space))
             if _canonical(space.get(index).to_json_dict()) == _canonical(config_json)),
            None,
        )
        if config_index is None or not space.get(config_index).valid():
            raise ValueError(
                f"ledger candidate config is not valid in current {template} space"
            )
        lookup[key] = config_index
    if refs:
        raise ValueError("ledger candidate contains unknown schedule spaces")
    all_indices = []
    for template, workload, _, space in layer.config_spaces:
        if template != "add.vta" and len(space) > 1:
            all_indices.append(lookup[(template, repr(workload))])
        else:
            all_indices.append(0)
    return all_indices, _portable_config_space_identity(layer)


def _export_ledger_selection(path, compute, ledgers, candidate_indices, *, mode):
    from schedule import _geometry_identity, export_schedule_snapshot

    layer_by_occurrence = {layer.occurrence: layer for layer in compute.layers}
    if set(ledgers) != set(candidate_indices):
        raise ValueError("ledger and selection occurrence sets differ")
    selections = {}
    measurements = {}
    statuses = []
    ledger_sources = []
    for occurrence in sorted(candidate_indices):
        layer = layer_by_occurrence.get(occurrence)
        if layer is None:
            raise ValueError(f"unknown occurrence {occurrence} in tuning ledger")
        ledger = ledgers[occurrence]
        identity = ledger.get("identity", {})
        candidate_index, candidate = _candidate_for_export(
            ledger, candidate_indices[occurrence]
        )
        expanded, config_space_identity = _expand_candidate_config(layer, candidate)
        if (identity.get("model_id") != compute.model_id
                or identity.get("model_sha256") != compute.model_sha256
                or identity.get("geometry_sha256") != _geometry_identity(compute)
                or identity.get("compute_sha256") != layer.compute_sha256
                or identity.get("config_space_sha256") != config_space_identity):
            raise ValueError(f"occurrence {occurrence} ledger identity does not match deployment")
        selections[occurrence] = expanded
        if _successful_tsim_candidate(candidate):
            measurement = candidate["measurements"]["tsim"]
            cycles = measurement["cycles"]
            timestamp = measurement.get("timestamp")
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                    or not timestamp > 0):
                raise ValueError(
                    f"occurrence {occurrence} TSIM candidate has no valid measurement timestamp"
                )
            measurements[occurrence] = {
                "backend": "tsim",
                "protocol": "tsim_single_call_v1",
                "units": "cycles",
                "results": [
                    {"costs": [cycles], "error_no": 0, "all_cost": 0.0,
                     "timestamp": timestamp}
                    for _ in expanded
                ],
            }
            validation_status = "measured_tsim"
        else:
            validation_status = "unmeasured_candidate"
        statuses.append({
            "occurrence": occurrence,
            "candidate_index": candidate_index,
            "validation_status": validation_status,
            "candidate_failures": [
                {"stage": failure.get("stage"), "type": failure.get("type"),
                 "message": str(failure.get("message", ""))[:512]}
                for failure in candidate.get("failures", [])
            ],
        })
        ledger_sources.append({
            "occurrence": occurrence,
            "ledger_sha256": _source_ledger_sha256(ledger),
        })
    provenance = {
        "kind": "actual_compute_tuning_export",
        "mode": mode,
        "source_ledgers": ledger_sources,
        "selections": statuses,
    }
    snapshot = export_schedule_snapshot(
        path, compute, selections, measurements=measurements, provenance=provenance
    )
    return {"snapshot": snapshot, "selections": statuses}


def export_candidate_snapshot(path, compute, ledger, occurrence, candidate_index):
    """Export one zero-based ledger candidate as a deployable partial snapshot."""
    if ledger.get("identity", {}).get("occurrence") != occurrence:
        raise ValueError("candidate ledger occurrence does not match the requested workload")
    return _export_ledger_selection(
        path, compute, {occurrence: ledger}, {occurrence: candidate_index}, mode="candidate"
    )


def export_best_snapshot(path, compute, ledgers):
    """Export the best measured TSIM candidate for every supplied occurrence."""
    candidate_indices = {
        occurrence: best_tsim_candidate(ledger)[0]
        for occurrence, ledger in ledgers.items()
    }
    return _export_ledger_selection(path, compute, ledgers, candidate_indices, mode="best")
