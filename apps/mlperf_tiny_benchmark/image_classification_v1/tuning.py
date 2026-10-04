"""Candidate search and native schedule-log contracts for IC V1 tuning."""

import hashlib
import itertools
import json
import random
import time
from dataclasses import dataclass

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
