"""Export and validate deployable snapshots of actual VTA layer schedules."""

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import tvm
from tvm import autotvm


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SelectedSchedule:
    occurrence: int
    symbol: str
    configs: tuple
    measured: bool
    measurement: dict


@dataclass(frozen=True)
class ScheduleSnapshot:
    path: Path | None
    model_id: str
    model_sha256: str
    geometry_sha256: str
    selected: dict

    def coverage(self, deployment):
        return tuple(
            (layer.occurrence, layer.symbol, layer.occurrence in self.selected)
            for layer in deployment.layers
        )


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _active_config_sha256():
    config_path = os.environ.get("VTA_CONFIG_FILE")
    if not config_path:
        raise ValueError("VTA_CONFIG_FILE must identify the active geometry config")
    try:
        return _sha256(Path(config_path).expanduser().resolve(strict=True).read_bytes())
    except OSError as error:
        raise ValueError(f"active VTA config cannot be read: {config_path}") from error


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _portable_target(target):
    return re.sub(r"(?<=-model=)(?:fsim|tsim)_", "sim_", str(target))


def _tunable_template(template, space):
    # VTA's add.vta task is backend plumbing: FSIM exposes a singleton
    # fallback space while TSIM exposes host fallback entities. It is not a
    # selectable accelerator schedule and must remain on normal lowering.
    return template != "add.vta" and len(space) > 1


def _geometry_identity(deployment):
    geometry = dict(deployment.geometry)
    for key in ("target", "model"):
        if key in geometry:
            target = str(geometry[key]) if key == "target" else f"-model={geometry[key]}"
            geometry[key] = _portable_target(target).removeprefix("-model=")
    return _sha256(_canonical_json(geometry).encode("utf-8"))


def _portable_config_space_identity(layer):
    spaces = [
        {
            "template": template,
            "workload": repr(workload),
            "target": _portable_target(target),
            "space_size": len(space) if template != "add.vta" else None,
            "entities": (
                [space.get(index).to_json_dict() for index in range(len(space))]
                if _tunable_template(template, space) else None
            ),
        }
        for template, workload, target, space in sorted(
            layer.config_spaces,
            key=lambda entry: (entry[0], repr(entry[1]), _portable_target(entry[2])),
        )
        if template != "add.vta"
    ]
    return _sha256(_canonical_json(spaces).encode("utf-8"))


def _sidecar_path(path):
    path = Path(path)
    if path.suffix != ".log":
        raise ValueError("schedule path must name a native AutoTVM .log file")
    return path.with_suffix(".json")


def _record_input(layer, space_entry, config):
    template, workload, target, _ = space_entry
    task = autotvm.task.Task(template, tuple(workload[1:]))
    return autotvm.measure.MeasureInput(target, task, config)


def _measurement_result(measurement):
    """Create a native record result without inventing an unmeasured cost."""
    if measurement is None:
        return autotvm.measure.MeasureResult((1e9,), 1, 0.0, time.time())
    costs = measurement.get("costs")
    error_no = measurement.get("error_no", 0)
    all_cost = measurement.get("all_cost", 0.0)
    timestamp = measurement.get("timestamp", time.time())
    if not isinstance(costs, (list, tuple)) or not costs:
        raise ValueError("measured schedule requires non-empty measurement costs")
    if any(isinstance(cost, bool) or not isinstance(cost, (int, float))
           or not math.isfinite(cost) or cost <= 0 for cost in costs):
        raise ValueError("measurement costs must be finite positive numbers")
    if isinstance(error_no, bool) or not isinstance(error_no, int) or error_no != 0:
        raise ValueError("measured schedule result must have error_no 0")
    if isinstance(all_cost, bool) or not isinstance(all_cost, (int, float)) or not math.isfinite(all_cost) or all_cost < 0:
        raise ValueError("measurement all_cost must be a finite non-negative number")
    if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp) or timestamp <= 0):
        raise ValueError("measurement timestamp must be finite and positive")
    return autotvm.measure.MeasureResult(tuple(costs), error_no, all_cost, timestamp)


def _validate_measured_provenance(measurement, occurrence):
    if (not isinstance(measurement, dict)
            or measurement.get("backend") != "tsim"
            or measurement.get("protocol") != "tsim_single_call_v1"
            or measurement.get("units") != "cycles"):
        raise ValueError(
            f"occurrence {occurrence} requires TSIM single-call cycle measurement provenance"
        )


def _layer_identity(layer):
    return {
        "occurrence": layer.occurrence,
        "symbol": layer.symbol,
        "compute_sha256": layer.compute_sha256,
        "config_space_identity": _portable_config_space_identity(layer),
        "workloads": [
            {"template": template, "workload": repr(workload), "target": _portable_target(target),
             "space_size": len(space) if template != "add.vta" else None}
            for template, workload, target, space in sorted(
                layer.config_spaces,
                key=lambda entry: (entry[0], repr(entry[1]), _portable_target(entry[2])),
            )
            if template != "add.vta"
        ],
    }


def export_schedule_snapshot(path, deployment, selections, *, measurements=None, provenance=None):
    """Write a native log and same-stem metadata for selected layer occurrences.

    `selections` maps occurrence integers to one config-space index per captured
    template. `measurements` maps occurrences to per-template AutoTVM result
    dictionaries. Omitted occurrences intentionally retain default schedules.
    """
    path = Path(path)
    sidecar = _sidecar_path(path)
    if not isinstance(selections, dict):
        raise TypeError("selections must map layer occurrence integers to config indices")
    measurements = {} if measurements is None else measurements
    if not isinstance(measurements, dict):
        raise TypeError("measurements must map layer occurrences to measured results")
    if provenance is not None and not isinstance(provenance, dict):
        raise TypeError("schedule provenance must be a JSON object")
    if provenance is not None:
        try:
            _canonical_json(provenance)
        except (TypeError, ValueError) as error:
            raise ValueError("schedule provenance must contain JSON-compatible values") from error

    layers = {layer.occurrence: layer for layer in deployment.layers}
    unknown = set(selections) - set(layers)
    if unknown:
        raise ValueError(f"unknown layer occurrences in schedule: {sorted(unknown)}")
    if set(measurements) - set(selections):
        raise ValueError("measurement data cannot reference an unselected occurrence")

    rows = []
    occurrences = []
    for occurrence in sorted(selections):
        layer = layers[occurrence]
        indices = selections[occurrence]
        if not isinstance(indices, (list, tuple)) or len(indices) != len(layer.config_spaces):
            raise ValueError(
                f"occurrence {occurrence} requires {len(layer.config_spaces)} config indices"
            )
        measurement_provenance = measurements.get(occurrence)
        if measurement_provenance is not None:
            _validate_measured_provenance(measurement_provenance, occurrence)
            result_specs = measurement_provenance.get("results")
            if not isinstance(result_specs, (list, tuple)) or len(result_specs) != len(indices):
                raise ValueError(f"occurrence {occurrence} measurement count does not match config count")
        else:
            result_specs = None
        record_refs = []
        recorded_results = []
        for index, (entry, raw_index) in enumerate(zip(layer.config_spaces, indices)):
            template, workload, target, space = entry
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError(f"occurrence {occurrence} config indices must be integers")
            if raw_index < 0 or raw_index >= len(space):
                raise ValueError(
                    f"occurrence {occurrence} config index {raw_index} is outside "
                    f"{template} space [0, {len(space)})"
                )
            config = space.get(raw_index)
            if not config.valid():
                raise ValueError(f"occurrence {occurrence} config {raw_index} is invalid for {template}")
            if not _tunable_template(template, space):
                continue
            spec = None if result_specs is None else result_specs[index]
            record_result = _measurement_result(spec)
            if spec is not None:
                recorded_results.append(spec)
            row = autotvm.record.encode(_record_input(layer, entry, config), record_result)
            row_index = len(rows)
            rows.append(row)
            record_refs.append({
                "template": template,
                "workload": repr(workload),
                "target": _portable_target(target),
                "record_index": row_index,
                "record_sha256": _sha256(row.encode("utf-8")),
                "config": config.to_json_dict(),
            })
        stored_measurement = None
        if measurement_provenance is not None:
            stored_measurement = dict(measurement_provenance)
            stored_measurement["results"] = recorded_results
        occurrences.append({
            **_layer_identity(layer),
            "measured": measurement_provenance is not None,
            "measurement": stored_measurement,
            "records": record_refs,
        })

    log_bytes = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": deployment.model_id,
        "model_sha256": deployment.model_sha256,
        "config_sha256": _active_config_sha256(),
        "geometry_sha256": _geometry_identity(deployment),
        "log_sha256": _sha256(log_bytes),
        "occurrences": occurrences,
    }
    if provenance is not None:
        metadata["provenance"] = provenance
    metadata_bytes = (_canonical_json(metadata) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    log_tmp = _write_temp(path.parent, log_bytes)
    metadata_tmp = _write_temp(sidecar.parent, metadata_bytes)
    try:
        # The metadata rename publishes the pair. A crash between replacements
        # leaves a hash mismatch, which the loader rejects instead of guessing.
        os.replace(log_tmp, path)
        log_tmp = None
        os.replace(metadata_tmp, sidecar)
        metadata_tmp = None
    finally:
        for temporary in (log_tmp, metadata_tmp):
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    return load_schedule_snapshot(path, deployment)


def _write_temp(directory, data):
    descriptor, name = tempfile.mkstemp(prefix=".schedule-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return name


def _decode_rows(log_bytes, path):
    try:
        text = log_bytes.decode("utf-8")
        lines = text.splitlines()
        if not lines and log_bytes:
            raise ValueError("empty AutoTVM log")
        records = [autotvm.record.decode(line) for line in lines]
    except Exception as error:
        raise ValueError(f"native AutoTVM log cannot be decoded: {path}") from error
    if any(record is None for record in records):
        raise ValueError(f"native AutoTVM log contains unsupported legacy records: {path}")
    return lines, records


def load_schedule_snapshot(path, deployment):
    """Validate a snapshot against the current actual deployment description."""
    if path is None or (isinstance(path, str) and path.lower() == "none"):
        return ScheduleSnapshot(
            None, deployment.model_id, deployment.model_sha256,
            _geometry_identity(deployment), {},
        )
    path = Path(path)
    sidecar = _sidecar_path(path)
    try:
        log_bytes = path.read_bytes()
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"schedule log or same-stem metadata is missing: {error.filename}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"schedule snapshot cannot be read: {path}") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schedule metadata schema")
    if "provenance" in metadata and not isinstance(metadata["provenance"], dict):
        raise ValueError("schedule provenance must be a JSON object")
    if metadata.get("model_id") != deployment.model_id:
        raise ValueError("schedule model identity does not match the prepared deployment")
    if metadata.get("model_sha256") != deployment.model_sha256:
        raise ValueError("schedule model content hash does not match the prepared deployment")
    if metadata.get("config_sha256") != _active_config_sha256():
        raise ValueError("schedule geometry config content hash does not match VTA_CONFIG_FILE")
    if metadata.get("geometry_sha256") != _geometry_identity(deployment):
        raise ValueError("schedule geometry identity does not match the active VTA geometry")
    if metadata.get("log_sha256") != _sha256(log_bytes):
        raise ValueError("schedule log hash does not match its metadata")
    lines, records = _decode_rows(log_bytes, path)

    layers = {layer.occurrence: layer for layer in deployment.layers}
    selected = {}
    seen_records = set()
    seen_occurrences = set()
    occurrence_rows = metadata.get("occurrences")
    if not isinstance(occurrence_rows, list):
        raise ValueError("schedule metadata occurrences must be a list")
    for row in occurrence_rows:
        if not isinstance(row, dict):
            raise ValueError("schedule occurrence metadata must be an object")
        occurrence = row.get("occurrence")
        if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence not in layers:
            raise ValueError(f"schedule contains unknown occurrence {occurrence!r}")
        if occurrence in seen_occurrences:
            raise ValueError(f"schedule contains duplicate occurrence {occurrence}")
        seen_occurrences.add(occurrence)
        layer = layers[occurrence]
        expected_identity = _layer_identity(layer)
        for key, value in expected_identity.items():
            if row.get(key) != value:
                raise ValueError(f"schedule occurrence {occurrence} {key} does not match deployment")
        record_rows = row.get("records")
        tunable_entries = [entry for entry in layer.config_spaces if _tunable_template(entry[0], entry[3])]
        if not isinstance(record_rows, list) or len(record_rows) != len(tunable_entries):
            raise ValueError(f"schedule occurrence {occurrence} must select each tunable captured template")
        configs = []
        refs_by_key = {}
        for ref in record_rows:
            if not isinstance(ref, dict):
                raise ValueError(f"schedule occurrence {occurrence} record reference is malformed")
            key = (ref.get("template"), ref.get("workload"), ref.get("target"))
            if key in refs_by_key:
                raise ValueError(f"schedule occurrence {occurrence} has duplicate template records")
            refs_by_key[key] = ref
        for entry in layer.config_spaces:
            template, workload, target, space = entry
            if not _tunable_template(template, space):
                configs.append(None)
                continue
            key = (template, repr(workload), _portable_target(target))
            ref = refs_by_key.pop(key, None)
            if ref is None:
                raise ValueError(f"schedule occurrence {occurrence} is missing the {template} record")
            record_index = ref.get("record_index")
            if isinstance(record_index, bool) or not isinstance(record_index, int) or not (0 <= record_index < len(records)):
                raise ValueError(f"schedule occurrence {occurrence} record index is invalid")
            if record_index in seen_records:
                raise ValueError(f"schedule log record {record_index} is referenced more than once")
            seen_records.add(record_index)
            if ref.get("record_sha256") != _sha256(lines[record_index].encode("utf-8")):
                raise ValueError(f"schedule occurrence {occurrence} native record hash mismatch")
            measure_input, measure_result = records[record_index]
            if (measure_input.task.name != template or tuple(measure_input.task.args) != tuple(workload[1:])
                    or _portable_target(measure_input.target) != _portable_target(target)):
                raise ValueError(f"schedule occurrence {occurrence} native record workload/target mismatch")
            config_dict = measure_input.config.to_json_dict()
            if (ref.get("template") != template or ref.get("workload") != repr(workload)
                    or ref.get("target") != _portable_target(target)):
                raise ValueError(f"schedule occurrence {occurrence} record identity metadata mismatch")
            if _canonical_json(ref.get("config")) != _canonical_json(config_dict):
                raise ValueError(f"schedule occurrence {occurrence} metadata config differs from native record")
            try:
                config = autotvm.task.ConfigEntity.from_json_dict(config_dict)
            except Exception as error:
                raise ValueError(f"schedule occurrence {occurrence} native config is invalid") from error
            matching_index = next(
                (index for index in range(len(space))
                if _canonical_json(space.get(index).to_json_dict()) == _canonical_json(config_dict)), None
            )
            if matching_index is None or not config.valid():
                raise ValueError(f"schedule occurrence {occurrence} config is invalid for {template}")
            configs.append(config)
        if refs_by_key:
            raise ValueError(f"schedule occurrence {occurrence} contains unknown template records")
        measured = row.get("measured")
        measurement = row.get("measurement")
        if not isinstance(measured, bool) or (measured != (measurement is not None)):
            raise ValueError(f"schedule occurrence {occurrence} measurement provenance is inconsistent")
        if measured:
            _validate_measured_provenance(measurement, occurrence)
            results = measurement.get("results")
            if not isinstance(results, list) or len(results) != len(record_rows):
                raise ValueError(f"schedule occurrence {occurrence} measurement provenance is malformed")
            for item, (_, result) in zip(results, [records[ref["record_index"]] for ref in record_rows]):
                if result.error_no != 0:
                    raise ValueError(
                        f"schedule occurrence {occurrence} native measurement result reports failure"
                    )
                if (not result.costs or any(
                        isinstance(cost, bool) or not isinstance(cost, (int, float))
                        or not math.isfinite(cost) or cost <= 0 for cost in result.costs)):
                    raise ValueError(
                        f"schedule occurrence {occurrence} native measurement costs must be finite positive numbers"
                    )
                if (isinstance(result.timestamp, bool) or not isinstance(result.timestamp, (int, float))
                        or not math.isfinite(result.timestamp) or result.timestamp <= 0):
                    raise ValueError(
                        f"schedule occurrence {occurrence} native measurement timestamp must be finite and positive"
                    )
                if (isinstance(result.all_cost, bool) or not isinstance(result.all_cost, (int, float))
                        or not math.isfinite(result.all_cost) or result.all_cost < 0):
                    raise ValueError(
                        f"schedule occurrence {occurrence} native measurement all_cost must be finite and non-negative"
                    )
                native_result = {
                    "costs": list(result.costs),
                    "error_no": result.error_no,
                    "all_cost": result.all_cost,
                    "timestamp": result.timestamp,
                }
                if not isinstance(item, dict) or _canonical_json(item) != _canonical_json(native_result):
                    raise ValueError(f"schedule occurrence {occurrence} measurement disagrees with native record")
        selected[occurrence] = SelectedSchedule(
            occurrence, layer.symbol, tuple(configs), measured, measurement
        )
    if seen_records != set(range(len(records))):
        raise ValueError("schedule log contains unreferenced native records")
    return ScheduleSnapshot(
        path, deployment.model_id, deployment.model_sha256,
        _geometry_identity(deployment), selected,
    )
