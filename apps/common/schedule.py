"""Export and validate deployable snapshots of actual VTA layer schedules."""

import hashlib
import json
import os
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


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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
    if any(isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0 for cost in costs):
        raise ValueError("measurement costs must be non-negative numbers")
    if isinstance(error_no, bool) or not isinstance(error_no, int) or error_no < 0:
        raise ValueError("measurement error_no must be a non-negative integer")
    if isinstance(all_cost, bool) or not isinstance(all_cost, (int, float)) or all_cost < 0:
        raise ValueError("measurement all_cost must be a non-negative number")
    return autotvm.measure.MeasureResult(tuple(costs), error_no, all_cost, timestamp)


def _layer_identity(layer):
    return {
        "occurrence": layer.occurrence,
        "symbol": layer.symbol,
        "compute_sha256": layer.compute_sha256,
        "config_space_identity": layer.config_space_identity,
        "workloads": [
            {"template": template, "workload": repr(workload), "target": target,
             "space_size": len(space)}
            for template, workload, target, space in layer.config_spaces
        ],
    }


def export_schedule_snapshot(path, deployment, selections, *, measurements=None):
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
            if not isinstance(measurement_provenance, dict) or measurement_provenance.get("backend") not in ("fsim", "tsim"):
                raise ValueError(f"occurrence {occurrence} measurement backend is invalid")
            if not isinstance(measurement_provenance.get("protocol"), str) or not measurement_provenance["protocol"]:
                raise ValueError(f"occurrence {occurrence} measurement protocol is required")
            if not isinstance(measurement_provenance.get("units"), str) or not measurement_provenance["units"]:
                raise ValueError(f"occurrence {occurrence} measurement units are required")
            result_specs = measurement_provenance.get("results")
            if not isinstance(result_specs, (list, tuple)) or len(result_specs) != len(indices):
                raise ValueError(f"occurrence {occurrence} measurement count does not match config count")
        else:
            result_specs = None
        record_refs = []
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
            spec = None if result_specs is None else result_specs[index]
            record_result = _measurement_result(spec)
            row = autotvm.record.encode(_record_input(layer, entry, config), record_result)
            row_index = len(rows)
            rows.append(row)
            record_refs.append({
                "template": template,
                "workload": repr(workload),
                "target": str(target),
                "record_index": row_index,
                "record_sha256": _sha256(row.encode("utf-8")),
                "config": config.to_json_dict(),
            })
        occurrences.append({
            **_layer_identity(layer),
            "measured": measurement_provenance is not None,
            "measurement": measurement_provenance,
            "records": record_refs,
        })

    log_bytes = ("\n".join(rows) + ("\n" if rows else "")).encode("utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": deployment.model_id,
        "model_sha256": deployment.model_sha256,
        "geometry_sha256": deployment.geometry_sha256,
        "log_sha256": _sha256(log_bytes),
        "occurrences": occurrences,
    }
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
            deployment.geometry_sha256, {},
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
    if metadata.get("model_id") != deployment.model_id:
        raise ValueError("schedule model identity does not match the prepared deployment")
    if metadata.get("model_sha256") != deployment.model_sha256:
        raise ValueError("schedule model content hash does not match the prepared deployment")
    if metadata.get("geometry_sha256") != deployment.geometry_sha256:
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
        if not isinstance(record_rows, list) or len(record_rows) != len(layer.config_spaces):
            raise ValueError(f"schedule occurrence {occurrence} must select every captured template")
        configs = []
        for ref, entry in zip(record_rows, layer.config_spaces):
            template, workload, target, space = entry
            if not isinstance(ref, dict):
                raise ValueError(f"schedule occurrence {occurrence} record reference is malformed")
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
                    or str(measure_input.target) != str(target)):
                raise ValueError(f"schedule occurrence {occurrence} native record workload/target mismatch")
            config_dict = measure_input.config.to_json_dict()
            if ref.get("template") != template or ref.get("workload") != repr(workload) or ref.get("target") != str(target):
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
        measured = row.get("measured")
        measurement = row.get("measurement")
        if not isinstance(measured, bool) or (measured != (measurement is not None)):
            raise ValueError(f"schedule occurrence {occurrence} measurement provenance is inconsistent")
        if measured:
            if (not isinstance(measurement, dict)
                    or measurement.get("backend") not in ("fsim", "tsim")
                    or not isinstance(measurement.get("protocol"), str)
                    or not measurement.get("protocol")
                    or not isinstance(measurement.get("units"), str)
                    or not measurement.get("units")):
                raise ValueError(f"schedule occurrence {occurrence} measurement provenance is malformed")
            results = measurement.get("results")
            if not isinstance(results, list) or len(results) != len(configs):
                raise ValueError(f"schedule occurrence {occurrence} measurement provenance is malformed")
            for item, (_, result) in zip(results, [records[ref["record_index"]] for ref in record_rows]):
                if not isinstance(item, dict) or result.error_no != item.get("error_no", 0):
                    raise ValueError(f"schedule occurrence {occurrence} measurement disagrees with native record")
        selected[occurrence] = SelectedSchedule(
            occurrence, layer.symbol, tuple(configs), measured, measurement
        )
    if seen_records != set(range(len(records))):
        raise ValueError("schedule log contains unreferenced native records")
    return ScheduleSnapshot(
        path, deployment.model_id, deployment.model_sha256,
        deployment.geometry_sha256, selected,
    )
