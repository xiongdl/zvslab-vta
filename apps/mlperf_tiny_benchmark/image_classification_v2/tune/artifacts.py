"""Self-contained native AutoTVM record export and validation helpers."""

import hashlib
import json
import os
from pathlib import Path


def _config_json(config):
    value = config.to_json_dict() if hasattr(config, "to_json_dict") else config
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def export_selected_record(records, config, cycles, destination, record_module):
    """Write one successful native record matching the selected TSIM result."""
    destination = Path(destination).expanduser().resolve()
    expected_config = _config_json(config)
    selected = []
    for measure_input, result in records:
        if _config_json(measure_input.config) != expected_config:
            continue
        if result.error_no != 0 or not result.costs:
            continue
        measured_cycles = result.costs[0]
        if isinstance(measured_cycles, bool) or not isinstance(measured_cycles, int):
            continue
        if measured_cycles == cycles:
            selected.append((measure_input, result))
    if len(selected) != 1:
        raise ValueError(
            "selected TSIM candidate must identify exactly one successful native record; "
            f"found {len(selected)}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = record_module.encode(*selected[0]) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"path": str(destination), "sha256": digest}


def load_validated_record(path, expected_sha256, record_module):
    """Verify a standalone native record's bytes and single-record contents."""
    path = Path(path).expanduser().resolve(strict=True)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("best native record SHA-256 does not match")
    records = list(record_module.load_from_file(str(path)))
    if len(records) != 1:
        raise ValueError(f"best native record must contain exactly one entry, got {len(records)}")
    measure_input, result = records[0]
    if result.error_no != 0 or not result.costs:
        raise ValueError("best native record is not a successful measurement")
    return measure_input, result


def write_json(path, value):
    """Atomically write a stable, human-readable JSON artifact."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
