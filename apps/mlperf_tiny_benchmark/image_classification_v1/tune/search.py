"""Durable state and selection helpers for the IC V1 two-stage search."""

import hashlib
import json
import os
from pathlib import Path


STATE_SCHEMA_VERSION = 1


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _identity_hash(identity):
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def create_state(identity, path):
    """Create an empty FSIM state bound to immutable task and run identity."""
    _validate_identity(identity)
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "identity": identity,
        "identity_sha256": _identity_hash(identity),
        "state_path": str(Path(path).expanduser().resolve()),
        "visited_indices": [],
        "attempted_count": 0,
        "successful_configurations": {},
        "failures": [],
        "infrastructure_errors": [],
        "stop_reason": None,
    }
    return state


def _validate_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("search identity must be a JSON object")
    for field in ("model_sha256", "geometry_sha256", "workload_sha256"):
        value = identity.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"search identity {field} must be a SHA-256 hex digest")
    for field in ("config_space_size", "trial_batch", "min_successful",
                  "fsim_timeout_seconds", "tsim_timeout_seconds"):
        value = identity.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"search identity {field} must be a positive integer")
    if identity["trial_batch"] <= 0:
        raise ValueError("trial batch must be positive")
    valid_size = identity.get("valid_config_space_size", identity["config_space_size"])
    if isinstance(valid_size, bool) or not isinstance(valid_size, int) or not 0 < valid_size <= identity["config_space_size"]:
        raise ValueError("search identity valid_config_space_size must fit the configuration space")


def load_state(path, expected_identity):
    """Load state only when all computation and measurement options match."""
    _validate_identity(expected_identity)
    path = Path(path).expanduser().resolve(strict=True)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"search state is not readable JSON: {path}") from error
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("unsupported or missing search-state schema")
    if state.get("identity") != expected_identity or state.get("identity_sha256") != _identity_hash(
        expected_identity
    ):
        raise ValueError("search state identity does not match model, geometry, workload, or options")
    visited = state.get("visited_indices")
    if not isinstance(visited, list) or any(
        isinstance(index, bool) or not isinstance(index, int)
        or index < 0 or index >= expected_identity["config_space_size"]
        for index in visited
    ) or len(set(visited)) != len(visited):
        raise ValueError("search state visited configuration indices are invalid")
    state["state_path"] = str(path)
    return state


def save_state(state):
    """Atomically persist state after a measured configuration."""
    path = Path(state["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def next_batch_size(state):
    """Number of new distinct configurations to attempt in the next batch."""
    if not should_continue(state):
        return 0
    remaining = _valid_space_size(state) - len(state["visited_indices"])
    return min(state["identity"]["trial_batch"], remaining)


def record_trial(state, config_index, config, *, successful, error=None):
    """Add one unique configuration outcome to durable search state."""
    size = state["identity"]["config_space_size"]
    if isinstance(config_index, bool) or not isinstance(config_index, int) or not 0 <= config_index < size:
        raise ValueError(f"configuration index must be in 0..{size - 1}")
    if config_index in state["visited_indices"]:
        raise ValueError(f"configuration index {config_index} was already visited")
    if not isinstance(config, dict):
        raise ValueError("configuration identity must be a JSON object")
    state["visited_indices"].append(config_index)
    state["attempted_count"] += 1
    if successful:
        key = _canonical(config)
        state["successful_configurations"].setdefault(
            key, {"config_index": config_index, "config": config}
        )
    else:
        state["failures"].append(
            {"config_index": config_index, "error": str(error or "measurement failed")}
        )
    state["stop_reason"] = stop_reason(state)


def record_infrastructure_error(state, backend, config_index, error):
    """Persist infrastructure failures separately from candidate failures."""
    state.setdefault("infrastructure_errors", []).append({
        "backend": backend,
        "config_index": config_index,
        "error": f"{type(error).__name__}: {error}",
    })


def successful_configurations(state):
    """Return successful distinct configs in discovery order."""
    return sorted(
        state["successful_configurations"].values(), key=lambda item: item["config_index"]
    )


def success_count(state):
    return len(state["successful_configurations"])


def should_continue(state):
    return (
        success_count(state) < state["identity"]["min_successful"]
        and len(state["visited_indices"]) < _valid_space_size(state)
    )


def stop_reason(state):
    if success_count(state) >= state["identity"]["min_successful"]:
        return "successful_schedule_quota"
    if len(state["visited_indices"]) >= _valid_space_size(state):
        return "configuration_space_exhausted"
    return None


def _valid_space_size(state):
    return state["identity"].get(
        "valid_config_space_size", state["identity"]["config_space_size"]
    )


def select_best_tsim(candidates):
    """Select the minimum valid positive single-call TSIM cycle count."""
    valid = [
        candidate for candidate in candidates
        if not candidate.get("error")
        and isinstance(candidate.get("tsim_cycles"), int)
        and not isinstance(candidate.get("tsim_cycles"), bool)
        and candidate["tsim_cycles"] > 0
    ]
    if not valid:
        raise ValueError("no successful TSIM candidate has a positive cycle count")
    return min(valid, key=lambda item: item["tsim_cycles"])
