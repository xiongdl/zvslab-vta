"""Shared durable controller primitives for two-stage Tiny operator tuning."""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
DEFAULT_TRIAL_BATCH = 100
DEFAULT_MIN_SUCCESSFUL = 20
TSIM_PROTOCOL = {"name": "tsim_single_call", "version": 1, "warmup_excluded": True}


@dataclass(frozen=True)
class SeedGateBinding:
    """Hashes returned only after the seed manifest and deployment report pass."""

    model: str
    model_sha256: str
    geometry_sha256: str
    seed_manifest_sha256: str
    seed_gate_report_sha256: str


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _valid_digest(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_identity(identity, *, require_seed_gate=False):
    if not isinstance(identity, dict):
        raise ValueError("run identity must be a JSON object")
    for field in ("model", "model_sha256", "geometry_sha256", "workload_sha256",
                  "fusion_sha256", "symbol"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError(f"run identity {field} must be a non-empty string")
    for field in ("model_sha256", "geometry_sha256", "workload_sha256", "fusion_sha256"):
        _valid_digest(identity[field], f"run identity {field}")
    for field in ("config_space_size", "trial_batch", "min_successful",
                  "fsim_timeout_seconds", "tsim_timeout_seconds"):
        _positive_int(identity.get(field), f"run identity {field}")
    occurrence = identity.get("occurrence")
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 0:
        raise ValueError("run identity occurrence must be a non-negative integer")
    valid_size = identity.get("valid_config_space_size", identity["config_space_size"])
    _positive_int(valid_size, "run identity valid_config_space_size")
    if valid_size > identity["config_space_size"]:
        raise ValueError("valid configuration space exceeds raw configuration space")
    if require_seed_gate:
        _valid_digest(identity.get("seed_manifest_sha256"), "run identity seed_manifest_sha256")
        _valid_digest(identity.get("seed_gate_report_sha256"), "run identity seed_gate_report_sha256")


def _state_payload(state):
    return {key: value for key, value in state.items() if key != "state_sha256"}


def create_state(identity, path, *, seed_gate):
    """Create a full-search state bound to the passing seed report and its manifest."""
    if not isinstance(seed_gate, SeedGateBinding):
        raise ValueError("full search requires a validated SeedGateBinding")
    bound = dict(identity)
    for field in ("model", "model_sha256", "geometry_sha256"):
        if bound.get(field) != getattr(seed_gate, field):
            raise ValueError(f"passing seed gate {field} does not match full-search identity")
    bound["seed_manifest_sha256"] = seed_gate.seed_manifest_sha256
    bound["seed_gate_report_sha256"] = seed_gate.seed_gate_report_sha256
    _validate_identity(bound, require_seed_gate=True)
    state = {
        "schema_version": SCHEMA_VERSION,
        "identity": bound,
        "identity_sha256": sha256_json(bound),
        "state_path": str(Path(path).expanduser().resolve()),
        "visited_indices": [],
        "attempted_config_hashes": [],
        "fsim_results": [],
        "tsim_results": [],
        "infrastructure_errors": [],
        "stop_reason": None,
    }
    state["state_sha256"] = sha256_json(_state_payload(state))
    return state


def load_state(path, expected_identity):
    """Load only an untampered state with an exact model/seed/options identity."""
    _validate_identity(expected_identity, require_seed_gate=True)
    state_path = Path(path).expanduser().resolve(strict=True)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"search state is not readable JSON: {state_path}") from error
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing search-state schema")
    if state.get("identity") != expected_identity or state.get("identity_sha256") != sha256_json(expected_identity):
        raise ValueError("search state identity does not match model, seed gate, geometry, or options")
    if state.get("state_sha256") != sha256_json(_state_payload(state)):
        raise ValueError("search state integrity hash does not match its contents")
    _validate_ledger(state, expected_identity)
    state["state_path"] = str(state_path)
    return state


def _validate_ledger(state, identity):
    visited = state.get("visited_indices")
    hashes = state.get("attempted_config_hashes")
    fsim = state.get("fsim_results")
    tsim = state.get("tsim_results")
    if not isinstance(visited, list) or any(
        isinstance(index, bool) or not isinstance(index, int)
        or index < 0 or index >= identity["config_space_size"] for index in visited
    ) or len(set(visited)) != len(visited):
        raise ValueError("search state visited configuration indices are invalid")
    if len(visited) != len(hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("search state contains duplicate or incomplete attempted configs")
    if not isinstance(fsim, list) or len(fsim) != len(visited):
        raise ValueError("search state FSIM ledger does not match attempted configurations")
    fsim_by_hash = {}
    for item in fsim:
        if not isinstance(item, dict) or item.get("config_sha256") not in hashes:
            raise ValueError("search state FSIM ledger contains an unknown config")
        if item["config_sha256"] in fsim_by_hash:
            raise ValueError("search state FSIM ledger contains duplicate config identities")
        fsim_by_hash[item["config_sha256"]] = item
    successes = {key for key, item in fsim_by_hash.items() if item.get("success") is True}
    tsim_by_hash = set()
    if not isinstance(tsim, list):
        raise ValueError("search state TSIM ledger must be a list")
    for item in tsim:
        key = item.get("config_sha256") if isinstance(item, dict) else None
        if key not in successes or key in tsim_by_hash:
            raise ValueError("search state TSIM ledger has a foreign or duplicate config")
        tsim_by_hash.add(key)
    if any(not isinstance(err, dict) for err in state.get("infrastructure_errors", [])):
        raise ValueError("search state infrastructure errors are malformed")


def save_state(state):
    """Atomically save durable progress after a candidate outcome."""
    _validate_ledger(state, state["identity"])
    path = Path(state["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    state["state_sha256"] = sha256_json(_state_payload(state))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def successful_fsim_count(state):
    return sum(result.get("success") is True for result in state["fsim_results"])


def should_continue(state):
    return (successful_fsim_count(state) < state["identity"]["min_successful"]
            and len(state["visited_indices"]) < state["identity"].get(
                "valid_config_space_size", state["identity"]["config_space_size"]))


def next_batch_size(state):
    if not should_continue(state):
        return 0
    remaining = state["identity"].get(
        "valid_config_space_size", state["identity"]["config_space_size"]
    ) - len(state["visited_indices"])
    return min(state["identity"]["trial_batch"], remaining)


def record_fsim_result(state, config_index, config, *, success, error=None):
    """Record one unique FSIM configuration; duplicate indices/configs are errors."""
    identity = state["identity"]
    if (isinstance(config_index, bool) or not isinstance(config_index, int)
            or not 0 <= config_index < identity["config_space_size"]):
        raise ValueError("FSIM config index is outside the configuration space")
    if config_index in state["visited_indices"]:
        raise ValueError(f"FSIM config index {config_index} was already visited")
    if not isinstance(config, dict):
        raise ValueError("FSIM config identity must be a JSON object")
    digest = sha256_json(config)
    if digest in state["attempted_config_hashes"]:
        raise ValueError("FSIM duplicate configuration identity")
    if not isinstance(success, bool):
        raise ValueError("FSIM success must be boolean")
    if not success and (not isinstance(error, str) or not error):
        raise ValueError("failed FSIM candidate requires a diagnostic")
    state["visited_indices"].append(config_index)
    state["attempted_config_hashes"].append(digest)
    state["fsim_results"].append({
        "config_index": config_index,
        "config": config,
        "config_sha256": digest,
        "success": success,
        "error": None if success else error,
    })
    if successful_fsim_count(state) >= identity["min_successful"]:
        state["stop_reason"] = "successful_schedule_quota"
    elif len(state["visited_indices"]) >= identity.get(
            "valid_config_space_size", identity["config_space_size"]):
        state["stop_reason"] = "configuration_space_exhausted"


def record_infrastructure_error(state, backend, config_index, error):
    if backend not in ("fsim", "tsim"):
        raise ValueError("backend must be fsim or tsim")
    state["infrastructure_errors"].append({
        "backend": backend,
        "config_index": config_index,
        "error": f"{type(error).__name__}: {error}",
    })


def pending_tsim_configs(state):
    """Return every successful FSIM config without a TSIM attempt."""
    attempted = {item["config_sha256"] for item in state["tsim_results"]}
    return [item for item in state["fsim_results"]
            if item["success"] and item["config_sha256"] not in attempted]


def record_tsim_result(state, config_sha256, *, cycles=None, error=None,
                       deployment_lowerable=False, lowering_error=None):
    """Record exactly one TSIM attempt for a successful FSIM config."""
    if config_sha256 not in {item["config_sha256"] for item in state["fsim_results"]
                             if item["success"]}:
        raise ValueError("TSIM candidate is not a successful FSIM configuration")
    if config_sha256 in {item["config_sha256"] for item in state["tsim_results"]}:
        raise ValueError("TSIM candidate was already attempted")
    if cycles is not None:
        _positive_int(cycles, "TSIM cycle count")
    if cycles is None and (not isinstance(error, str) or not error):
        raise ValueError("failed TSIM candidate requires a diagnostic")
    if not isinstance(deployment_lowerable, bool):
        raise ValueError("deployment_lowerable must be boolean")
    if lowering_error is not None and not isinstance(lowering_error, str):
        raise ValueError("lowering_error must be a string")
    state["tsim_results"].append({
        "config_sha256": config_sha256,
        "tsim_cycles": cycles,
        "error": error,
        "deployment_lowerable": deployment_lowerable,
        "lowering_error": lowering_error,
    })


def select_best_tsim(state):
    """Select the minimum positive TSIM cycle among real-lowerable configs."""
    pending = pending_tsim_configs(state)
    if pending:
        raise ValueError(f"{len(pending)} successful FSIM schedules still need TSIM attempts")
    lowerable = {item["config_sha256"] for item in state["tsim_results"]
                 if item["tsim_cycles"] is not None and item["deployment_lowerable"]}
    candidates = [item for item in state["tsim_results"] if item["config_sha256"] in lowerable]
    if not candidates:
        raise ValueError("no positive TSIM candidate lowers through real deployment")
    return min(candidates, key=lambda item: item["tsim_cycles"])


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def write_seed_manifest(path, identity, entries):
    """Export separate, complete seed evidence before any full search starts."""
    if not isinstance(identity, dict) or not isinstance(entries, list) or not entries:
        raise ValueError("seed manifest needs an identity and non-empty entries")
    expected = identity.get("occurrences")
    if not isinstance(expected, list) or not expected:
        raise ValueError("seed identity must enumerate expected occurrences")
    actual = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("seed entries must be objects")
        actual.append(entry.get("occurrence"))
        for field in ("symbol", "fusion_sha256", "workload_sha256", "config"):
            if field not in entry:
                raise ValueError(f"seed entry requires {field}")
        _valid_digest(entry["fusion_sha256"], "seed fusion_sha256")
        _valid_digest(entry["workload_sha256"], "seed workload_sha256")
        _positive_int(entry.get("autotvm_cycles"), "seed AutoTVM TSIM cycles")
        _nonnegative_int(entry.get("config_index"), "seed config index")
        if entry.get("fsim_success") is not True:
            raise ValueError("seed entry must contain a successful FSIM schedule")
        config_hash = sha256_json(entry["config"])
        if entry.get("config_sha256", config_hash) != config_hash:
            raise ValueError("seed config SHA-256 does not match its configuration")
        entry["config_sha256"] = config_hash
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("seed entries do not cover expected occurrences in order")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "vta_seed_schedules_v1",
        "status": "complete",
        "identity": identity,
        "identity_sha256": sha256_json(identity),
        "entries": entries,
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return sha256_file(destination)


def validate_seed_gate(report_path, manifest_path, expected_identity, occurrences):
    """Validate a passing one-sample deployment report and bind its two hashes."""
    report = _read_json(report_path, "seed deployment report")
    manifest = _read_json(manifest_path, "seed schedule manifest")
    manifest_hash = sha256_file(manifest_path)
    report_hash = sha256_file(report_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("artifact_kind") != "vta_seed_schedules_v1":
        raise ValueError("unsupported seed schedule manifest")
    if manifest.get("status") != "complete":
        raise ValueError("seed schedule manifest is incomplete")
    identity = manifest.get("identity")
    if identity != expected_identity or manifest.get("identity_sha256") != sha256_json(identity):
        raise ValueError("seed manifest belongs to another model, geometry, or graph")
    if (report.get("schema_version") != SCHEMA_VERSION
            or report.get("artifact_kind") != "vta_deployment_profile_v1"
            or report.get("phase") != "seed" or report.get("status") != "passed"):
        raise ValueError("seed deployment report is not a passing seed-gate artifact")
    if report.get("sample_count") != 1:
        raise ValueError("seed deployment report must validate exactly one sample")
    protocol = report.get("measurement_protocol")
    if not isinstance(protocol, dict) or any(
            protocol.get(key) != value for key, value in TSIM_PROTOCOL.items()):
        raise ValueError("seed deployment report requires TSIM single-call v1 measurements")
    for field in ("model", "model_sha256", "geometry_sha256"):
        if report.get(field) != identity.get(field):
            raise ValueError(f"seed deployment {field} does not match seed manifest")
    if report.get("selected_manifest_sha256") != manifest_hash:
        raise ValueError("seed deployment report does not bind the seed schedule manifest")
    expected_rows = {item["occurrence"]: item for item in occurrences}
    if len(expected_rows) != len(occurrences) or list(expected_rows) != [
            entry.get("occurrence") for entry in manifest["entries"]]:
        raise ValueError("prepared occurrence identity list is duplicated or unordered")
    entries = manifest["entries"]
    if [entry.get("occurrence") for entry in entries] != list(expected_rows):
        raise ValueError("seed manifest occurrence coverage differs from prepared graph")
    rows = report.get("occurrences")
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise ValueError("seed deployment occurrence coverage is incomplete")
    seen = set()
    by_entry = {entry["occurrence"]: entry for entry in entries}
    for row in rows:
        occurrence = row.get("occurrence") if isinstance(row, dict) else None
        if occurrence not in expected_rows or occurrence in seen:
            raise ValueError("seed deployment contains an unexpected or duplicate occurrence")
        seen.add(occurrence)
        expected = expected_rows[occurrence]
        entry = by_entry[occurrence]
        for field in ("symbol", "fusion_sha256", "workload_sha256", "config_sha256"):
            if row.get(field) != expected.get(field, entry.get(field)):
                raise ValueError(f"seed deployment {field} mismatch at occurrence {occurrence}")
        auto_cycles = _positive_int(row.get("autotvm_cycles"), "seed AutoTVM TSIM cycles")
        if auto_cycles != entry["autotvm_cycles"]:
            raise ValueError(f"seed AutoTVM cycles disagree at occurrence {occurrence}")
        deployed = _positive_int(row.get("deployment_cycles"), "seed deployment TSIM cycles")
        if 10 * abs(deployed - auto_cycles) > auto_cycles:
            raise ValueError(f"seed deployment cycle deviation exceeds 10% at occurrence {occurrence}")
        if row.get("passed") is not True:
            raise ValueError(f"seed deployment gate did not pass at occurrence {occurrence}")
    return SeedGateBinding(
        model=identity["model"],
        model_sha256=identity["model_sha256"],
        geometry_sha256=identity["geometry_sha256"],
        seed_manifest_sha256=manifest_hash,
        seed_gate_report_sha256=report_hash,
    )


def validate_replay_manifest(path, expected_identity, occurrences):
    """Validate selected exports, their self-contained file hashes and coverage."""
    manifest_path = Path(path).expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path, "selected schedule manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("artifact_kind") != "vta_selected_schedules_v1":
        raise ValueError("unsupported selected schedule manifest")
    identity = manifest.get("identity")
    if identity != expected_identity or manifest.get("identity_sha256") != sha256_json(identity):
        raise ValueError("selected manifest identity does not match model, graph, seed gate, or search")
    if manifest.get("measurement_protocol") != TSIM_PROTOCOL:
        raise ValueError("selected manifest TSIM protocol is not single-call v1")
    expected = list(occurrences)
    entries = manifest.get("entries")
    expected_indices = [item.get("occurrence") if isinstance(item, dict) else item
                        for item in expected]
    if not isinstance(entries, list) or [item.get("occurrence") for item in entries] != expected_indices:
        raise ValueError("selected manifest occurrence coverage is incomplete or unordered")
    for entry in entries:
        expected_item = next(
            (item for item in expected if isinstance(item, dict)
             and item.get("occurrence") == entry["occurrence"]),
            None,
        )
        if expected_item is not None:
            for field in ("symbol", "fusion_sha256", "workload_sha256"):
                if entry.get(field) != expected_item.get(field):
                    raise ValueError(f"selected manifest {field} mismatch at occurrence {entry['occurrence']}")
        for key, digest_key in (("result_json", "result_sha256"), ("native_record", "native_record_sha256")):
            name = entry.get(key)
            digest = _valid_digest(entry.get(digest_key), f"selected {digest_key}")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"selected {key} must be a local filename")
            artifact = manifest_path.parent / name
            if not artifact.is_file() or sha256_file(artifact) != digest:
                raise ValueError(f"selected artifact {key} is missing or has a mismatched hash")
            if key == "result_json":
                result = _read_json(artifact, "selected result JSON")
                if result.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("selected result JSON has an unsupported schema")
                for field in ("model", "model_sha256", "geometry_sha256"):
                    if result.get(field) != identity.get(field):
                        raise ValueError(f"selected result {field} does not match run identity")
                for field in ("occurrence", "symbol", "fusion_sha256", "workload_sha256",
                              "config_sha256", "tsim_cycles"):
                    if result.get(field) != entry.get(field):
                        raise ValueError(f"selected result {field} mismatch at occurrence {entry['occurrence']}")
        _positive_int(entry.get("tsim_cycles"), "selected TSIM cycles")
    return manifest


def worker_environment(backend, base_environment=None, geometry_path=None, python_paths=()):
    """Build an explicit backend environment for one isolated worker process."""
    if backend not in ("fsim", "tsim"):
        raise ValueError("worker backend must be fsim or tsim")
    environment = dict(os.environ if base_environment is None else base_environment)
    geometry = geometry_path or environment.get("VTA_CONFIG_FILE")
    if not geometry:
        raise ValueError("worker requires VTA_CONFIG_FILE")
    geometry = Path(geometry).expanduser().resolve(strict=True)
    environment["VTA_CONFIG_FILE"] = str(geometry)
    environment["VTA_BACKEND"] = backend
    existing = [item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item]
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*(str(x) for x in python_paths), *existing]))
    return environment


def run_isolated_worker(command, backend, *, base_environment=None, geometry_path=None,
                        python_paths=(), cwd=None):
    """Run one backend worker as a new process with an explicit environment."""
    if not isinstance(command, (list, tuple)) or not command or not all(
            isinstance(item, str) for item in command):
        raise ValueError("worker command must be a non-empty string argument list")
    return subprocess.run(
        list(command),
        env=worker_environment(backend, base_environment, geometry_path, python_paths),
        cwd=cwd,
        check=False,
    )
