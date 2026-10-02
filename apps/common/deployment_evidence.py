"""Shared validation and reporting for one-sample real VTA deployments."""

import hashlib
import json
import os
from pathlib import Path


REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "vta_deployment_profile_v1"
PROTOCOL = {"name": "tsim_single_call", "version": 1, "warmup_excluded": True}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cycles_within_ten_percent(deployment_cycles, autotvm_cycles):
    """Use the approved inclusive 10% threshold with exact integer arithmetic."""
    for label, value in (("deployment", deployment_cycles), ("AutoTVM", autotvm_cycles)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} cycles must be a positive integer")
    return 10 * abs(deployment_cycles - autotvm_cycles) <= autotvm_cycles


def validate_selected_configs(manifest, expected_occurrences, *, phase):
    """Match each exact selected configuration to its prepared symbol occurrence."""
    if phase not in ("seed", "selected"):
        raise ValueError("deployment phase must be seed or selected")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported selected schedule manifest")
    if manifest.get("status") != "complete":
        raise ValueError("deployment requires complete selected schedule coverage")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected_occurrences):
        raise ValueError("selected manifest does not cover every prepared VTA occurrence")
    expected = {item["occurrence"]: item for item in expected_occurrences}
    if len(expected) != len(expected_occurrences):
        raise ValueError("prepared graph contains duplicate occurrence identities")
    actual = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("selected manifest entries must be objects")
        occurrence = entry.get("occurrence")
        if (isinstance(occurrence, bool) or not isinstance(occurrence, int)
                or occurrence not in expected or occurrence in actual):
            raise ValueError(f"selected manifest has an unexpected or duplicate occurrence {occurrence}")
        actual.add(occurrence)
        identity = expected[occurrence]
        for field in ("symbol", "fusion_sha256", "workload_sha256"):
            if entry.get(field) != identity.get(field):
                raise ValueError(f"selected manifest {field} mismatch at occurrence {occurrence}")
        config = entry.get("config", entry.get("conv_config"))
        if not isinstance(config, dict):
            raise ValueError(f"selected manifest config is missing at occurrence {occurrence}")
        config_hash = sha256_json(config)
        if entry.get("config_sha256", config_hash) != config_hash:
            raise ValueError(f"selected manifest config hash mismatch at occurrence {occurrence}")
    if actual != set(expected):
        raise ValueError("selected manifest occurrence coverage is incomplete")
    return entries


def validate_seed_alignment_report(path, *, model_id, model_sha256, geometry_path,
                                   geometry_sha256, expected_occurrences):
    """Validate a complete deployment gate and the exact seed manifest it measured."""
    try:
        path = Path(path).expanduser().resolve(strict=True)
    except (OSError, TypeError) as error:
        raise ValueError("alignment report path does not name an existing file") from error
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"alignment report is not readable JSON: {path}") from error
    protocol = {
        **PROTOCOL,
        "operator_counted_invocations": 1,
        "full_model_counted_invocations": 1,
    }
    if (not isinstance(report, dict) or report.get("schema_version") != REPORT_SCHEMA_VERSION
            or report.get("artifact_kind") != REPORT_KIND or report.get("phase") != "seed"
            or report.get("status") != "passed" or report.get("completion_label") != "SEED_ALIGNMENT"
            or report.get("model_id") != model_id or report.get("model") != model_id
            or report.get("model_sha256") != model_sha256 or report.get("backend") != "tsim"
            or report.get("sample_count") != 1 or report.get("measurement_protocol") != protocol):
        raise ValueError("alignment report is not a passing one-sample seed report for this model")
    geometry_path = Path(geometry_path).expanduser().resolve(strict=True)
    if (not isinstance(report.get("geometry"), dict)
            or report["geometry"].get("path") != str(geometry_path)
            or report["geometry"].get("sha256") != geometry_sha256
            or sha256_file(geometry_path) != geometry_sha256):
        raise ValueError("alignment report geometry identity does not match the active configuration")

    try:
        manifest_path = Path(report.get("selected_manifest", "")).expanduser().resolve(strict=True)
    except (OSError, TypeError) as error:
        raise ValueError("alignment report seed manifest path does not exist") from error
    if (report.get("selected_manifest_sha256") != sha256_file(manifest_path)):
        raise ValueError("alignment report seed manifest hash does not match the referenced file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("alignment report seed manifest is not readable JSON") from error
    if (not isinstance(manifest, dict) or manifest.get("model") != model_id
            or manifest.get("model_id") != model_id or manifest.get("phase") != "seed"
            or manifest.get("status") != "complete" or manifest.get("bounded") is not False
            or manifest.get("completion_label") != "SEED_COMPLETE"
            or manifest.get("model_sha256") != model_sha256
            or manifest.get("geometry_sha256") != geometry_sha256
            or manifest.get("measurement_protocol") != {
                "name": PROTOCOL["name"], "version": PROTOCOL["version"],
                "warmup_excluded": True, "counted_invocations": 1,
            }):
        raise ValueError("alignment report does not reference a complete, matching seed manifest")
    entries = validate_selected_configs(manifest, expected_occurrences, phase="seed")
    expected_indices = list(range(len(expected_occurrences)))
    if (manifest.get("workload_count") != len(expected_indices)
            or manifest.get("selected_workload_indices") != expected_indices
            or manifest.get("failures") != []):
        raise ValueError("seed manifest does not cover every prepared occurrence")
    seed_by_occurrence = {entry["occurrence"]: entry for entry in entries}
    expected = {item["occurrence"]: item for item in expected_occurrences}
    for occurrence, seed in seed_by_occurrence.items():
        for field in ("result_json", "native_record"):
            name = seed.get(field)
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"seed manifest {field} must be a local filename")
            artifact_path = manifest_path.parent / name
            if not artifact_path.is_file():
                raise ValueError(f"seed manifest {field} is missing at occurrence {occurrence}")
        native_path = manifest_path.parent / seed["native_record"]
        native_hash = seed.get("native_record_sha256")
        if (not isinstance(native_hash, str) or sha256_file(native_path) != native_hash):
            raise ValueError(f"seed native record hash mismatch at occurrence {occurrence}")
        try:
            result = json.loads((manifest_path.parent / seed["result_json"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"seed result JSON is unreadable at occurrence {occurrence}") from error
        expected_identity = expected[occurrence]
        if (not isinstance(result, dict)
                or result.get("model") != model_id
                or result.get("model_sha256") != model_sha256
                or result.get("geometry_sha256") != geometry_sha256
                or result.get("workload_index") != occurrence
                or result.get("symbol") != expected_identity["symbol"]
                or result.get("fusion_sha256") != expected_identity["fusion_sha256"]
                or result.get("workload_sha256") != expected_identity["workload_sha256"]
                or result.get("tsim_cycles") != seed.get("tsim_cycles")
                or result.get("best_native_record") != seed.get("native_record")
                or result.get("best_native_record_sha256") != native_hash
                or sha256_json(result.get("conv_config", result.get("config")))
                != seed.get("config_sha256")):
            raise ValueError(f"seed result identity mismatch at occurrence {occurrence}")
    rows = report.get("occurrences")
    if not isinstance(rows, list) or len(rows) != len(expected_occurrences):
        raise ValueError("alignment report does not cover every prepared occurrence")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("alignment report occurrences must be objects")
        occurrence = row.get("occurrence")
        if isinstance(occurrence, bool) or not isinstance(occurrence, int):
            raise ValueError("alignment report occurrence must be an integer")
        identity = expected.get(occurrence)
        seed = seed_by_occurrence.get(occurrence)
        if identity is None or seed is None or occurrence in seen:
            raise ValueError(f"alignment report has an unexpected or duplicate occurrence {occurrence}")
        for field in ("symbol", "fusion_sha256", "workload_sha256"):
            if row.get(field) != identity.get(field) or seed.get(field) != identity.get(field):
                raise ValueError(f"alignment report {field} mismatch at occurrence {occurrence}")
        for field in ("config_sha256",):
            if row.get(field) != seed.get(field):
                raise ValueError(f"alignment report {field} mismatch at occurrence {occurrence}")
        deployment_cycles = row.get("deployment_cycles")
        autotvm_cycles = row.get("autotvm_cycles")
        if not cycles_within_ten_percent(deployment_cycles, autotvm_cycles):
            raise ValueError(f"alignment report cycle difference exceeds 10% at occurrence {occurrence}")
        if (row.get("passed") is not True or row.get("counted_invocations") != 1
                or seed.get("tsim_cycles") != autotvm_cycles
                or row.get("relative_cycle_difference")
                != abs(deployment_cycles - autotvm_cycles) / autotvm_cycles):
            raise ValueError(f"alignment report gate or seed cycles mismatch at occurrence {occurrence}")
        seen.add(occurrence)
    if seen != set(expected):
        raise ValueError("alignment report occurrence coverage is incomplete")
    return path


def validate_replay_manifest(manifest, *, model_id, model_sha256, geometry_sha256,
                             expected_occurrences):
    """Require a complete, model-bound artifact manifest before replaying records."""
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != REPORT_SCHEMA_VERSION
            or manifest.get("model") != model_id or manifest.get("model_id") != model_id):
        raise ValueError("unsupported or mismatched best manifest")
    phase = manifest.get("phase")
    completion = "SEED_COMPLETE" if phase == "seed" else "FULL_SEARCH"
    if (phase not in ("seed", "selected") or manifest.get("status") != "complete"
            or manifest.get("bounded") is not False
            or manifest.get("completion_label") != completion
            or manifest.get("model_sha256") != model_sha256
            or manifest.get("geometry_sha256") != geometry_sha256
            or manifest.get("measurement_protocol") != {
                "name": PROTOCOL["name"], "version": PROTOCOL["version"],
                "warmup_excluded": True, "counted_invocations": 1,
            }):
        raise ValueError("best manifest is incomplete or has a mismatched model, geometry, or protocol")
    entries = validate_selected_configs(manifest, expected_occurrences, phase=phase)
    indices = list(range(len(expected_occurrences)))
    if (manifest.get("workload_count") != len(indices)
            or manifest.get("selected_workload_indices") != indices
            or manifest.get("failures") != []):
        raise ValueError("best manifest does not cover every prepared occurrence")
    for entry in entries:
        config = entry.get("config", entry.get("conv_config"))
        if entry.get("config_sha256") != sha256_json(config):
            raise ValueError(f"best manifest config hash mismatch at occurrence {entry['occurrence']}")
        _cycles = entry.get("tsim_cycles")
        if isinstance(_cycles, bool) or not isinstance(_cycles, int) or _cycles <= 0:
            raise ValueError(f"best manifest TSIM cycles are invalid at occurrence {entry['occurrence']}")
    return entries


def lower_selected_configs(manifest, expected_occurrences, lowerer, *, phase):
    """Lower each exact exported config against its matching prepared symbol."""
    if not callable(lowerer):
        raise ValueError("selected config lowerer must be callable")
    entries = validate_selected_configs(manifest, expected_occurrences, phase=phase)
    identity_by_occurrence = {item["occurrence"]: item for item in expected_occurrences}
    lowered = {}
    for entry in entries:
        occurrence = entry["occurrence"]
        config = entry.get("config", entry.get("conv_config"))
        try:
            schedule = lowerer(identity_by_occurrence[occurrence], config)
        except Exception as error:
            raise ValueError(
                f"selected config lowering failed for {entry['symbol']} occurrence {occurrence}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if schedule is None or getattr(schedule, "schedule", schedule) is None:
            raise ValueError(f"selected config lowering returned no schedule at occurrence {occurrence}")
        lowered[occurrence] = schedule
    return lowered


def assert_counter_agreement(ordinary_stats, debug_stats):
    """Require exact TSIM counter equality for the same graph, sample and state."""
    if not isinstance(ordinary_stats, dict) or not isinstance(debug_stats, dict):
        raise ValueError("ordinary and debug TSIM stats must be mappings")
    if ordinary_stats != debug_stats:
        raise ValueError(
            "ordinary and debug full-run TSIM counters disagree: "
            f"ordinary={ordinary_stats}, debug={debug_stats}"
        )
    cycles = ordinary_stats.get("cycle_count")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("full-run TSIM cycle_count must be a positive integer")
    return cycles


def resolve_graph_nodes(graph_json, expected_occurrences):
    """Map every expected VTA symbol to one reloaded graph-executor node."""
    graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("graph executor JSON must contain a nodes list")
    nodes = graph["nodes"]
    mapping = {}
    for expected in expected_occurrences:
        occurrence, symbol = expected.get("occurrence"), expected.get("symbol")
        matches = []
        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or node.get("op") != "tvm_op":
                continue
            attrs = node.get("attrs", {})
            if attrs.get("func_name") == symbol or node.get("name") == symbol:
                matches.append(index)
        if len(matches) != 1:
            raise ValueError(
                f"graph symbol {symbol!r} maps to {len(matches)} nodes; expected exactly one"
            )
        mapping[occurrence] = matches[0]
    if len(mapping) != len(expected_occurrences):
        raise ValueError("graph node occurrence identities are duplicated")
    return mapping


def profile_graph_resident_nodes(graph_json, expected_occurrences, debug_graph, session, simulator):
    """Execute each VTA graph node once on resident tensors with a cleared TSIM window."""
    mapping = resolve_graph_nodes(graph_json, expected_occurrences)
    graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
    rows = []
    for identity in expected_occurrences:
        occurrence = identity["occurrence"]
        node_index = mapping[occurrence]
        session.clear_and_validate(simulator)
        debug_graph._execute_node(node_index)
        stats = session.read_stats(simulator=simulator)
        session.validate_activity(stats)
        cycles = stats.get("cycle_count") if isinstance(stats, dict) else None
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError(f"graph node occurrence {occurrence} has invalid cycle_count")
        rows.append({
            "occurrence": occurrence,
            "symbol": identity["symbol"],
            "graph_node_index": node_index,
            "graph_node_name": graph["nodes"][node_index].get("name"),
            "graph_func_name": graph["nodes"][node_index].get("attrs", {}).get("func_name"),
            "deployment_cycles": cycles,
            "counted_invocations": 1,
        })
    return rows


def validate_one_sample(sample_id, sample_sha256, reference_output, deployed_output, checker):
    """Run the existing model reference checker exactly once for a named sample."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("deployment sample_id must be a non-empty string")
    if (not isinstance(sample_sha256, str) or len(sample_sha256) != 64
            or any(char not in "0123456789abcdef" for char in sample_sha256)):
        raise ValueError("deployment sample_sha256 must be a lowercase SHA-256 digest")
    if not callable(checker):
        raise ValueError("deployment reference checker must be callable")
    checker(reference_output, deployed_output)
    return {"sample_id": sample_id, "sha256": sample_sha256, "sample_count": 1}


def validate_occurrence_rows(rows, expected):
    """Check exact identity coverage and the inclusive per-operator 10% gate."""
    if not isinstance(rows, list) or not isinstance(expected, list) or not expected:
        raise ValueError("deployment rows and expected occurrences must be non-empty lists")
    expected_by_id = {item.get("occurrence"): item for item in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError("expected occurrence identities are duplicated")
    seen = set()
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("deployment occurrence rows must be objects")
        occurrence = row.get("occurrence")
        if occurrence not in expected_by_id or occurrence in seen:
            raise ValueError(f"deployment has unexpected or duplicate occurrence {occurrence}")
        seen.add(occurrence)
        identity = expected_by_id[occurrence]
        for field in ("symbol", "fusion_sha256", "workload_sha256", "config_sha256"):
            if row.get(field) != identity.get(field):
                raise ValueError(f"deployment {field} mismatch at occurrence {occurrence}")
        if row.get("counted_invocations") != 1:
            raise ValueError(f"occurrence {occurrence} must contain one counted node invocation")
        deployment_cycles = row.get("deployment_cycles")
        autotvm_cycles = row.get("autotvm_cycles")
        if not cycles_within_ten_percent(deployment_cycles, autotvm_cycles):
            raise ValueError(f"deployment cycle difference exceeds 10% at occurrence {occurrence}")
        difference = abs(deployment_cycles - autotvm_cycles) / autotvm_cycles
        normalized.append({
            **row,
            "relative_cycle_difference": difference,
            "passed": True,
        })
    if seen != set(expected_by_id):
        raise ValueError("deployment occurrence coverage is incomplete")
    return normalized


def build_deployment_report(*, phase, model_id, model_sha256, geometry_path,
                            sample, selected_manifest_path, full_model,
                            occurrences, expected_occurrences, peak_macs_per_cycle,
                            completion_label=None):
    """Build a calculator-compatible report after all one-sample gates pass."""
    if phase not in ("seed", "selected"):
        raise ValueError("deployment phase must be seed or selected")
    if completion_label is None:
        completion_label = "SEED_ALIGNMENT" if phase == "seed" else "OPTIMAL_DEPLOYMENT"
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("deployment model_id is required")
    for label, value in (("model_sha256", model_sha256), ("sample.sha256", sample.get("sha256"))):
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    geometry_path = Path(geometry_path).expanduser().resolve(strict=True)
    selected_manifest_path = Path(selected_manifest_path).expanduser().resolve(strict=True)
    selected_manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    entries = validate_selected_configs(selected_manifest, expected_occurrences, phase=phase)
    manifest_identity = selected_manifest.get("identity", {})
    if selected_manifest.get("model_sha256", manifest_identity.get("model_sha256")) != model_sha256:
        raise ValueError("selected manifest model identity does not match deployment")
    geometry_sha256 = sha256_file(geometry_path)
    if selected_manifest.get("geometry_sha256", manifest_identity.get("geometry_sha256")) != geometry_sha256:
        raise ValueError("selected manifest geometry identity does not match deployment")
    with geometry_path.open(encoding="utf-8") as stream:
        geometry = json.load(stream)
    if not isinstance(geometry, dict):
        raise ValueError("VTA geometry must be a JSON object")
    exponents = [geometry.get("LOG_BATCH"), geometry.get("LOG_BLOCK")]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in exponents):
        raise ValueError("VTA geometry needs non-negative integer LOG_BATCH and LOG_BLOCK")
    expected_peak = 2 ** exponents[0] * 2 ** exponents[1] * 2 ** exponents[1]
    if (isinstance(peak_macs_per_cycle, bool) or not isinstance(peak_macs_per_cycle, int)
            or peak_macs_per_cycle != expected_peak):
        raise ValueError("peak MACs/cycle does not match the selected VTA geometry")
    if sample.get("sample_count") != 1:
        raise ValueError("deployment report requires exactly one validated sample")
    for key in ("baseline_cycles", "tuned_cycles"):
        value = full_model.get(key) if isinstance(full_model, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"full-model {key} must be a positive integer")
    normalized = validate_occurrence_rows(occurrences, expected_occurrences)
    entry_by_occurrence = {entry["occurrence"]: entry for entry in entries}
    mac_rows = []
    for row in normalized:
        entry = entry_by_occurrence[row["occurrence"]]
        config = entry.get("config", entry.get("conv_config"))
        config_hash = sha256_json(config)
        if row["config_sha256"] != config_hash:
            raise ValueError(f"deployed config identity mismatch at occurrence {row['occurrence']}")
        macs = entry.get("logical_macs_per_invocation", entry.get("mac_count"))
        if isinstance(macs, bool) or not isinstance(macs, int) or macs <= 0:
            raise ValueError(f"selected manifest MAC derivation is invalid at occurrence {row['occurrence']}")
        autotvm_cycles = row["autotvm_cycles"]
        mac_rows.append({
            "occurrence": row["occurrence"],
            "symbol": row["symbol"],
            "fusion_sha256": row["fusion_sha256"],
            "workload_sha256": row["workload_sha256"],
            "config_sha256": config_hash,
            "logical_macs_per_invocation": macs,
            "counted_invocations": 1,
            "deployment_cycles": row["deployment_cycles"],
            "autotvm_cycles": autotvm_cycles,
            "relative_cycle_difference": row["relative_cycle_difference"],
            "passed": True,
        })
    manifest_hash = sha256_file(selected_manifest_path)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_KIND,
        "phase": phase,
        "status": "passed",
        "model_id": model_id,
        "model": model_id,
        "model_sha256": model_sha256,
        "backend": "tsim",
        "geometry": {
            "path": str(geometry_path),
            "sha256": geometry_sha256,
            "peak_macs_per_cycle": expected_peak,
        },
        "sample_count": sample["sample_count"],
        "sample": {"sample_id": sample["sample_id"], "sha256": sample["sha256"]},
        "measurement_protocol": {
            **PROTOCOL,
            "operator_counted_invocations": 1,
            "full_model_counted_invocations": 1,
        },
        "scope": {
            "full_model_cycles": "uninstrumented_complete_deployment",
            "host_operations": "excluded_from_vta_mac_totals",
        },
        "full_model": {
            "invocation_count": 1,
            "baseline_cycles": full_model["baseline_cycles"],
            "tuned_cycles": full_model["tuned_cycles"],
        },
        "completion_label": completion_label,
        "occurrence_base": 0,
        "selected_manifest": str(selected_manifest_path),
        "selected_manifest_sha256": manifest_hash,
        "occurrences": mac_rows,
    }


def write_failure_report(path, *, phase, model_id, stage, error, details=None):
    """Atomically preserve failure diagnostics without creating a passing report."""
    if not isinstance(error, BaseException):
        error_message = str(error)
    else:
        error_message = f"{type(error).__name__}: {error}"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": "vta_deployment_failure_v1",
        "phase": phase,
        "model_id": model_id,
        "stage": stage,
        "status": "failed",
        "error": error_message,
        "details": details or {},
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
