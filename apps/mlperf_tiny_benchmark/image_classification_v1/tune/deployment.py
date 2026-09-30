#!/usr/bin/env python3
"""Apply selected IC V1 schedules and export measured TSIM deployment evidence."""

import argparse
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path


TUNE_DIR = Path(__file__).resolve().parent
APP_ROOT = TUNE_DIR.parent
REPO_ROOT = APP_ROOT.parents[3]
PROTOCOL = {"name": "tsim_single_call", "version": 1, "warmup_excluded": True}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_cycles(deployment_cycles, autotvm_cycles):
    """Validate one positive cycle pair and enforce the approved 10% bound."""
    for name, value in (("deployment_cycles", deployment_cycles), ("autotvm_cycles", autotvm_cycles)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    difference = abs(deployment_cycles - autotvm_cycles) / autotvm_cycles
    result = {
        "deployment_cycles": deployment_cycles,
        "autotvm_cycles": autotvm_cycles,
        "relative_cycle_difference": difference,
    }
    if difference > 0.10:
        raise ValueError(
            "deployment versus selected AutoTVM cycles exceeds 10%: "
            f"deployment={deployment_cycles}, autotvm={autotvm_cycles}, relative={difference:.6%}"
        )
    return result


def build_occurrence_config_map(identities, entries):
    """Map every deployed occurrence to its own config without key-based overwrite."""
    by_index = {entry.get("workload_index"): entry for entry in entries}
    if len(by_index) != len(entries):
        raise ValueError("selected workload indices must be unique")
    selected = {}
    for index, identity in enumerate(identities):
        entry = by_index.get(index)
        if entry is None:
            raise ValueError(f"selected schedule is missing deployed occurrence {index}")
        if entry.get("occurrence") != identity.get("occurrence") or entry.get("symbol") != identity.get("symbol"):
            raise ValueError(f"selected schedule identity mismatch at occurrence {index}")
        selected[index] = entry["config"]
    if len(by_index) != len(identities):
        raise ValueError("best manifest must cover every deployed fusion occurrence exactly once")
    return selected


def _lower_selected_relay_module(
    module, config_by_symbol, tvm_module, vta_module, autotvm_module, compiler
):
    """Run VTA's normal Relay-to-TIR boundary with one AutoTVM config per symbol."""
    transform = vta_module.relay.transform
    compiler_config = transform.VTACompilerConfig.from_env(vta_module.get_env())
    for function in transform._collect_vta_relay_functions(module):
        transform._validate_vta_function(function, compiler_config)
    if not transform._collect_vta_relay_functions(module):
        raise ValueError("prepared module has no VTA functions to lower")
    outlined = tvm_module.relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols("vta")(module)
    global_functions = transform._global_vta_relay_functions(outlined)
    global_handles = {function.handle.value for _, function in global_functions}
    nested = [
        function for function in transform._collect_vta_relay_functions(outlined)
        if function.handle.value not in global_handles
    ]
    if nested:
        raise ValueError("all nested Compiler='vta' functions must be directly outlineable")
    symbols = {function.attrs.get_str("global_symbol") for _, function in global_functions}
    if symbols != set(config_by_symbol):
        raise ValueError("selected schedules do not cover the outlined VTA function symbols")
    for global_var, function in global_functions:
        symbol = function.attrs.get_str("global_symbol")
        # The standard VTA compiler lowers all functions in one callback. Here
        # each function gets its validated occurrence's config and an empty TE
        # cache, so equal shape keys cannot select another occurrence's config.
        compiler.clear()
        with autotvm_module.task.ApplyConfig(config_by_symbol[symbol]):
            primfunc = transform.lower_vta_function(function, compiler_config)
        outlined.update_func(global_var, primfunc)
        compiler.clear()
    return outlined


@contextmanager
def _selected_occurrence_lowering(tvm_module, vta_module, autotvm_module, compiler, selected_by_symbol):
    """Temporarily override the VTA target hook for occurrence-scoped dispatch."""
    name = "vta.relay._relay_to_tir"
    previous = tvm_module.get_global_func(name, allow_missing=True)
    if previous is None:
        raise RuntimeError("VTA Relay-to-TIR callback is unavailable")

    def lower(module):
        return _lower_selected_relay_module(
            module, selected_by_symbol, tvm_module, vta_module, autotvm_module, compiler
        )

    tvm_module.register_func(name, lower, override=True)
    try:
        yield
    finally:
        tvm_module.register_func(name, previous, override=True)


def _positive_integer(obj, field, location):
    value = obj.get(field) if isinstance(obj, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location}.{field} must be a positive integer")
    return value


def validate_deployment_report(report):
    """Validate the versioned, model-independent real deployment contract."""
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("unsupported deployment report schema_version")
    if report.get("artifact_kind") != "vta_deployment_profile_v1":
        raise ValueError("unsupported deployment report artifact_kind")
    for key in ("model_id", "model_sha256"):
        if not isinstance(report.get(key), str) or not report[key]:
            raise ValueError(f"deployment report {key} must be a non-empty string")
    geometry = report.get("geometry")
    if not isinstance(geometry, dict) or not isinstance(geometry.get("sha256"), str):
        raise ValueError("deployment report geometry.sha256 is required")
    _positive_integer(geometry, "peak_macs_per_cycle", "geometry")
    if report.get("backend") != "tsim":
        raise ValueError("deployment report backend must be tsim")
    protocol = report.get("measurement_protocol")
    if not isinstance(protocol, dict) or protocol.get("name") != PROTOCOL["name"] \
            or protocol.get("version") != PROTOCOL["version"] \
            or protocol.get("warmup_excluded") is not True:
        raise ValueError("deployment report requires the TSIM single-call v1 protocol")
    operator_invocations = _positive_integer(protocol, "operator_counted_invocations", "measurement_protocol")
    full_invocations = _positive_integer(protocol, "full_model_counted_invocations", "measurement_protocol")
    full_model = report.get("full_model")
    if not isinstance(full_model, dict):
        raise ValueError("deployment report full_model must be an object")
    if _positive_integer(full_model, "invocation_count", "full_model") != full_invocations:
        raise ValueError("invocation counts must align between protocol and full_model")
    for field in ("baseline_cycles", "tuned_cycles"):
        _positive_integer(full_model, field, "full_model")
    occurrences = report.get("occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError("deployment report occurrences must be a non-empty list")
    seen_occurrences = set()
    for position, item in enumerate(occurrences):
        if not isinstance(item, dict):
            raise ValueError(f"occurrences[{position}] must be an object")
        occurrence = _positive_integer(item, "occurrence", f"occurrences[{position}]") - 1
        symbol = item.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"occurrences[{position}].symbol is required")
        if occurrence in seen_occurrences:
            raise ValueError("occurrence identity must be unique")
        seen_occurrences.add(occurrence)
        for field in ("fusion_sha256", "workload_sha256", "config_sha256"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ValueError(f"occurrences[{position}].{field} is required")
        _positive_integer(item, "logical_macs_per_invocation", f"occurrences[{position}]")
        _positive_integer(item, "deployment_cycles", f"occurrences[{position}]")
        _positive_integer(item, "autotvm_cycles", f"occurrences[{position}]")
        if item.get("counted_invocations") != operator_invocations:
            raise ValueError("operator invocation counts must align with measurement protocol")
        expected = abs(item["deployment_cycles"] - item["autotvm_cycles"]) / item["autotvm_cycles"]
        if expected > 0.10:
            raise ValueError(f"occurrence {occurrence} comparison exceeds 10%")
        if abs(float(item.get("relative_cycle_difference", -1)) - expected) > 1e-12:
            raise ValueError(f"occurrence {occurrence} relative cycle difference is inconsistent")
    if report.get("scope", {}).get("full_model_cycles") != "uninstrumented_complete_deployment":
        raise ValueError("full_model_cycles must come from an uninstrumented complete deployment")
    if report.get("scope", {}).get("host_operations") != "excluded_from_vta_mac_totals":
        raise ValueError("deployment report must identify host operations as excluded")
    return report


def example_valid_report():
    """Small synthetic report used by focused contract tests."""
    return {
        "schema_version": 1,
        "artifact_kind": "vta_deployment_profile_v1",
        "model_id": "arbitrary-model",
        "model_sha256": "model-hash",
        "geometry": {"sha256": "geometry-hash", "peak_macs_per_cycle": 64},
        "backend": "tsim",
        "measurement_protocol": {
            **PROTOCOL, "operator_counted_invocations": 1, "full_model_counted_invocations": 1
        },
        "full_model": {"invocation_count": 1, "baseline_cycles": 500, "tuned_cycles": 400},
        "occurrences": [{
            "occurrence": 1, "symbol": "fusion_a", "fusion_sha256": "fusion-hash",
            "workload_sha256": "workload-hash", "config_sha256": "config-hash",
            "logical_macs_per_invocation": 1000, "counted_invocations": 1,
            "deployment_cycles": 100, "autotvm_cycles": 100,
            "relative_cycle_difference": 0.0,
        }],
        "scope": {
            "full_model_cycles": "uninstrumented_complete_deployment",
            "host_operations": "excluded_from_vta_mac_totals",
        },
    }


def load_best_manifest(path, prepared, identities):
    """Validate self-contained best records and return occurrence-specific entries."""
    import importlib.util

    legacy_path = APP_ROOT / "tune.py"
    spec = importlib.util.spec_from_file_location("ic_v1_deployment_legacy", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    manifest_path = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("model") != "image_classification_v1":
        raise ValueError("unsupported or mismatched selected-schedule manifest")
    if manifest.get("model_sha256") != prepared.imported.model_sha256:
        raise ValueError("selected-schedule model hash does not match deployed model")
    geometry_path = Path(manifest.get("geometry_path", "")).expanduser().resolve(strict=True)
    if manifest.get("geometry_sha256") != _sha256(geometry_path):
        raise ValueError("selected-schedule geometry hash does not match deployed geometry")
    if manifest.get("measurement_protocol") != legacy.shared.TSIM_MEASUREMENT_PROTOCOL:
        raise ValueError("selected-schedule TSIM protocol does not match deployment")
    selected = manifest.get("entries")
    if not isinstance(selected, list) or len(selected) != len(identities):
        raise ValueError("selected-schedule manifest must cover every deployed occurrence")
    indexed = {entry.get("workload_index"): entry for entry in selected}
    if len(indexed) != len(selected) or set(indexed) != set(range(len(identities))):
        raise ValueError("selected-schedule manifest workload indices are incomplete or duplicated")
    identities_json = [json.loads(identity.canonical_json()) for identity in identities]
    result_entries = []
    from tvm import autotvm

    for index, identity in enumerate(identities):
        entry = indexed[index]
        result_path = manifest_path.parent / entry["result_json"]
        replay = legacy.replay_result(result_path, expected_workload_index=index)
        result = replay["result"]
        if (entry.get("occurrence") != identity.occurrence or entry.get("symbol") != identity.symbol
                or entry.get("fusion_sha256") != identity.sha256
                or result.get("fusion_sha256") != identity.sha256
                or entry.get("tsim_cycles") != result.get("tsim_cycles")):
            raise ValueError(f"selected schedule identity mismatch at occurrence {index}")
        record_path = manifest_path.parent / result["best_native_record"]
        records = list(autotvm.record.load_from_file(str(record_path)))
        if len(records) != 1:
            raise ValueError(f"selected native record must contain one entry for occurrence {index}")
        measure_input, measure_result = records[0]
        if measure_result.error_no != 0 or int(measure_result.costs[0]) != result["tsim_cycles"]:
            raise ValueError(f"selected native record cycles mismatch at occurrence {index}")
        result_entries.append({
            "workload_index": index,
            "result": result,
            "config": measure_input.config,
            "config_json": measure_input.config.to_json_dict(),
            "identity": identities_json[index],
            "record_path": record_path,
        })
    return manifest, result_entries


def _import_runtime():
    spec = importlib.util.spec_from_file_location("ic_v1_deployment_runtime", APP_ROOT / "runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(runtime)
    finally:
        sys.path.pop(0)
    return runtime


def execute_deployment(manifest_path, output_path, build_dir):
    """Build baseline and selected variants, profile real occurrences, and export evidence."""
    import tvm
    import vta
    from tvm import autotvm
    from tvm.contrib.debugger import debug_executor

    runtime = _import_runtime()
    from fused_tasks import extract_fused_identities
    prepared = runtime.prepare_model(runtime.MODEL_PATH)
    identities = extract_fused_identities(prepared)
    best_manifest, selected = load_best_manifest(manifest_path, prepared, identities)
    occurrence_configs = build_occurrence_config_map(
        [{"occurrence": identity.occurrence, "symbol": identity.symbol} for identity in identities],
        [{"workload_index": entry["workload_index"], "occurrence": entry["result"]["occurrence"],
          "symbol": entry["result"]["symbol"], "config": entry["config"]}
         for entry in selected],
    )

    output_path = Path(output_path).expanduser().resolve()
    build_dir = Path(build_dir).expanduser().resolve()
    samples = runtime.committed_sample_paths()
    inputs = tuple((path, runtime.load_sample(path)) for path in samples)
    ref_factory = tvm.relay.build(prepared.reference_module, target="llvm")

    def build_variant(name, selected_context=False):
        compiler = tvm.relay.backend.te_compiler.get()
        compiler.clear()
        try:
            if selected_context:
                selected_by_symbol = {
                    identity.symbol: occurrence_configs[index]
                    for index, identity in enumerate(identities)
                }
                with _selected_occurrence_lowering(
                    tvm, vta, autotvm, compiler, selected_by_symbol
                ):
                    with vta.build_config():
                        factory = tvm.relay.build(
                            prepared.mixed_module, target=runtime._mixed_target()
                        )
            else:
                with vta.build_config():
                    factory = tvm.relay.build(
                        prepared.mixed_module, target=runtime._mixed_target()
                    )
        finally:
            compiler.clear()
        bundle = runtime.export_graph_bundle if hasattr(runtime, "export_graph_bundle") else None
        from graph_artifacts import export_graph_bundle
        exported = export_graph_bundle(
            factory, build_dir / name, "mixed", artifact_name=f"c3-{name}",
            artifact_role="mixed", model_sha256=prepared.imported.model_sha256,
            host_codegen="llvm", simulator="tsim", expected_vta_symbols=prepared.routing.symbols,
        )
        return exported

    # Build actual reference and both deployed mixed models under the same graph contract.
    reference = runtime._reload_artifact(
        ref_factory, build_dir / "baseline" / "reference", "reference", "llvm", "tsim",
        model_sha256=prepared.imported.model_sha256,
    ) if hasattr(runtime, "_reload_artifact") else None
    if reference is None:
        from graph_artifacts import export_graph_bundle
        reference = export_graph_bundle(
            ref_factory, build_dir / "baseline" / "reference", "reference",
            artifact_name="c3-reference", artifact_role="reference",
            model_sha256=prepared.imported.model_sha256, host_codegen="llvm", simulator="tsim",
            forbidden_vta_symbols=prepared.routing.symbols,
        )
    baseline = build_variant("baseline", False)
    tuned = build_variant("tuned", True)
    simulator = runtime._simulator_session("tsim").load()
    session = runtime._simulator_session("tsim")

    def make_executor(bundle, debug=False, device=None):
        device = device or tvm.ext_dev(0)
        creator = debug_executor.create if debug else tvm.contrib.graph_executor.create
        return creator(bundle.graph_json, bundle.module, device, **({"dump_root": str(build_dir / "debug") } if debug else {}))

    def run_model(bundle):
        graph = make_executor(bundle)
        graph.load_params(bundle.params)
        outputs = []
        session.clear_and_validate(simulator)
        for path, data in inputs:
            graph.set_input(runtime.INPUT_NAME, data)
            graph.run()
            outputs.append((path, graph.get_output(0).numpy()))
        cycles = session.read_stats(simulator=simulator)
        session.validate_activity(cycles)
        return outputs, int(cycles["cycle_count"])

    refs = []
    reference_graph = make_executor(reference, device=tvm.cpu(0))
    reference_graph.load_params(reference.params)
    for path, data in inputs:
        reference_graph.set_input(runtime.INPUT_NAME, data)
        reference_graph.run()
        refs.append((path, reference_graph.get_output(0).numpy()))

    baseline_outputs, baseline_cycles = run_model(baseline)
    tuned_outputs, tuned_cycles = run_model(tuned)
    for expected, baseline_actual, tuned_actual in zip(refs, baseline_outputs, tuned_outputs):
        runtime.compare_outputs(expected[0], expected[1], baseline_actual[1])
        runtime.compare_outputs(expected[0], expected[1], tuned_actual[1])

    # The debug executor executes each real graph node over graph-resident tensors. Its
    # complete-run counter must match ordinary GraphExecutor before node counters are trusted.
    debug_graph = make_executor(tuned, debug=True)
    debug_graph.load_params(tuned.params)
    first_path, first_data = inputs[0]
    debug_graph.set_input(runtime.INPUT_NAME, first_data)
    session.clear_and_validate(simulator)
    debug_graph._run_per_layer()
    debug_full = session.read_stats(simulator=simulator)
    session.validate_activity(debug_full)
    ordinary_graph = make_executor(tuned)
    ordinary_graph.load_params(tuned.params)
    ordinary_graph.set_input(runtime.INPUT_NAME, first_data)
    session.clear_and_validate(simulator)
    ordinary_graph.run()
    ordinary_full = session.read_stats(simulator=simulator)
    session.validate_activity(ordinary_full)
    if debug_full["cycle_count"] != ordinary_full["cycle_count"]:
        raise ValueError(
            "debug profiling changes full-model TSIM cycles: "
            f"debug={debug_full['cycle_count']}, ordinary={ordinary_full['cycle_count']}"
        )

    graph_nodes = debug_graph.debug_datum.get_graph_nodes()
    node_by_symbol = {
        node.get("attrs", {}).get("global_symbol"): index
        for index, node in enumerate(graph_nodes)
        if node.get("attrs", {}).get("global_symbol")
    }
    occurrences = []
    for identity, entry in zip(identities, selected):
        index = node_by_symbol.get(identity.symbol)
        if index is None:
            raise ValueError(f"debug deployment graph omitted VTA symbol {identity.symbol}")
        session.clear_and_validate(simulator)
        debug_graph._execute_node(index)
        stats = session.read_stats(simulator=simulator)
        session.validate_activity(stats)
        cycles = int(stats["cycle_count"])
        compared = compare_cycles(cycles, int(entry["result"]["tsim_cycles"]))
        occurrences.append({
            "occurrence": identity.occurrence + 1,
            "symbol": identity.symbol,
            "fusion_sha256": identity.sha256,
            "workload_sha256": entry["result"]["workload_sha256"],
            "config_sha256": hashlib.sha256(_canonical(entry["config_json"]).encode()).hexdigest(),
            "logical_macs_per_invocation": int(entry["result"]["mac_count"]),
            "counted_invocations": 1,
            **compared,
        })
    report = {
        "schema_version": 1,
        "artifact_kind": "vta_deployment_profile_v1",
        "model_id": "image_classification_v1",
        "model_sha256": prepared.imported.model_sha256,
        "geometry": {
            "path": str(Path(best_manifest["geometry_path"]).resolve()),
            "sha256": best_manifest["geometry_sha256"],
            "peak_macs_per_cycle": (
                2 ** vta.get_env().LOG_BATCH * (2 ** vta.get_env().LOG_BLOCK) ** 2
            ),
        },
        "backend": "tsim",
        "measurement_protocol": {
            **PROTOCOL,
            "operator_counted_invocations": 1,
            "full_model_counted_invocations": len(samples),
            "warmup_invocations_excluded": True,
        },
        "full_model": {
            "invocation_count": len(samples),
            "baseline_cycles": baseline_cycles,
            "tuned_cycles": tuned_cycles,
            "operator_cycle_sum_per_invocation": sum(
                item["deployment_cycles"] for item in occurrences
            ),
            "tuned_residual_cycles_per_invocation": (
                int(ordinary_full["cycle_count"])
                - sum(item["deployment_cycles"] for item in occurrences)
            ),
            "debug_profile_full_model_cycles": int(debug_full["cycle_count"]),
            "ordinary_full_model_single_sample_cycles": int(ordinary_full["cycle_count"]),
        },
        "occurrences": occurrences,
        "scope": {
            "operator_cycles": "one selected VTA graph node invocation using real graph inputs",
            "full_model_cycles": "uninstrumented_complete_deployment",
            "host_operations": "excluded_from_vta_mac_totals",
            "correctness_samples": [path.name for path in samples],
            "profiling_equivalence": "debug and ordinary full-model single-sample cycle counts match exactly",
        },
        "selected_manifest": str(Path(manifest_path).resolve()),
        "selected_manifest_sha256": _sha256(manifest_path),
        "completion_label": (
            "BOUNDED_C3_PROFILE_INCOMPLETE"
            if best_manifest.get("bounded") else "FULL_SEARCH_PROFILE"
        ),
    }
    validate_deployment_report(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=TUNE_DIR / "deployment.json")
    parser.add_argument("--build-dir", type=Path, default=APP_ROOT / "build" / "deployment-validation")
    args = parser.parse_args(argv)
    try:
        report = execute_deployment(args.best_manifest, args.output, args.build_dir)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(f"Deployment report: {args.output}")
    print(f"Occurrences validated: {len(report['occurrences'])}")
    print(f"Baseline full-model cycles ({report['full_model']['invocation_count']} samples): {report['full_model']['baseline_cycles']}")
    print(f"Tuned full-model cycles ({report['full_model']['invocation_count']} samples): {report['full_model']['tuned_cycles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
