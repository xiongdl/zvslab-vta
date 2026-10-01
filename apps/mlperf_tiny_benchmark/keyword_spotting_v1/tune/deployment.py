#!/usr/bin/env python3
"""Apply KWS V1 seed or selected schedules in one real TSIM deployment."""

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


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_runtime():
    app_path = str(APP_ROOT)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    for name in ("runtime", "model_pipeline", "graph_artifacts"):
        sys.modules.pop(name, None)
    return _load_module("ad_v1_deployment_runtime", APP_ROOT / "runtime.py")


def _load_fused_tasks():
    path = APP_ROOT.parent / "fused_tasks.py"
    for name in ("fused_tasks", "mlperf_tiny_fused_tasks"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() == path.resolve():
            sys.modules["fused_tasks"] = module
            return module
    module = _load_module("fused_tasks", path)
    return module


def _load_legacy():
    return _load_module("ad_v1_deployment_legacy", APP_ROOT / "tune.py")


def _canonical_config(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def compare_cycles(deployment_cycles, autotvm_cycles):
    from deployment_evidence import cycles_within_ten_percent

    if not cycles_within_ten_percent(deployment_cycles, autotvm_cycles):
        raise ValueError(
            "deployment versus selected AutoTVM cycles exceeds 10%: "
            f"deployment={deployment_cycles}, autotvm={autotvm_cycles}"
        )
    return {
        "deployment_cycles": deployment_cycles,
        "autotvm_cycles": autotvm_cycles,
        "relative_cycle_difference": abs(deployment_cycles - autotvm_cycles) / autotvm_cycles,
        "passed": True,
    }


def config_entries_by_symbol(identities, entries):
    """Bind configs to exact KWS symbols and occurrences without workload-key aliasing."""
    if not isinstance(entries, list) or len(entries) != len(identities):
        raise ValueError("selected schedule must cover every KWS VTA occurrence")
    by_occurrence = {entry.get("occurrence"): entry for entry in entries}
    if len(by_occurrence) != len(entries) or set(by_occurrence) != set(range(len(identities))):
        raise ValueError("selected schedule occurrence coverage is invalid")
    configs = {}
    for identity in identities:
        entry = by_occurrence[identity.occurrence]
        if entry.get("symbol") != identity.symbol or entry.get("fusion_sha256") != identity.sha256:
            raise ValueError(f"selected schedule identity mismatch at occurrence {identity.occurrence}")
        config = entry.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"selected config is missing at occurrence {identity.occurrence}")
        config_sha = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if entry.get("config_sha256") != config_sha:
            raise ValueError(f"selected config hash mismatch at occurrence {identity.occurrence}")
        configs[identity.symbol] = config
    return configs


def validate_seed_manifest(path, prepared, identities):
    """Validate model-bound manifest and standalone native records for deployment."""
    from deployment_evidence import validate_selected_configs

    path = Path(path).expanduser().resolve(strict=True)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("model") != "keyword_spotting_v1"
            or manifest.get("model_id", "keyword_spotting_v1") != "keyword_spotting_v1"):
        raise ValueError("unsupported or mismatched selected-schedule manifest")
    if manifest.get("status") != "complete":
        raise ValueError("selected-schedule manifest is incomplete")
    if manifest.get("model_sha256") != prepared.imported.model_sha256:
        raise ValueError("selected-schedule model hash does not match deployed model")
    geometry_path = Path(manifest.get("geometry_path", "")).expanduser().resolve(strict=True)
    from deployment_evidence import sha256_file
    geometry_sha = sha256_file(geometry_path)
    if manifest.get("geometry_sha256") != geometry_sha:
        raise ValueError("selected-schedule geometry hash does not match deployed geometry")
    if manifest.get("measurement_protocol") != _load_legacy().shared.TSIM_MEASUREMENT_PROTOCOL:
        raise ValueError("selected-schedule TSIM protocol does not match deployment")
    expected = []
    tuner = _load_legacy()
    _, model_dir, model_filename = tuner.shared.MODEL_PIPELINES["keyword_spotting_v1"]
    if tuner.shared._sha256_file(APP_ROOT / model_dir / model_filename) != prepared.imported.model_sha256:
        raise ValueError("prepared graph model identity does not match committed KWS V1 model")
    for identity in identities:
        expected.append({
            "occurrence": identity.occurrence,
            "symbol": identity.symbol,
            "fusion_sha256": identity.sha256,
            "workload_sha256": None,
        })
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(identities):
        raise ValueError("selected-schedule manifest must cover every deployed occurrence")
    indexed = {entry.get("occurrence"): entry for entry in entries}
    if len(indexed) != len(entries) or set(indexed) != set(range(len(identities))):
        raise ValueError("selected-schedule occurrence coverage is invalid")
    tasks = [ _load_fused_tasks().create_task(identity, __import__("vta").get_env().target)
              for identity in identities ]
    for identity, task in zip(identities, tasks):
        entry = indexed[identity.occurrence]
        expected[identity.occurrence]["workload_sha256"] = tuner.shared._task_workload_id(task)
        for field in ("symbol", "fusion_sha256", "workload_sha256"):
            if entry.get(field) != expected[identity.occurrence][field]:
                raise ValueError(f"selected manifest {field} mismatch at occurrence {identity.occurrence}")
        result_path = path.parent / entry.get("result_json", "")
        replay = tuner.replay_result(result_path, expected_workload_index=identity.occurrence)
        result = replay["result"]
        if result.get("model") != "keyword_spotting_v1" or result.get("fusion_sha256") != identity.sha256:
            raise ValueError(f"selected native record identity mismatch at occurrence {identity.occurrence}")
        if _canonical_config(replay["config"].to_json_dict()) != _canonical_config(entry.get("config")):
            raise ValueError(f"selected config differs from its native record at occurrence {identity.occurrence}")
    validate_selected_configs(manifest, expected, phase=manifest.get("phase", "seed"))
    return manifest, [indexed[index] for index in range(len(identities))], tasks


def _lower_selected_module(module, config_by_symbol, tvm, vta, autotvm, compiler):
    transform = vta.relay.transform
    compiler_config = transform.VTACompilerConfig.from_env(vta.get_env())
    functions = transform._collect_vta_relay_functions(module)
    if not functions:
        raise ValueError("prepared module has no VTA functions to lower")
    for function in functions:
        transform._validate_vta_function(function, compiler_config)
    outlined = tvm.relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols("vta")(module)
    global_functions = transform._global_vta_relay_functions(outlined)
    symbols = {function.attrs.get_str("global_symbol") for _, function in global_functions}
    if symbols != set(config_by_symbol):
        raise ValueError("selected schedules do not cover the deployed VTA symbols")
    for global_var, function in global_functions:
        symbol = function.attrs.get_str("global_symbol")
        compiler.clear()
        with autotvm.task.ApplyConfig(config_by_symbol[symbol]):
            primfunc = transform.lower_vta_function(function, compiler_config)
        outlined.update_func(global_var, primfunc)
        compiler.clear()
    return outlined


@contextmanager
def _selected_lowering(tvm, vta, autotvm, compiler, config_by_symbol):
    name = "vta.relay._relay_to_tir"
    previous = tvm.get_global_func(name, allow_missing=True)
    if previous is None:
        raise RuntimeError("VTA Relay-to-TIR callback is unavailable")
    tvm.register_func(
        name,
        lambda module: _lower_selected_module(module, config_by_symbol, tvm, vta, autotvm, compiler),
        override=True,
    )
    try:
        yield
    finally:
        tvm.register_func(name, previous, override=True)


class _EvidenceProfileSession:
    """Adapt the runtime status-callback API to the shared profile helper."""

    def __init__(self, runtime_session):
        self.runtime_session = runtime_session

    def clear_and_validate(self, simulator):
        return self.runtime_session.clear_and_validate(simulator)

    def read_stats(self, simulator):
        return self.runtime_session.read_stats(simulator.stats)

    def validate_activity(self, stats):
        return self.runtime_session.validate_activity(stats)


def execute_deployment(manifest_path, output_path, build_dir):
    import hashlib
    import tvm
    import vta
    from tvm import autotvm
    from tvm.contrib import graph_executor
    from tvm.contrib.debugger import debug_executor
    from tvm.relay.backend import te_compiler

    from deployment_evidence import (
        assert_counter_agreement, build_deployment_report,
        profile_graph_resident_nodes, validate_one_sample,
    )
    runtime = _import_runtime()
    fused = _load_fused_tasks()
    prepared = runtime.prepare_model(runtime.MODEL_PATH)
    identities = fused.extract_fused_identities(prepared)
    manifest, entries, tasks = validate_seed_manifest(manifest_path, prepared, identities)
    config_by_symbol = config_entries_by_symbol(identities, entries)
    record_by_occurrence = {}
    tuner = _load_legacy()
    for identity, entry in zip(identities, entries):
        replay = tuner.replay_result(Path(manifest_path).resolve().parent / entry["result_json"],
                                     expected_workload_index=identity.occurrence)
        if _canonical_config(replay["config"].to_json_dict()) != _canonical_config(config_by_symbol[identity.symbol]):
            raise ValueError(f"selected native config mismatch at occurrence {identity.occurrence}")
        config_by_symbol[identity.symbol] = replay["config"]
        record_by_occurrence[identity.occurrence] = replay["result"]

    build_dir = Path(build_dir).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    from graph_artifacts import export_graph_bundle
    reference_factory = tvm.relay.build(prepared.reference_module, target="llvm")
    compiler = te_compiler.get()
    compiler.clear()
    try:
        with _selected_lowering(tvm, vta, autotvm, compiler, config_by_symbol):
            with vta.build_config():
                mixed_factory = tvm.relay.build(
                    prepared.mixed_module, target=runtime._mixed_target("llvm")
                )
    finally:
        compiler.clear()
    reference = export_graph_bundle(
        reference_factory, build_dir / "reference", "reference",
        artifact_name="kws-v1-seed-reference", artifact_role="reference",
        model_sha256=prepared.imported.model_sha256, host_codegen="llvm", simulator="tsim",
        forbidden_vta_symbols=prepared.routing.symbols,
    )
    mixed = export_graph_bundle(
        mixed_factory, build_dir / "mixed", "mixed",
        artifact_name="kws-v1-seed-selected", artifact_role="mixed",
        model_sha256=prepared.imported.model_sha256, host_codegen="llvm", simulator="tsim",
        expected_vta_symbols=prepared.routing.symbols,
    )

    records = runtime.committed_sample_records()
    sample = records[0]
    input_data = runtime.load_sample(sample.path)
    if input_data.shape != runtime.INPUT_SHAPE or input_data.dtype.name != runtime.INPUT_DTYPE:
        raise ValueError("KWS V1 sample input violates the model tensor contract")

    simulator_session = runtime._simulator_session("tsim").validate_environment()
    simulator = simulator_session.load()
    device = tvm.ext_dev(0)

    def make_graph(bundle, debug=False):
        creator = debug_executor.create if debug else graph_executor.create
        kwargs = {"dump_root": str(build_dir / "debug")} if debug else {}
        graph = creator(bundle.graph_json, bundle.module, device, **kwargs)
        graph.load_params(bundle.params)
        graph.set_input(runtime.INPUT_NAME, input_data)
        return graph

    ref_graph = graph_executor.create(reference.graph_json, reference.module, tvm.cpu(0))
    ref_graph.load_params(reference.params)
    ref_graph.set_input(runtime.INPUT_NAME, input_data)
    ref_graph.run()
    reference_output = ref_graph.get_output(0).numpy()

    def ordinary_run(bundle):
        graph = make_graph(bundle)
        simulator_session.clear_and_validate(simulator)
        graph.run()
        output = graph.get_output(0).numpy()
        stats = simulator_session.read_stats(simulator.stats)
        simulator_session.validate_activity(stats)
        return graph, output, stats

    # Baseline is built without selected dispatch, then compared with the tuned graph.
    compiler.clear()
    with vta.build_config():
        baseline_factory = tvm.relay.build(prepared.mixed_module, target=runtime._mixed_target("llvm"))
    baseline_bundle = export_graph_bundle(
        baseline_factory, build_dir / "baseline", "mixed", artifact_name="kws-v1-baseline",
        artifact_role="mixed", model_sha256=prepared.imported.model_sha256,
        host_codegen="llvm", simulator="tsim", expected_vta_symbols=prepared.routing.symbols,
    )
    baseline_graph, baseline_output, baseline_stats = ordinary_run(baseline_bundle)
    tuned_graph, tuned_output, tuned_stats = ordinary_run(mixed)
    baseline_cycles = int(baseline_stats["cycle_count"])
    tuned_cycles = int(tuned_stats["cycle_count"])

    import numpy as np
    def check_output(expected, actual):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        try:
            runtime.compare_outputs(sample.path, expected, actual)
        except RuntimeError as error:
            raise ValueError(f"KWS V1 output differs from HOST reference: {error}") from error
    validate_one_sample(sample.filename, hashlib.sha256(sample.path.read_bytes()).hexdigest(),
                        reference_output, baseline_output, check_output)
    validate_one_sample(sample.filename, hashlib.sha256(sample.path.read_bytes()).hexdigest(),
                        reference_output, tuned_output, check_output)

    debug_graph = make_graph(mixed, debug=True)
    simulator_session.clear_and_validate(simulator)
    debug_graph._run_per_layer()
    debug_stats = simulator_session.read_stats(simulator.stats)
    simulator_session.validate_activity(debug_stats)
    ordinary_graph, _, ordinary_stats = ordinary_run(mixed)
    assert_counter_agreement(ordinary_stats, debug_stats)
    ordinary_cycles = int(ordinary_stats["cycle_count"])
    expected_occurrences = []
    for identity, entry, task in zip(identities, entries, tasks):
        expected_occurrences.append({
            "occurrence": identity.occurrence,
            "symbol": identity.symbol,
            "fusion_sha256": identity.sha256,
            "workload_sha256": tuner.shared._task_workload_id(task),
            "config_sha256": hashlib.sha256(json.dumps(entry["config"], sort_keys=True,
                                                         separators=(",", ":")).encode()).hexdigest(),
        })
    profiled = profile_graph_resident_nodes(
        mixed.graph_json, expected_occurrences, debug_graph,
        _EvidenceProfileSession(simulator_session), simulator
    )
    rows = []
    for identity, entry, task, node in zip(identities, entries, tasks, profiled):
        result = record_by_occurrence[identity.occurrence]
        cycles = int(result["tsim_cycles"])
        gate = compare_cycles(node["deployment_cycles"], cycles)
        rows.append({
            **expected_occurrences[identity.occurrence],
            "logical_macs_per_invocation": tuner._logical_mac_count(task),
            "autotvm_cycles": cycles,
            "deployment_cycles": node["deployment_cycles"],
            "counted_invocations": 1,
            **gate,
        })
    model_path = runtime.MODEL_PATH
    sample_data = {
        "sample_id": sample.filename,
        "sha256": hashlib.sha256(sample.path.read_bytes()).hexdigest(),
        "sample_count": 1,
    }
    report = build_deployment_report(
        phase="seed" if manifest.get("phase") == "seed" else "selected",
        model_id="keyword_spotting_v1", model_sha256=prepared.imported.model_sha256,
        geometry_path=manifest["geometry_path"], sample=sample_data,
        selected_manifest_path=manifest_path,
        full_model={"baseline_cycles": baseline_cycles, "tuned_cycles": tuned_cycles},
        occurrences=rows, expected_occurrences=expected_occurrences,
        peak_macs_per_cycle=2 ** vta.get_env().LOG_BATCH * (2 ** vta.get_env().LOG_BLOCK) ** 2,
    )
    report["full_model"]["debug_profile_full_model_cycles"] = int(debug_stats["cycle_count"])
    report["full_model"]["ordinary_full_model_single_sample_cycles"] = ordinary_cycles
    report["scope"]["sample_preprocessing"] = "existing KWS V1 WAV-to-MFCC preprocessing"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-dir", type=Path, default=APP_ROOT / "build" / "tune-deployment")
    args = parser.parse_args(argv)
    try:
        report = execute_deployment(args.best_manifest, args.output, args.build_dir)
    except Exception as error:
        from deployment_evidence import write_failure_report
        phase = "seed"
        try:
            phase = json.loads(args.best_manifest.read_text(encoding="utf-8")).get("phase", phase)
            if phase == "full":
                phase = "selected"
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        write_failure_report(
            args.output, phase=phase, model_id="keyword_spotting_v1",
            stage="deployment", error=error,
        )
        raise
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "occurrences": len(report["occurrences"]), "sample": report["sample"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
