#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deploy IC V2 with exported best schedules and profile each VTA occurrence."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


TUNE_DIR = Path(__file__).resolve().parent
APP_ROOT = TUNE_DIR.parent
REPO_ROOT = APP_ROOT.parents[3]
DEFAULT_OUTPUT = TUNE_DIR / "deployment.json"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cycles_within_strict_ten_percent(deployed_cycles, autotvm_cycles):
    """Check positive integer counts with the approved strict rational bound."""
    for label, value in (("deployment", deployed_cycles), ("AutoTVM", autotvm_cycles)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} cycle count must be a positive integer")
    return 10 * abs(deployed_cycles - autotvm_cycles) < autotvm_cycles


def validate_occurrence_rows(rows, expected):
    """Require one uniquely identified, passing cycle pair per expected fusion."""
    expected_keys = {(item["occurrence"], item["symbol"]) for item in expected}
    actual_keys = []
    for row in rows:
        key = (row.get("occurrence"), row.get("symbol"))
        actual_keys.append(key)
        if key not in expected_keys:
            raise ValueError(f"deployment contains unexpected occurrence {key}")
        if not cycles_within_strict_ten_percent(
            row.get("deployment_cycles"), row.get("autotvm_cycles")
        ):
            raise ValueError(f"strict <10% cycle gate failed for occurrence {key}")
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("deployment contains duplicate occurrence rows")
    if set(actual_keys) != expected_keys:
        raise ValueError("deployment occurrence coverage is incomplete")
    return rows


def _load_tuning_entry():
    return _load_module("ic_v2_deployment_tuning", TUNE_DIR / "tune.py")


def _native_log_from_manifest(manifest_path, manifest, tuning, destination):
    """Combine the manifest's independently validated native records for dispatch."""
    from tvm import autotvm

    native_log = Path(destination)
    native_log.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in manifest["entries"]:
        result_path = manifest_path.parent / entry["result_json"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        best_path = manifest_path.parent / result["result"]["best_native_record"]
        tuning.artifacts.load_validated_record(
            best_path,
            result["result"]["best_native_record_sha256"],
            tuning.legacy.shared.autotvm.record,
        )
        records.extend(autotvm.record.load_from_file(str(best_path)))
    native_log.write_text(
        "".join(autotvm.record.encode(measure_input, result) + "\n" for measure_input, result in records),
        encoding="utf-8",
    )
    return native_log


def _import_runtime():
    app_path = str(APP_ROOT)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    return _load_module("ic_v2_deployment_runtime", APP_ROOT / "runtime.py")


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(args):
    import numpy as np
    import tvm
    from tvm.contrib import debug_executor, graph_executor

    output_path = args.output.expanduser().resolve()
    output_path.unlink(missing_ok=True)
    tuning = _load_tuning_entry()
    manifest_path = args.best_manifest.expanduser().resolve(strict=True)
    tuning._replay_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("deployment requires complete occurrence coverage in the selected manifest")
    if manifest.get("completion_label") not in ("FULL_SEARCH", "BOUNDED_SMOKE_INCOMPLETE"):
        raise ValueError("selected manifest has an invalid completion label")

    runtime = _import_runtime()
    prepared = runtime.prepare_model(runtime.MODEL_PATH)
    identities = tuning.legacy.fused.extract_fused_identities(prepared)
    expected = [
        {"occurrence": identity.occurrence, "symbol": identity.symbol,
         "fusion_sha256": identity.sha256}
        for identity in identities
    ]
    if len(expected) != 8:
        raise ValueError(f"prepared IC V2 must route eight VTA occurrences, got {len(expected)}")
    rows = []
    for entry in manifest["entries"]:
        idx = entry["workload_index"]
        identity = identities[idx]
        result = json.loads((manifest_path.parent / entry["result_json"]).read_text())
        if entry["symbol"] != identity.symbol or entry["fusion_sha256"] != identity.sha256:
            raise ValueError("selected manifest occurrence does not match prepared IC V2 routing")
        rows.append({
            **expected[idx],
            "workload_index": idx,
            "config_index": result["result"]["config_index"],
            "config": result["result"]["selected_config"],
            "config_sha256": hashlib.sha256(json.dumps(
                result["result"]["selected_config"], sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "autotvm_cycles": result["result"]["tsim_cycles"],
            "result_json": entry["result_json"],
            "native_record_sha256": entry["native_record_sha256"],
        })
    if {(row["occurrence"], row["symbol"]) for row in rows} != {
        (item["occurrence"], item["symbol"]) for item in expected
    }:
        raise ValueError("selected manifest does not cover all eight prepared VTA occurrences")

    native_log = _native_log_from_manifest(
        manifest_path, manifest, tuning, args.output_dir / "selected-native-records.log"
    )
    artifacts = runtime.build_host_artifacts(
        prepared, args.output_dir / "tuned", host_codegen="llvm", simulator="tsim",
        selected_config_log=native_log,
    )
    baseline_artifacts = runtime.build_host_artifacts(
        prepared, args.output_dir / "baseline", host_codegen="llvm", simulator="tsim",
    )
    session, simulator = runtime._load_simulator("tsim")
    sample_paths = runtime.committed_sample_paths()
    inputs = [(path, runtime.load_sample(path)) for path in sample_paths]
    reference_outputs = [runtime._run_graph(artifacts.reference, data) for _, data in inputs]
    baseline_cycles = []
    tuned_cycles = []
    for (path, data), reference in zip(inputs, reference_outputs):
        session.clear_and_validate(simulator)
        baseline = runtime._run_graph(baseline_artifacts.mixed, data)
        runtime.compare_outputs(path, reference, baseline)
        baseline_cycles.append(session.read_stats(simulator=simulator)["cycle_count"])

        session.clear_and_validate(simulator)
        tuned = runtime._run_graph(artifacts.mixed, data)
        runtime.compare_outputs(path, reference, tuned)
        tuned_cycles.append(session.read_stats(simulator=simulator)["cycle_count"])

    # Compare full-run counters from ordinary and debug graph runtimes on one
    # identical tuned input before using the debug executor's resident tensors.
    first_path, first_input = inputs[0]
    session.clear_and_validate(simulator)
    runtime._run_graph(artifacts.mixed, first_input)
    ordinary_stats = session.read_stats(simulator=simulator)
    debug_graph = debug_executor.create(
        artifacts.mixed.graph_json, artifacts.mixed.module, artifacts.mixed.device
    )
    debug_graph.load_params(artifacts.mixed.params)
    debug_graph.set_input(runtime.INPUT_NAME, first_input)
    session.clear_and_validate(simulator)
    debug_graph.run()
    debug_stats = session.read_stats(simulator=simulator)
    if debug_stats != ordinary_stats:
        raise RuntimeError(
            f"debug and ordinary complete-run TSIM counters disagree: "
            f"debug={debug_stats}, ordinary={ordinary_stats}"
        )

    graph = json.loads(artifacts.mixed.graph_json)
    graph_nodes = graph["nodes"]
    target_nodes = {}
    for node_index, node in enumerate(graph_nodes):
        if node.get("op") != "tvm_op":
            continue
        attrs = node.get("attrs", {})
        function_name = attrs.get("func_name", "")
        for identity in identities:
            if identity.symbol == function_name or identity.symbol in node.get("name", ""):
                if identity.occurrence in target_nodes:
                    raise ValueError(f"multiple graph nodes map to {identity.symbol}")
                target_nodes[identity.occurrence] = node_index
    if set(target_nodes) != set(range(len(identities))):
        raise ValueError(
            f"reloaded graph VTA node mapping is incomplete: {target_nodes}; "
            f"expected symbols={[item.symbol for item in identities]}"
        )

    for row in rows:
        node_index = target_nodes[row["occurrence"]]
        node = graph_nodes[node_index]
        # Outputs already reside on device after debug_graph.run(); execute only
        # the selected VTA node between profiler clear/read calls.
        session.clear_and_validate(simulator)
        debug_graph._execute_node(node_index)
        stats = session.read_stats(simulator=simulator)
        cycles = stats.get("cycle_count")
        session.validate_activity(stats)
        row["deployment_cycles"] = cycles
        row["graph_node"] = node.get("name")
        row["graph_func_name"] = node.get("attrs", {}).get("func_name")
        row["difference_numerator"] = abs(cycles - row["autotvm_cycles"])
        row["difference_denominator"] = row["autotvm_cycles"]
        row["difference_percent"] = 100.0 * row["difference_numerator"] / row["difference_denominator"]
        row["passed"] = cycles_within_strict_ten_percent(cycles, row["autotvm_cycles"])

    validate_occurrence_rows(rows, expected)
    report = {
        "schema_version": 1,
        "model": "image_classification_v2",
        "model_sha256": manifest["model_sha256"],
        "geometry_path": manifest["geometry_path"],
        "geometry_sha256": manifest["geometry_sha256"],
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_completion_label": manifest["completion_label"],
        "measurement_protocol": manifest["measurement_protocol"],
        "debug_ordinary_counters_agree": True,
        "debug_full_run_stats": debug_stats,
        "ordinary_full_run_stats": ordinary_stats,
        "baseline_full_model_cycles": baseline_cycles,
        "tuned_full_model_cycles": tuned_cycles,
        "sample_count": len(sample_paths),
        "outputs_passed": len(sample_paths),
        "occurrences": rows,
        "status": "passed",
    }
    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Deployment report: {output}")
    print("Deployment status: passed (8/8 VTA occurrences, 10/10 samples)")
    return 0


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "build" / "selected_deployment")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as error:
        failure = args.output.expanduser().resolve().with_suffix(".failure.json")
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
