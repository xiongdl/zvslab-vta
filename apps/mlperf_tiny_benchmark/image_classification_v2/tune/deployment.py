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


def _import_runtime():
    app_path = str(APP_ROOT)
    if app_path not in sys.path:
        sys.path.insert(0, app_path)
    return _load_module("ic_v2_deployment_runtime", APP_ROOT / "runtime.py")


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(args):
    from tvm.contrib.debugger import debug_executor

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
    selected_configs_by_symbol = {}
    for entry in manifest["entries"]:
        idx = entry["workload_index"]
        identity = identities[idx]
        native_path = manifest_path.parent / entry["native_record"]
        native_input, native_result = tuning.artifacts.load_validated_record(
            native_path, entry["native_record_sha256"], tuning.legacy.shared.autotvm.record
        )
        if entry["symbol"] != identity.symbol or entry["fusion_sha256"] != identity.sha256:
            raise ValueError("selected manifest occurrence does not match prepared IC V2 routing")
        if len(native_result.costs) != 1 or int(native_result.costs[0]) != entry["tsim_cycles"]:
            raise ValueError("selected native record cycles do not match its manifest entry")
        selected_configs_by_symbol[identity.symbol] = native_input.config
        rows.append({
            **expected[idx],
            "workload_index": idx,
            "config_index": int(native_input.config.index),
            "config": native_input.config.to_json_dict(),
            "config_sha256": hashlib.sha256(json.dumps(
                native_input.config.to_json_dict(), sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "autotvm_cycles": entry["tsim_cycles"],
            "result_json": entry["result_json"],
            "native_record_sha256": entry["native_record_sha256"],
        })
    if {(row["occurrence"], row["symbol"]) for row in rows} != {
        (item["occurrence"], item["symbol"]) for item in expected
    }:
        raise ValueError("selected manifest does not cover all eight prepared VTA occurrences")

    artifacts = runtime.build_host_artifacts(
        prepared, args.output_dir / "tuned", host_codegen="llvm", simulator="tsim",
        selected_configs_by_symbol=selected_configs_by_symbol,
    )
    baseline_artifacts = runtime.build_host_artifacts(
        prepared, args.output_dir / "baseline", host_codegen="llvm", simulator="tsim",
    )
    session, simulator = runtime._load_simulator("tsim")
    sample_paths = runtime.committed_sample_paths()
    inputs = [(path, runtime.load_sample(path)) for path in sample_paths]
    if len(inputs) != 10:
        raise RuntimeError(f"IC V2 deployment requires ten committed samples, got {len(inputs)}")

    # Full-model performance and all per-node alignment use the first sample.
    # The selected graph runs every sample below for output correctness only.
    first_path, first_input = inputs[0]
    first_reference = runtime._run_graph(artifacts.reference, first_input)
    session.clear_and_validate(simulator)
    baseline_output = runtime._run_graph(baseline_artifacts.mixed, first_input)
    runtime.compare_outputs(first_path, first_reference, baseline_output)
    baseline_stats = session.read_stats(simulator=simulator)
    session.validate_activity(baseline_stats)

    session.clear_and_validate(simulator)
    first_tuned_output = runtime._run_graph(artifacts.mixed, first_input)
    runtime.compare_outputs(first_path, first_reference, first_tuned_output)
    ordinary_stats = session.read_stats(simulator=simulator)
    session.validate_activity(ordinary_stats)

    # Compare full-run counters from ordinary and debug graph runtimes on one
    # identical tuned input before using the debug executor's resident tensors.
    debug_graph = debug_executor.create(
        artifacts.mixed.graph_json, artifacts.mixed.module, artifacts.mixed.device
    )
    debug_graph.load_params(artifacts.mixed.params)
    debug_graph.set_input(runtime.INPUT_NAME, first_input)
    session.clear_and_validate(simulator)
    debug_graph._run_per_layer()
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
        # Use the resident graph tensors and one counted deployed invocation.
        session.clear_and_validate(simulator)
        debug_graph._execute_node(node_index)
        stats = session.read_stats(simulator=simulator)
        session.validate_activity(stats)
        cycles = int(stats["cycle_count"])
        row["deployment_cycles"] = cycles
        row["graph_node"] = node.get("name")
        row["graph_func_name"] = node.get("attrs", {}).get("func_name")
        row["difference_numerator"] = abs(cycles - row["autotvm_cycles"])
        row["difference_denominator"] = row["autotvm_cycles"]
        row["difference_percent"] = 100.0 * row["difference_numerator"] / row["difference_denominator"]
        row["passed"] = cycles_within_strict_ten_percent(cycles, row["autotvm_cycles"])

    try:
        validate_occurrence_rows(rows, expected)
    except ValueError as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error": str(error),
            "model": "image_classification_v2",
            "model_sha256": manifest["model_sha256"],
            "geometry_sha256": manifest["geometry_sha256"],
            "manifest_sha256": _sha256_file(manifest_path),
            "manifest_completion_label": manifest["completion_label"],
            "debug_ordinary_counters_agree": debug_stats == ordinary_stats,
            "debug_full_run_stats": debug_stats,
            "ordinary_full_run_stats": ordinary_stats,
            "outputs_passed": 1,
            "occurrences": rows,
        }
        failure_path = output_path.with_suffix(".failure.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise

    # The strict one-sample performance gate passed. Now run the selected graph
    # over all ten samples for correctness, without collecting performance data.
    outputs_passed = 0
    for sample_index, (path, data) in enumerate(inputs):
        reference = (
            first_reference
            if sample_index == 0
            else runtime._run_graph(artifacts.reference, data)
        )
        tuned = runtime._run_graph(artifacts.mixed, data)
        runtime.compare_outputs(path, reference, tuned)
        outputs_passed += 1

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
        "baseline_full_model_cycles": [baseline_stats["cycle_count"]],
        "tuned_full_model_cycles": [ordinary_stats["cycle_count"]],
        "sample_count": len(sample_paths),
        "outputs_passed": outputs_passed,
        "performance_sample_count": 1,
        "performance_sample": first_path.name,
        "occurrences": rows,
        "status": "passed",
    }
    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Deployment report: {output}")
    print("Deployment status: passed (8/8 VTA occurrences on 1 sample; 10/10 outputs)")
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
        if not failure.exists():
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(
                json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n"
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
