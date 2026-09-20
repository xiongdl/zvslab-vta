#!/usr/bin/env python3
"""Run the MLPerf Tiny anomaly autoencoder on HOST or FSIM."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import runtime


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("host", "fsim"), default="host",
        help="execution backend (default: host)",
    )
    parser.add_argument(
        "--simulator", choices=("host", "fsim", "tsim"),
        help="compatibility alias for --mode; tsim is reserved for checkpoint 3",
    )
    parser.add_argument(
        "--build-dir", "--output-dir", dest="build_dir",
        default=str(runtime.DEFAULT_BUILD_DIR),
        help="directory for generated reference/mixed artifact bundles",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(runtime.MANIFEST_PATH),
        help="sample manifest; default is the committed ten-sample manifest",
    )
    parser.add_argument(
        "--host-codegen", choices=("llvm", "c", "all"), default="llvm",
        help="host code generator; all is supported for FSIM matrices",
    )
    parser.add_argument(
        "--output-json", type=str,
        help="write deterministic result JSON to this path",
    )
    return parser


def _json_result(result):
    execution = result.execution
    return {
        "mode": execution.mode,
        "host_codegen": execution.host_codegen,
        "model": {
            "sha256": execution.model_metadata["model_sha256"],
            "artifact_dirs": {
                "reference": str(result.artifacts.reference.artifact_dir),
                "mixed": str(result.artifacts.mixed.artifact_dir),
            },
        },
        "input": execution.model_metadata["input"],
        "output": execution.model_metadata["output"],
        "score_semantics": execution.model_metadata["score"],
        "per_sample_results": [asdict(item) for item in execution.samples],
        "summary": execution.summary,
        "profiler_stats": execution.profiler_stats,
    }


def _write_or_print(payload, output_json):
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv=None):
    args = _parser().parse_args(argv)
    mode = args.mode
    if args.simulator:
        if args.simulator == "tsim":
            print("TSIM is reserved for checkpoint 3; use --mode host or --mode fsim", file=sys.stderr)
            return 2
        if args.mode != "host" and args.mode != args.simulator:
            print("--mode and --simulator select different backends", file=sys.stderr)
            return 2
        mode = args.simulator

    try:
        if args.host_codegen == "all":
            if mode != "fsim":
                raise ValueError("--host-codegen all currently requires --mode fsim")
            prepared, artifacts, executions = runtime.deploy_matrix(args.build_dir, mode, args.manifest)
            payloads = []
            for current_artifacts, execution in zip(artifacts, executions):
                result = runtime.DeploymentResult(prepared, current_artifacts, execution)
                payloads.append(_json_result(result))
            payload = {"mode": mode, "host_codegen": "all", "runs": payloads}
        else:
            result = runtime.deploy(args.build_dir, args.host_codegen, mode, args.manifest)
            payload = _json_result(result)
        _write_or_print(payload, args.output_json)
        return 0
    except Exception as error:
        print(f"anomaly deployment failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
