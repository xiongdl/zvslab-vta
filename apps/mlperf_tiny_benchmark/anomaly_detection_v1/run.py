#!/usr/bin/env python3
"""Run the MLPerf Tiny anomaly autoencoder on HOST, FSIM, or TSIM."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import runtime


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("host", "fsim", "tsim"), default="host",
        help="execution backend (default: host)",
    )
    parser.add_argument(
        "--simulator", choices=("host", "fsim", "tsim"),
        help="compatibility alias for --mode",
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
        help="host code generator; all builds the ordered LLVM/C simulator matrix",
    )
    parser.add_argument(
        "--output-json", type=str,
        help="write deterministic result JSON to this path",
    )
    parser.add_argument(
        "--tsim-window-budget", type=_positive_int, default=None,
        help=(
            "TSIM-only maximum representative windows per sample; default 1, "
            f"or {runtime.TSIM_WINDOW_BUDGET_ENV}"
        ),
    )
    parser.add_argument(
        "--schedule", type=str, default=None,
        help="actual-deployment schedule snapshot (.log plus same-stem .json); omitted or none uses defaults",
    )
    parser.add_argument("--deployment-report", type=Path)
    parser.add_argument(
        "--validate-schedule-evidence", action="store_true",
        help="on TSIM, require complete measured coverage and per-layer cycle alignment",
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
        "execution": execution.model_metadata["execution"],
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


def _print_coverage(artifacts):
    for occurrence, symbol, selected in getattr(artifacts, "schedule_coverage", ()):
        kind = "snapshot" if selected else "default"
        print(f"occurrence {occurrence} ({symbol}): {kind} schedule", file=sys.stderr)


def main(argv=None):
    args = _parser().parse_args(argv)
    mode = args.mode
    if args.simulator:
        if args.mode != "host" and args.mode != args.simulator:
            print("--mode and --simulator select different backends", file=sys.stderr)
            return 2
        mode = args.simulator

    if args.validate_schedule_evidence and args.deployment_report is None:
        print("--validate-schedule-evidence requires --deployment-report", file=sys.stderr)
        return 2
    if args.deployment_report is not None and args.host_codegen == "all":
        print("--deployment-report requires one host codegen, not --host-codegen all", file=sys.stderr)
        return 2

    try:
        if args.host_codegen == "all":
            if mode not in ("fsim", "tsim"):
                raise ValueError("--host-codegen all requires --mode fsim or tsim")
            if args.tsim_window_budget is None:
                prepared, artifacts, executions = runtime.deploy_matrix(
                    args.build_dir, mode, args.manifest, schedule=args.schedule
                )
            else:
                prepared, artifacts, executions = runtime.deploy_matrix(
                    args.build_dir, mode, args.manifest,
                    tsim_window_budget=args.tsim_window_budget,
                    schedule=args.schedule,
                )
            payloads = []
            for current_artifacts, execution in zip(artifacts, executions):
                _print_coverage(current_artifacts)
                result = runtime.DeploymentResult(prepared, current_artifacts, execution)
                payloads.append(_json_result(result))
            payload = {"mode": mode, "host_codegen": "all", "runs": payloads}
        else:
            result = runtime.deploy(
                args.build_dir, args.host_codegen, mode, args.manifest,
                tsim_window_budget=args.tsim_window_budget,
                schedule=args.schedule,
            )
            payload = _json_result(result)
            _print_coverage(result.artifacts)
            if args.deployment_report is not None:
                report = runtime.write_deployment_report(
                    result, args.deployment_report, schedule=args.schedule,
                    validate_schedule_evidence=args.validate_schedule_evidence,
                )
                payload["deployment_report"] = str(args.deployment_report.resolve())
                payload["deployment_report_status"] = report["status"]
        _write_or_print(payload, args.output_json)
        return 0
    except Exception as error:
        print(f"anomaly deployment failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
