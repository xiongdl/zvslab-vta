#!/usr/bin/env python3
"""Run the fixed MLPerf Tiny Keyword Spotting v1 deployment."""

import argparse
import sys
from pathlib import Path

import runtime as deploy_runtime

DEFAULT_OUTPUT_DIR = deploy_runtime.DEFAULT_OUTPUT_DIR
deploy = deploy_runtime.deploy
deploy_fsim_matrix = deploy_runtime.deploy_fsim_matrix
deploy_tsim_matrix = deploy_runtime.deploy_tsim_matrix


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for generated graph-executor bundles",
    )
    parser.add_argument(
        "--host-codegen",
        choices=("llvm", "c", "all"),
        default="llvm",
        help="host code generator (all builds the ordered LLVM/C matrix)",
    )
    parser.add_argument(
        "--simulator",
        choices=("host", "fsim", "tsim"),
        default="fsim",
        help="execution mode (default: fsim)",
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


def _print_execution(prefix, execution):
    print(f"{prefix} compared samples: {len(execution.comparisons)}")
    for comparison in execution.comparisons:
        mixed_top1 = (
            "not-run"
            if comparison.mixed is None
            else str(int(comparison.mixed.argmax(axis=1)[0]))
        )
        print(
            f"{prefix} sample: {comparison.sample_path.name} "
            f"reference top-1: {comparison.top1} mixed top-1: {mixed_top1}"
        )
    print(f"{prefix} profiler: {execution.profiler_stats}")


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.validate_schedule_evidence and args.deployment_report is None:
        print("--validate-schedule-evidence requires --deployment-report", file=sys.stderr)
        return 2
    if args.deployment_report is not None and args.host_codegen == "all":
        print("--deployment-report requires one host codegen, not --host-codegen all", file=sys.stderr)
        return 2
    if args.schedule not in (None, "none") and args.simulator == "host":
        print("schedule replay requires --simulator fsim or tsim", file=sys.stderr)
        return 2
    if args.validate_schedule_evidence and args.simulator != "tsim":
        print("--validate-schedule-evidence requires --simulator tsim", file=sys.stderr)
        return 2
    if args.host_codegen == "all":
        if args.simulator == "host":
            _parser().error("--host-codegen all requires --simulator fsim or tsim")
        deployer = deploy_fsim_matrix if args.simulator == "fsim" else deploy_tsim_matrix
        result = deployer(args.output_dir, schedule=args.schedule)
        for artifacts, execution in zip(result.artifacts, result.executions):
            prefix = f"{artifacts.host_codegen}-{args.simulator}"
            print(f"{prefix} partitions: {len(result.prepared.routing.symbols)}")
            _print_execution(prefix, execution)
            print(f"{prefix} reference bundle: {artifacts.reference.artifact_dir}")
            print(f"{prefix} mixed bundle: {artifacts.mixed.artifact_dir}")
        print(f"MLPerf KWS v1 LLVM/C {args.simulator.upper()} matrix passed")
        return 0

    result = deploy(
        args.output_dir,
        host_codegen=args.host_codegen,
        simulator=args.simulator,
        schedule=args.schedule,
    )
    prefix = f"{args.host_codegen}-{args.simulator}"
    _print_execution(prefix, result.execution)
    for occurrence, symbol, selected in result.artifacts.schedule_coverage:
        kind = "snapshot" if selected else "default"
        print(f"occurrence {occurrence} ({symbol}): {kind} schedule", file=sys.stderr)
    if args.deployment_report is not None:
        report = deploy_runtime.write_deployment_report(
            result,
            args.deployment_report,
            schedule=args.schedule,
            validate_schedule_evidence=args.validate_schedule_evidence,
        )
        print(f"deployment report: {args.deployment_report.resolve()} ({report['status']})")
    print(f"{prefix} reference bundle: {result.artifacts.reference.artifact_dir}")
    print(f"{prefix} mixed bundle: {result.artifacts.mixed.artifact_dir}")
    print("MLPerf KWS v1 deployment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
