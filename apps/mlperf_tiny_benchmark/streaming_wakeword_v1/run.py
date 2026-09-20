#!/usr/bin/env python3
"""Run the fixed MLPerf Tiny streaming wakeword v1 deployment."""

import argparse

import runtime


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(runtime.DEFAULT_OUTPUT_DIR),
        help="directory for generated graph-executor bundles",
    )
    parser.add_argument(
        "--host-codegen",
        choices=("llvm", "c", "all"),
        default="llvm",
        help="host code generator; all runs LLVM then C",
    )
    parser.add_argument(
        "--simulator",
        choices=("host", "fsim", "tsim"),
        default="fsim",
        help="execution mode (default: fsim)",
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
    if args.host_codegen == "all":
        if args.simulator == "host":
            _parser().error("--host-codegen all requires --simulator fsim or tsim")
        deployer = (
            runtime.deploy_fsim_matrix
            if args.simulator == "fsim"
            else runtime.deploy_tsim_matrix
        )
        result = deployer(args.output_dir)
        for artifacts, execution in zip(result.artifacts, result.executions):
            prefix = f"{artifacts.host_codegen}-{args.simulator}"
            print(f"{prefix} partitions: {len(result.prepared.routing.symbols)}")
            _print_execution(prefix, execution)
            print(f"{prefix} reference bundle: {artifacts.reference.artifact_dir}")
            print(f"{prefix} mixed bundle: {artifacts.mixed.artifact_dir}")
        print(
            f"MLPerf Tiny streaming wakeword v1 LLVM/C "
            f"{args.simulator.upper()} matrix passed"
        )
        return 0

    result = runtime.deploy(
        args.output_dir,
        host_codegen=args.host_codegen,
        simulator=args.simulator,
    )
    prefix = f"{args.host_codegen}-{args.simulator}"
    _print_execution(prefix, result.execution)
    print(f"{prefix} reference bundle: {result.artifacts.reference.artifact_dir}")
    print(f"{prefix} mixed bundle: {result.artifacts.mixed.artifact_dir}")
    print("MLPerf Tiny streaming wakeword v1 deployment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
