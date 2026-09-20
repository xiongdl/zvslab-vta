#!/usr/bin/env python3
"""Run the fixed MLPerf Tiny Keyword Spotting v1 deployment."""

import argparse

from runtime import DEFAULT_OUTPUT_DIR, deploy, deploy_fsim_matrix, deploy_tsim_matrix


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
    return parser


def _print_execution(prefix, execution):
    print(f"{prefix} compared samples: {len(execution.comparisons)}")
    print(f"{prefix} profiler: {execution.profiler_stats}")


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.host_codegen == "all":
        if args.simulator == "host":
            _parser().error("--host-codegen all requires --simulator fsim or tsim")
        deployer = deploy_fsim_matrix if args.simulator == "fsim" else deploy_tsim_matrix
        result = deployer(args.output_dir)
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
    )
    prefix = f"{args.host_codegen}-{args.simulator}"
    _print_execution(prefix, result.execution)
    print(f"{prefix} reference bundle: {result.artifacts.reference.artifact_dir}")
    print(f"{prefix} mixed bundle: {result.artifacts.mixed.artifact_dir}")
    print("MLPerf KWS v1 deployment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
