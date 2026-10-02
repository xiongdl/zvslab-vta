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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Run the fixed MLPerf Tiny VWW deployment with an optional actual schedule."""

import argparse
import sys
from pathlib import Path

import runtime

DEFAULT_OUTPUT_DIR = runtime.DEFAULT_OUTPUT_DIR
deploy = runtime.deploy
deploy_fsim_matrix = runtime.deploy_fsim_matrix
deploy_tsim_matrix = runtime.deploy_tsim_matrix


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="directory for generated graph-executor bundles",
    )
    parser.add_argument(
        "--host-codegen", choices=("llvm", "c", "all"), default="llvm",
        help="host code generator; all runs the ordered LLVM/C matrix",
    )
    parser.add_argument(
        "--simulator", choices=("fsim", "tsim"), default="fsim",
        help="simulator to execute (must match VTA_BACKEND)",
    )
    parser.add_argument(
        "--schedule", type=str, default=None,
        help="actual-deployment schedule snapshot (.log plus same-stem .json); omitted or none uses defaults",
    )
    parser.add_argument("--deployment-report", type=Path)
    parser.add_argument(
        "--validate-schedule-evidence", action="store_true",
        help="on TSIM, require measured complete coverage and per-layer cycle alignment",
    )
    return parser


def _print_execution(prefix, execution):
    print(f"{prefix} compared samples: {len(execution.comparisons)}")
    for comparison in execution.comparisons:
        mixed_top1 = (
            "not-run" if comparison.mixed is None
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
    if args.schedule not in (None, "none") and args.host_codegen == "all":
        print("schedule replay requires one --host-codegen: llvm or c", file=sys.stderr)
        return 2
    if args.validate_schedule_evidence and args.simulator != "tsim":
        print("--validate-schedule-evidence requires --simulator tsim", file=sys.stderr)
        return 2
    if args.host_codegen == "all":
        deployer = deploy_fsim_matrix if args.simulator == "fsim" else deploy_tsim_matrix
        result = deployer(args.output_dir)
        for artifacts, execution in zip(result.artifacts, result.executions):
            prefix = f"{artifacts.host_codegen}-{args.simulator}"
            print(f"{prefix} partitions: {len(result.prepared.routing.symbols)}")
            _print_execution(prefix, execution)
            print(f"{prefix} reference bundle: {artifacts.reference.artifact_dir}")
            print(f"{prefix} mixed bundle: {artifacts.mixed.artifact_dir}")
        print(f"MLPerf VWW LLVM/C {args.simulator.upper()} matrix passed")
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
        source = "snapshot" if selected else "default"
        print(f"occurrence {occurrence} ({symbol}): {source} schedule", file=sys.stderr)
    if args.deployment_report is not None:
        report = runtime.write_deployment_report(
            result, args.deployment_report, schedule=args.schedule,
            validate_schedule_evidence=args.validate_schedule_evidence,
        )
        print(f"deployment report: {args.deployment_report.resolve()} ({report['status']})")
    print(f"{prefix} reference bundle: {result.artifacts.reference.artifact_dir}")
    print(f"{prefix} mixed bundle: {result.artifacts.mixed.artifact_dir}")
    print("MLPerf VWW deployment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
