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

"""Run the fixed MLPerf Tiny ResNet-8 deployment."""

import argparse
from pathlib import Path

from runtime import DEFAULT_OUTPUT_DIR, deploy, deploy_fsim_matrix, deploy_tsim_matrix
from runtime import write_deployment_report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--host-codegen", choices=("llvm", "c", "all"), default="llvm")
    parser.add_argument("--simulator", choices=("fsim", "tsim"), default="fsim")
    parser.add_argument(
        "--schedule", type=str, default=None,
        help="actual-deployment schedule snapshot (.log plus same-stem .json); omitted or none uses defaults",
    )
    parser.add_argument("--deployment-report", type=Path)
    parser.add_argument(
        "--validate-schedule-evidence", action="store_true",
        help="on TSIM, require complete measured coverage and strict per-layer cycle alignment",
    )
    return parser


def _print_coverage(artifacts):
    for occurrence, symbol, selected in getattr(artifacts, "schedule_coverage", ()):
        kind = "snapshot" if selected else "default"
        print(f"occurrence {occurrence} ({symbol}): {kind} schedule")


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.validate_schedule_evidence and args.deployment_report is None:
        raise ValueError("--validate-schedule-evidence requires --deployment-report")
    if args.host_codegen == "all" and args.deployment_report is not None:
        raise ValueError("--deployment-report requires one host codegen, not --host-codegen all")
    schedule = None if args.schedule is None else args.schedule
    if args.host_codegen == "all":
        result = (
            deploy_fsim_matrix(args.output_dir, schedule=schedule)
            if args.simulator == "fsim"
            else deploy_tsim_matrix(args.output_dir, schedule=schedule)
        )
        for artifacts, execution in zip(result.artifacts, result.executions):
            _print_coverage(artifacts)
            print(f"{artifacts.host_codegen}-{args.simulator} compared samples: {len(execution.comparisons)}")
            print(f"{artifacts.host_codegen}-{args.simulator} profiler: {execution.profiler_stats}")
        print(f"MLPerf ResNet LLVM/C {args.simulator.upper()} matrix passed")
        return 0
    if args.host_codegen == "llvm" and args.simulator == "fsim" and schedule is None:
        result = deploy(args.output_dir)
    else:
        result = deploy(
            args.output_dir,
            host_codegen=args.host_codegen,
            simulator=args.simulator,
            schedule=schedule,
        )
    if hasattr(result, "artifacts"):
        _print_coverage(result.artifacts)
    print(f"Compared samples: {len(result.execution.comparisons)}")
    print(f"{args.simulator.upper()} profiler: {result.execution.profiler_stats}")
    if args.deployment_report is not None:
        report = write_deployment_report(
            result, args.deployment_report, schedule=schedule,
            validate_schedule_evidence=args.validate_schedule_evidence,
        )
        print(f"Deployment report: {args.deployment_report.resolve()}")
        if report.get("occurrences"):
            print(f"Measured occurrences validated: {len(report['occurrences'])}")
    print("MLPerf ResNet HOST deployment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
