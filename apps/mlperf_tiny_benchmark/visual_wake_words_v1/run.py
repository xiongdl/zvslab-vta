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

"""Run the fixed MLPerf Tiny VWW HOST deployment."""

import argparse
import json
from pathlib import Path

from runtime import DEFAULT_OUTPUT_DIR, deploy, deploy_fsim_matrix, deploy_tsim_matrix


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for the two generated host libraries",
    )
    parser.add_argument("--autotvm-log", type=Path, help="native AutoTVM log for tuned replay")
    parser.add_argument("--autotvm-sidecar", type=Path, help="matching JSON sidecar")
    parser.add_argument(
        "--host-codegen",
        choices=("llvm", "c", "all"),
        default="llvm",
        help="host code generator to execute (all builds the ordered LLVM/C matrix)",
    )
    parser.add_argument(
        "--simulator",
        choices=("fsim", "tsim"),
        default="fsim",
        help=(
            "simulator to execute (default: fsim; tsim uses the shared "
            "vta_64mac.json geometry with VTA_BACKEND=tsim)"
        ),
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if (args.autotvm_log is None) != (args.autotvm_sidecar is None):
        raise SystemExit("--autotvm-log and --autotvm-sidecar must be provided together")
    if args.autotvm_log is not None and args.host_codegen == "all":
        raise SystemExit("AutoTVM replay requires one --host-codegen: llvm or c")
    if args.host_codegen == "all":
        result = (
            deploy_fsim_matrix(args.output_dir)
            if args.simulator == "fsim"
            else deploy_tsim_matrix(args.output_dir)
        )
        for artifacts, execution in zip(result.artifacts, result.executions):
            print(f"{artifacts.host_codegen}-{args.simulator} partitions: {len(result.prepared.routing.symbols)}")
            print(f"{artifacts.host_codegen}-{args.simulator} compared samples: {len(execution.comparisons)}")
            print(f"{artifacts.host_codegen}-{args.simulator} profiler: {execution.profiler_stats}")
            print(f"{artifacts.host_codegen}-{args.simulator} reference bundle: {artifacts.reference.artifact_dir}")
            print(f"{artifacts.host_codegen}-{args.simulator} mixed bundle: {artifacts.mixed.artifact_dir}")
        print(f"MLPerf VWW LLVM/C {args.simulator.upper()} matrix passed")
    elif args.host_codegen == "llvm":
        # Keep the original call shape for callers that wrap the compatibility API.
        if args.autotvm_log is not None:
            result = deploy(
                args.output_dir,
                simulator=args.simulator,
                autotvm_log=args.autotvm_log,
                autotvm_sidecar=args.autotvm_sidecar,
            )
        else:
            result = (
                deploy(args.output_dir)
                if args.simulator == "fsim"
                else deploy(args.output_dir, simulator=args.simulator)
            )
        print(f"Compared samples: {len(result.execution.comparisons)}")
        print(f"{args.simulator.upper()} profiler: {result.execution.profiler_stats}")
        if args.autotvm_sidecar is not None:
            metadata = json.loads(args.autotvm_sidecar.read_text(encoding="utf-8"))
            print(f"AutoTVM model: {metadata['model_id']} ({metadata['model_sha256']})")
            print(f"AutoTVM log: {args.autotvm_log.resolve()}")
            print(f"AutoTVM sidecar: {args.autotvm_sidecar.resolve()}")
        print("MLPerf VWW HOST deployment passed")
    else:
        result = deploy(
            args.output_dir, host_codegen=args.host_codegen, simulator=args.simulator,
            autotvm_log=args.autotvm_log, autotvm_sidecar=args.autotvm_sidecar,
        )
        print(f"{args.host_codegen}-{args.simulator} compared samples: {len(result.execution.comparisons)}")
        print(f"{args.host_codegen}-{args.simulator} profiler: {result.execution.profiler_stats}")
        if args.autotvm_sidecar is not None:
            metadata = json.loads(args.autotvm_sidecar.read_text(encoding="utf-8"))
            print(f"AutoTVM model: {metadata['model_id']} ({metadata['model_sha256']})")
            print(f"AutoTVM log: {args.autotvm_log.resolve()}")
            print(f"AutoTVM sidecar: {args.autotvm_sidecar.resolve()}")
        print("MLPerf VWW HOST deployment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
