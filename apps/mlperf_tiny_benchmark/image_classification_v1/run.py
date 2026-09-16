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

"""Run the fixed MLPerf Tiny ResNet-8 HOST deployment."""

import argparse

from runtime import DEFAULT_OUTPUT_DIR, deploy, deploy_fsim_matrix


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for the two generated host libraries",
    )
    parser.add_argument(
        "--host-codegen",
        choices=("llvm", "c", "all"),
        default="llvm",
        help="host code generator to execute (all builds the ordered LLVM/C FSIM matrix)",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.host_codegen == "all":
        result = deploy_fsim_matrix(args.output_dir)
        for artifacts, execution in zip(result.artifacts, result.executions):
            print(f"{artifacts.host_codegen}-fsim partitions: {len(result.prepared.routing.symbols)}")
            print(f"{artifacts.host_codegen}-fsim compared samples: {len(execution.comparisons)}")
            print(f"{artifacts.host_codegen}-fsim profiler: {execution.profiler_stats}")
            print(f"{artifacts.host_codegen}-fsim reference bundle: {artifacts.reference.artifact_dir}")
            print(f"{artifacts.host_codegen}-fsim mixed bundle: {artifacts.mixed.artifact_dir}")
        print("MLPerf ResNet LLVM/C FSIM matrix passed")
    elif args.host_codegen == "llvm":
        # Keep the original call shape for callers that wrap the compatibility API.
        result = deploy(args.output_dir)
        print(f"Compared samples: {len(result.execution.comparisons)}")
        print(f"FSIM profiler: {result.execution.profiler_stats}")
        print("MLPerf ResNet HOST deployment passed")
    else:
        result = deploy(args.output_dir, host_codegen=args.host_codegen)
        print(f"{args.host_codegen}-fsim compared samples: {len(result.execution.comparisons)}")
        print(f"{args.host_codegen}-fsim profiler: {result.execution.profiler_stats}")
        print("MLPerf ResNet HOST deployment passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
