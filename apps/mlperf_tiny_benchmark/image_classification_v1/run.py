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

"""Compile and run one float ResNet-8 target using one RGB image."""

import argparse
import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_ROOT / "model" / "pretrainedResnet.tflite"
DEFAULT_INPUT = APP_ROOT / "samples" / "00-airplane.png"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build"


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="float ResNet-8 TFLite model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="one 32x32 RGB PNG or JPEG image")
    parser.add_argument(
        "--target", choices=("c", "llvm", "vta,c", "vta,llvm"), default="vta,llvm",
        help="VTA partition followed by CPU fallback, or CPU only",
    )
    parser.add_argument("--simulator", choices=("fsim", "tsim"), default="fsim")
    parser.add_argument("--schedule", type=Path, help="selected AutoTVM schedule log and same-stem metadata")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deployment-report", type=Path, help="write a Markdown deployment report")
    parser.add_argument("--export-workloads", type=Path, help="export actual VTA computations and activations")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.export_workloads is not None and not args.target.startswith("vta,"):
        raise ValueError("--export-workloads requires a target that includes VTA")
    if args.target.startswith("vta,") and not os.environ.get("VTA_BACKEND"):
        raise RuntimeError("VTA target requires VTA_BACKEND to match --simulator")
    if args.target.startswith("vta,") and os.environ["VTA_BACKEND"] != args.simulator:
        raise RuntimeError("VTA_BACKEND must match --simulator")

    # Keep CPU startup independent from VTA and simulator package initialization.
    import runtime

    result = runtime.run_selected(
        target=args.target,
        simulator=args.simulator,
        schedule=args.schedule,
        output_dir=args.output_dir,
        model_path=args.model,
        input_path=args.input,
        export_workloads=args.export_workloads,
    )
    if args.deployment_report is not None:
        runtime.write_deployment_report(result, args.deployment_report)
        print(f"Deployment report: {args.deployment_report.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
