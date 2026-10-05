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

"""Tune exported streaming wakeword workloads in separate FSIM and TSIM stages."""

import argparse
import math
import os
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--workload", type=int, default=-1,
                        help="-1 selects every exported VTA occurrence; otherwise select one index")
    parser.add_argument("--simulator", choices=("fsim", "tsim"), default="fsim")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--trial-batch", type=int)
    parser.add_argument("--min-successful", type=int)
    parser.add_argument("--input-logs", type=Path)
    parser.add_argument("--output-logs", type=Path, required=True)
    return parser


def validate_args(args):
    if args.workload < -1:
        raise ValueError("--workload must be -1 or a non-negative occurrence index")
    if args.timeout is None:
        args.timeout = 60 if args.simulator == "fsim" else 120
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite positive seconds")
    if args.simulator == "fsim":
        if args.input_logs is not None:
            raise ValueError("--input-logs is only valid for TSIM")
        args.trial_batch = 100 if args.trial_batch is None else args.trial_batch
        args.min_successful = 20 if args.min_successful is None else args.min_successful
        if args.trial_batch <= 0 or args.min_successful <= 0:
            raise ValueError("--trial-batch and --min-successful must be positive")
    else:
        if args.input_logs is None:
            raise ValueError("TSIM requires --input-logs from a successful FSIM stage")
        if args.trial_batch is not None or args.min_successful is not None:
            raise ValueError("--trial-batch and --min-successful are only valid for FSIM")
    return args



def main(argv=None):
    args = validate_args(_parser().parse_args(argv))
    selected = os.environ.get("VTA_BACKEND")
    if selected != args.simulator:
        raise ValueError(
            f"VTA_BACKEND={selected!r} must match --simulator {args.simulator}"
        )
    config = os.environ.get("VTA_CONFIG_FILE")
    if not config or not Path(config).is_absolute() or not Path(config).is_file():
        raise ValueError("VTA tuning requires an absolute existing VTA_CONFIG_FILE")
    from python.tuning import run

    return run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"tune.py: {error}") from error
