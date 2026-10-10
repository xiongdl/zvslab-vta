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
"""Host-side coverage for Xilinx raw instruction prevalidation."""

import os
from pathlib import Path
import shlex
import subprocess
import sys


def test_pynq_raw_alu_guard(tmp_path):
    root = Path(__file__).resolve().parents[3]
    config = os.environ.get("VTA_CONFIG_FILE", str(root / "config/vta_64mac.json"))
    cfg = [sys.executable, str(root / "config/vta_config.py"), "--use-cfg=" + config]
    abi_header = tmp_path / "abi_config.h"
    subprocess.run(cfg + ["--abi-header=" + str(abi_header)], check=True)
    flags = shlex.split(subprocess.check_output(cfg + ["--backend-contract", "--defs"], text=True))
    binary = tmp_path / "pynq_alu_guard_probe"
    subprocess.run([
        os.environ.get("CXX", "c++"), "-std=c++17", *flags,
        "-include", str(abi_header), "-I" + str(root / "include"),
        "-I" + str(root / "src/pynq"),
        str(Path(__file__).with_name("pynq_alu_guard_probe.cc")), "-o", str(binary),
    ], check=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
    assert "legacy ALU, RMUL, rounding, CMA bounds" in result.stdout
