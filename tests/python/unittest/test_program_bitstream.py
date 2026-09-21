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
"""Regression tests for canonical bitstream backend names."""

import importlib.util
import sys
from pathlib import Path

import pytest


PROGRAM_PATH = Path(__file__).resolve().parents[3] / "python" / "vta" / "program_bitstream.py"


def _load_program_bitstream():
    spec = importlib.util.spec_from_file_location("program_bitstream_test_module", PROGRAM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_fsim_and_rejects_legacy_sim(tmp_path, monkeypatch):
    module = _load_program_bitstream()
    bitstream = tmp_path / "bitstream.bin"
    bitstream.write_bytes(b"test")
    calls = []
    monkeypatch.setattr(module, "bitstream_program", lambda target, path: calls.append((target, path)))

    monkeypatch.setattr(sys, "argv", ["program_bitstream.py", "fsim", str(bitstream)])
    module.main()
    assert calls == [("fsim", str(bitstream))]

    monkeypatch.setattr(sys, "argv", ["program_bitstream.py", "sim", str(bitstream)])
    with pytest.raises(RuntimeError, match="Unknown target sim"):
        module.main()
