"""Real runtime instruction encoding and cache identity checks."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
from conftest import VTA_ROOT


@pytest.fixture(scope="module")
def runtime_probe(tmp_path_factory):
    tvm_root = Path(os.environ["TVM_PATH"])
    build_dir = tmp_path_factory.mktemp("alu_runtime")
    abi_header = build_dir / "abi_config.h"
    config = os.environ["VTA_CONFIG_FILE"]
    cfg = [sys.executable, str(VTA_ROOT / "config" / "vta_config.py"), "--use-cfg=" + config]
    subprocess.run(cfg + ["--abi-header=" + str(abi_header)], check=True)
    flags = shlex.split(subprocess.check_output(cfg + ["--backend-contract", "--defs"], text=True))
    binary = build_dir / "runtime_probe"
    command = [
        os.environ.get("CXX", "c++"), "-std=c++17", *flags,
        "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>", "-include", str(abi_header),
        "-I" + str(VTA_ROOT / "include"), "-I" + str(tvm_root / "include"),
        "-I" + str(tvm_root / "3rdparty/dlpack/include"),
        "-I" + str(tvm_root / "3rdparty/dmlc-core/include"),
        str(VTA_ROOT / "src/runtime/runtime.cc"), str(VTA_ROOT / "src/sim/sim_tlpp.cc"),
        str(Path(__file__).with_name("runtime_probe.cc")),
        "-L" + str(tvm_root / "build"), "-ltvm", "-Wl,-rpath," + str(tvm_root / "build"),
        "-o", str(binary),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    return binary


def invoke(probe, kind, opcode=5, rounding=0, slot=4):
    return subprocess.run([str(probe), kind, str(opcode), str(rounding), str(slot)],
                          check=False, capture_output=True, text=True, timeout=15)


def alu_rows(result):
    return [[int(value) for value in line.split()] for line in result.stdout.splitlines()
            if line.strip()]


def test_old_uop_api_keeps_rounding_zero(runtime_probe):
    result = invoke(runtime_probe, "legacy", opcode=5)
    assert result.returncode == 0, result.stderr
    rows = alu_rows(result)
    assert [(row[2], row[3], row[4]) for row in rows] == [(5, 0, 37)]


def test_rounding_is_part_of_cache_key_for_same_caller_signature(runtime_probe):
    result = invoke(runtime_probe, "cache", opcode=5, slot=13)
    assert result.returncode == 0, result.stderr
    assert [(row[2], row[3], row[4]) for row in alu_rows(result)] == [(5, 1, 37), (5, 2, 37)]


def test_opcode_is_part_of_cache_key_for_same_signature_and_rounding(runtime_probe):
    result = invoke(runtime_probe, "cache-opcode", slot=13)
    assert result.returncode == 0, result.stderr
    assert [(row[2], row[3], row[4]) for row in alu_rows(result)] == [(5, 0, 37), (6, 0, 37)]


@pytest.mark.parametrize("kind,opcode,rounding", [
    ("new", 5, 3),
    ("legacy-mismatch", 2, 1),
    ("expected-mismatch", 5, 1),
    ("opcode-mismatch", 6, 0),
    ("mixed", 5, 1),
])
def test_invalid_rounding_or_kernel_mismatch_fails(runtime_probe, kind, opcode, rounding):
    result = invoke(runtime_probe, kind, opcode, rounding)
    assert result.returncode != 0
    assert "Check failed" in result.stderr or "Check failed" in result.stdout
    if kind == "opcode-mismatch":
        assert "ALU initializer opcode does not match expected opcode" in result.stderr
