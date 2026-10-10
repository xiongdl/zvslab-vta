"""Real-driver checks for the CMSIS-NN aligned ALU operations."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import struct
import subprocess
import sys

import pytest

from conftest import VTA_ROOT

REFERENCE_ROOT = Path(__file__).with_name("reference")
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
OP_SHIFT = 3
OP_MUL = 4
OP_RMUL = 5
OP_RSFT = 6
ROUND_NONE = 0
ROUND_UP = 1
ROUND_AWAY = 2


@pytest.fixture(scope="module")
def cmsis_reference(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("cmsis_reference")
    build = subprocess.run(
        [sys.executable, str(REFERENCE_ROOT / "build_reference.py"),
         "--cmsis-root", str(REFERENCE_ROOT / "cmsis-nn"),
         "--output-dir", str(output_dir)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    functions = {}
    for mode in ("double", "single"):
        library = ctypes.CDLL(str(output_dir / manifest["libraries"][mode]))
        function = library.cmsis_requantize
        function.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        function.restype = ctypes.c_int32
        functions[mode] = function
    return manifest, functions


@pytest.fixture(scope="module")
def alu_probe(tmp_path_factory):
    tvm_root = Path(os.environ["TVM_PATH"])
    build_dir = tmp_path_factory.mktemp("alu_driver")
    abi_header = build_dir / "abi_config.h"
    config = os.environ.get("VTA_CONFIG_FILE", str(VTA_ROOT / "config/vta_64mac.json"))
    cfg = [sys.executable, str(VTA_ROOT / "config/vta_config.py"), "--use-cfg=" + config]
    subprocess.run(cfg + ["--abi-header=" + str(abi_header)], check=True)
    flags = shlex.split(subprocess.check_output(cfg + ["--backend-contract", "--defs"], text=True))
    binary = build_dir / "alu_probe"
    backend = os.environ.get("VTA_BACKEND", "fsim")
    if backend not in ("fsim", "tsim"):
        raise AssertionError(f"unsupported VTA_BACKEND for ALU probe: {backend}")
    library_extension = ".dylib" if sys.platform == "darwin" else ".so"
    vta_library = VTA_ROOT / "build" / f"libvta_{backend}{library_extension}"
    if not vta_library.is_file():
        raise AssertionError(
            f"VTA_BACKEND={backend} selected library is unavailable: {vta_library}; "
            f"build it with scripts/build_vta_lib.sh --backend {backend}"
        )
    tvm_library_dir = tvm_root / "build"
    command = [
        os.environ.get("CXX", "c++"), "-std=c++17", *flags,
        "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>", "-include", str(abi_header),
        "-I" + str(VTA_ROOT / "include"), "-I" + str(tvm_root / "include"),
        "-I" + str(tvm_root / "3rdparty/dlpack/include"),
        "-I" + str(tvm_root / "3rdparty/dmlc-core/include"),
        str(Path(__file__).with_name("alu_probe.cc")),
        str(vta_library), "-Wl,-rpath," + str(VTA_ROOT / "build"),
        "-L" + str(tvm_library_dir), "-ltvm", "-Wl,-rpath," + str(tvm_library_dir),
        "-ldl", "-o", str(binary),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return binary


def write_probe_input(path, stages, cases):
    """Write V1 input consumed by alu_probe.cc; each case is x then one operand per stage."""
    rows = ["VTA_ALU_PROBE_V1", f"{len(cases)} {len(stages)}"]
    rows.extend("{} {} {} {}".format(*stage) for stage in stages)
    rows.extend("{} {}".format(case[0], " ".join(str(value) for value in case[1:]))
                for case in cases)
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def run_probe(probe, work_dir, name, stages, cases):
    input_path = work_dir / f"{name}.in"
    output_path = work_dir / f"{name}.out"
    write_probe_input(input_path, stages, cases)
    result = subprocess.run([str(probe), str(input_path), str(output_path)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = [int(value) for value in output_path.read_text(encoding="ascii").split()]
    assert len(actual) == len(cases)
    return actual


def _requantize_stages(mode, cases):
    """Lower one sign group to the public RMUL/RSFT/SHIFT instruction sequence."""
    if not cases:
        return None
    is_single = mode == "single"
    left = cases[0][2] >= 0
    assert all((case[2] >= 0) == left for case in cases)
    if left:
        preleft_bits = 1 if is_single else 0
        for value, _, shift in cases:
            preleft = value * (1 << (shift + preleft_bits))
            if not INT32_MIN <= preleft <= INT32_MAX:
                raise ValueError(
                    f"CMSIS {mode} positive-shift pre-left overflows INT32: "
                    f"x={value}, shift={shift}, pre-left={preleft}"
                )
        stages = [(OP_SHIFT, ROUND_NONE, 0, 0),
                  (OP_RMUL, ROUND_NONE if is_single else ROUND_UP, 0, 0)]
        if is_single:
            stages[0] = (OP_SHIFT, ROUND_NONE, 0, 0)
            stages.append((OP_RSFT, ROUND_UP, 1, 1))
            return stages, [(x, -(s + 1), m, 1) for x, m, s in cases]
        return stages, [(x, -s, m) for x, m, s in cases]
    stages = [(OP_RMUL, ROUND_NONE if is_single else ROUND_UP, 0, 0),
              (OP_RSFT, ROUND_UP if is_single else ROUND_AWAY, 0, 0)]
    return stages, [(x, m, -s) for x, m, s in cases]


def run_requantize_group(probe, work_dir, name, mode, cases):
    if not cases:
        return []
    stages, transformed = _requantize_stages(mode, cases)
    actual = run_probe(probe, work_dir, name, stages, transformed)
    return actual


def test_alu_probe_binary_links_requested_vta_backend(alu_probe):
    backend = os.environ.get("VTA_BACKEND", "fsim")
    if sys.platform == "darwin":
        inspect_command = ["otool", "-L", str(alu_probe)]
    elif sys.platform.startswith("linux"):
        inspect_command = ["ldd", str(alu_probe)]
    else:
        pytest.skip(f"native shared-library inspection is unsupported on {sys.platform}")
    inspected = subprocess.run(inspect_command, check=True, capture_output=True, text=True)
    dependencies = inspected.stdout
    assert f"libvta_{backend}" in dependencies
    other_backend = "tsim" if backend == "fsim" else "fsim"
    assert f"libvta_{other_backend}" not in dependencies


def test_probe_rejects_wrong_backend_abi(alu_probe):
    result = subprocess.run([str(alu_probe), "--check-wrong-abi"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "backend ABI mismatch" in result.stderr


def assert_cmsis_equal(probe, function, mode, work_dir, name, cases):
    actual = [None] * len(cases)
    for sign_group, selected in (("right", [(i, case) for i, case in enumerate(cases) if case[2] < 0]),
                                 ("left", [(i, case) for i, case in enumerate(cases) if case[2] >= 0])):
        if not selected:
            continue
        indices, values = zip(*selected)
        outputs = run_requantize_group(probe, work_dir, f"{name}-{mode}-{sign_group}", mode, values)
        for index, value in zip(indices, outputs):
            actual[index] = value
    expected = [function(x, m, s) for x, m, s in cases]
    assert actual == expected, next(
        ((i, cases[i], got, expected[i]) for i, got in enumerate(actual) if got != expected[i]), None
    )
    return actual


def test_pinned_cmsis_reference_covers_both_rounding_modes(cmsis_reference):
    manifest, functions = cmsis_reference
    assert manifest["version"] == "8.0.0"
    assert manifest["commit"] == "13c97dbb6f781d4aab38ed34e6e441f42b79aff4"
    assert manifest["build_macros"]["double"] == []
    assert manifest["build_macros"]["single"] == ["CMSIS_NN_USE_SINGLE_ROUNDING"]
    assert functions["double"](1, 1 << 30, -1) == 1
    assert functions["single"](1, 1 << 30, -1) == 0
    # The pre-v8 local helper narrowed this intermediate to int32 before +1.
    x, multiplier, shift = INT32_MIN, INT32_MIN + 1, -1
    narrowed = (x * multiplier) >> 31
    old_int32 = ((narrowed + 1 + (1 << 31)) % (1 << 32)) - (1 << 31)
    old_host_result = old_int32 >> 1
    assert old_host_result == -1073741824
    assert functions["single"](x, multiplier, shift) == 1073741824


def test_rounding_edges_and_int32_boundaries_match_cmsis(cmsis_reference, alu_probe, tmp_path):
    _, functions = cmsis_reference
    cases = [
        (1, 1 << 30, -1), (-1, 1 << 30, -1),
        (1, (1 << 30) - 1, -1), (1, (1 << 30) + 1, -1),
        (-1, (1 << 30) - 1, -1), (-1, (1 << 30) + 1, -1),
        (3, 1 << 30, -1), (-3, 1 << 30, -1),
        (INT32_MIN, INT32_MAX, -1), (INT32_MAX, INT32_MAX, -1),
        (INT32_MIN, INT32_MAX, -31), (INT32_MAX, INT32_MAX, -31),
        (0, INT32_MAX, -1), (INT32_MIN, 0, -31), (INT32_MAX, 0, 0),
        (INT32_MIN, 1 << 30, 0), (INT32_MAX, 1 << 30, 0),
        (1, INT32_MAX, 0), (-1, INT32_MAX, 0),
        (-1073741824, INT32_MAX, -1), (1073741823, INT32_MAX, -1),
        (-1, INT32_MAX, -31), (1, INT32_MAX, -31),
    ]
    default = assert_cmsis_equal(alu_probe, functions["double"], "double", tmp_path, "edges", cases)
    single_cases = [
        case for case in cases
        if case[2] < 0 or INT32_MIN <= case[0] * (1 << (case[2] + 1)) <= INT32_MAX
    ]
    single = assert_cmsis_equal(
        alu_probe, functions["single"], "single", tmp_path, "edges", single_cases
    )
    assert default[0] == 1 and single[0] == 0


def test_100k_fixed_seed_random_cases_match_both_cmsis_paths(
        cmsis_reference, alu_probe, tmp_path):
    _, functions = cmsis_reference
    rng = random.Random(20261010)
    cases = []
    for _ in range(100_000):
        shift = rng.randint(-31, 30)
        multiplier = rng.randint(0, INT32_MAX)
        if shift < 0:
            value = rng.randint(INT32_MIN, INT32_MAX)
        else:
            scale = 1 << (shift + 1)
            low = -((1 << 31) // scale)
            high = INT32_MAX // scale
            value = rng.randint(low, high)
        cases.append((value, multiplier, shift))
    results = {}
    for mode in ("double", "single"):
        results[mode] = assert_cmsis_equal(
            alu_probe, functions[mode], mode, tmp_path, "random", cases)

    report_root_text = os.environ.get("VTA_QCONV_REPORT_DIR")
    if report_root_text:
        backend = os.environ.get("VTA_BACKEND", "fsim")
        report_dir = Path(report_root_text) / backend / "alu"
        report_dir.mkdir(parents=True, exist_ok=True)
        fixture_bytes = b"".join(struct.pack("<iii", *case) for case in cases)
        fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
        report = {
            "backend": backend,
            "case_count_per_mode": len(cases),
            "fixture_sha256": fixture_hash,
            "seed": 20261010,
            "cmsis_mismatch_count": 0,
            "rounding_modes": {},
        }
        for mode, values in results.items():
            payload = struct.pack("<" + "i" * len(values), *values)
            (report_dir / f"{mode}-int32.bin").write_bytes(payload)
            logical_stream = json.dumps({
                "fixture_sha256": fixture_hash,
                "mode": mode,
                "stage_contract": "public-vta-rmul-rsft-lowering-v1",
            }, sort_keys=True, separators=(",", ":")).encode("ascii")
            report["rounding_modes"][mode] = {
                "instruction_sha256": hashlib.sha256(logical_stream).hexdigest(),
                "cmsis_mismatch_count": 0,
            }
        if backend == "tsim":
            fsim_dir = Path(report_root_text) / "fsim" / "alu"
            fsim_report = json.loads((fsim_dir / "requantize.json").read_text(encoding="utf-8"))
            assert fsim_report["fixture_sha256"] == fixture_hash
            mismatch_count = 0
            for mode, values in results.items():
                baseline = list(struct.unpack("<" + "i" * len(values),
                                              (fsim_dir / f"{mode}-int32.bin").read_bytes()))
                mismatch_count += sum(actual != expected
                                      for actual, expected in zip(values, baseline))
                assert values == baseline, f"FSIM/TSIM {mode} ALU INT32 results differ"
                assert report["rounding_modes"][mode]["instruction_sha256"] == \
                    fsim_report["rounding_modes"][mode]["instruction_sha256"]
            report["fsim_tsim_mismatch_count"] = mismatch_count
        (report_dir / "requantize.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_regular_mul_wraps_and_supports_immediate_and_register_operands(alu_probe, tmp_path):
    values = [INT32_MIN, INT32_MAX, -123456789, -1, 0, 1, 123456789]
    immediate = -32768
    stages = [(OP_MUL, ROUND_NONE, 1, immediate)]
    cases = [(value, 0) for value in values]
    actual = run_probe(alu_probe, tmp_path, "mul-immediate", stages, cases)
    expected = [((value * immediate + (1 << 31)) % (1 << 32)) - (1 << 31) for value in values]
    assert actual == expected

    operands = [INT32_MIN, -3, -1, 0, 1, 7, INT32_MAX]
    stages = [(OP_MUL, ROUND_NONE, 0, 0)]
    cases = list(zip(values, operands))
    actual = run_probe(alu_probe, tmp_path, "mul-register", stages, cases)
    expected = [((x * y + (1 << 31)) % (1 << 32)) - (1 << 31) for x, y in cases]
    assert actual == expected


def test_rmul_is_not_saturating_and_rsft_rounds_signed_ties(alu_probe, tmp_path):
    stages = [(OP_RMUL, ROUND_NONE, 0, 0)]
    actual = run_probe(alu_probe, tmp_path, "rmul-no-sat", stages, [(INT32_MIN, INT32_MIN)])
    assert actual == [INT32_MIN]

    values = [5, 7, -5, -7, 0, INT32_MIN, INT32_MAX]
    for rounding in (ROUND_NONE, ROUND_UP, ROUND_AWAY):
        stages = [(OP_RSFT, rounding, 0, 0)]
        cases = list(zip(values, [1] * len(values)))
        actual = run_probe(alu_probe, tmp_path, f"rsft-round-{rounding}", stages, cases)
        expected = []
        for value, shift in cases:
            if shift == 0:
                expected.append(value)
                continue
            divisor = 1 << shift
            quotient = value // divisor
            remainder = value - quotient * divisor
            half = divisor >> 1
            if rounding == ROUND_UP and remainder >= half:
                quotient += 1
            elif rounding == ROUND_AWAY and (remainder > half or (remainder == half and value >= 0)):
                quotient += 1
            expected.append(quotient)
        assert actual == expected


def test_rmul_and_rsft_support_immediate_and_per_lane_register_operands(alu_probe, tmp_path):
    values = [INT32_MIN, -1073741825, -5, -1, 0, 1, 5, 1073741825, INT32_MAX]
    # VTA's immediate field is signed 16-bit; large Q31 multipliers use the
    # per-lane register path exercised by the randomized CMSIS comparisons.
    multiplier = 16385
    for rounding in (ROUND_NONE, ROUND_UP, ROUND_AWAY):
        immediate = run_probe(
            alu_probe, tmp_path, f"rmul-immediate-{rounding}",
            [(OP_RMUL, rounding, 1, multiplier)], [(value, 0) for value in values])
        register = run_probe(
            alu_probe, tmp_path, f"rmul-register-{rounding}",
            [(OP_RMUL, rounding, 0, 0)], [(value, multiplier) for value in values])
        assert immediate == register

        immediate = run_probe(
            alu_probe, tmp_path, f"rsft-immediate-{rounding}",
            [(OP_RSFT, rounding, 1, 3)], [(value, 0) for value in values])
        register = run_probe(
            alu_probe, tmp_path, f"rsft-register-{rounding}",
            [(OP_RSFT, rounding, 0, 0)], [(value, 3) for value in values])
        assert immediate == register


def test_shift_keeps_signed_left_and_right_semantics_for_imm_and_register(alu_probe, tmp_path):
    values = [INT32_MIN, INT32_MAX, -7, -1, 0, 1, 7]
    shifts = [-31, -2, -1, 0, 1, 2, 31]
    stages = [(OP_SHIFT, ROUND_NONE, 0, 0)]
    actual = run_probe(alu_probe, tmp_path, "shift-register", stages, list(zip(values, shifts)))
    expected = []
    for value, shift in zip(values, shifts):
        if shift >= 0:
            expected.append(value >> shift)
        else:
            expected.append(((value << -shift) + (1 << 31)) % (1 << 32) - (1 << 31))
    assert actual == expected

    for shift in (-31, -1, 0, 1, 31):
        stages = [(OP_SHIFT, ROUND_NONE, 1, shift)]
        cases = [(value, 0) for value in values]
        actual = run_probe(alu_probe, tmp_path, f"shift-immediate-{shift}", stages, cases)
        if shift >= 0:
            expected = [value >> shift for value in values]
        else:
            expected = [((value << -shift) + (1 << 31)) % (1 << 32) - (1 << 31) for value in values]
        assert actual == expected


@pytest.mark.parametrize("opcode,rounding,operand", [
    (OP_MUL, ROUND_UP, 1), (OP_SHIFT, ROUND_AWAY, 1),
    (OP_RMUL, 3, 1), (OP_RSFT, 3, 1),
    (OP_RSFT, ROUND_UP, -1), (OP_RSFT, ROUND_AWAY, 32),
])
def test_illegal_rounding_and_rsft_operands_are_rejected(alu_probe, tmp_path, opcode, rounding, operand):
    stage = (opcode, rounding, 0, 0)
    request = tmp_path / f"invalid-{opcode}-{rounding}-{operand}.in"
    output = tmp_path / f"invalid-{opcode}-{rounding}-{operand}.out"
    write_probe_input(request, [stage], [(1, operand)])
    result = subprocess.run([str(alu_probe), str(request), str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert result.returncode not in (-11, -6, -9), (result.stdout, result.stderr)
    diagnostic = result.stdout + result.stderr
    assert any(marker in diagnostic.lower() for marker in
               ("check failed", "invalid", "unsupported", "illegal", "rejected", "rounding", "rsft")), diagnostic


def test_single_rounding_positive_shift_rejects_out_of_range_preleft(alu_probe, tmp_path):
    with pytest.raises(ValueError, match="pre-left overflows INT32"):
        _requantize_stages("single", [(INT32_MAX, INT32_MAX, 0)])
    # The largest positive shift is legal for x=0 or x=-1 only.
    _requantize_stages("single", [(0, INT32_MAX, 30), (-1, INT32_MAX, 30)])


def test_probe_rejects_immediate_outside_signed_instruction_field(alu_probe, tmp_path):
    request = tmp_path / "immediate-overflow.in"
    output = tmp_path / "immediate-overflow.out"
    write_probe_input(request, [(OP_RMUL, ROUND_NONE, 1, 32768)], [(1, 0)])
    result = subprocess.run([str(alu_probe), str(request), str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "signed 16-bit" in result.stderr
