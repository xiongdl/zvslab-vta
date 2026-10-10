"""Acceptance checks for the extracted real first-convolution fixture."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from conftest import VTA_ROOT
from analyze_rounding import analyze
from fixture import check_requantize_range, load_fixture, quantize_multiplier

FIXTURE_DIR = Path(__file__).with_name("fixtures") / "resnet-first-conv-cifar0"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_fixture_has_expected_per_channel_first_convolution():
    fixture = load_fixture(FIXTURE_DIR)
    assert fixture["input"].shape == (1, 32, 32, 3)
    assert fixture["weight"].shape == (16, 3, 3, 3)
    assert fixture["bias"].shape == (16,)
    assert fixture["multiplier"].shape == (16,)
    assert fixture["shift"].shape == (16,)
    assert fixture["tflite_output"].shape == (1, 32, 32, 16)
    metadata = json.loads((FIXTURE_DIR / "metadata.json").read_text())
    assert len(metadata["quantization"]["weight_scale"]) == 16
    assert len(set(fixture["multiplier"].tolist())) > 1
    assert len(set(fixture["shift"].tolist())) > 1
    q = metadata["quantization"]
    expected = [quantize_multiplier(q["input_scale"] * scale / q["output_scale"])
                for scale in q["weight_scale"]]
    assert fixture["multiplier"].tolist() == [item[0] for item in expected]
    assert fixture["shift"].tolist() == [item[1] for item in expected]


def test_multiplier_uses_tflite_q31_scale_conversion():
    multiplier, shift = quantize_multiplier(0.5)
    assert (multiplier, shift) == (1 << 30, 0)


@pytest.mark.parametrize("scale,expected", [
    ((2**30 + 0.5) / 2**31, (1073741825, 0)),
    (2**-33, (0, 0)),
])
def test_multiplier_matches_tensorflow_215_edge_rules(scale, expected):
    # Expected values are taken from TensorFlow 2.15 QuantizeMultiplier:
    # TfLiteRound is std::round (positive ties away), and shifts below -31 flush.
    assert quantize_multiplier(scale) == expected


def test_single_rounding_rejects_positive_shift_preleft_overflow():
    values = np.array([2**30], dtype=np.int64)
    shifts = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="pre-left"):
        check_requantize_range(values, shifts, "single")


@pytest.mark.parametrize("value", [2**30, -(2**30) - 1])
def test_double_rounding_rejects_positive_shift_preleft_overflow(value):
    with pytest.raises(ValueError, match="pre-left"):
        check_requantize_range(np.array([value], dtype=np.int32),
                               np.array([1], dtype=np.int32), "double")


@pytest.mark.parametrize("mode", ["double", "single"])
@pytest.mark.parametrize("shift", [-32, 31])
def test_requantize_range_rejects_unsupported_shift(mode, shift):
    with pytest.raises(ValueError, match="shift.*-31.*30"):
        check_requantize_range(np.array([0], dtype=np.int32),
                               np.array([shift], dtype=np.int32), mode)


def test_requantize_range_rejects_int64_wrapping_preleft_input():
    # The legacy int64 multiply computes 1 * 2**65, which wraps to zero.
    with pytest.raises(ValueError, match="shift.*-31.*30"):
        check_requantize_range(np.array([1], dtype=np.int32),
                               np.array([64], dtype=np.int32), "single")


def test_requantize_range_rejects_non_int32_accumulator():
    with pytest.raises(ValueError, match="INT32"):
        check_requantize_range(np.array([2**31], dtype=np.int64),
                               np.array([0], dtype=np.int32), "single")


def test_requantize_range_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown requantize mode"):
        check_requantize_range(np.array([0], dtype=np.int32),
                               np.array([0], dtype=np.int32), "other")


def test_requantize_range_accepts_supported_int32_boundaries():
    int32_limits = np.array([-(2**31), 2**31 - 1, -1, 0], dtype=np.int64)
    check_requantize_range(int32_limits, np.array([-31, -31, 30, 30]), "single")
    positive_shift_edges = np.array([[-(2**30), 2**30 - 1]], dtype=np.int32)
    check_requantize_range(positive_shift_edges, np.array([0, 0]), "single")
    check_requantize_range(positive_shift_edges, np.array([1, 1]), "double")


def test_rounding_attribution_rejects_accumulator_mismatch(tmp_path):
    fsim_dir, cmsis_dir = tmp_path / "fsim", tmp_path / "cmsis"
    fsim_dir.mkdir()
    cmsis_dir.mkdir()
    zero_acc = np.zeros(32 * 32 * 16, dtype=np.int32)
    wrong_acc = zero_acc.copy()
    wrong_acc[0] = 1
    zero_out = np.zeros(32 * 32 * 16, dtype=np.int8)
    for mode in ("double", "single"):
        wrong_acc.tofile(fsim_dir / f"{mode}-accumulator.bin")
        zero_acc.tofile(cmsis_dir / f"{mode}-accumulator.bin")
        zero_out.tofile(fsim_dir / f"{mode}-output.bin")
        zero_out.tofile(cmsis_dir / f"{mode}-output.bin")
    with pytest.raises(ValueError, match="accumulator differs"):
        analyze(FIXTURE_DIR, fsim_dir, cmsis_dir)


def test_rounding_attribution_rejects_one_unit_from_wrong_multiplier(
        cmsis_conv_reference, tmp_path):
    fixture = load_fixture(FIXTURE_DIR)
    ref_dir, manifest = cmsis_conv_reference
    bad_fixture = {key: value.copy() for key, value in fixture.items()}
    bad_fixture["multiplier"][6] -= 150000
    wrong_acc, wrong_single_output = _run_cmsis(
        ref_dir / manifest["libraries"]["single"], bad_fixture
    )
    correct_acc, _ = _run_cmsis(
        ref_dir / manifest["libraries"]["single"], fixture
    )
    assert np.array_equal(wrong_acc, correct_acc)
    delta = wrong_single_output.astype(np.int16) - fixture["tflite_output"].astype(np.int16)
    assert delta[0, 0, 24, 6] == -1
    assert np.max(np.abs(delta)) == 1

    fsim_dir, cmsis_dir = tmp_path / "wrong-param-fsim", tmp_path / "wrong-param-cmsis"
    fsim_dir.mkdir()
    cmsis_dir.mkdir()
    double_acc, double_output = _run_cmsis(
        ref_dir / manifest["libraries"]["double"], fixture
    )
    for directory in (fsim_dir, cmsis_dir):
        double_acc.tofile(directory / "double-accumulator.bin")
        double_output.tofile(directory / "double-output.bin")
        wrong_acc.tofile(directory / "single-accumulator.bin")
        wrong_single_output.tofile(directory / "single-output.bin")
    with pytest.raises(ValueError, match="stage arithmetic"):
        analyze(FIXTURE_DIR, fsim_dir, cmsis_dir)


@pytest.fixture(scope="module")
def cmsis_conv_reference(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("cmsis_conv")
    subprocess.run([
        sys.executable, str(FIXTURE_DIR.parents[1] / "reference" / "build_reference.py"),
        "--cmsis-root", str(FIXTURE_DIR.parents[1] / "reference" / "cmsis-nn"),
        "--output-dir", str(output_dir),
    ], check=True, capture_output=True, text=True)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["version"] == "8.0.0"
    assert manifest["commit"] == "13c97dbb6f781d4aab38ed34e6e441f42b79aff4"
    return output_dir, manifest


@pytest.fixture(scope="module")
def conv_probe(tmp_path_factory):
    tvm_root = Path(os.environ["TVM_PATH"])
    build_dir = tmp_path_factory.mktemp("conv_probe")
    abi_header = build_dir / "abi_config.h"
    config = os.environ.get("VTA_CONFIG_FILE", str(VTA_ROOT / "config/vta_64mac.json"))
    cfg = [sys.executable, str(VTA_ROOT / "config/vta_config.py"), "--use-cfg=" + config]
    subprocess.run(cfg + ["--abi-header=" + str(abi_header)], check=True)
    flags = shlex.split(subprocess.check_output(cfg + ["--backend-contract", "--defs"], text=True))
    backend = os.environ.get("VTA_BACKEND", "fsim")
    if backend not in ("fsim", "tsim"):
        raise AssertionError(f"unsupported VTA_BACKEND: {backend}")
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    library = VTA_ROOT / "build" / f"libvta_{backend}{extension}"
    if not library.is_file():
        raise AssertionError(f"missing selected backend library: {library}")
    binary = build_dir / "conv_probe"
    command = [
        os.environ.get("CXX", "c++"), "-std=c++17", *flags,
        "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>", "-include", str(abi_header),
        "-I" + str(VTA_ROOT / "include"), "-I" + str(tvm_root / "include"),
        "-I" + str(tvm_root / "3rdparty/dlpack/include"),
        "-I" + str(tvm_root / "3rdparty/dmlc-core/include"),
        str(Path(__file__).with_name("conv_probe.cc")), str(library),
        "-Wl,-rpath," + str(VTA_ROOT / "build"), "-L" + str(tvm_root / "build"),
        "-ltvm", "-Wl,-rpath," + str(tvm_root / "build"), "-ldl", "-o", str(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def _run_cmsis(library_path, fixture):
    lib = ctypes.CDLL(str(library_path))
    function = lib.cmsis_conv2d
    byte_pointer = ctypes.POINTER(ctypes.c_int8)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    function.argtypes = [byte_pointer, byte_pointer, int_pointer, int_pointer,
                         int_pointer, int_pointer, byte_pointer]
    function.restype = ctypes.c_int
    arrays = [fixture[key].copy() for key in
              ("input", "weight", "bias", "multiplier", "shift")]
    input_data, weights, bias, multiplier, shift = arrays
    accumulator = np.empty((1, 32, 32, 16), dtype=np.int32)
    output = np.empty((1, 32, 32, 16), dtype=np.int8)
    as_byte = lambda value: value.ctypes.data_as(byte_pointer)
    as_int = lambda value: value.ctypes.data_as(int_pointer)
    status = function(as_byte(input_data), as_byte(weights), as_int(bias),
                      as_int(multiplier), as_int(shift), as_int(accumulator), as_byte(output))
    assert status == 0
    return accumulator, output


def test_real_per_channel_qconv2d_matches_cmsis_and_explains_tflite(
        cmsis_conv_reference, conv_probe, tmp_path):
    fixture = load_fixture(FIXTURE_DIR)
    ref_dir, manifest = cmsis_conv_reference
    backend = os.environ.get("VTA_BACKEND", "fsim")
    results = {}
    cmsis_results = {}
    analyzer_fsim = tmp_path / "analyzer-fsim"
    analyzer_cmsis = tmp_path / "analyzer-cmsis"
    analyzer_fsim.mkdir()
    analyzer_cmsis.mkdir()
    for mode in ("double", "single"):
        output_dir = tmp_path / f"{backend}-{mode}"
        output_dir.mkdir()
        debug = os.environ.get("VTA_QCONV_DEBUG")
        run = subprocess.run([str(conv_probe), "--fixture", str(FIXTURE_DIR), "--mode", mode,
                              "--output-dir", str(output_dir)], capture_output=not debug,
                             text=not debug, timeout=10 if debug else 300)
        assert run.returncode == 0, (run.stdout or "") + (run.stderr or "")
        fsim_acc = np.fromfile(output_dir / "accumulator.bin", dtype=np.int32)
        fsim_out = np.fromfile(output_dir / "output.bin", dtype=np.int8)
        assert fsim_acc.size == 32 * 32 * 16
        assert fsim_out.size == fsim_acc.size
        results[mode] = (fsim_acc.reshape(1, 32, 32, 16),
                         fsim_out.reshape(1, 32, 32, 16))
        cmsis_results[mode] = _run_cmsis(
            ref_dir / manifest["libraries"][mode], fixture
        )
        results[mode][0].tofile(analyzer_fsim / f"{mode}-accumulator.bin")
        results[mode][1].tofile(analyzer_fsim / f"{mode}-output.bin")
        cmsis_results[mode][0].tofile(analyzer_cmsis / f"{mode}-accumulator.bin")
        cmsis_results[mode][1].tofile(analyzer_cmsis / f"{mode}-output.bin")
        np.testing.assert_array_equal(results[mode][0], cmsis_results[mode][0])
        np.testing.assert_array_equal(results[mode][1], cmsis_results[mode][1])
    # The actual TFLite output selects the default double-rounding path. The
    # 13 single-rounding deltas are independently checked by analyze_rounding.py.
    np.testing.assert_array_equal(results["double"][1], fixture["tflite_output"])
    single_delta = results["single"][1].astype(np.int16) - fixture["tflite_output"].astype(np.int16)
    assert np.count_nonzero(single_delta) == 13
    assert np.max(np.abs(single_delta)) == 1
    report = analyze(FIXTURE_DIR, analyzer_fsim, analyzer_cmsis)
    assert report["cmsis_fsim"]["double"] == {
        "accumulator_differences": 0, "output_differences": 0}
    assert report["cmsis_fsim"]["single"] == {
        "accumulator_differences": 0, "output_differences": 0}
    assert report["tflite_comparison"]["double"] == {
        "difference_count": 0, "max_absolute_difference": 0}
    assert report["tflite_comparison"]["single"] == {
        "difference_count": 13, "max_absolute_difference": 1}
    assert len(report["tflite_single_difference_attribution"]) == 13
    report_dir_text = os.environ.get("VTA_QCONV_REPORT_DIR")
    if report_dir_text:
        backend_report_dir = Path(report_dir_text) / backend
        conv_report_dir = backend_report_dir / "conv"
        cmsis_report_dir = backend_report_dir / "cmsis"
        conv_report_dir.mkdir(parents=True, exist_ok=True)
        cmsis_report_dir.mkdir(parents=True, exist_ok=True)
        for mode in ("double", "single"):
            results[mode][0].tofile(conv_report_dir / f"{mode}-accumulator.bin")
            results[mode][1].tofile(conv_report_dir / f"{mode}-output.bin")
            cmsis_results[mode][0].tofile(cmsis_report_dir / f"{mode}-accumulator.bin")
            cmsis_results[mode][1].tofile(cmsis_report_dir / f"{mode}-output.bin")
        fixture_metadata = json.loads((FIXTURE_DIR / "metadata.json").read_text(encoding="utf-8"))
        fixture_hash = fixture_metadata["fixture_sha256"]
        instruction_hash = hashlib.sha256(json.dumps({
            "fixture_sha256": fixture_hash,
            "geometry": {"tile": 64, "channels": 16, "taps": 9},
            "logical_instruction_contract": "qconv2d-gemm-rmul-rsft-out-v1",
            "rounding_modes": ["double", "single"],
        }, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
        report.update({
            "backend": backend,
            "fixture_sha256": fixture_hash,
            "instruction_sha256": instruction_hash,
            "cmsis_fsim_mismatch_count": 0 if backend == "fsim" else None,
            "tflite_fsim_mismatch_count": report["tflite_comparison"]["double"]["difference_count"],
            "tflite_max_abs_diff": report["tflite_comparison"]["double"]["max_absolute_difference"],
            "tflite_single_fsim_mismatch_count": report["tflite_comparison"]["single"]["difference_count"],
            "tflite_max_abs_diff_by_rounding_mode": {
                mode: report["tflite_comparison"][mode]["max_absolute_difference"]
                for mode in ("double", "single")
            },
            "rounding_evidence": report["tflite_single_difference_attribution"],
        })
        if backend == "tsim":
            fsim_dir = Path(report_dir_text) / "fsim"
            fsim_report = json.loads((fsim_dir / "conv" / "qconv2d-rounding.json").read_text(
                encoding="utf-8"))
            assert fsim_report["fixture_sha256"] == fixture_hash
            assert fsim_report["instruction_sha256"] == instruction_hash
            parity_differences = 0
            for mode in ("double", "single"):
                for tensor_name, actual in (("accumulator", results[mode][0]),
                                             ("output", results[mode][1])):
                    baseline = np.fromfile(fsim_dir / "conv" / f"{mode}-{tensor_name}.bin",
                                            dtype=actual.dtype).reshape(actual.shape)
                    differences = int(np.count_nonzero(actual != baseline))
                    parity_differences += differences
                    np.testing.assert_array_equal(actual, baseline)
            report["fsim_tsim_mismatch_count"] = parity_differences
            report["cmsis_fsim_mismatch_count"] = fsim_report["cmsis_fsim_mismatch_count"]
            report["tflite_fsim_mismatch_count"] = fsim_report["tflite_fsim_mismatch_count"]
        (conv_report_dir / "qconv2d-rounding.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if backend == "tsim":
            alu_report = json.loads((backend_report_dir / "alu" / "requantize.json").read_text(
                encoding="utf-8"))
            vta_path = Path(os.environ["VTA_PATH"])
            library_suffix = "dylib" if sys.platform == "darwin" else "so"
            selected_library = vta_path / "build" / f"libvta_{backend}.{library_suffix}"
            fsim_library = vta_path / "build" / f"libvta_fsim.{library_suffix}"
            hardware_library = vta_path / "build" / f"libvta_hw.{library_suffix}"
            config_path = Path(os.environ["VTA_CONFIG_FILE"])
            import tvm
            (backend_report_dir / "task5-validation.json").write_text(json.dumps({
                "cmsis_fsim_mismatch_count": report["cmsis_fsim_mismatch_count"],
                "fsim_tsim_mismatch_count": (
                    report["fsim_tsim_mismatch_count"] +
                    alu_report["fsim_tsim_mismatch_count"]),
                "tflite_fsim_mismatch_count": report["tflite_fsim_mismatch_count"],
                "tflite_max_abs_diff": report["tflite_max_abs_diff"],
                "tflite_single_fsim_mismatch_count": report[
                    "tflite_single_fsim_mismatch_count"],
                "tflite_max_abs_diff_by_rounding_mode": report[
                    "tflite_max_abs_diff_by_rounding_mode"],
                "fixture_sha256": fixture_hash,
                "instruction_sha256": instruction_hash,
                "rounding_evidence": report["rounding_evidence"],
                "source_runtime_build": {
                    "cmsis_nn": "8.0.0 @ 13c97dbb6f781d4aab38ed34e6e441f42b79aff4",
                    "model_sha256": fixture_metadata["model_sha256"],
                    "input_sha256": fixture_metadata["input_sha256"],
                    "fixture_source": {
                        "model": fixture_metadata["model"],
                        "input": fixture_metadata["input_source"],
                    },
                    "tvm_version": tvm.__version__,
                    "tvm_path": os.environ["TVM_PATH"],
                    "vta_path": str(vta_path),
                    "vta_config": str(config_path),
                    "vta_config_sha256": _sha256_file(config_path),
                    "vta_backend": "tsim",
                    "backend_library": str(selected_library),
                    "backend_library_sha256": _sha256_file(selected_library),
                    "fsim_backend_library": str(fsim_library),
                    "fsim_backend_library_sha256": _sha256_file(fsim_library),
                    "hardware_library": str(hardware_library),
                    "hardware_library_sha256": _sha256_file(hardware_library),
                },
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
