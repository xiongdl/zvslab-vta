#!/usr/bin/env python3
"""Validate stage-level rounding attribution for the real convolution sample."""
import argparse
import json
from pathlib import Path

import numpy as np

from fixture import load_fixture


def _read_arrays(directory: Path, mode: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = np.fromfile(directory / f"{mode}-accumulator.bin", dtype=np.int32)
    output = np.fromfile(directory / f"{mode}-output.bin", dtype=np.int8)
    expected = 32 * 32 * 16
    if accumulator.size != expected or output.size != expected:
        raise ValueError(f"{directory} {mode} output has invalid shape")
    return accumulator.reshape(1, 32, 32, 16), output.reshape(1, 32, 32, 16)


def _round_shift(value: int, shift: int, mode: str) -> tuple[int, int, int]:
    if shift == 0:
        return value, 0, 0
    divisor = 1 << shift
    quotient = value // divisor
    remainder = value - quotient * divisor
    half = divisor >> 1
    increment = remainder >= half if mode == "up" else (
        remainder > half or (remainder == half and value >= 0)
    )
    return quotient + int(increment), remainder, int(increment)


def analyze(fixture_dir: Path, fsim_dir: Path, cmsis_dir: Path) -> dict:
    fixture = load_fixture(fixture_dir)
    tflite = fixture["tflite_output"].astype(np.int16)
    multipliers = fixture["multiplier"].astype(np.int64)
    shifts = fixture["shift"].astype(np.int32)
    fsim, cmsis = {}, {}
    for mode in ("double", "single"):
        fsim[mode] = _read_arrays(fsim_dir, mode)
        cmsis[mode] = _read_arrays(cmsis_dir, mode)
        if not np.array_equal(fsim[mode][0], cmsis[mode][0]):
            raise ValueError(f"cannot attribute rounding: {mode} accumulator differs from CMSIS")
        if not np.array_equal(fsim[mode][1], cmsis[mode][1]):
            raise ValueError(f"cannot attribute rounding: {mode} output differs from CMSIS")
    if not np.array_equal(fsim["double"][0], fsim["single"][0]):
        raise ValueError("cannot attribute rounding: double and single accumulators differ")

    double_output = fsim["double"][1].astype(np.int16)
    single_output = fsim["single"][1].astype(np.int16)
    double_delta = double_output - tflite
    single_delta = single_output - tflite
    if np.max(np.abs(double_delta)) != 0:
        raise ValueError("double-rounding path did not reproduce the TFLite output")
    if np.max(np.abs(single_delta)) > 1:
        raise ValueError("single-rounding TFLite difference exceeds one quantized unit")

    accumulator = fsim["double"][0].reshape(-1, 16)
    report_differences = []
    for raw_coord in np.argwhere(single_delta != 0):
        n, y, x, channel = (int(value) for value in raw_coord)
        index = y * 32 + x
        acc_value = int(accumulator[index, channel])
        multiplier = int(multipliers[channel])
        shift = int(shifts[channel])
        product = acc_value * multiplier
        q31_double, rem31_double, inc31 = _round_shift(product, 31, "up")
        q31_single = product // (1 << 31)
        rem31_single = product - q31_single * (1 << 31)
        double_shifted, double_rem, double_inc = _round_shift(q31_double, -shift, "away")
        single_shifted, single_rem, single_inc = _round_shift(q31_single, -shift, "up")
        double_value = min(max(double_shifted - 128, -128), 0)
        single_value = min(max(single_shifted - 128, -128), 0)
        if double_value != int(double_output[n, y, x, channel]) or \
                single_value != int(single_output[n, y, x, channel]):
            raise ValueError("stage arithmetic failed to reproduce the CMSIS/FSIM output")
        report_differences.append({
            "coordinate": [n, y, x, channel], "accumulator": acc_value,
            "multiplier": multiplier, "shift": shift, "product": product,
            "double": {"rmul": q31_double, "rmul_remainder": rem31_double,
                       "rmul_increment": inc31, "rsft": double_shifted,
                       "rsft_remainder": double_rem, "rsft_increment": double_inc,
                       "output": int(double_output[n, y, x, channel])},
            "single": {"rmul": q31_single, "rmul_remainder": rem31_single,
                       "rsft": single_shifted, "rsft_remainder": single_rem,
                       "rsft_increment": single_inc,
                       "output": int(single_output[n, y, x, channel])},
            "tflite_output": int(tflite[n, y, x, channel]),
            "difference_source": "rounding strategy; matched input, weights, bias, accumulator, multiplier, shift, zero point, activation",
        })
    return {
        "cmsis_version": "8.0.0",
        "cmsis_commit": "13c97dbb6f781d4aab38ed34e6e441f42b79aff4",
        "cmsis_fsim": {
            mode: {"accumulator_differences": int(np.count_nonzero(fsim[mode][0] != cmsis[mode][0])),
                  "output_differences": int(np.count_nonzero(fsim[mode][1] != cmsis[mode][1]))}
            for mode in ("double", "single")
        },
        "tflite_comparison": {
            "double": {"difference_count": int(np.count_nonzero(double_delta)),
                       "max_absolute_difference": int(np.max(np.abs(double_delta)))},
            "single": {"difference_count": int(np.count_nonzero(single_delta)),
                       "max_absolute_difference": int(np.max(np.abs(single_delta)))},
        },
        "tflite_single_difference_attribution": report_differences,
        "attribution_basis": "Both FSIM paths match pinned CMSIS accumulators and outputs exactly. The recorded stages recompute Q31 product/remainders from each common accumulator and per-channel parameter; double matches TFLite at every output while single differs only at the listed rounding boundaries.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--fsim-dir", required=True, type=Path)
    parser.add_argument("--cmsis-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.fixture, args.fsim_dir, args.cmsis_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
