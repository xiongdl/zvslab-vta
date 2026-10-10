"""Shared loading and range checks for the pinned first-convolution sample."""
from pathlib import Path

import numpy as np


def quantize_multiplier(scale: float) -> tuple[int, int]:
    """Convert a positive real scale to TFLite's signed Q31 multiplier/shift."""
    if not np.isfinite(scale) or scale < 0:
        raise ValueError(f"scale must be finite and nonnegative: {scale}")
    if scale == 0:
        return 0, 0
    significand, shift = np.frexp(float(scale))
    multiplier = int(np.rint(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1
    return multiplier, int(shift)


def load_fixture(path: Path) -> dict:
    path = Path(path)
    with np.load(path / "fixture.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def check_requantize_range(acc: np.ndarray, shifts: np.ndarray, mode: str) -> None:
    """Reject values that violate the CMSIS scalar single-rounding pre-left range."""
    acc = np.asarray(acc, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    if mode not in ("double", "single"):
        raise ValueError(f"unknown requantize mode: {mode}")
    if acc.shape[-1] != shifts.shape[-1]:
        raise ValueError("accumulator and per-channel shifts must have matching final axes")
    if mode == "double":
        return
    for channel, shift in enumerate(shifts.reshape(-1)):
        if shift >= 0:
            scaled = acc[..., channel] * (1 << (int(shift) + 1))
            if np.any((scaled < -(1 << 31)) | (scaled > (1 << 31) - 1)):
                raise ValueError(
                    f"single-rounding pre-left overflows INT32 in channel {channel} "
                    f"for shift {int(shift)}"
                )
