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
    # TensorFlow 2.15 TfLiteRound delegates to std::round, so positive ties
    # round away from zero rather than to even.
    multiplier = int(np.floor(significand * (1 << 31) + 0.5))
    if multiplier == (1 << 31):
        multiplier //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return multiplier, int(shift)


def load_fixture(path: Path) -> dict:
    path = Path(path)
    with np.load(path / "fixture.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def check_requantize_range(acc: np.ndarray, shifts: np.ndarray, mode: str) -> None:
    """Validate supported CMSIS INT32 shifts and single-rounding pre-left range."""
    if mode not in ("double", "single"):
        raise ValueError(f"unknown requantize mode: {mode}")
    acc = np.asarray(acc)
    shifts = np.asarray(shifts)
    if acc.ndim == 0 or shifts.ndim != 1:
        raise ValueError("accumulator must have a channel axis and shifts must be one-dimensional")
    if not np.issubdtype(acc.dtype, np.integer):
        raise ValueError("accumulator values must be integers in the INT32 range")
    if not np.issubdtype(shifts.dtype, np.integer):
        raise ValueError("shifts must be integers in [-31, 30]")
    if acc.shape[-1] != shifts.size:
        raise ValueError("accumulator and per-channel shifts must have matching final axes")
    if np.any(acc < -(1 << 31)) or np.any(acc > (1 << 31) - 1):
        raise ValueError("accumulator values must fit INT32")
    if np.any(shifts < -31) or np.any(shifts > 30):
        raise ValueError("shifts must be in [-31, 30]")
    # Cast only after validating bounds, so no out-of-range unsigned values can
    # wrap during conversion to the signed intermediate type.
    acc = acc.astype(np.int64, copy=False)
    shifts = shifts.astype(np.int64, copy=False)
    for channel, shift in enumerate(shifts.reshape(-1)):
        preleft_shift = int(shift) + (1 if mode == "single" else 0)
        if preleft_shift <= 0:
            continue
        factor = 1 << preleft_shift
        min_input = -((1 << 31) // factor)
        max_input = ((1 << 31) - 1) // factor
        channel_acc = acc[..., channel]
        if np.any((channel_acc < min_input) | (channel_acc > max_input)):
            raise ValueError(
                f"{mode}-rounding pre-left overflows INT32 in channel {channel} "
                f"for shift {int(shift)}"
            )
