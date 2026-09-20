"""Deterministic streaming wakeword preprocessing.

The feature path is a NumPy transcription of the training ``get_lfbe_func``
graph.  Model import and Relay preparation are kept below this self-contained
audio path so runtime preprocessing never needs the training environment.
"""

from functools import lru_cache
from pathlib import Path
import wave

import numpy as np


SAMPLE_RATE = 16000
CLIP_FRAMES = SAMPLE_RATE
WINDOW_SIZE_SAMPLES = 1024  # 64 ms; TensorFlow's implicit FFT length is 1024.
WINDOW_STRIDE_SAMPLES = 512  # 32 ms
FFT_LENGTH = 1024
MEL_BINS = 40
POWER_OFFSET = 52.0
INPUT_SCALE = 0.003701042616739869
INPUT_ZERO_POINT = -128
INPUT_SHAPE = (1, 30, 1, 40)


@lru_cache(maxsize=1)
def _mel_filterbank():
    """Return TensorFlow's linear-to-mel triangular filterbank in float32."""
    def hertz_to_mel(hertz):
        return np.float64(1127.0) * np.log1p(np.float64(hertz) / 700.0)

    def mel_to_hertz(mel):
        return 700.0 * np.expm1(np.asarray(mel, dtype=np.float64) / 1127.0)

    lower_edge_mel = hertz_to_mel(0.0)
    upper_edge_mel = hertz_to_mel(SAMPLE_RATE / 2.0)
    band_edges_mel = np.linspace(lower_edge_mel, upper_edge_mel, MEL_BINS + 2)
    band_edges_hertz = mel_to_hertz(band_edges_mel)
    spectrogram_hertz = np.linspace(
        0.0, SAMPLE_RATE / 2.0, FFT_LENGTH // 2 + 1, dtype=np.float64
    )

    lower_edges = band_edges_hertz[:-2, None]
    center_edges = band_edges_hertz[1:-1, None]
    upper_edges = band_edges_hertz[2:, None]
    lower_slopes = (spectrogram_hertz[None, :] - lower_edges) / (
        center_edges - lower_edges
    )
    upper_slopes = (upper_edges - spectrogram_hertz[None, :]) / (
        upper_edges - center_edges
    )
    return np.maximum(0.0, np.minimum(lower_slopes, upper_slopes)).astype(
        np.float32
    )


def _read_wav(sample_path):
    """Read strict mono PCM16/16 kHz WAV and normalize to float32."""
    sample_path = Path(sample_path)
    try:
        with wave.open(str(sample_path), "rb") as wav:
            if (
                wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getframerate() != SAMPLE_RATE
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError(
                    f"{sample_path} must be mono 16-bit {SAMPLE_RATE} Hz WAV"
                )
            samples = np.frombuffer(
                wav.readframes(wav.getnframes()), dtype="<i2"
            ).copy()
    except ValueError:
        raise
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"unable to read WAV sample {sample_path}") from error

    if samples.size == 0:
        raise ValueError(f"{sample_path} contains no PCM frames")
    samples = samples[:CLIP_FRAMES]
    if samples.size < CLIP_FRAMES:
        samples = np.pad(samples, (0, CLIP_FRAMES - samples.size))
    return samples.astype(np.float32) / np.float32(32768.0)


def _log_mel_features(samples):
    """Compute the training graph's 30x40 normalized log-mel matrix."""
    if samples.shape != (CLIP_FRAMES,) or samples.dtype != np.float32:
        raise ValueError(f"unexpected normalized audio shape/dtype: {samples.shape}, {samples.dtype}")

    preemphasis = np.float32(1.0 - 2.0 ** -5)
    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - preemphasis * samples[:-1]

    starts = np.arange(
        0, CLIP_FRAMES - WINDOW_SIZE_SAMPLES + 1, WINDOW_STRIDE_SAMPLES
    )
    frames = np.stack(
        [emphasized[start : start + WINDOW_SIZE_SAMPLES] for start in starts]
    )
    window = np.hamming(WINDOW_SIZE_SAMPLES).astype(np.float32)
    magnitudes = np.abs(np.fft.rfft(frames * window, n=FFT_LENGTH)).astype(
        np.float32
    )
    power = (np.square(magnitudes) / np.float32(WINDOW_SIZE_SAMPLES)).astype(
        np.float32
    )
    peak = max(float(power.max()), 1e-30)
    power = np.clip(power, np.float32(1e-30), np.float32(peak))
    mel = np.tensordot(power, _mel_filterbank(), axes=([-1], [1])).astype(
        np.float32
    )
    mel = np.maximum(mel, np.float32(1e-30))
    log_mel = np.float32(10.0) * np.log10(mel).astype(np.float32)
    log_mel = (log_mel + np.float32(POWER_OFFSET) - 32.0 + 32.0) / 64.0
    return np.clip(log_mel, 0.0, 1.0).astype(np.float32)


def quantize_features(features, scale=INPUT_SCALE, zero_point=INPUT_ZERO_POINT):
    """Quantize a validated 30x40 normalized log-mel matrix once."""
    features = np.asarray(features)
    if features.shape != (30, 40) or features.dtype.kind not in "fc":
        raise ValueError(f"unexpected log-mel feature matrix: {features.shape}, {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("log-mel features must be finite")
    if float(features.min()) < 0.0 or float(features.max()) > 1.0:
        raise ValueError("log-mel features must be clipped to [0, 1]")
    if scale <= 0:
        raise ValueError("input quantization scale must be positive")
    quantized = np.rint(features / np.float32(scale) + np.float32(zero_point))
    return np.clip(quantized, -128, 127).astype(np.int8)[None, :, None, :]


def load_sample(sample_path):
    """Load one strict WAV and return the model's int8 input tensor."""
    return quantize_features(_log_mel_features(_read_wav(sample_path)))
