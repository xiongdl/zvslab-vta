"""Focused preprocessing contracts for streaming wakeword v1."""

import importlib.util
from pathlib import Path
import sys
import wave

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = APP_ROOT / "model_pipeline.py"
SAMPLES = tuple(sorted((APP_ROOT / "samples").glob("*.wav")))


@pytest.fixture(scope="module")
def model_pipeline():
    spec = importlib.util.spec_from_file_location(
        "streaming_wakeword_model_pipeline", PIPELINE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_samples_produce_deterministic_int8_feature_contract(model_pipeline):
    for sample_path in SAMPLES:
        first = model_pipeline.load_sample(sample_path)
        second = model_pipeline.load_sample(sample_path)

        np.testing.assert_array_equal(first, second)
        assert first.shape == (1, 30, 1, 40)
        assert first.dtype == np.int8
        assert int(first.min()) >= -128
        assert int(first.max()) <= 127


def test_preprocessing_rejects_non_mono_16khz_pcm(tmp_path, model_pipeline):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(np.zeros((16000, 2), dtype="<i2").tobytes())

    with pytest.raises(ValueError, match="mono 16-bit 16000 Hz WAV"):
        model_pipeline.load_sample(path)


def test_preprocessing_rejects_empty_wav(tmp_path, model_pipeline):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)

    with pytest.raises(ValueError, match="no PCM frames"):
        model_pipeline.load_sample(path)
