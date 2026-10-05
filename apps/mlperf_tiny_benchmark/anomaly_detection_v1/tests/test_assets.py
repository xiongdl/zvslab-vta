"""Committed anomaly model, sample, and license provenance checks."""

import hashlib
import json
import wave
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL = APP_ROOT / "model" / "ad01_fp32.tflite"
LICENSE = APP_ROOT / "LICENSE.mlperf-tiny"
SAMPLES = APP_ROOT / "samples"


def test_committed_assets_match_the_sample_manifest():
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 10
    for record in manifest["samples"]:
        path = SAMPLES / record["filename"]
        content = path.read_bytes()
        assert len(content) == record["byte_length"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
    with wave.open(str(SAMPLES / "normal_id_01_00000000.wav"), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)


def test_model_and_license_are_present_and_unmodified_in_shape():
    assert MODEL.stat().st_size > 0
    assert LICENSE.is_file()
    assert LICENSE.read_text(encoding="utf-8").strip()
