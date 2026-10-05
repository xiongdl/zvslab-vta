"""Provenance checks for the preserved VWW model and image samples."""

import hashlib
import json
from pathlib import Path

from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL = APP_ROOT / "model" / "vww_96_float.tflite"
LICENSE = APP_ROOT / "LICENSE.mlperf-tiny"
SAMPLES = APP_ROOT / "samples"
MODEL_SHA256 = "115bbc094d2119561320a21f01b6500a18bea8cc8589282ab007097bec8af38c"
LICENSE_SHA256 = "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_model_and_license_bytes_are_preserved():
    assert _sha256(MODEL) == MODEL_SHA256
    assert _sha256(LICENSE) == LICENSE_SHA256


def test_sample_manifest_matches_every_committed_jpeg():
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    files = {path.name for path in SAMPLES.glob("*.jpg")}
    entries = manifest["samples"]
    assert {entry["filename"] for entry in entries} == files
    for entry in entries:
        path = SAMPLES / entry["filename"]
        assert _sha256(path) == entry["sha256"]


def test_default_image_is_a_96_by_96_rgb_jpeg():
    with Image.open(SAMPLES / "00-non-person-000000000009.jpg") as image:
        assert image.size == (96, 96)
        assert image.mode in ("RGB", "RGBA")
