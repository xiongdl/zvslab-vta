"""Provenance checks for the application-owned model and sample bundle."""

import hashlib
import json
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"
LICENSE = APP_ROOT / "LICENSE.mlperf-tiny"
SAMPLES = APP_ROOT / "samples"
MODEL_SHA256 = "fb17ae9c1b6d0e5bd97f0f35024f207556261d7310b249716c87cc0628214b0e"
LICENSE_SHA256 = "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_model_and_license_bytes_are_preserved():
    assert _sha256(MODEL) == MODEL_SHA256
    assert _sha256(LICENSE) == LICENSE_SHA256


def test_sample_manifest_matches_every_committed_image():
    manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
    files = {path.name for path in SAMPLES.glob("*.png")}
    entries = manifest["samples"]
    assert {entry["filename"] for entry in entries} == files
    for entry in entries:
        path = SAMPLES / entry["filename"]
        assert _sha256(path) == entry["png_sha256"]


def test_default_image_is_a_32_by_32_rgb_png():
    from PIL import Image

    with Image.open(SAMPLES / "00-airplane.png") as image:
        assert image.size == (32, 32)
        assert image.mode in ("RGB", "RGBA")
