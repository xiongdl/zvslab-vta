"""Byte-level provenance and structural contracts for anomaly assets."""

import hashlib
import json
from pathlib import Path
import wave

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "ad01_fp32.tflite"
LICENSE_PATH = APP_ROOT / "LICENSE.mlperf-tiny"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"

MODEL_SHA256 = "c66636f4d7f8af8b10518e7be750a22c9d8d46ec97326b40b0d94c097e0aad9b"
LICENSE_SHA256 = "0f8a66094ba89816073810e65a86366fa962a61b8d474c8a4bbec7d7ac6fe3db"
EXPECTED_SAMPLES = [
    ("normal_id_01_00000000.wav", "normal", 0, 352044, "0385da04d6cf8c1f9d0df775f98fda55409a71890c02ed53bb5d2c66171f6828"),
    ("normal_id_01_00000001.wav", "normal", 0, 352044, "9288e70692964b005c6247ecb9ce689ad00171f569b1a7752e497f07347fb243"),
    ("normal_id_01_00000002.wav", "normal", 0, 352044, "bfb7b845cf3b21e1dc20a47d5b032141da289e959e11c1ecb9f68fd1bf398331"),
    ("normal_id_01_00000003.wav", "normal", 0, 352044, "6769c42c426ffedda9ff7ac1ed6da9937b2668d0518ef5e40ae48ee6e8302bd2"),
    ("normal_id_01_00000004.wav", "normal", 0, 352044, "e0f1974c00eee5a3793a68f0fcadc15f0b1ddb3c3a751938a00bed9f7c9dd6b1"),
    ("anomaly_id_01_00000000.wav", "anomaly", 1, 352044, "bb9d793188bcc1ed7d0f48124cf49a06b082ac14184f683913374182e31df9d8"),
    ("anomaly_id_01_00000001.wav", "anomaly", 1, 352044, "e6fde8cf4f2b8b6c8d3956f1ffbe00cebaef21d6a8a1e09e8f27506a87daba23"),
    ("anomaly_id_01_00000002.wav", "anomaly", 1, 352044, "a26c520bd0fb0bbf7d8438aa3646729487bcc64ff1f85810411574ed1b36894a"),
    ("anomaly_id_01_00000003.wav", "anomaly", 1, 352044, "6f582a6c775b3e97eb8d6c2eb451902d19c336d51c04d7dc04691284ed040c1a"),
    ("anomaly_id_01_00000004.wav", "anomaly", 1, 352044, "3baddd82cb4335efed8db89351e735d2c01959e901d6e855efef96d352978c33"),
]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_model_and_license_are_authenticated_assets():
    assert _sha256(MODEL_PATH) == MODEL_SHA256
    assert _sha256(LICENSE_PATH) == LICENSE_SHA256
    assert "MIT License" in LICENSE_PATH.read_text(encoding="utf-8")


def test_manifest_has_exact_balanced_order_and_safe_provenance():
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["class_mapping"] == {"0": "normal", "1": "anomaly"}
    assert len(manifest["samples"]) == 10
    assert [item["order"] for item in manifest["samples"]] == list(range(10))
    assert [
        (item["filename"], item["class_name"], item["label"], item["byte_length"], item["sha256"])
        for item in manifest["samples"]
    ] == EXPECTED_SAMPLES

    filenames = [item["filename"] for item in manifest["samples"]]
    assert len(set(filenames)) == len(filenames)
    for item in manifest["samples"]:
        filename = Path(item["filename"])
        source_path = Path(item["source_relative_path"])
        assert not filename.is_absolute()
        assert filename.name == item["filename"]
        assert ".." not in filename.parts
        assert not source_path.is_absolute()
        assert ".." not in source_path.parts


def test_committed_wavs_match_manifest_hashes_and_audio_contract():
    manifest = _manifest()
    assert {
        path.name for path in MANIFEST_PATH.parent.glob("*.wav")
    } == {item["filename"] for item in manifest["samples"]}
    for item in manifest["samples"]:
        path = MANIFEST_PATH.parent / item["filename"]
        data = path.read_bytes()
        assert len(data) == item["byte_length"]
        assert _sha256(path) == item["sha256"]
        with wave.open(str(path), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000
            assert wav.getnframes() > 0


@pytest.mark.parametrize("filename", [item[0] for item in EXPECTED_SAMPLES])
def test_each_expected_sample_is_present(filename):
    assert (MANIFEST_PATH.parent / filename).is_file()
