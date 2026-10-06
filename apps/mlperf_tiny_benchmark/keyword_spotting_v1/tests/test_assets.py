"""Byte-level provenance and structural contracts for KWS assets."""

import hashlib
import json
from pathlib import Path
import wave

import tflite


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "kws_ref_model_float32.tflite"
LICENSE_PATH = APP_ROOT / "LICENSE.mlperf-tiny"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"

MODEL_SHA256 = "738a9f29d175aaa3928db9c8281265be5ec3406598fd3d30018b26084a3d5536"
EXPECTED_LABELS = (
    "Down",
    "Go",
    "Left",
    "No",
    "Off",
    "On",
    "Right",
    "Stop",
    "Up",
    "Yes",
    "Silence",
    "Unknown",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_model_and_license_are_authenticated_assets():
    assert _sha256(MODEL_PATH) == MODEL_SHA256
    assert {path.name for path in MODEL_PATH.parent.glob("*.tflite")} == {MODEL_PATH.name}
    license_text = LICENSE_PATH.read_text(encoding="utf-8")
    assert "Apache License" in license_text


def test_model_flatbuffer_has_float_tensors_and_no_quantization_metadata():
    model = tflite.Model.GetRootAsModel(MODEL_PATH.read_bytes(), 0)
    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)
    assert graph.Tensors(graph.Inputs(0)).Type() == tflite.TensorType.FLOAT32
    assert graph.Tensors(graph.Outputs(0)).Type() == tflite.TensorType.FLOAT32
    types = [graph.Tensors(index).Type() for index in range(graph.TensorsLength())]
    assert set(types) <= {tflite.TensorType.FLOAT32, tflite.TensorType.INT32}
    assert types.count(tflite.TensorType.INT32) == 1
    assert all(
        graph.Tensors(index).Quantization().ScaleLength() == 0
        and graph.Tensors(index).Quantization().ZeroPointLength() == 0
        for index in range(graph.TensorsLength())
    )


def test_manifest_has_exact_canonical_order_and_safe_provenance():
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["model_sha256"] == MODEL_SHA256
    assert manifest["model_source"] == "MLPerf Tiny v1.4 benchmark/training/keyword_spotting/trained_models/kws_ref_model_float32.tflite"
    assert manifest["sample_rate"] == 16000
    assert manifest["clip_frames"] == 16000
    assert [item["order"] for item in manifest["samples"]] == list(range(12))
    assert tuple(item["label_name"] for item in manifest["samples"]) == EXPECTED_LABELS
    assert tuple(item["label"] for item in manifest["samples"]) == tuple(range(12))

    filenames = [item["filename"] for item in manifest["samples"]]
    assert len(filenames) == 12
    assert len(set(filenames)) == len(filenames)
    for item in manifest["samples"]:
        filename = Path(item["filename"])
        source_path = Path(item["source_relative_path"])
        assert not filename.is_absolute()
        assert filename.name == item["filename"]
        assert ".." not in filename.parts
        assert not source_path.is_absolute()
        assert ".." not in source_path.parts
        assert len(item["sha256"]) == 64
        assert item["byte_length"] > 44


def test_committed_wavs_match_manifest_hashes_and_audio_contract():
    manifest = _manifest()
    sample_dir = MANIFEST_PATH.parent
    assert {path.name for path in sample_dir.glob("*.wav")} == {
        item["filename"] for item in manifest["samples"]
    }
    for item in manifest["samples"]:
        path = sample_dir / item["filename"]
        assert path.read_bytes().__len__() == item["byte_length"]
        assert _sha256(path) == item["sha256"]
        with wave.open(str(path), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000
            assert wav.getnframes() == 16000
