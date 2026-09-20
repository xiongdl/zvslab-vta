"""Byte-level and structural contracts for streaming wakeword v1 assets."""

import hashlib
import json
from pathlib import Path
import wave

import tflite


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "model" / "str_ww_ref_model.tflite"
MODEL_README_PATH = APP_ROOT / "model" / "README.md"
LICENSE_PATH = APP_ROOT / "LICENSE.mlperf-tiny"
SAMPLES_ROOT = APP_ROOT / "samples"
MANIFEST_PATH = SAMPLES_ROOT / "manifest.json"

MODEL_SHA256 = "3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b"
LICENSE_SHA256 = "0d542e0c8804e39aa7f37eb00da5a762149dc682d7829451287e11b938e94594"
EXPECTED_CLASSES = ("Marvin", "Silence", "Unknown")
EXPECTED_SAMPLES = (
    {
        "order": 0,
        "label": 0,
        "label_name": "Marvin",
        "filename": "marvin-00176480_nohash_0.wav",
        "byte_length": 32044,
        "sha256": "b95e103110b89a0d4dff88023edd537a92834f565cb8e3f38b16f725f3d58451",
        "source_relative_path": "marvin/00176480_nohash_0.wav",
        "source_start_frame": 0,
        "source_frame_count": 16000,
        "selection_provenance": "lexicographically first WAV in the marvin directory",
    },
    {
        "order": 1,
        "label": 1,
        "label_name": "Silence",
        "filename": "silence-doing_the_dishes-00000000.wav",
        "byte_length": 32044,
        "sha256": "372026037b1d1615e5335eae7149b90f7ac9b75b0a7016279d9f11adff15cdce",
        "source_relative_path": "_background_noise_/doing_the_dishes.wav",
        "source_start_frame": 0,
        "source_frame_count": 16000,
        "selection_provenance": "first 16000 frames (one second) from the lexicographically first WAV in the _background_noise_ directory",
    },
    {
        "order": 2,
        "label": 2,
        "label_name": "Unknown",
        "filename": "unknown-0165e0e8_nohash_0.wav",
        "byte_length": 32044,
        "sha256": "cf65db13461bb82fe5edff74b21e38d4ea7e54b895be59d13f698c0f616688db",
        "source_relative_path": "backward/0165e0e8_nohash_0.wav",
        "source_start_frame": 0,
        "source_frame_count": 16000,
        "selection_provenance": "lexicographically first WAV in the first non-marvin class directory (backward)",
    },
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _assert_safe_relative_path(value, *, basename_only=False):
    path = Path(value)
    assert not path.is_absolute(), value
    assert ".." not in path.parts, value
    if basename_only:
        assert path.name == value, value


def test_model_is_the_authenticated_mlperf_tiny_v14_artifact():
    assert MODEL_PATH.is_file()
    assert not MODEL_PATH.is_symlink()
    assert MODEL_PATH.stat().st_size == 74520
    assert _sha256(MODEL_PATH) == MODEL_SHA256
    assert _sha256(LICENSE_PATH) == LICENSE_SHA256

    readme = MODEL_README_PATH.read_text(encoding="utf-8")
    for expected in (
        "MLPerf Tiny v1.4",
        "benchmark/training/streaming_wakeword/trained_models/str_ww_ref_model.tflite",
        MODEL_SHA256,
        "byte-for-byte",
        "Apache License",
        "LICENSE.mlperf-tiny",
    ):
        assert expected in readme


def test_model_flatbuffer_has_the_approved_input_output_contract():
    model = tflite.Model.GetRootAsModel(MODEL_PATH.read_bytes(), 0)
    assert model.Version() == 3
    assert model.SubgraphsLength() == 1

    graph = model.Subgraphs(0)
    assert graph.InputsLength() == 1
    assert graph.OutputsLength() == 1

    input_tensor = graph.Tensors(graph.Inputs(0))
    output_tensor = graph.Tensors(graph.Outputs(0))
    assert input_tensor.Name() == b"serving_default_input_1:0"
    assert list(input_tensor.ShapeAsNumpy()) == [1, 30, 1, 40]
    assert input_tensor.Type() == int(tflite.TensorType.INT8)
    assert input_tensor.Quantization().ScaleAsNumpy().tolist() == [
        0.003701042616739869
    ]
    assert input_tensor.Quantization().ZeroPointAsNumpy().tolist() == [-128]

    assert output_tensor.Name() == b"StatefulPartitionedCall:0"
    assert list(output_tensor.ShapeAsNumpy()) == [1, 3]
    assert output_tensor.Type() == int(tflite.TensorType.INT8)
    assert output_tensor.Quantization().ScaleAsNumpy().tolist() == [0.00390625]
    assert output_tensor.Quantization().ZeroPointAsNumpy().tolist() == [-128]


def test_manifest_has_exact_class_order_and_authenticated_provenance():
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert manifest["source_dataset"] == "Speech Commands v0.02"
    assert manifest["selection"] == (
        "lexicographically first WAV in marvin, deterministic first second of "
        "lexicographically first WAV in _background_noise_, and lexicographically "
        "first WAV in the first non-marvin class directory"
    )
    assert manifest["sample_rate"] == 16000
    assert manifest["sample_width_bytes"] == 2
    assert manifest["channels"] == 1
    assert manifest["clip_frames"] == 16000
    assert manifest["class_mapping"] == {
        "0": "Marvin",
        "1": "Silence",
        "2": "Unknown",
    }
    assert tuple(item["order"] for item in manifest["samples"]) == (0, 1, 2)
    assert tuple(item["label"] for item in manifest["samples"]) == (0, 1, 2)
    assert tuple(item["label_name"] for item in manifest["samples"]) == EXPECTED_CLASSES
    assert [
        {key: item[key] for key in expected}
        for item, expected in zip(manifest["samples"], EXPECTED_SAMPLES)
    ] == list(EXPECTED_SAMPLES)

    model = manifest["model"]
    _assert_safe_relative_path(model["filename"])
    assert model["filename"] == "model/str_ww_ref_model.tflite"
    assert model["byte_length"] == MODEL_PATH.stat().st_size
    assert model["sha256"] == MODEL_SHA256
    assert _sha256(MODEL_PATH) == model["sha256"]
    _assert_safe_relative_path(model["source_relative_path"])

    for item in manifest["samples"]:
        _assert_safe_relative_path(item["filename"], basename_only=True)
        _assert_safe_relative_path(item["source_relative_path"])
        assert len(item["sha256"]) == 64
        assert item["byte_length"] == 32044
        assert item["source_frame_count"] == 16000
        assert item["source_start_frame"] == 0
        assert item["selection_provenance"]
        assert ".envs" not in item["source_relative_path"]


def test_committed_wavs_are_exactly_the_three_manifest_samples():
    manifest = _manifest()
    expected_names = {item["filename"] for item in manifest["samples"]}
    actual_names = {path.name for path in SAMPLES_ROOT.glob("*.wav")}
    assert actual_names == expected_names
    assert {path.name for path in SAMPLES_ROOT.iterdir()} == expected_names | {
        "manifest.json"
    }

    for item in manifest["samples"]:
        path = SAMPLES_ROOT / item["filename"]
        assert path.is_file()
        assert not path.is_symlink()
        assert path.stat().st_size == item["byte_length"]
        assert _sha256(path) == item["sha256"]
        with wave.open(str(path), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 16000
            assert audio.getnframes() == 16000
            assert audio.getcomptype() == "NONE"


def test_runtime_assets_are_repository_owned_and_do_not_reference_envs():
    assert APP_ROOT.is_dir()
    assert not APP_ROOT.is_symlink()
    assert ".envs" not in MANIFEST_PATH.read_text(encoding="utf-8")

    runtime_files = [
        path
        for path in APP_ROOT.rglob("*")
        if path.is_file() and "tests" not in path.relative_to(APP_ROOT).parts
    ]
    for path in runtime_files:
        assert not path.is_symlink()
        if path.suffix.lower() not in {".wav", ".tflite", ".a", ".so", ".dylib"}:
            assert ".envs" not in path.read_text(encoding="utf-8")

    for item in _manifest()["samples"]:
        local_path = (SAMPLES_ROOT / item["filename"]).resolve()
        assert APP_ROOT in local_path.parents
        assert ".envs" not in str(local_path)
