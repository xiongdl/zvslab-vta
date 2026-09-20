"""Focused preprocessing contracts for streaming wakeword v1."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = APP_ROOT / "model_pipeline.py"
MODEL_PATH = APP_ROOT / "model" / "str_ww_ref_model.tflite"
SAMPLES = tuple(sorted((APP_ROOT / "samples").glob("*.wav")))
MODEL_SHA256 = "3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b"
EXPECTED_TFLITE_OPERATORS = (
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "CONV_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
)


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


def test_import_validates_authenticated_tflite_and_relay_contract(model_pipeline):
    imported = model_pipeline.import_model(MODEL_PATH)

    assert imported.model_sha256 == MODEL_SHA256
    assert imported.input_name == "serving_default_input_1:0"
    assert imported.input_shape == (1, 30, 1, 40)
    assert imported.input_dtype == "int8"
    assert imported.input_scale == 0.003701042616739869
    assert imported.input_zero_point == -128
    assert imported.output_name == "StatefulPartitionedCall:0"
    assert imported.output_shape == (1, 3)
    assert imported.output_dtype == "int8"
    assert imported.output_scale == 0.00390625
    assert imported.output_zero_point == -128
    assert imported.tflite_operator_names == EXPECTED_TFLITE_OPERATORS


def test_import_rejects_changed_model_bytes(model_pipeline, tmp_path):
    changed = tmp_path / "changed.tflite"
    contents = bytearray(MODEL_PATH.read_bytes())
    contents[-1] ^= 1
    changed.write_bytes(contents)

    with pytest.raises(ValueError, match="model SHA-256"):
        model_pipeline.import_model(changed)


def test_prepare_calls_vta_partition_exactly_once(model_pipeline, monkeypatch):
    imported = SimpleNamespace(module=object(), params={})
    partition_input = object()

    class Normalized:
        def clone(self):
            return partition_input

    normalized = Normalized()
    mixed = object()
    routing = object()
    calls = []

    monkeypatch.setattr(model_pipeline, "import_model", lambda path: imported)
    monkeypatch.setattr(
        model_pipeline,
        "normalize_model",
        lambda model: calls.append(("normalize", model)) or normalized,
    )

    def partition(model, params, mod_name):
        calls.append(("partition", model, params, mod_name))
        return mixed

    monkeypatch.setattr(model_pipeline.vta.relay, "partition_for_vta", partition)
    monkeypatch.setattr(
        model_pipeline,
        "inspect_partitioning",
        lambda reference, mixed_module: calls.append(
            ("inspect", reference, mixed_module)
        )
        or routing,
    )

    prepared = model_pipeline.prepare_model(MODEL_PATH)

    assert prepared.imported is imported
    assert prepared.reference_module is normalized
    assert prepared.mixed_module is mixed
    assert prepared.routing is routing
    assert calls == [
        ("normalize", imported),
        ("partition", partition_input, {}, model_pipeline.VTA_MODULE_NAME),
        ("inspect", normalized, mixed),
    ]


def test_inspect_rejects_missing_vta_partition(model_pipeline, monkeypatch):
    monkeypatch.setattr(model_pipeline, "_external_functions", lambda module: ())

    with pytest.raises(ValueError, match="no VTA partitions"):
        model_pipeline.inspect_partitioning(object(), object())


def test_inspect_rejects_partition_without_convolution(model_pipeline, monkeypatch):
    external = (
        (f"tvmgen_{model_pipeline.VTA_MODULE_NAME}_vta_main_0", object(), object()),
    )
    monkeypatch.setattr(model_pipeline, "_external_functions", lambda module: external)
    monkeypatch.setattr(model_pipeline, "_relay_operator_names", lambda function: ())

    with pytest.raises(ValueError, match="convolution"):
        model_pipeline.inspect_partitioning(object(), object())


def test_real_model_partition_keeps_int8_io_and_vta_convolutions(model_pipeline):
    prepared = model_pipeline.prepare_model(MODEL_PATH)

    assert prepared.reference_module["main"].params[0].checked_type.dtype == "int8"
    assert tuple(
        int(d) for d in prepared.reference_module["main"].params[0].checked_type.shape
    ) == (1, 30, 1, 40)
    assert prepared.mixed_module["main"].ret_type.dtype == "int8"
    assert tuple(int(d) for d in prepared.mixed_module["main"].ret_type.shape) == (1, 3)
    assert prepared.routing.symbols
    assert all(count > 0 for count in prepared.routing.convolutions_per_partition)


def test_normalization_preserves_all_per_axis_fixed_point_nodes(model_pipeline):
    imported = model_pipeline.import_model(MODEL_PATH)
    canonical = model_pipeline.relay.qnn.transform.CanonicalizeOps()(imported.module)
    canonical = model_pipeline.relay.transform.InferType()(canonical)
    canonical_count = model_pipeline._relay_operator_names(canonical["main"]).count(
        "fixed_point_multiply_per_axis"
    )

    normalized = model_pipeline.normalize_model(imported)

    assert canonical_count == 8
    assert model_pipeline._relay_operator_names(normalized["main"]).count(
        "fixed_point_multiply_per_axis"
    ) == canonical_count
