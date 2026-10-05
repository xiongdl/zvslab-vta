"""Model-aware registry contract for the MLPerf Tiny applications."""

from pathlib import Path

from mlperf_tiny_benchmark.model_registry import MODEL_PIPELINES


def test_registry_lists_exact_model_assets_in_application_order():
    expected = {
        "image_classification_v1": ("image_classification_v1", "model", "pretrainedResnet.tflite"),
        "anomaly_detection_v1": ("anomaly_detection_v1", "model", "ad01_fp32.tflite"),
        "streaming_wakeword_v1": (
            "streaming_wakeword_v1", "model", "str_ww_ref_model.tflite"
        ),
    }

    assert MODEL_PIPELINES == expected
    app_root = Path(__file__).resolve().parents[1]
    for model_id, (_, directory, filename) in MODEL_PIPELINES.items():
        assert (app_root / model_id / directory / filename).is_file()
