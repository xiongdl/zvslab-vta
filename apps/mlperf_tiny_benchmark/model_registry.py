"""MLPerf Tiny model-specific asset registration."""

MODEL_PIPELINES = {
    "image_classification_v1": ("image_classification_v1", "model", "pretrainedResnet.tflite"),
    "streaming_wakeword_v1": (
        "streaming_wakeword_v1", "model", "str_ww_ref_model.tflite"
    ),
}
