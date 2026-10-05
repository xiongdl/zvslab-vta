"""MLPerf Tiny model-specific asset registration."""

MODEL_PIPELINES = {
    "image_classification_v1": ("image_classification_v1", "model", "pretrainedResnet.tflite"),
    "anomaly_detection_v1": ("anomaly_detection_v1", "model", "ad01_fp32.tflite"),
    "keyword_spotting_v1": ("keyword_spotting_v1", "model", "kws_ref_model.tflite"),
    "streaming_wakeword_v1": (
        "streaming_wakeword_v1", "model", "str_ww_ref_model.tflite"
    ),
}
