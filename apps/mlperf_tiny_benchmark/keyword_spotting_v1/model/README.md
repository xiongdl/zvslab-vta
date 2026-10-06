# MLPerf Tiny KWS v1 float32 model

This directory contains only `kws_ref_model_float32.tflite`, exported from the
upstream MLPerf Tiny v1.4 float32 SavedModel at
`benchmark/training/keyword_spotting/trained_models/kws_ref_model/` with
`scripts/convert_kws_model.py`. No training, `Optimize.DEFAULT`, representative
dataset, or weight rewriting is used. The SavedModel contains 56 float32
learned variables. The generated FlatBuffer is 105,296 bytes with SHA-256
`738a9f29d175aaa3928db9c8281265be5ec3406598fd3d30018b26084a3d5536`.

The model has one float32 input `[1, 49, 10, 1]`, one float32 output `[1, 12]`,
35 tensors, 13 operators, and one int32 reshape-shape constant. Learned
weights and model activations are float32. The source SavedModel files have
these SHA-256 values:

- `saved_model.pb`: `80665dd19eeb03d1152fdc098b5635261731b23478d09f6ff2166da799f98138`
- `variables/variables.data-00000-of-00001`:
  `5ebfddd34e85a2d35e87e27a8a803353534019bb460907b16db5fa72d757e8c5`
- `variables/variables.index`:
  `3f21dd9ea6736e20e70c5ed2f40ed9d8a7d12b98316ac0e4ab8f4eb7e303d738`

The upstream externally supplied float-I/O TFLite is not used because its
convolution weights were dynamic-range quantized. The deployed model is
generated from the all-float SavedModel instead.
