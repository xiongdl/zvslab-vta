# MLPerf Tiny KWS v1 source model

`kws_ref_model.tflite` is copied byte-for-byte from the MLPerf Tiny v1.4
source tree:

```text
benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite
```

The committed model SHA-256 is:

```text
aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae
```

The model is distributed as part of MLPerf Tiny v1.4 under the repository's
Apache License 2.0; the applicable license text is copied to
`../LICENSE.mlperf-tiny`. Its FlatBuffer contains one subgraph, one input with
shape `[1, 49, 10, 1]` and dtype `int8`, and one output with shape `[1, 12]`
and dtype `int8`. The application preserves this pre-quantized model and
validates the contract before Relay import.

## All-float deployment model

`kws_ref_model_float32.tflite` is exported from the upstream SavedModel at
`benchmark/training/keyword_spotting/trained_models/kws_ref_model/` with
`scripts/convert_kws_model.py` using `.envs/sww-env`. No training,
`Optimize.DEFAULT`, representative dataset, or weight rewriting is used. The
SavedModel contains 56 float32 learned variables. The generated FlatBuffer is
105,296 bytes with SHA-256
`738a9f29d175aaa3928db9c8281265be5ec3406598fd3d30018b26084a3d5536`.

The FlatBuffer has one float32 input `[1, 49, 10, 1]`, one float32 output
`[1, 12]`, 35 tensors, 13 operators, and one int32 reshape shape constant;
learned weights and activations are float32. Its source-file SHA-256 values
are:

- `saved_model.pb`: `80665dd19eeb03d1152fdc098b5635261731b23478d09f6ff2166da799f98138`
- `variables/variables.data-00000-of-00001`:
  `5ebfddd34e85a2d35e87e27a8a803353534019bb460907b16db5fa72d757e8c5`
- `variables/variables.index`:
  `3f21dd9ea6736e20e70c5ed2f40ed9d8a7d12b98316ac0e4ab8f4eb7e303d738`

The original externally supplied float-I/O TFLite is not used for deployment
because its convolution weights were dynamic-range quantized.
