# MLPerf Tiny streaming wakeword v1 source model

`str_ww_ref_model.tflite` is copied byte-for-byte from the MLPerf Tiny v1.4
source tree:

```text
benchmark/training/streaming_wakeword/trained_models/str_ww_ref_model.tflite
```

The committed model is 74,520 bytes and has SHA-256:

```text
3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b
```

The model is distributed as part of MLPerf Tiny v1.4 under the Apache License
2.0. The applicable license text is copied to
`../LICENSE.mlperf-tiny`. The FlatBuffer contains one subgraph with the
following preserved int8 contract:

| Tensor | Name | Shape | Quantization |
| --- | --- | --- | --- |
| Input | `serving_default_input_1:0` | `[1, 30, 1, 40]` | scale `0.003701042616739869`, zero point `-128` |
| Output | `StatefulPartitionedCall:0` | `[1, 3]` | scale `0.00390625`, zero point `-128` |

The committed application owns this model copy. Runtime code must read it from
the application directory and must not depend on the source environment or
dataset.

## Float32 conversion source

`str_ww_ref_model_floag32.tflite` is generated from the upstream H5 reference
model at
`.envs/tiny-v1.4/benchmark/training/streaming_wakeword/trained_models/str_ww_ref_model.h5`
by `scripts/convert_sww_model.py`. The converter uses the upstream
`quantize.py` Keras load and TFLite conversion with float defaults; it does not
load calibration data or apply the INT8 conversion block. The source H5 and
temporary adapted Python file are kept outside this application directory.

Conversion provenance:

- H5 SHA-256: `b0f267a8ba0bcb911c1098229c32fac21996e4191c9c60d1fda80adaa70a8add`
- upstream `quantize.py` SHA-256: `6303e820a13ce6d50ea26f2e6d19ee3cbbfac2fae99cb520c0737afc578742f2`
- float32 TFLite size: 191,428 bytes
- float32 TFLite SHA-256: `c735ab47248df7648d9cb4397c0e7d161fe2e88ede17ad900f34a4163d89b267`

FlatBuffer inspection with the repository's `tflite` schema reports one
subgraph, 29 tensors, and 11 operators. Input
`serving_default_input_1:0` is float32 `[1, 30, 1, 40]`; output
`StatefulPartitionedCall:0` is float32 `[1, 3]`. Model weights and activations
are float32; the only integer tensor is the int32 shape constant consumed by
RESHAPE. No INT8/UINT8/INT16 tensors or quantized operators are present. The
app tree stores only TFLite model files; the H5 source remains in the ignored
MLPerf Tiny source tree.
