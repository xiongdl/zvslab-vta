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
