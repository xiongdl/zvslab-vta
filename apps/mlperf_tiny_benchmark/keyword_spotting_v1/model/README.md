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
