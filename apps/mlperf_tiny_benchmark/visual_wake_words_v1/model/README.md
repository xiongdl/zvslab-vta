# MLPerf Tiny Visual Wake Words source model

`vww_96_float.tflite` is an unmodified floating-point model copied
byte-for-byte from the MLPerf Tiny v1.4 source tree at:

```text
benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite
```

The local source used for this copy was:

```text
.envs/tiny-v1.4/benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite
```

Its SHA-256 is:

```text
115bbc094d2119561320a21f01b6500a18bea8cc8589282ab007097bec8af38c
```

The FlatBuffer has one `float32` NHWC input named `input_1` with shape
`[1, 96, 96, 3]` and one `float32` output named `Identity` with shape
`[1, 2]`. Output index `0` is `non_person`; output index `1` is `person`.
The model is distributed by MLPerf Tiny under Apache-2.0; the applicable text
is copied verbatim to `../LICENSE.mlperf-tiny`.

The committed bytes remain floating point. Deployment preprocessing converts
RGB JPEG samples to `float32` and divides by `255.0`. TVM performs the one
deployment-time quantization with this fixed policy:

```python
with relay.quantize.qconfig(
    calibrate_mode="global_scale",
    global_scale=8.0,
    skip_conv_layers=[0],
):
    quantized = relay.quantize.quantize(mod, params=params)
```

The reference graph and mixed VTA graph are both derived from this one
quantized module. The mixed graph has twelve VTA regions; unsupported
depthwise convolutions remain on the host.
