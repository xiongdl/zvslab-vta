# MLPerf Tiny ResNet-8 Large source model

`pretrainedResnet_large_float.tflite` is an unmodified floating-point model
copied byte-for-byte from the MLPerf Tiny v1.4 source tree at:

```text
benchmark/training/image_classification/trained_models/pretrainedResnet_large_float.tflite
```

Its SHA-256 is:

```text
fb17ae9c1b6d0e5bd97f0f35024f207556261d7310b249716c87cc0628214b0e
```

The FlatBuffer contract is:

- Input tensor: `serving_default_input_5:0`, float32 NHWC `[1, 32, 32, 3]`.
- Output tensor: `StatefulPartitionedCall:0`, float32 `[1, 10]`.
- Ordered operators: `CONV_2D, CONV_2D, CONV_2D, ADD, CONV_2D, CONV_2D, CONV_2D, ADD, CONV_2D, CONV_2D, CONV_2D, ADD, AVERAGE_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX`.
- Convolution output channels: `40/40/40/80/80/80/160/160/160`.

The model is distributed by MLPerf Tiny under Apache-2.0; the applicable text
is copied verbatim to `../LICENSE.mlperf-tiny`.

The committed bytes remain floating point. Quantization is performed by TVM at
deployment time with the following fixed policy:

```python
with relay.quantize.qconfig(
    calibrate_mode="global_scale",
    global_scale=8.0,
    skip_conv_layers=[0],
):
    quantized = relay.quantize.quantize(mod, params=params)
```

CIFAR-10 samples are not calibration input.
