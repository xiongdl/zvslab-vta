# Quantized convolution requantization acceptance

`fixtures/resnet-first-conv-cifar0/` is extracted from the first `CONV_2D` in
the pinned Tiny-v1.4 ResNet TFLite model and CIFAR-10 `test_batch`, sample 0.
It contains `fixture.npz` for Python analysis, `fixture.bin` for the C++ driver,
and `metadata.json` with tensor indices, quantization parameters, SHA256 hashes,
and runtime versions. Input preprocessing is the TFLite test program's uint8
pixel value minus 128, cast to int8. Extraction uses TensorFlow 2.15 with
delegates disabled and intermediate tensors preserved.

Recreate the fixture from the repository root with:

```sh
.envs/sww-env/bin/python vta/tests/qconv2d/extract_conv_fixture.py \
  --model .envs/tiny-v1.4/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite \
  --cifar-batch .envs/cifar-10-batches-py/test_batch \
  --sample-index 0 \
  --output-dir vta/tests/qconv2d/fixtures/resnet-first-conv-cifar0
```

The actual CMSIS-NN v8.0.0 `arm_convolve_s8` implementation is compiled twice:
once with default double rounding and once with
`CMSIS_NN_USE_SINGLE_ROUNDING`. The wrapper also produces an independent int64
accumulator observer. `conv_probe` runs the real VTA GEMM over 64-position tiles,
then runs per-lane RMUL/RSFT, output offset, and activation clipping through the
selected FSIM or TSIM driver. The host only prepares im2col input vectors and
corrected bias; it does not perform the convolution.

Run the complete acceptance suite, including ALU edge/random checks and both
convolution rounding modes, with:

```sh
bash vta/tests/qconv2d/run_tests.sh --backend fsim
```

Use `--backend tsim` to run the same fixture against TSIM. `--conv-only` runs
just the convolution tests. JUnit results and the generated
`reports/<backend>-qconv2d/qconv2d-rounding.json` are local ignored reports.
The JSON records per-mode CMSIS/driver difference counts and, for each of the
13 single-rounding versus TFLite output differences, the coordinate,
accumulator, channel multiplier/shift, complete product, Q31 remainder, and
both rounding-stage results. Default double rounding matches TFLite exactly;
single rounding remains an independently named CMSIS mode and matches CMSIS
exactly while differing from TFLite at those 13 rounding boundaries by one.
