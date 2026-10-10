# Task 3 report: per-channel qconv2d FSIM acceptance

## Result

Implemented a reproducible first-convolution fixture and exercised a real VTA
GEMM plus the per-channel ALU requantization sequence through the FSIM driver.
The hardware accumulator and final output both match the pinned CMSIS-NN
reference exactly in double- and single-rounding modes. Default double rounding
also matches the TFLite output exactly. Single rounding differs from TFLite at
13 elements by one quantized unit; those are explained by the two different
rounding sequences below.

## Fixture and reference identity

- Input: `.envs/cifar-10-batches-py/test_batch`, sample 0, reshaped NHWC and
  converted from uint8 to int8 by subtracting 128.
- Model: `.envs/tiny-v1.4/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite`.
- Model SHA256: `3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`.
- Preprocessed input SHA256: `9f2b799d8a7d23ea057764d98841e08038753212d8fcb06b4b8f28a859926511`.
- Extracted op: tensor indices input 0, weight 8, bias 3, output 22; shapes
  `[1,32,32,3]`, `[16,3,3,3]`, `[16]`, and `[1,32,32,16]`.
- Runtime: TensorFlow 2.15.0, NumPy 1.26.4, Python 3.11.16.
- CMSIS-NN: pinned v8.0.0, commit
  `13c97dbb6f781d4aab38ed34e6e441f42b79aff4`. The double library has no extra
  macros; the single library is compiled with
  `CMSIS_NN_USE_SINGLE_ROUNDING`. Host scalar configuration has DSP, MVEI, and
  requantize inline assembly disabled.
- Per-channel shifts generated from the interpreter scales are
  `[-8,-10,-9,-8,-9,-9,-9,-8,-9,-9,-10,-9,-8,-11,-10,-7]`.

## Driver path and checks

The extractor saves both `fixture.npz` and the C++ driver's fixed-layout
`fixture.bin`. Host code only forms 64-position im2col tiles. Each of the two
8-channel output blocks runs nine actual GEMM taps. Input lanes 3–7 and weights
for those lanes are zero; border im2col values are explicitly -128. Bias is
corrected per channel as `bias + 128 * sum(weights)`. The 64-position tile
requires 576 input vectors, 256 accumulator vectors for data/parameters/scratch,
and at most 640 uops; the probe checks these against configured SRAM capacity.
It stores the GEMM accumulator through four low-byte stores and reconstructs
the full INT32 values before uploading those readbacks for the ALU run.

The pinned `arm_convolve_s8` implementation runs independently for each CMSIS
rounding build. Its wrapper also computes an int64 scalar accumulator observer
using the input offset and padded zero values. This confirms the hardware bias
correction and padding transformation element by element.

| Comparison | Accumulator differences | Output differences |
| --- | ---: | ---: |
| CMSIS double vs FSIM double | 0 | 0 |
| CMSIS single vs FSIM single | 0 | 0 |

The FSIM double output vs TFLite has 0 differences (maximum absolute
difference 0). FSIM single vs TFLite has 13 differences (maximum absolute
difference 1). For every such coordinate, the analyzer first verifies equal
CMSIS/FSIM accumulators and outputs, then recomputes the Q31 product, quotient,
remainder, and both requantization stages from the shared accumulator and
channel parameters. The double sequence matches TFLite at every element; the
single sequence has an RMUL quotient one lower at each listed coordinate, and
the final RSFT preserves that one-unit difference. This establishes rounding
as the source after checking inputs, convolution accumulators, multipliers,
shifts, output offset, and activation bounds.

| N,H,W,C | Accumulator | Multiplier | Shift | Product | Double RMUL→RSFT | Single RMUL→RSFT | TFLite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,0,24,6 | 11781 | 2099891493 | -9 | 24738821679033 | 11520→23 (-105) | 11519→22 (-106) | -105 |
| 0,1,16,6 | 11781 | 2099891493 | -9 | 24738821679033 | 11520→23 (-105) | 11519→22 (-106) | -105 |
| 0,1,24,15 | 4130 | 1164674270 | -7 | 4810104735100 | 2240→18 (-110) | 2239→17 (-111) | -110 |
| 0,2,3,9 | 4880 | 1464338136 | -9 | 7145970103680 | 3328→7 (-121) | 3327→6 (-122) | -121 |
| 0,7,26,7 | 1845 | 1638298993 | -8 | 3022661642085 | 1408→6 (-122) | 1407→5 (-123) | -122 |
| 0,9,23,0 | 221 | 1242405367 | -8 | 274571586107 | 128→1 (-127) | 127→0 (-128) | -127 |
| 0,11,8,12 | 36699 | 1101019486 | -8 | 40406314116714 | 18816→74 (-54) | 18815→73 (-55) | -54 |
| 0,13,16,0 | 1548 | 1242405367 | -8 | 1923243508116 | 896→4 (-124) | 895→3 (-125) | -124 |
| 0,14,12,9 | 10887 | 1464338136 | -9 | 15942249286632 | 7424→15 (-113) | 7423→14 (-114) | -113 |
| 0,20,13,7 | 4194 | 1638298993 | -8 | 6871025976642 | 3200→13 (-115) | 3199→12 (-116) | -115 |
| 0,21,6,12 | 8738 | 1101019486 | -8 | 9620708268668 | 4480→18 (-110) | 4479→17 (-111) | -110 |
| 0,22,9,15 | 6962 | 1164674270 | -7 | 8108462267740 | 3776→30 (-98) | 3775→29 (-99) | -98 |
| 0,31,12,8 | 19277 | 1454413568 | -9 | 28036730350336 | 13056→26 (-102) | 13055→25 (-103) | -102 |

The complete report, including the Q31 and final-shift remainders/increments,
is generated at `vta/tests/qconv2d/reports/<backend>-qconv2d/qconv2d-rounding.json`
and is ignored as a generated result.

## Verification and scope

`bash vta/tests/qconv2d/run_tests.sh --backend fsim` passed all 39 tests,
including encoding/runtime checks, 100,000 fixed-seed cases in each CMSIS
requantization mode, and this real convolution acceptance. `git diff --check`
passed. No files under `vta/apps` were modified. TSIM convolution comparison was
not part of this Task 3 FSIM run.


## Review follow-up

The range checker rejects non-integer or out-of-INT32 accumulators, unsupported
shifts outside [-31,30], and unknown rounding modes before doing any
requantization range arithmetic. This avoids NumPy int64 wraparound for
pre-left overflow checks. Tests include the wrapped `1 * 2**65` case,
supported shift and legal INT32 boundaries, as well as unknown mode and
non-INT32 accumulator rejection.

The analyzer rejection test changes one channel multiplier, then runs the pinned
CMSIS single-rounding implementation with the altered value. The resulting
accumulators remain identical, while output coordinate `[0,0,24,6]` differs
from TFLite by one unit and all differences stay within one. The analyzer
rejects this as a stage-arithmetic mismatch instead of attributing it to
rounding.

Final verification after these fixes: `bash tests/qconv2d/run_tests.sh --backend
fsim` completed with 39 passed.
