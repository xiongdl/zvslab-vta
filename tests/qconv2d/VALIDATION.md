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

An initial run without worktree write authorization failed because the Python
encoding-check subprocess could not create `.pkl_memoize_py3`; pytest teardown
also could not write `reports/fsim-all.xml`. The suite was rerun with managed
worktree write authorization, which allowed both artifacts to be created; the
encoding check then passed and the complete result was 39 passed.

## Task 5: FSIM/TSIM exact parity and entry points

Final code was validated in independent backend processes, in this order:

- `bash vta/tests/qconv2d/run_tests.sh --backend fsim`: 40 passed in 14.83 s.
- `bash vta/tests/qconv2d/run_tests.sh --backend tsim`: 40 passed in 36.98 s.
- `bash scripts/test_vta_fsim.sh`: 71 passed in 37.12 s.
- `bash scripts/test_vta_tsim.sh`: 52 passed in 45.10 s.

Both ALU rounding modes ran 100,000 fixed-seed full-INT32 inputs. The
convolution compared every INT32 accumulator and final INT8 output. The TSIM
JSON records zero CMSIS/FSIM, FSIM/TSIM, and TFLite/FSIM differences for the
default double-rounding mode; the single-rounding TFLite comparison has 13
differences, each exactly one output unit. The generic TFLite count and maximum
absolute difference refer to default double rounding; separate single-rounding
fields record its count and per-mode maximum. Its 13-entry evidence table matches
the input, weights, bias, accumulator, multiplier, shift, zero point, and
activation and shows the differing Q31/final-shift rounding increments.

Final fixture SHA256 is
`91affaa44efd90848be4214cce93d8ca86f4e0110fd62b1ced6d5b4aab83638d`; logical
instruction SHA256 is
`300a7619e44448203bbd61a5e4d717469b07a725775b9c501ecd6be92a55e9a9`. The
report identifies CMSIS-NN 8.0.0 at
`13c97dbb6f781d4aab38ed34e6e441f42b79aff4`, model SHA256
`3c002613d1b2475eb51dd78dfb85a546c8ae658dee71cf6ade43b022fe205415`, input
SHA256 `9f2b799d8a7d23ea057764d98841e08038753212d8fcb06b4b8f28a859926511`,
TVM 0.17.0, and the VTA 64mac config SHA256
`23b338eacdf5747610d90fd17296e3d0d4236ce416191b7c1cfc597cd67991fa`. The
selected TSIM library is `vta/build/libvta_tsim.dylib` (SHA256
`5b3fd9b613ba46eca279f25aea6f42f32761bdaf21caaff08537147d9ae861f8`); its
hardware library is `vta/build/libvta_hw.dylib` (SHA256
`4570aa8efc3b054bc6feabf1ccfbb8df6674c1831b3b4743dc9d3ab1694ed53c`). Full
machine-readable metadata and all rounding evidence are retained in the
ignored generated file `vta/tests/qconv2d/reports/tsim/task5-validation.json`.

The probes initialize `vta.tsim` in their own process and verify the selected
backend ABI. Immediate uops use matching source/destination indices for the
FSIM/RTL immediate-read convention; snapshot and convolution clamp operations
use dedicated in-place uops. Chunk size includes per-stage, copy, and snapshot
uop banks. The non-target runtime rejects unsupported opcode/rounding on
Xilinx and Intel targets. Intel's old kernel has opcode-4 MUL but not RMUL,
RSFT, or rounding. Xilinx `VTADeviceRun` now scans raw instructions in the
tracked CMA buffer and rejects unsupported ALU opcodes or nonzero rounding
before enqueue. HLS assertions remain a C/HLS simulation defense; the
pre-enqueue check does not depend on synthesized HLS assertion behavior.

The final Task 5 run modified no files under `vta/apps`; generated reports and
logs remain ignored. `git diff --check` passed.

## Review follow-up

The default `scripts/test_vta_fsim.sh` and `scripts/test_vta_tsim.sh` lists do
not include `test_dwc.py`; the DwC evidence was therefore collected separately
after the Task 5 changes using the final `vta_64mac` libraries:

- `env VTA_BACKEND=fsim VTA_PATH="$PWD/vta" TVM_PATH="$PWD/tvm" VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" PYTHONPATH="$PWD/vta/python:$PWD/tvm/python" /Users/xdl/Projects/codex-tvm-vta/.envs/tvm-vta-env/bin/python -m pytest -q vta/tests/python/unittest/test_dwc.py`: 29 passed in 10.64 s.
- The same command with `VTA_BACKEND=tsim`: 29 passed in 8.21 s.
- `test_depthwise_conv2d.py::test_real_kws_layer_signed_int32` on FSIM and TSIM, with the same `vta_64mac` config and geometry `[8,8]`: 1 passed on each backend (2.96 s and 3.23 s). The FSIM acceptance summary reported 9,000 DwC operations and 8,000 output bytes.

Xilinx `VTADeviceRun` now resolves the raw instruction physical range only
within registered CMA allocations, checks the full stream length including
offset, and rejects unsupported opcode or nonzero rounding before writing
device registers. Allocation registration, lookup, and removal are protected
by a mutex. The focused host test
`pytest -q vta/tests/python/unittest/test_pynq_alu_guard.py` passed and covers
legacy opcodes 0–4, RMUL rejection, nonzero-rounding rejection, and CMA range
boundaries. The PYNQ driver passed host `-fsyntax-only` compilation with the
64mac ABI and temporary declarations for the unavailable board-only CMA API;
no Xilinx vendor/device toolchain was available.

After correcting the final report summary, `bash
vta/tests/qconv2d/run_tests.sh --backend tsim --conv-only` passed 14 tests and
regenerated `reports/tsim/task5-validation.json`. The required generic TFLite
count and max-difference now both describe default double rounding (0); explicit
single-rounding fields report 13 differences and max absolute difference 1.
CMSIS/FSIM and FSIM/TSIM counts remain zero.

## Final whole-branch review fixwave

The final review's four Important findings and one Minor finding are resolved.
FSIM SHIFT now follows the RTL low-five-bit count contract for both directions;
unsigned magnitude arithmetic defines `INT32_MIN` without signed overflow. The
new `VTAPushALUOpEx(..., rounding, expected_opcode)` API keys cached kernels by
signature, opcode, and rounding and checks the initializer against both
expected values. The legacy `VTAPushALUOp` and `VTAUopPush` signatures and
rounding-zero behavior remain covered. The shared range checker validates the
default double-rounding pre-left shift as well as the single-rounding shift,
using division bounds for both signed endpoints. The multiplier converter now
matches TensorFlow 2.15 positive ties-away rounding and flushes shifts below
-31, with literal edge expectations independent of the converter. The extra
blank line at `backend_init.h` EOF has been removed.

TDD red runs reproduced the defects: the range/multiplier selection failed on
both TF 2.15 edge values and accepted overflowing double pre-left; the FSIM
ALU/runtime run had 22 passes and failed only on large-count SHIFT and opcode
cache reuse. After the fixes:

- `bash scripts/build_vta_lib.sh --config "$PWD/vta/config/vta_64mac.json" --backend fsim --jobs 4` built the private FSIM library successfully.
- `bash vta/tests/qconv2d/run_tests.sh --backend fsim --alu-only`: 25 passed.
- `bash vta/tests/qconv2d/run_tests.sh --backend tsim --alu-only`: 25 passed.
- Focused runtime capture tests: 8 passed, including same-handle/same-signature/same-rounding RMUL then RSFT capture, explicit initializer-opcode mismatch diagnostic, and legacy API behavior.
- Focused range, converter, and pinned-fixture parameter checks: 13 passed. Full `--conv-only` runs also passed the new helper cases.
- `bash vta/tests/qconv2d/run_tests.sh --backend fsim --conv-only`: 18 passed.
- `bash vta/tests/qconv2d/run_tests.sh --backend tsim --conv-only`: 18 passed.
- Reprofiled `/private/tmp/vta-final-review-checks/check.py`: FSIM and TSIM both returned `1 1 0 2 -7 -7 0 1` for the eight large-count SHIFT cases; runtime capture produced opcodes 5 then 6 at rounding 0 under the same handle and signature.
- Root and VTA full-range `git diff --check` passed, including the earlier committed VTA changes.

The regenerated ignored `reports/tsim/task5-validation.json` records 0
CMSIS/FSIM and 0 FSIM/TSIM differences; default double rounding has 0
TFLite/FSIM differences with maximum absolute difference 0. Single rounding
retains 13 individually attributed TFLite differences, each with maximum
absolute difference 1. The converter fix leaves the pinned 16 per-channel
multiplier/shift values unchanged. Fixture SHA256 remains
`91affaa44efd90848be4214cce93d8ca86f4e0110fd62b1ced6d5b4aab83638d` and
logical instruction SHA256 remains
`300a7619e44448203bbd61a5e4d717469b07a725775b9c501ecd6be92a55e9a9`. Final
runtime/build hashes are recorded for both backend libraries: FSIM
`0cfa583deb1e5362421fa838cd444b07d3a4b8b46132ed5938e9fe2840c48ebf`, TSIM
`5b3fd9b613ba46eca279f25aea6f42f32761bdaf21caaff08537147d9ae861f8`, and
unchanged Chisel hardware library
`4570aa8efc3b054bc6feabf1ccfbb8df6674c1831b3b4743dc9d3ab1694ed53c`.


## TSIM completion timeout follow-up

The final TSIM acceptance run exposed an intermittent 100,000-case mismatch
whose original cause is still unproven. During diagnosis, a one-status-poll
TSIM run showed that the driver returned success even though the completion bit
was clear (`done=0`, `cycle_count=0`). The driver now returns the existing
nonzero timeout status and prints a diagnostic containing the status-poll
budget, instruction count, final status, and hardware `cycle_count`; the
existing pause behavior is unchanged. The runtime already checks that
`VTADeviceRun` returns zero before resetting instruction buffers.

Both ALU and convolution probes now use a default budget of 100,000 status
polls and still stop at the first observed completion. A diagnostic run of
full 2,048-case chunks observed 8,407 hardware cycles and 394–974 host status
reads before completion. These are different units; the budget is more than
102 times the highest observed status-read count. The original zero-output
chunk did not recur during exact replay or the final fixed-seed run, so timeout
is a demonstrated false-success mode but is not established as the cause of
that earlier mismatch.

- A one-case TSIM probe with `--wait-cycles 1` fails with a nonzero status and
  the `TSIM timeout` diagnostic. Before the fix, this same regression returned
  success with completion clear.
- `bash vta/tests/qconv2d/run_tests.sh --backend tsim`: 47 passed in 39.94 s,
  including the timeout regression, 100,000 fixed-seed cases per rounding
  mode, and the convolution checks.
- The FSIM ABI/link smoke passed (1 test); no FSIM library rebuild was needed
  for this TSIM-driver-only follow-up.
- Final backend hashes: TSIM
  `206bbe5e0a114a4c710b8c65711c8b9c477d312bdc110c6c9ac8ea8f7635b048`, FSIM
  `0cfa583deb1e5362421fa838cd444b07d3a4b8b46132ed5938e9fe2840c48ebf`, and
  unchanged Chisel hardware `4570aa8efc3b054bc6feabf1ccfbb8df6674c1831b3b4743dc9d3ab1694ed53c`.
  The ignored `reports/tsim/task5-validation.json` metadata matches these
  library hashes and records zero FSIM/TSIM mismatches.
