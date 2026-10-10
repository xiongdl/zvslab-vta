# Keyword Spotting V1

This application deploys the all-float MLPerf Tiny v1.4 KWS model. Its only
model asset is `model/kws_ref_model_float32.tflite`, exported from the upstream
float32 SavedModel and authenticated by SHA-256
`738a9f29d175aaa3928db9c8281265be5ec3406598fd3d30018b26084a3d5536`. The
FlatBuffer has float32 input `[1, 49, 10, 1]` and output `[1, 12]`. Audio is
mono PCM16 at 16 kHz, padded or trimmed to one second, then transformed to
float32 MFCC features. Relay applies the image-classification VTA policy
`global_scale=8.0, skip_conv_layers=[0]` once; CPU and mixed execution use that
same quantized Relay graph.

The model partitions into four genuine VTA convolution regions. A mixed FSIM
deployment compiles and executes all four regions, exports their real
workloads and input activations, and matches the CPU graph's float32 scores.
The committed `down` sample predicts class 0 (`Down`). Workload export and
tuning are tied to the selected model hash; `tune.py` requires `--model` and
rejects a workload snapshot from another model.

## Running

Run from the repository root with initialized `tvm/` and `vta/`, the pinned
`.envs/tvm-vta-env` environment, and built TVM/VTA libraries:

```bash
export APP="$PWD/vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1"
export PYTHON="$PWD/.envs/tvm-vta-env/bin/python"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$APP"
export CONFIG="$PWD/vta/config/vta_64mac.json"
```

CPU deployment does not require a VTA backend:

```bash
env -u VTA_BACKEND -u VTA_CONFIG_FILE PYTHONPATH="$PYTHONPATH" \
  "$PYTHON" "$APP/deploy.py" --target llvm
```

Run a mixed FSIM deployment and export the workloads it actually executes:

```bash
VTA_BACKEND=fsim VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
  "$PYTHON" "$APP/deploy.py" --target vta,llvm --simulator fsim \
  --model "$APP/model/kws_ref_model_float32.tflite" \
  --input "$APP/samples/down-00176480_nohash_0.wav" \
  --export-workloads "$APP/build/workloads.json" \
  --deployment-report "$APP/build/fsim.md"
```

Both `deploy.py` and `tune.py` accept a float TFLite `--model`. `Makefile`
forwards `MODEL` to deployment and both tuning stages. For example:

```bash
make -C "$APP" deploy TARGET=llvm MODEL="$APP/model/kws_ref_model_float32.tflite"
make -C "$APP" tune-fsim MODEL="$APP/model/kws_ref_model_float32.tflite" \
  WORKLOADS="$APP/build/workloads.json"
```

Tuning requires a workload snapshot exported from the same float32 TFLite
model. FSIM candidates are checked against the deployed output. TSIM measures
the candidates in cycles and selects one schedule per VTA occurrence. The
logs under `tune/vta_64mac/` record the model, workload, and geometry identities;
deployment validates those identities before replay.

```bash
# Deploy and export the actual four VTA occurrences.
VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=fsim make -C "$APP" deploy \
  MODEL="$APP/model/kws_ref_model_float32.tflite" TARGET=vta,llvm \
  SIMULATOR=fsim EXPORT_WORKLOADS="$APP/build/workloads.json"

# Bounded smoke search: one verified FSIM candidate per occurrence.
VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=fsim make -C "$APP" tune-fsim \
  MODEL="$APP/model/kws_ref_model_float32.tflite" \
  WORKLOADS="$APP/build/workloads.json" WORKLOAD=-1 \
  TRIAL_BATCH=1 MIN_SUCCESSFUL=1 TIMEOUT=60

# Measure each candidate in TSIM, select the minimum-cycle schedules, then replay.
VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=tsim make -C "$APP" tune-tsim \
  MODEL="$APP/model/kws_ref_model_float32.tflite" \
  WORKLOADS="$APP/build/workloads.json" \
  INPUT_LOGS="$APP/tune/vta_64mac/fsim.tmp" WORKLOAD=-1 TIMEOUT=120
VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=tsim make -C "$APP" deploy \
  MODEL="$APP/model/kws_ref_model_float32.tflite" TARGET=vta,llvm \
  SIMULATOR=tsim SCHEDULE="$APP/tune/vta_64mac/best.log"
```

The C3 smoke run exported the same four real occurrences, verified one FSIM
candidate for each, and measured one TSIM candidate for each. The selected
occurrence measurements were 565,916 cycles each; TSIM replay produced the
same Down scores as CPU and FSIM. This is bounded smoke evidence, not an
exhaustive search or a performance claim against another implementation. See
the initiative's `CHECKPOINT-C3.md` for hashes, all target selectors, and
reproduction commands. `make tune` runs export, FSIM search, and TSIM selection
in one command; `make clean` removes only local build and Python cache output,
preserving the model, samples, license, and validated tuning files.

## Validation

Run the focused suite with the project environment:

```bash
VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=fsim PYTHONPATH="$PYTHONPATH" \
  "$PYTHON" -m pytest -q "$APP/tests"
```

The tests cover the float FlatBuffer contract and provenance, deterministic
float feature preprocessing, quantized CPU graph preparation, four actual
VTA regions, model-hash validation, Make argument forwarding, and the selected
deployment/report workflow. `make clean` removes build output and Python
caches while preserving the model, audio samples, and license.

### Native depthwise layer acceptance

The first quantized depthwise convolution (TFLite operator 1) can be extracted
from the checked-in float model and `samples/right-00b01445_nohash_0.wav`:

```python
from keyword_spotting_v1.python.dwc_sample import extract_dwc_sample
sample = extract_dwc_sample()
```

Extraction uses this application's existing MFCC preprocessing and Relay
quantization policy. It returns signed int8 activation `[1,25,5,64]`, signed
int8 weights `[3,3,64,1]`, stride `(1,1)`, four-side padding `(1,1,1,1)`, and
an int32 CPU reference `[1,25,5,64]`. The observed Relay layouts are NHWC and
HWOI; depth multiplier is one. A separate scalar integer convolution must
match the Relay output exactly. SHA256 hashes identify the model, WAV, logical
activation, weights, and reference.

From the project root, run all three physical geometries (8×8, 8×16, 16×8):

```sh
scripts/test_vta_dwc.sh
# Optional custom artifact directory:
scripts/test_vta_dwc.sh /absolute/path/to/acceptance-artifacts
```

The runner rebuilds each matching compiler extension/runtime in fresh
processes. All three FSIM numerical runs and regressions finish before TSIM
starts. TSIM regenerates Chisel and Verilator output for each geometry;
existing dependency caches are reused. Each layer uses the production native
TOP compute/schedule, all real input channels, and nine kernel taps. Four
ordinary int8 stores after shifts by 0/8/16/24 reconstruct all 8000 signed
int32 outputs and compare them exactly. Every physical channel retains real
nonzero data, including expanded upper channels and both reverse subblocks.

The default artifact directory is `vta/build/dwc-acceptance`, outside the
implementation planning workspace. Per-run artifacts include the full interleaved instruction/queue trace,
configuration, library copies and SHA256 fingerprints, logical/packed tensors
and result, profiler summary, existing GEMM benchmark results, and backend/ISA
results. FSIM must report native DwC counts (9000, 4500, 9000 per byte for the
three geometries) with zero GEMM work in the depthwise layer. TSIM must complete
all four commands and the numerical comparison with positive cycle counts;
loading or initializing its libraries alone is insufficient. Opcode 5 is
visible as `DWC` in runtime instruction dumps. Bridged STORE→COMPUTE→LOAD
queue tokens and completion remain in the trace. The runner rebuilds default
8×8 libraries on exit. It accepts `DWC_JOBS` to change build parallelism.

This is independent-layer acceptance; it does not route the complete KWS
graph through native depthwise execution or add tuning records.

Acceptance on 2026-10-10: all six runs compared all 8000 int32 outputs exactly
(range −8205 to 6860). Each backend/geometry also passed the existing numerical
GEMM benchmark and 31 backend/ISA tests.

| BI×BO | FSIM native DwC updates per byte | TSIM cycles for bytes 0/1/2/3 |
| --- | ---: | --- |
| 8×8 | 9000 | 35893 / 43755 / 43755 / 43755 |
| 8×16 | 4500 | 20753 / 24712 / 24712 / 24712 |
| 16×8 | 9000 | 35559 / 43718 / 43718 / 43718 |

Final default-geometry checks passed 412 VTA Python unit tests, 10 focused
TOP/extraction tests, and all 83 Chisel tests. The complete-project bare pytest
collection has known unrelated duplicate-module/board-host errors (recorded
in the Task5 report); it was not repeated for this acceptance.
