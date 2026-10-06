# Streaming Wakeword V1

This app takes one mono, 16-bit, 16 kHz WAV, computes the existing normalized
log-mel features as float32 `[1, 30, 1, 40]`, and runs the app-owned float32
TFLite model. See [model provenance](model/README.md). The app directory stores
only that float32 TFLite model; its H5 source stays in the ignored MLPerf Tiny
training tree.

Run from the repository root with `.envs/tvm-vta-env` and initialized `tvm/`
and `vta/` submodules. CPU deployment needs TVM. VTA deployment also needs the
VTA extension built with an absolute geometry file and matching backend:

```bash
bash scripts/build_tvm_lib_macos.sh
bash scripts/build_vta_lib.sh --config "$PWD/vta/config/vta_64mac.json" --backend fsim
```

The Makefile defaults to the committed float TFLite, Marvin WAV, and
`TARGET=vta,llvm SIMULATOR=fsim`. `MODEL` is passed to both deploy and tune
CLIs.

```bash
APP=vta/apps/mlperf_tiny_benchmark/streaming_wakeword_v1
env -u VTA_BACKEND -u VTA_CONFIG_FILE make -C "$APP" deploy TARGET=llvm
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
  EXPORT_WORKLOADS="$APP/build/workloads.json" \
  REPORT="$APP/build/fsim-report.md"
```

The direct deployment CLI accepts all four targets `c`, `llvm`, `vta,c`, and
`vta,llvm`. Both CPU and mixed deployment import and validate the float TFLite
`--model`; their CPU reference and mixed graph use the same TVM global-scale
quantized Relay graph. Audio features are passed to the model as float32 with
no former int8 scale/zero-point conversion.

```bash
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/$APP" \
  .envs/tvm-vta-env/bin/python "$APP/deploy.py" \
  --model "$APP/model/str_ww_ref_model_float32.tflite" \
  --input "$APP/samples/marvin-00176480_nohash_0.wav" \
  --target vta,llvm --simulator fsim \
  --export-workloads "$APP/build/workloads.json" \
  --deployment-report "$APP/build/fsim-report.md"
```

## Tuning

Tuning imports the supplied float32 TFLite model and validates its SHA-256
against the deployment-exported workload snapshot before searching or replay.
FSIM verifies candidate outputs; TSIM measures those candidates in cycles and
selects a schedule per VTA occurrence. The schedule sidecar ties the native
log to the float model, actual occurrence compute, VTA geometry, and simulator
measurements.

```bash
# Export actual workloads from the deployed float model.
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  make -C "$APP" deploy MODEL="$APP/model/str_ww_ref_model_float32.tflite" \
  TARGET=vta,llvm SIMULATOR=fsim \
  EXPORT_WORKLOADS="$APP/build/workloads.json"

# Bounded smoke search: one verified candidate for each occurrence.
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  make -C "$APP" tune-fsim MODEL="$APP/model/str_ww_ref_model_float32.tflite" \
  WORKLOADS="$APP/build/workloads.json" WORKLOAD=-1 \
  TRIAL_BATCH=1 MIN_SUCCESSFUL=1 TIMEOUT=60

# Select by measured TSIM cycles, then replay the selected schedule.
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
  make -C "$APP" tune-tsim MODEL="$APP/model/str_ww_ref_model_float32.tflite" \
  WORKLOADS="$APP/build/workloads.json" \
  INPUT_LOGS="$APP/tune/vta_64mac/fsim.tmp" WORKLOAD=-1 TIMEOUT=120
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
  make -C "$APP" deploy MODEL="$APP/model/str_ww_ref_model_float32.tflite" \
  TARGET=vta,llvm SIMULATOR=tsim SCHEDULE="$APP/tune/vta_64mac/best.log"
```

The bounded C3 smoke run verified one FSIM candidate and measured one TSIM
candidate for each of four real occurrences. Their selected cycle counts were
166,099, 407,683, 254,683, and 4,357. TSIM replay predicted Marvin and matched
the CPU and FSIM scores. This is smoke evidence, not an exhaustive search or a
performance claim against another implementation. The initiative's
`CHECKPOINT-C3.md` records hashes and the four deployment targets. `make tune`
runs export, FSIM search, and TSIM selection together. `make clean` removes
generated build output and Python caches while preserving the float32 TFLite
model, WAV samples, license, and validated tune evidence.
