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
