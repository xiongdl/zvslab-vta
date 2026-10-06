# Anomaly Detection V1

This application demonstrates selected-target deployment and workload-based
VTA tuning for the MLPerf Tiny ToyCar autoencoder. It processes one WAV,
selects the first available 640-value log-mel feature vector, executes the
model once, and reports the reconstruction and its MSE. The MSE is a workflow
measurement only; this app does not apply a normal/anomaly threshold or score
multiple windows.

The local `python/` package owns model import and preprocessing, selected
compilation and execution, authenticated graph/workload artifacts, isolated
measurements, tuning, schedule validation, and transactional publication. It
does not import another app or `vta/apps/common`. The imported float32 model is
checked for one `(1, 640)` input, one `(1, 640)` output, and the supported ten
fully-connected operator topology. Its default asset hash is provenance;
custom model paths are accepted when they satisfy that contract.

## Prerequisites and setup

Run commands from the repository root. Use the existing `.envs/tvm-vta-env`,
initialized `tvm/` and `vta/` submodules, built TVM libraries, and FSIM/TSIM
libraries. Do not reinstall dependencies for this workflow.

```bash
APP=vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1
PYTHON="$PWD/.envs/tvm-vta-env/bin/python"
CONFIG="$PWD/vta/config/vta_64mac.json"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/$APP"
```

If libraries have not yet been built, follow the repository commands in
[`scripts/README.md`](../../../../scripts/README.md). FSIM and TSIM runtime
commands below select their backend explicitly; `CONFIG` contains geometry
only.

Default assets are `model/ad01_fp32.tflite` and
`samples/normal_id_01_00000000.wav`. Input audio must be mono PCM16 at 16 kHz.
The preprocessing code uses the app's deterministic log-mel calculation and
reports both the total feature vectors available and the one executed vector.

## Manual acceptance

Run the full local contract suite first. It covers model and sample provenance,
preprocessing, one-window selection, CPU startup without VTA, selected-target
validation, graph/workload integrity, Make argument quoting/order, cleanup,
candidate-process isolation, and tuning storage behavior.

```bash
VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=fsim \
  "$PYTHON" -m pytest -q --import-mode=importlib "$APP/tests"
```

Run CPU deployment for each supported host code generator. CPU startup works
with VTA variables unset and does not import VTA or load a simulator. Each run
prints one MSE and the total available versus executed windows.

```bash
env -u VTA_BACKEND -u VTA_CONFIG_FILE "$PYTHON" "$APP/deploy.py" --target c
env -u VTA_BACKEND -u VTA_CONFIG_FILE "$PYTHON" "$APP/deploy.py" --target llvm
```

Run the VTA-selected target matrix. This model currently produces real VTA
partitions. Each command compiles only the selected VTA-plus-host target and
executes one feature vector. Reports distinguish CPU placement from VTA
placement; FSIM cycle counts are `N/A`.

```bash
for backend in fsim tsim; do
  export VTA_BACKEND="$backend" VTA_CONFIG_FILE="$CONFIG"
  for host in c llvm; do
    "$PYTHON" "$APP/deploy.py" --target "vta,$host" --simulator "$backend" \
      --deployment-report "$APP/build/${backend}-${host}.md"
  done
done
```

Export actual pre-schedule VTA Relay functions and the activation captured at
each outlined workload. The exported file contains model, input,
preprocessing/quantization, raw-config, geometry, and TVM/VTA provenance. Then
remove or rename the model and WAV temporarily, and load the snapshot in a
separate tuning process to confirm tuning uses only the export. Restore both
files before deployment tests.

```bash
export VTA_BACKEND=fsim VTA_CONFIG_FILE="$CONFIG"
"$PYTHON" "$APP/deploy.py" --target vta,llvm --simulator fsim \
  --export-workloads "$APP/build/workloads.json"
```

Run all nine real VTA occurrences through FSIM candidate search and TSIM
selection, then replay the complete schedule. `WORKLOAD=-1`, one candidate
batch, and one successful candidate per occurrence keep the manual run
bounded. TSIM chooses the lowest measured cycles for each workload and replay
reports full selected coverage plus per-layer and whole-model cycles.

```bash
"$PYTHON" "$APP/tune.py" --workloads "$APP/build/workloads.json" \
  --workload -1 --simulator fsim --trial-batch 1 --min-successful 1 \
  --timeout 120 --output-logs "$APP/tune/vta_64mac/fsim.tmp"
VTA_BACKEND=tsim "$PYTHON" "$APP/tune.py" \
  --workloads "$APP/build/workloads.json" --workload -1 --simulator tsim \
  --input-logs "$APP/tune/vta_64mac/fsim.tmp" \
  --timeout 120 --output-logs "$APP/tune/vta_64mac/best.log"
VTA_BACKEND=tsim "$PYTHON" "$APP/deploy.py" --target vta,llvm --simulator tsim \
  --schedule "$APP/tune/vta_64mac/best.log" \
  --deployment-report "$APP/build/replay.md"
```

The same split and full workflow are available through Make. `tune` exports
workloads only when `WORKLOADS` is absent, runs FSIM then TSIM, and stops after
TSIM; it does not silently redeploy the selected schedule. Variables accept
paths with spaces and relative paths resolve from the command's working
 directory.

```bash
make -C "$APP" deploy TARGET=llvm
make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
  EXPORT_WORKLOADS=build/workloads.json REPORT=build/default.md
make -C "$APP" tune-fsim WORKLOADS=build/workloads.json WORKLOAD=-1 \
  TRIAL_BATCH=1 MIN_SUCCESSFUL=1
make -C "$APP" tune-tsim WORKLOADS=build/workloads.json WORKLOAD=-1 \
  INPUT_LOGS=tune/vta_64mac/fsim.tmp
make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=tsim \
  SCHEDULE=tune/vta_64mac/best.log REPORT=build/replay.md
make -C "$APP" tune OUTPUT_DIR=build/full-tune WORKLOAD=-1 \
  TRIAL_BATCH=1 MIN_SUCCESSFUL=1 FSIM_TIMEOUT=120 TSIM_TIMEOUT=120
```

Check that `fsim.tmp` contains native AutoTVM candidate records and a same-stem
metadata file; `best.log` and its metadata identify the model, config, and
selected occurrence. The deployment report lists hashes, selected target,
preprocessing and fixed quantization policies, total/selected windows, reconstruction MSE, CPU/VTA
placement, dense-derived MAC counts, schedule coverage, and available
measurements. TSIM replay must agree with the prepared CPU reference output;
measured selected-layer cycles must align with candidate TSIM records to
strictly less than 10 percent.

Negative checks fail before publishing an artifact: CPU workload export is
rejected; an invalid audio format or mismatched model shape/dtype/topology is
rejected; altered workload, model/config identity, or schedule metadata is
rejected; and TSIM tuning requires FSIM candidate logs. Unsupported operators
remain on CPU. If a future model/config yields no real VTA partitions, normal
VTA-target execution reports zero coverage and `N/A` cycles, while export and
schedule replay fail with `no real VTA workloads`; no empty winner is emitted.

Finally, verify safe cleanup. It removes only local build output and Python
caches. It preserves model, WAV samples, licenses, and persistent tune data,
including when given a custom output directory.

```bash
make -C "$APP" clean
make -C "$APP" clean
 test -f "$APP/model/ad01_fp32.tflite"
 test -f "$APP/samples/normal_id_01_00000000.wav"
 test -f "$APP/LICENSE.mlperf-tiny"
```
