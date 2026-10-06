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
  --model "$APP/model/str_ww_ref_model_floag32.tflite" \
  --input "$APP/samples/marvin-00176480_nohash_0.wav" \
  --target vta,llvm --simulator fsim \
  --export-workloads "$APP/build/workloads.json" \
  --deployment-report "$APP/build/fsim-report.md"
```

## C2 SWW verification

The supplied float model imports as float32 input/output, and TVM quantization
uses `calibrate_mode=global_scale`, `global_scale=8.0`,
`skip_conv_layers=[0]`. The mixed graph contains four outlined VTA convolution
occurrences. Remaining depthwise operations run on CPU.

With the committed Marvin WAV and `vta/config/vta_64mac.json`, the FSIM mixed
deployment executed all four real regions and exported their actual Relay
functions and int8 VTA activations. The reported float32 scores were
`[0.99998331, 1.570615e-09, 1.6730595e-05]` (class 0, Marvin). The LLVM CPU
deployment produced the same scores. This compares the CPU and mixed forms of
the TVM-quantized graph; these scores are not claimed to equal the float TFLite
source output bit-for-bit.

Workload snapshots record the float model hash, decoded float feature dtype,
quantization policy, WAV hash, VTA geometry, actual Relay functions, and
captured activations. `tune.py` requires both `--model` and `--workloads`,
validates the float TFLite contract, and rejects a model hash that differs
from the snapshot before search or replay. Tuning is gated on both KWS and SWW
having executable VTA regions; KWS import is currently escalated, so no tuning
search or schedule is claimed for this initiative yet.

```text
tune.py --model FLOAT32_TFLITE --workloads SNAPSHOT.json
        --workload -1|INDEX --simulator fsim|tsim
        --timeout SECONDS --output-logs PATH
FSIM: --trial-batch N --min-successful N
TSIM: --input-logs PATH
```

`clean` removes generated build output and Python caches. It retains the float
TFLite model, WAV samples, license, and any valid tune evidence.
