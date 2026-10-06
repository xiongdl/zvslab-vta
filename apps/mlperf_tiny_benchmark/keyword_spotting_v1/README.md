# Keyword Spotting V1

This application deploys one WAV sample with the committed int8 KWS model. It
owns MFCC preprocessing, QNN-preserving graph preparation, selected deployment,
workload and schedule validation, tuning interfaces, reports, and Make
orchestration. The input is mono PCM16 at 16 kHz, padded or trimmed to one
second, then transformed to int8 `(1, 49, 10, 1)` using input scale `0.5847029`
and zero point `83`. Output is 12 raw int8 scores and the highest-scoring
keyword label. Accuracy is not an acceptance gate.

The imported TFLite graph contains fixed-point QNN multiplier and per-axis
shift operations. Preparation uses Relay QNN canonicalization only. Its int8
outputs match the imported graph exactly on all 12 committed WAV samples. With
that arithmetic preserved, the current VTA partitioner finds no real VTA
partitions in this model. All computation stays on CPU for `vta,c` and
`vta,llvm`; reports state zero VTA coverage and N/A cycles, and the simulator
is not loaded. Workload export and schedule replay stop with `no real VTA
workloads`. The tune CLI accepts only authenticated workload snapshots and
never creates a synthetic winner.

## Prerequisites and paths

Run from the repository root with initialized `tvm/` and `vta/`, the pinned
`.envs/tvm-vta-env` environment, and built TVM libraries. The four VTA target
selectors also require the absolute geometry config
`$PWD/vta/config/vta_64mac.json` and matching `VTA_BACKEND=fsim|tsim`; the
zero-coverage fallback itself does not load either simulator. Do not install
packages for this application.

```bash
export APP="$PWD/vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1"
export PYTHON="$PWD/.envs/tvm-vta-env/bin/python"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$APP"
export CONFIG="$PWD/vta/config/vta_64mac.json"
```

Defaults are `model/kws_ref_model.tflite`,
`samples/down-00176480_nohash_0.wav`, target `vta,llvm`, simulator `fsim`, and
local `build/`. Custom models must have one int8 input of shape
`(1, 49, 10, 1)`, one int8 output of shape `(1, 12)`, the supported operator
topology and input quantization. Custom input paths must contain mono PCM16
16 kHz WAV audio.

## Manual acceptance

Run each command from the repository root. Successful deployment prints one
keyword index/name and 12 raw int8 scores. Reports include model, WAV, and
configuration hashes, selected target, CPU/VTA placement, coverage, raw output,
and available measurements.

1. Run local tests. They check preprocessing, supported model structure,
   exact imported-QNN/prepared-CPU agreement for all committed samples, graph
   bundle integrity, CLI/Make behavior, tuning input validation, and safe
   cleanup:

   ```bash
   VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=fsim PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" -m pytest -q "$APP/tests"
   ```

2. Run CPU deployment without any VTA environment. Each command must print a
   keyword and 12 scores; the report must show CPU placement and N/A cycles.
   CPU workload export must fail before runtime startup:

   ```bash
   env -u VTA_BACKEND -u VTA_CONFIG_FILE PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target c --deployment-report "$APP/build/c.md"
   env -u VTA_BACKEND -u VTA_CONFIG_FILE PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target llvm --deployment-report "$APP/build/llvm.md"
   env -u VTA_BACKEND -u VTA_CONFIG_FILE PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target c --export-workloads "$APP/build/c.json"
   ```

   The first two succeed; the last command fails because CPU-only deployment
   has no VTA workload to export.

3. Exercise both VTA host code generators and both backend selectors:

   ```bash
   for backend in fsim tsim; do
     for host in c llvm; do
       VTA_BACKEND="$backend" VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
         "$PYTHON" "$APP/deploy.py" --target "vta,$host" --simulator "$backend" \
         --deployment-report "$APP/build/$backend-$host.md"
     done
   done
   ```

   Each succeeds through CPU fallback, prints the same int8 output as its CPU
   counterpart, and reports `0` VTA partitions/coverage, CPU execution, and
   N/A cycles. No simulator profiler or cycle count appears. A mismatched
   backend and `--simulator` must fail before model compilation.

4. Verify that requests requiring actual accelerator computations fail
   without publishing an empty snapshot or a schedule claim:

   ```bash
   VTA_BACKEND=fsim VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target vta,llvm --simulator fsim \
     --export-workloads "$APP/build/workloads.json"
   VTA_BACKEND=tsim VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target vta,llvm --simulator tsim \
     --schedule "$APP/tune/vta_64mac/best.log"
   ```

   Both fail with `no real VTA workloads`; no workload, schedule, or deployment
   bundle is created. Running `make tune` without a workload snapshot reaches
   the same export failure before FSIM or TSIM tuning starts. A tune request
   with a nonexistent, tampered, or foreign-model workload file also fails and
   publishes no winner.

5. Check direct custom paths and Make routing, including paths with spaces:

   ```bash
   mkdir -p "$APP/build/custom path"
   cp "$APP/model/kws_ref_model.tflite" "$APP/build/custom path/model.tflite"
   cp "$APP/samples/down-00176480_nohash_0.wav" "$APP/build/custom path/input.wav"
   env -u VTA_BACKEND -u VTA_CONFIG_FILE PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target llvm \
     --model "$APP/build/custom path/model.tflite" \
     --input "$APP/build/custom path/input.wav" \
     --output-dir "$APP/build/custom path/direct bundle" \
     --deployment-report "$APP/build/custom path/direct report.md"
   make -C "$APP" deploy TARGET=c REPORT="$APP/build/make report.md"
   make -C "$APP" deploy TARGET=llvm \
     MODEL="$APP/build/custom path/model.tflite" \
     INPUT="$APP/build/custom path/input.wav" \
     OUTPUT_DIR="$APP/build/custom path/make bundle" \
     REPORT="$APP/build/custom path/make report.md"
   make -C "$APP" tune
   ```

   Direct and Make CPU deployments succeed and preserve the quoted paths.
   `make tune` fails with the same clear zero-coverage explanation and does not
   enter either tuning stage.

6. Confirm safe, idempotent cleanup:

   ```bash
   make -C "$APP" clean
   make -C "$APP" clean
   ```

   Both commands succeed. Local build output and Python caches are removed;
   model, samples, and licenses remain. No VTA schedule is retained because this
   graph currently produces zero real VTA workloads.

## CLI and Make contract

`deploy.py` accepts `--model`, `--input`, `--target c|llvm|vta,c|vta,llvm`,
`--simulator fsim|tsim`, `--schedule`, `--output-dir`, `--deployment-report`,
and `--export-workloads`. It compiles one selected graph. CPU startup does not
import VTA. Requested VTA targets require a matching `VTA_BACKEND` and
`--simulator`, plus an absolute existing `VTA_CONFIG_FILE`. Given the current
zero real partitions, those targets use CPU fallback and report the limitation.

`tune.py` consumes only an exported workload snapshot. Its FSIM stage accepts
`--trial-batch`, `--min-successful`, and `--timeout`; TSIM requires
`--input-logs` and rejects FSIM-only options. The real KWS graph currently
cannot produce workloads, so both tuning stages reject missing or empty
workload data instead of inventing candidates or cycles. The former
arithmetic-rewriting flow produced schedule records, but they do not apply to
the current exact-QNN graph and have been removed. This model has no usable VTA
schedule until the compiler produces a real VTA workload. `make deploy`,
`make tune-fsim`, `make tune-tsim`, `make tune`, and `make clean` are the local
entrypoints. Cleanup preserves model assets.
