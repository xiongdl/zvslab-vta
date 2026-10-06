# Streaming Wakeword V1

This directory owns one standalone deployment and tuning workflow for the
MLPerf Tiny streaming wakeword model. It prepares one mono, 16-bit, 16 kHz WAV
as one `[1, 30, 1, 40]` int8 input, runs the model without carrying state, and
prints the winning class index (`Marvin`, `Silence`, or `Unknown`) with its raw
int8 scores. The default model and sample are the committed files under
`model/` and `samples/`.

## Prerequisites

Run commands from the repository root. Use the existing `.envs/tvm-vta-env`
Python environment and initialized `tvm/` and `vta/` submodules. CPU targets
need the TVM runtime and libraries. A VTA request also needs a built VTA
extension, an absolute geometry file, and matching `VTA_BACKEND` and
`--simulator` values. Build FSIM and TSIM with the repository instructions in
[`scripts/README.md`](../../../../scripts/README.md).

```bash
bash scripts/build_tvm_lib_macos.sh
bash scripts/build_vta_lib.sh --config "$PWD/vta/config/vta_64mac.json" --backend all
```

## Selected deployment

The Makefile defaults to `TARGET=vta,llvm`, `SIMULATOR=fsim`, and the committed
model and Marvin WAV. Each command builds and runs only its selected target.

```bash
APP=vta/apps/mlperf_tiny_benchmark/streaming_wakeword_v1
make -C "$APP" deploy TARGET=c
make -C "$APP" deploy TARGET=llvm
make -C "$APP" deploy TARGET=vta,c SIMULATOR=fsim
make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim
```

For CPU-only use, remove VTA selectors from the environment:

```bash
env -u VTA_BACKEND -u VTA_CONFIG_FILE make -C "$APP" deploy TARGET=llvm
```

The direct CLI accepts the same four targets. Relative model, input, report,
output, schedule, and workload paths resolve from the directory where the
command is run.

```bash
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/$APP" \
  .envs/tvm-vta-env/bin/python "$APP/deploy.py" \
  --model "$APP/model/str_ww_ref_model.tflite" \
  --input "$APP/samples/silence-doing_the_dishes-00000000.wav" \
  --target llvm --output-dir "$APP/build/custom" \
  --deployment-report "$APP/build/custom/report.md"
```

A completed deployment prints the class index and label, then the three raw
int8 scores. The Markdown report records model and input hashes, selected and
actual execution targets, CPU/VTA placement, logical MAC counts, schedule
coverage, raw output, and available measurements. CPU and FSIM cycle counts
are reported as N/A.

The VTA request currently finds **zero real VTA partitions** after exact QNN
canonicalization. It runs the selected CPU fallback without loading an FSIM or
TSIM simulator, reports zero VTA coverage and N/A cycles, and does not claim
that model computation ran on VTA. Exact imported QNN and canonicalized CPU
outputs match on all three committed WAVs. Per-axis multipliers, shifts, zero
points, and output additions remain in the original canonicalized arithmetic;
no neutral VTA probe is attached.

## Workload export and tuning

The workload interface is ready for snapshots containing actual outlined VTA
functions and their captured int8 activations. This model currently produces
no such functions. Consequently, workload export and schedule replay stop with
`no real VTA workloads` before publishing a workload or schedule. There is no
FSIM candidate log, TSIM winner, or replay result for this model today.

Run these commands to verify the supported rejection paths:

```bash
CONFIG="$PWD/vta/config/vta_64mac.json" \
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
  EXPORT_WORKLOADS="$APP/build/workloads.json"

CONFIG="$PWD/vta/config/vta_64mac.json" \
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
  SCHEDULE="$APP/tune/vta_64mac/best.log"

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  make -C "$APP" tune
```

Each command exits with an explicit `no real VTA workloads` error. The full
`make tune` path stops during deployment workload export before FSIM search or
TSIM selection. The separate `tune-fsim` and `tune-tsim` targets require a
workload snapshot; a missing or empty snapshot is rejected by the loader. The
standalone `tune.py` interface is:

```text
--workloads PATH --workload -1|INDEX --simulator fsim|tsim
--timeout SECONDS --output-logs PATH
FSIM: --trial-batch N --min-successful N
TSIM: --input-logs PATH
```

FSIM searches exported occurrences; TSIM measures those candidates and
selects a schedule. Neither stage reloads a model or source WAV. Because no
real VTA occurrence is available, no positive tuning invocation is valid for
the committed model. Do not create empty logs or a synthetic winner to bypass
the rejection.

## Manual acceptance

From the repository root, execute the commands below. VTA build prerequisites
are described above; all deployments use the committed default model unless a
custom path is supplied.

1. Verify both CPU code generators work without VTA environment variables:

   ```bash
   env -u VTA_BACKEND -u VTA_CONFIG_FILE make -C "$APP" deploy TARGET=c
   env -u VTA_BACKEND -u VTA_CONFIG_FILE make -C "$APP" deploy TARGET=llvm
   ```

   Each command prints a class index and label plus three raw int8 scores.

2. Verify both VTA host code generators and both backend selections use the
   truthful CPU fallback. These four commands do not load simulator libraries:

   ```bash
   for backend in fsim tsim; do
     for host in c llvm; do
       VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND="$backend" \
         make -C "$APP" deploy TARGET="vta,$host" SIMULATOR="$backend" \
         REPORT="$APP/build/$backend-$host.md"
     done
   done
   ```

   Each report says `VTA coverage: 0`, identifies the selected host codegen as
   the actual execution target, gives the fallback reason, and reports N/A
   cycles. No simulator activity or VTA measurements are claimed.

3. Verify workload export and schedule replay reject the zero-workload model:

   ```bash
   VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
     make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
     EXPORT_WORKLOADS="$APP/build/workloads.json"
   VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
     make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
     SCHEDULE="$APP/tune/vta_64mac/best.log"
   ```

   Both commands exit nonzero with `no real VTA workloads`; no workload file,
   candidate log, schedule, or winner is published.

4. Verify the full tuning sequence stops at export, before search or selection:

   ```bash
   VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
     make -C "$APP" tune
   ```

   It exits nonzero with the same explicit reason and produces no tuning
   result. A positive FSIM/TSIM/replay sequence cannot be demonstrated until
   the compiler supports at least one real operation while preserving the
   model's fixed-point arithmetic.

5. Verify cleanup is repeatable and retains persistent assets and tune history:

   ```bash
   make -C "$APP" clean
   make -C "$APP" clean
   test -f "$APP/model/str_ww_ref_model.tflite"
   test -f "$APP/samples/marvin-00176480_nohash_0.wav"
   test -f "$APP/LICENSE.mlperf-tiny"
   test ! -e "$APP/build"
   ```

`clean` removes local build output and Python caches only. It keeps model,
samples, and license assets. This graph currently produces zero real VTA
workloads, so there is no usable VTA tuning schedule to retain.
