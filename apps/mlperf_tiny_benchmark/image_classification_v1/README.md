# MLPerf Tiny ResNet-8 HOST deployment

This fixed-purpose application imports the committed floating MLPerf Tiny v1.4
ResNet-8 model, applies the documented TVM quantization policy once, and builds
both a pure host reference and a mixed VTA Graph Executor artifact. It reloads
the host libraries, compares their output tensors exactly for the ten committed
PNG samples, and requires positive simulator activity. The matrix mode builds
LLVM and C variants below separate `llvm-fsim/` and `c-fsim/` (FSIM) or
`llvm-tsim/` and `c-tsim/` (TSIM) bundle roots.

Importing `vta` loads and validates the compiler target extension. The mixed
branch explicitly applies `vta.relay.partition_for_vta()` once, then passes
`tvm.target.Target("vta")` to `relay.build`; unsupported operators remain in the
LLVM host portion of the same standard runtime module.

The application is an execution-equivalence example. It does not report model
accuracy, performance, energy, or MLPerf submission results.

## Prerequisites

From the repository root, prepare the pinned Python environment and build the
compiler extension and simulator libraries:

```bash
# Use the existing project environment at .envs/tvm-vta-env; do not recreate it
# during verification.
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" --backend all
```

The active contract is the shared absolute `VTA_CONFIG_FILE` plus
`VTA_BACKEND=fsim|tsim`. The build script uses `--backend fsim|tsim|all` and
this runner uses `--simulator fsim|tsim`; the values must match. The CPU
reference branch is part of the FSIM matrix and is not a separate VTA backend.
`TARGET=sim`, `TARGET=tsim`, and `--target libvta_*` are retired; use the
shared geometry file and explicit backend selectors. FPGA backends such as
`pynq` and `zcu104` remain deferred.

## Run

From the repository root:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/run.py
```

The optional `--output-dir PATH` changes only the generated-artifact location.
The default `build/` directory is ignored by the repository. The default CLI
mode is LLVM FSIM; pass `--host-codegen c` for a single C-host run or
`--host-codegen all` to build and execute the complete ordered LLVM/C matrix.
Pass `--simulator tsim` with `VTA_CONFIG_FILE` set to the shared
`vta/config/vta_64mac.json` and `VTA_BACKEND=tsim` for the Verilated hardware
model. Matrix bundles
are published as `<host>-<simulator>/{reference,mixed}/`, with each directory
containing its Graph JSON, parameters, DSO, manifest, and generated host
source. A partial export is removed if the build fails.

## AutoTVM replay and comparison

The existing commands above build the untuned schedule. To compare that
baseline with history-best schedules, pass both a backend-matched native log
and its JSON sidecar. Replay validates the model hash, backend, config hash,
log hash, and task coverage before compiling the tuned VTA graph. Both builds
run in a fresh process on the same ten committed samples; each output is
compared exactly with the pure LLVM reference and the baseline/tuned tensors
must agree. TSIM also requires the tuned `cycle_count` to be lower.

Example TSIM replay (use the matching FSIM log and `VTA_BACKEND=fsim` for
FSIM):

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/run.py \
  --simulator tsim \
  --autotvm-log vta/apps/mlperf_tiny_benchmark/build/autotvm/image_classification_v1-tsim-<run>.log \
  --autotvm-sidecar vta/apps/mlperf_tiny_benchmark/build/autotvm/image_classification_v1-tsim-<run>.json
```

The result prints model, backend, config, log, and sidecar identities, the
number of compared samples, and baseline/tuned TSIM cycles. These are
simulator cycle counts; they are not FPGA latency or an MLPerf score. Replay
artifacts are written under `build/autotvm-comparison/` by default. Pass
`--output-dir PATH` to choose another artifact location. The untuned default
commands retain their existing behavior.

## Tune IC V1 workloads

The maintained two-stage entry point is `tune/tune.py`. By default it searches
each prepared VTA fusion occurrence in 100-distinct-configuration FSIM batches
until at least 20 distinct successful schedules are found or that task's
configuration space is exhausted. It measures every distinct FSIM success on
TSIM, then selects the lowest positive native `cycle_count`. FSIM and TSIM
workers run as separate processes with 60-second and 120-second per-candidate
timeouts. A failed schedule remains in the progress state and does not stop
later candidates. A bounded run is labeled `BOUNDED_SMOKE_INCOMPLETE`.

Run from the repository root with both simulator libraries built for the same
geometry file. The controller starts each worker with its required backend:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/image_classification_v1" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/tune/tune.py --all
```

For a two-workload smoke with one successful schedule per occurrence:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/image_classification_v1" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/tune/tune.py \
  --all --max-workloads 2 --trial-batch 1 --min-successful 1
```

Use `--workload-index N` to select one zero-based fusion occurrence. `--resume-manifest PATH` continues a compatible run; it rejects changed model, geometry, workload, timeout, or search options and reuses completed FSIM/TSIM records. FSIM/TSIM native logs, per-trial errors, progress, and resume state live below `image_classification_v1/build/two_stage_tuning/`. Self-contained selected native records and their manifest are exported below `tune/optimal/<run-id>/` by default. Set `--artifact-dir PATH` to change that output directory.

Replay checks the model and geometry hashes, full fusion occurrence identity,
workload and schedule configuration, TSIM protocol and selected native-record
hash; it applies the saved Conv configuration to the real outlined fusion. The
replay uses only the files under the artifact directory and does not need the
intermediate `build/` directory:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/image_classification_v1" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/tune/tune.py \
  --replay-manifest vta/apps/mlperf_tiny_benchmark/image_classification_v1/tune/optimal/<run-id>/best-manifest.json
```

The previous single-workload `image_classification_v1/tune.py` command and its
`--workload-index`, `--trials`, `--timeout`, `--output-dir`, and
`--replay-result` options remain available for compatibility. New all-workload
tuning and self-contained replay use `tune/tune.py`.

FSIM matrix:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/run.py \
  --simulator fsim --host-codegen all
```

Complete TSIM matrix (fresh process):

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/run.py \
  --simulator tsim --host-codegen all
```

Successful output reports eight deterministic VTA regions, ten exact output
comparisons per host, and positive simulator counters. FSIM validates GEMM,
weight-load, and output-store counters; TSIM validates its supported
`cycle_count` counter only. Missing libraries, unexpected model or routing
structure, output differences, wrong configuration, and absent accelerator
activity cause a nonzero exit. TSIM initialization and hardware loading remain
lazy until all four bundles have been built, exported, and reloaded.
