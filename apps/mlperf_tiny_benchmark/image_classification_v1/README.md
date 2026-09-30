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

## Tune one VTA workload

`tune.py` selects one supported VTA task in the prepared graph's extraction
order using a zero-based workload index. It uses AutoTVM random search with a
local FSIM runner, at most 32 trials, and a 120-second timeout for each
measurement. Each candidate gets a fresh local runner so an FSIM RPC worker
crash is recorded as a failed candidate and does not prevent later trials.
It then builds and measures the best successful FSIM record with TSIM,
preserving that record's exact AutoTVM configuration. The output prints
the task template, workload SHA-256, logical MAC count, TSIM `cycle_count`,
and generated artifact paths. FSIM timing is only the search signal; reported
cycles come from the separate TSIM measurement.

Run from the repository root with both simulator libraries built for the same
geometry file. Start with `VTA_BACKEND=fsim`; the command switches its own
process to `tsim` only for the final measurement:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/tune.py \
  --workload-index 0
```

`--workload-index` is required. `--trials N` and `--timeout SECONDS` override
the defaults for a bounded run; the trial count is capped at the selected
task's configuration-space size. `--output-dir PATH` changes the artifact
directory. By default, the FSIM log, best-record log, and JSON result are
written under the ignored
`vta/apps/mlperf_tiny_benchmark/build/autotvm/image_classification_v1/`
directory. An invalid index reports the valid range before creating a runner
or writing output.

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
