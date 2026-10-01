# MLPerf Tiny ResNet-8 Large HOST deployment

This fixed-purpose application imports the committed floating MLPerf Tiny v1.4
ResNet-8 Large model, applies the documented TVM quantization policy once, and builds
both a pure host reference and a mixed VTA Graph Executor artifact. It reloads
the host libraries, compares their output tensors exactly for the ten committed
PNG samples, and requires positive simulator activity. The matrix mode builds
LLVM and C variants below separate `llvm-fsim/` and `c-fsim/` (FSIM) or
`llvm-tsim/` and `c-tsim/` (TSIM) bundle roots.

The mixed artifacts use the `resnet8_large` identity and contain exactly eight
single-convolution VTA regions (`tvmgen_mlperf_resnet_large_vta_main_0` through
`_7`) plus five HOST convolutions.

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
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py
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

FSIM matrix:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator fsim --host-codegen all
```

Complete TSIM matrix (fresh process):

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator tsim --host-codegen all
```

Successful output reports eight deterministic VTA regions and five HOST convolutions, ten exact output
comparisons per host, and positive simulator counters. FSIM validates GEMM,
weight-load, and output-store counters; TSIM validates its supported
`cycle_count` counter only. Missing libraries, unexpected model or routing
structure, output differences, wrong configuration, and absent accelerator
activity cause a nonzero exit. TSIM initialization and hardware loading remain
lazy until all four reference/mixed bundles have been built, exported, and reloaded.

## AutoTVM tuning and replay

Tune the V2 prepared mixed graph independently for FSIM and TSIM. Each command
writes a backend-specific native log and JSON sidecar under the ignored shared
AutoTVM output directory. The sidecar includes the V2 model hash and a report
of supported and unsupported VTA task templates; unsupported operators are
not counted as tuned.

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model image_classification_v2 --backend fsim --trials-per-task 1

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model image_classification_v2 --backend tsim --trials-per-task 1
```

Replay a matching pair with the same simulator selector and geometry. The
runtime validates the model/backend/config/log pairing before compiling the
mixed graph with history-best; it clears the TECompiler cache before tuned
lowering so an earlier baseline compile cannot mask the selected schedule.
Output tensors are still checked against the existing reference path.

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator tsim \
  --autotvm-log <image_classification_v2-tsim.log> \
  --autotvm-sidecar <image_classification_v2-tsim.json>
```

## Complete VTA fusion tuning

The V2-local `tune/tune.py` extracts the eight actual prepared VTA Conv
occurrences. Each task identity includes the V2 symbol and occurrence plus the
Conv shapes, layouts, dtypes, bias, right shift, clip and cast. The default
`--all` search tests distinct configurations in 100-trial FSIM batches until
each occurrence has 20 successful schedules or its valid configuration space
is exhausted. It measures every FSIM success with TSIM in a separate process,
using the single-call cycle protocol and 60-second FSIM/120-second TSIM
candidate timeouts. The selected record is the minimum positive TSIM cycle
candidate that also lowers for the real prepared fusion.

Prerequisites are the project Python environment, the V2 model artifact,
`vta_64mac.json`, and built FSIM and TSIM simulator libraries. The controller
starts in an explicit FSIM environment and selects the matching backend for
each isolated worker:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/image_classification_v2" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/tune/tune.py --all
```

Use `--workload-index N` for one occurrence, or `--max-workloads N`,
`--trial-batch N`, or `--min-successful N` for bounded work. Runs with any
bounded selection or non-default search limits are labeled
`BOUNDED_SMOKE_INCOMPLETE`. `--fsim-timeout` and `--tsim-timeout` override
per-candidate limits. Intermediate native logs, failures, and resumable state
are written under the ignored `build/two_stage_tuning/<run-id>/`; exported
best JSON and native records go to `tune/optimal/<run-id>/`. The manifest
records model, geometry, fusion and workload identities, cycle protocol,
record hashes, selected configuration, candidate failures, and completion
status. Use `--resume-manifest PATH` only with the same model, geometry,
occurrences and search options. Replay validates the standalone export and
lowers the selected configuration without depending on intermediate build
files:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/image_classification_v2" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/tune/tune.py \
  --replay-manifest vta/apps/mlperf_tiny_benchmark/image_classification_v2/tune/optimal/<run-id>/best-manifest.json
```

## Selected-schedule deployment profile

`tune/deployment.py` validates the complete V2 best manifest, applies each
selected configuration to its corresponding VTA symbol while lowering, and
builds and reloads untuned and selected mixed Graph Executor bundles. It uses
the first committed sample for baseline and selected full-model cycles, debug
versus ordinary counter agreement, and all eight graph-resident VTA node cycle
comparisons. Each node uses one counted invocation after clearing warmup. The
strict integer gate is `10 * abs(deployment_cycles - autotvm_cycles) <
autotvm_cycles`; equality at 10 percent fails. After the one-sample performance
gate passes, the selected graph runs all ten committed samples for output
correctness only. Output JSON is published only after all ten outputs and all
eight cycle pairs pass. Build bundles are written under
`build/selected_deployment/`; the report defaults to `tune/deployment.json`
and a failed run writes `tune/deployment.failure.json`.

Prerequisites are the project Python environment, model and samples, matching
absolute `vta_64mac.json`, and built TVM/VTA TSIM libraries. Use a complete
full-search manifest for final acceptance; a complete-coverage bounded
manifest is suitable for integration verification and remains labeled
`BOUNDED_SMOKE_INCOMPLETE` in the output.

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/image_classification_v2" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/tune/deployment.py \
  --best-manifest vta/apps/mlperf_tiny_benchmark/image_classification_v2/tune/optimal/<run-id>/best-manifest.json \
  --output vta/apps/mlperf_tiny_benchmark/image_classification_v2/tune/deployment.json
```
