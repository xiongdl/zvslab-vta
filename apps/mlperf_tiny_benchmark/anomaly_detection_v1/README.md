# MLPerf Tiny anomaly detection v1

This application deploys the committed MLPerf Tiny v1.4 ToyCar autoencoder
through the local TVM/VTA Graph Executor flow. It is a fixed ten-sample
deployment check, not an MLPerf accuracy, performance, energy, or submission
report.

## Preparation

From the repository root, use the pinned environment and the checked-out TVM
and VTA Python trees:

```bash
export VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json"
export VTA_BACKEND=fsim
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python"
PYTHON="$PWD/.envs/tvm-vta-env/bin/python"
```

The active contract is the shared absolute `VTA_CONFIG_FILE` plus
`VTA_BACKEND=fsim|tsim`. The build script uses `--backend fsim|tsim|all`;
this runner uses `--mode host|fsim|tsim` (or its `--simulator` runner flag),
with `VTA_BACKEND` matching the selected VTA backend. `TARGET=sim`,
`TARGET=tsim`, and `--target libvta_*` are retired; migrate to the shared
geometry file and explicit backend selectors. FPGA backends such as `pynq` and
`zcu104` are deferred and are not implemented here.

The model and ten WAV files are committed under this directory. No runtime
step reads `.envs`, downloads data, or requires `librosa`; preprocessing uses
Python's `wave` module and NumPy. The current model contract is one float32
`(1, 640)` input and one float32 `(1, 640)` reconstructed output.

For FSIM, build the repository's VTA FSIM library first when it is not already
available:

```bash
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" --backend fsim
```

## HOST

HOST builds and reloads the artifact bundle but executes the CPU reference
graph, so it does not initialize the VTA simulator:

```bash
"$PYTHON" vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1/run.py \
  --mode host \
  --build-dir vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1/build \
  --output-json /tmp/anomaly-host.json
```

## FSIM

FSIM builds reference and mixed bundles, then lazily loads
`vta.testing.simulator` only for mixed execution:

```bash
"$PYTHON" vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1/run.py \
  --mode fsim \
  --host-codegen llvm \
  --build-dir vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1/build \
  --output-json /tmp/anomaly-fsim.json
```

Use `--host-codegen c` for the C host variant. `--host-codegen all` runs the
ordered LLVM/C FSIM matrix. `--output-dir` is accepted as an alias for
`--build-dir`; `--simulator fsim` selects the same FSIM runner path as
`--mode fsim`.

Each build directory contains `<host-codegen>-<mode>/reference` and
`mixed` bundles. Each bundle includes graph JSON, serialized parameters,
the host library, a hash-authenticated manifest, and inspectable host source.
Generated build outputs are ignored and should not be committed.

## Samples and score semantics

The default manifest contains exactly ten committed WAV files in fixed order:
five `normal` samples labeled `0`, followed by five `anomaly` samples labeled
`1`. HOST and FSIM preserve the complete feature-window contract: each WAV
produces a deterministic `(N, 640)` feature matrix and one score, the mean
squared error between all input feature vectors and their reconstructed output
vectors. Higher MSE is treated as more anomalous.
The reported `label` is the manifest label; `predicted_label` uses the
documented fixed-run threshold equal to the largest normal score. This is a
deployment score demonstration and does not claim classification accuracy.

The JSON result is sorted and contains model/input/output metadata,
per-sample results, the summary, and profiler counters when applicable. Each
sample records its total `feature_shape` and `total_window_count`, the actual
`executed_feature_shape` and `executed_window_count`, plus `sampled` and
`score_scope` so a limited run cannot be mistaken for a complete-window score.

## TSIM

TSIM uses the existing VTA software-simulation flow and must run in a fresh
process with the shared geometry file and `VTA_BACKEND=tsim`. Build the
hardware/TSIM library from the same geometry file before running it:

```bash
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" --backend tsim
```

Then run the complete LLVM/C matrix from the repository root:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1/run.py \
  --simulator tsim --host-codegen all
```

By default TSIM executes one deterministic representative window per sample.
This is an intentional TSIM smoke/representative-window run: it executes the
real mixed VTA graph for every one of the ten samples, but it does not cover
every audio feature window and its score is not a complete-window MSE. The
default keeps the aggregate matrix within the simulator time budget while
preserving five normal and five anomaly samples. Request a larger positive
integer budget with either CLI or environment configuration:

```bash
VTA_ANOMALY_TSIM_WINDOW_BUDGET=4 \
  VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
  PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1/run.py \
  --simulator tsim --host-codegen all

# CLI takes precedence over the environment variable.
.../run.py --simulator tsim --host-codegen all --tsim-window-budget 4
```

`--tsim-window-budget` and `VTA_ANOMALY_TSIM_WINDOW_BUDGET` accept only positive
integers; invalid values fail with a clear error. The selection is deterministic
and evenly spaced across the full feature matrix. If the budget reaches the
sample's total window count, the result records `sampled: false` and uses the
complete-window MSE semantics.

The command builds and reloads both host variants before one lazy TSIM
initialization. Successful output contains ten results per host, with
`normal_count: 5`, `anomaly_count: 5`, the actual per-sample window counts, and
a positive integer `cycle_count`. Missing TSIM registries, a non-TSIM
`VTA_BACKEND`, or absent `libvta_hw` causes a nonzero exit. The scores and
labels are deployment contracts only; they do not claim classification
accuracy.

## AutoTVM tuning and replay

The tuner extracts from the prepared anomaly mixed graph and writes a separate
native log and JSON sidecar for each backend. The sidecar records the model
hash and supported/unsupported VTA task templates; only supported templates
are measured and represented in history-best replay.

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model anomaly_detection_v1 --backend fsim --trials-per-task 1

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model anomaly_detection_v1 --backend tsim --trials-per-task 1
```

Pass a matching log/sidecar pair to the runner to compile and execute with
history-best. TSIM replay keeps the existing bounded-window policy; the default
is one deterministic representative feature window per sample, which is a
smoke score rather than a full-window anomaly score.

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/anomaly_detection_v1/run.py \
  --mode tsim --tsim-window-budget 1 \
  --autotvm-log <anomaly_detection_v1-tsim.log> \
  --autotvm-sidecar <anomaly_detection_v1-tsim.json>
```

## Complete-fusion two-stage tuning

The AD V1 two-stage adapter prepares this model's own quantized mixed graph,
extracts all nine deployed VTA fusion occurrences, and preserves the existing
HOST/VTA partition and audio preprocessing. A seed run searches one successful
FSIM schedule per occurrence and measures each on TSIM. A full run requires a
passing seed deployment alignment report and defaults to batches of 100
configurations until each occurrence has 20 successful schedules or its valid
space is exhausted. `--workload-index` supports one-occurrence execution;
`--resume-manifest`, `--replay-manifest`, and `--artifact-dir` support resume
and standalone validation.

Run from the repository root with the project Python environment and the same
absolute geometry file for both backend processes:

```bash
MODEL=anomaly_detection_v1
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/$MODEL"
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim PYTHONPATH="$PYTHONPATH" \
  ./.envs/tvm-vta-env/bin/python "vta/apps/mlperf_tiny_benchmark/$MODEL/tune/tune.py" --seed
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim PYTHONPATH="$PYTHONPATH" \
  ./.envs/tvm-vta-env/bin/python "vta/apps/mlperf_tiny_benchmark/$MODEL/tune/tune.py" --all --alignment-report <seed-deployment.json>
```

The adapter uses 60-second FSIM and 120-second TSIM candidate timeouts. Search
state and intermediate logs go below the model's ignored
`build/two_stage_tuning/`; self-contained native records and manifests are
written below `tune/optimal/` by default. Seed artifacts use a separate
`--artifact-dir` and remain distinct from full-search evidence.
