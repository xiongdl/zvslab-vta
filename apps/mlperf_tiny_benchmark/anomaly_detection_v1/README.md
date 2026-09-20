# MLPerf Tiny anomaly detection v1

This application deploys the committed MLPerf Tiny v1.4 ToyCar autoencoder
through the local TVM/VTA Graph Executor flow. It is a fixed ten-sample
deployment check, not an MLPerf accuracy, performance, energy, or submission
report.

## Preparation

From the repository root, use the pinned environment and the checked-out TVM
and VTA Python trees:

```bash
export VTA_CONFIG_FILE="$PWD/vta/config/vta_config.json"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python"
PYTHON="$PWD/.envs/tvm-vta-env/bin/python"
```

The model and ten WAV files are committed under this directory. No runtime
step reads `.envs`, downloads data, or requires `librosa`; preprocessing uses
Python's `wave` module and NumPy. The current model contract is one float32
`(1, 640)` input and one float32 `(1, 640)` reconstructed output.

For FSIM, build the repository's VTA FSIM library first when it is not already
available:

```bash
bash scripts/build_vta_lib.sh --target libvta_fsim
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
`--build-dir`; `--simulator fsim` is accepted as a compatibility alias for
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
process with `VTA_CONFIG_FILE` set to `vta/config/tsim_sample.json`. Build the
hardware/TSIM library before running it:

```bash
bash scripts/build_vta_lib.sh --target libvta_hw
```

Then run the complete LLVM/C matrix from the repository root:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/tsim_sample.json" \
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
  VTA_CONFIG_FILE="$PWD/vta/config/tsim_sample.json" \
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
a positive integer `cycle_count`. Missing TSIM registries, a non-TSIM VTA
target, or absent `libvta_hw` causes a nonzero exit. The scores and labels are
deployment contracts only; they do not claim classification accuracy.
Missing TSIM registries, a non-TSIM VTA target, or absent `libvta_hw` causes a
nonzero exit. The scores and labels are deployment contracts only; they do not
claim classification accuracy.
