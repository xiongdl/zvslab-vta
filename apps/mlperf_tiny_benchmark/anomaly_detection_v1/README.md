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
`1`. Each WAV produces a deterministic `(N, 640)` feature matrix and one
sample score: the mean squared error between all input feature vectors and
their reconstructed output vectors. Higher MSE is treated as more anomalous.
The reported `label` is the manifest label; `predicted_label` uses the
documented fixed-run threshold equal to the largest normal score. This is a
deployment score demonstration and does not claim classification accuracy.

The JSON result is sorted and contains model/input/output metadata,
per-sample results, the summary, and FSIM profiler counters when applicable.

## TSIM follow-up

TSIM is intentionally not implemented in checkpoint 2. The planned follow-up
will use `vta/config/tsim_sample.json`, validate lazy TSIM initialization and
positive cycle activity, and add the corresponding CLI matrix after the
checkpoint 3 runtime work is completed.
