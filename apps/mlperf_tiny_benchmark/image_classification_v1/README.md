# ResNet-8 image classification

This application imports a floating-point ResNet-8 TFLite model, applies one
fixed quantization policy, compiles exactly one selected CPU or VTA/CPU target,
and runs one image. It prints a CIFAR-10 class and raw output scores. Scores are
not calibrated probabilities, and this example does not claim an official
MLPerf result.

## Requirements

Use the existing `.envs/tvm-vta-env`, initialized `tvm/` and `vta/`
submodules, and previously built libraries. The CPU target needs TVM; a VTA
target also needs the selected VTA simulator library and an absolute geometry
configuration in `VTA_CONFIG_FILE`. A direct VTA invocation sets matching
`VTA_BACKEND` and `--simulator` values. CPU invocation does not require either
variable or simulator libraries.

Run from the repository root with:

```bash
export VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python"
APP=vta/apps/mlperf_tiny_benchmark/image_classification_v1
```

## Deployment

```bash
# CPU-only deployment; no VTA_BACKEND is needed.
env -u VTA_BACKEND ./.envs/tvm-vta-env/bin/python "$APP/run.py" --target llvm

# Choose one VTA plus CPU target and backend.
VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$APP/run.py" \
  --target vta,llvm --simulator fsim

# C host codegen with TSIM and a selected schedule.
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$APP/run.py" \
  --target vta,c --simulator tsim --schedule "$APP/tune/vta_64mac/best.log" \
  --deployment-report "$APP/build/deployment-report.md"
```

Supported targets are `c`, `llvm`, `vta,c`, and `vta,llvm`; the default is
`vta,llvm`. The two VTA targets partition supported computation for VTA first
and leave other operations on the selected CPU code generator. The default
simulator is FSIM. CPU targets ignore `--simulator` and `--schedule`.

`--model PATH` defaults to `model/pretrainedResnet.tflite` and accepts a float
ResNet-8 TFLite file with the supported input, output, and operator topology.
`--input PATH` defaults to `samples/00-airplane.png` and accepts one 32x32 RGB
PNG or JPEG. Input pixels are converted to float32 NHWC without rescaling,
matching the model's existing input convention. Quantization uses
`global_scale=8.0` and `skip_conv_layers=[0]` for every target.

The default artifact directory is `build/`; `--output-dir PATH` selects another
location. Explicit relative paths are resolved from the calling directory.
`--deployment-report PATH` optionally writes UTF-8 Markdown with model/input
hashes, target, schedule coverage, raw scores, logical convolution MACs, and
available cycle data. CPU and FSIM cycle counts and utilization are marked
`N/A`. TSIM reports one counted graph invocation after a warmup, per-VTA-node
cycles, peak MAC/cycle, and the documented whole-graph and layer-cycle scopes.

The graph bundle still validates exported library, graph, parameters, source
files, and required VTA symbols before publication and after reload.

`--export-workloads PATH` optionally saves the actual outlined VTA Relay
functions, their constants, deployment input activations, and hardware/config
provenance as a validated JSON snapshot. It requires a VTA target. The export
uses the default schedule even when `--schedule` is supplied, then continues
the requested deployment with that schedule. This lets tuning consume the same
computations and real activations used by deployment without reopening the
model or image. The loader checks the snapshot, hardware geometry, and current
VTA config spaces before use. For example:

```bash
VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$APP/run.py" \
  --target vta,llvm --simulator fsim \
  --export-workloads "$APP/build/workloads.json"
```

## Tuning during the transition

The current tuning entry point still uses the prior intermediate interface.
Checkpoint C3 replaces it with workloads exported by deployment; do not use
these commands as the new tuning workflow:

```bash
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$APP/tune.py" --seed --all
```

Historical evidence is retained under `tune/` and is not a fresh measurement
of the new single-image deployment path.
