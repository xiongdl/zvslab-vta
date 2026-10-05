# ResNet-8 image classification

This application imports a floating-point ResNet-8 TFLite model, applies one
fixed quantization policy, compiles exactly one selected CPU or VTA/CPU target,
and runs one image. It prints a CIFAR-10 class and raw output scores. Scores are
not calibrated probabilities, and this example does not claim an official
MLPerf result.

## Requirements

The application code, ResNet-8 model, and default input image are contained in
this directory. Running it still requires the repository's `.envs/tvm-vta-env`
and initialized `tvm/` and `vta/` checkouts. CPU targets need TVM; VTA targets
also need the selected simulator library and an absolute geometry configuration
in `VTA_CONFIG_FILE`. A direct VTA invocation sets matching `VTA_BACKEND` and
`--simulator` values. CPU invocation does not require either variable or VTA
simulator libraries. The Makefile uses the current repository layout to find
these external components.

The application's Python code does not import `apps/common` or neighboring
benchmark applications. Its tests use the project TVM/VTA environment and
libraries; they do not need the original CIFAR-10 archive or repository setup
and sample-extraction scripts.

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

The local Makefile wraps deployment and sets the matching simulator backend:

```bash
make deploy
make deploy TARGET=llvm
make deploy TARGET=vta,c SIMULATOR=tsim
make deploy MODEL='models/custom resnet.tflite' INPUT='images/cat one.png' \
  SCHEDULE=tune/vta_64mac/best.log REPORT=build/deployment-report.md
```

Run `make deploy` from this directory, or use `make -C` from another
directory. Explicit relative paths are resolved from Make's working directory;
defaults remain relative to the model directory. Variables are `MODEL`,
`INPUT`, `TARGET`, `SIMULATOR`, `SCHEDULE`, `OUTPUT_DIR`, `REPORT`,
`EXPORT_WORKLOADS`, and `CONFIG`. `CONFIG` defaults to the repository's
`vta/config/vta_64mac.json`. The Makefile requires the prebuilt project Python
and TVM/VTA libraries; it does not create environments, install packages, or
build libraries.

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

## Tuning

The Makefile provides individual stages and the full sequence:

```bash
# Export workloads during deployment, then run each stage explicitly.
make deploy EXPORT_WORKLOADS=build/workloads.json
make tune-fsim WORKLOADS=build/workloads.json WORKLOAD=0 \
  TRIAL_BATCH=1 MIN_SUCCESSFUL=1
make tune-tsim WORKLOADS=build/workloads.json \
  INPUT_LOGS=tune/vta_64mac/fsim.tmp WORKLOAD=0

# Or export, search and select in one command.
make tune
make tune WORKLOAD=0
make tune WORKLOADS=build/workloads.json
```

`tune-fsim` requires `WORKLOADS` and defaults to all occurrences, 100 trials
per batch, 20 successful candidates, and a 60-second timeout. `tune-tsim`
requires `WORKLOADS` and `INPUT_LOGS`, and defaults to a 120-second timeout.
Both stages accept `OUTPUT_LOGS`; defaults are the configuration's
`tune/<config-name>/fsim.tmp` and `best.log`. `make tune` accepts `MODEL`,
`INPUT`, `WORKLOAD`, `TRIAL_BATCH`, `MIN_SUCCESSFUL`, `FSIM_TIMEOUT`,
`TSIM_TIMEOUT`, `OUTPUT_DIR`, optional `WORKLOADS`, and `CONFIG`. `OUTPUT_DIR`
contains only generated build intermediates. The persistent schedules remain
under the tracked configuration directory. Full tuning stops after TSIM and
does not automatically deploy the winner.

Tuning consumes the exact pre-schedule VTA functions and real activations
exported by deployment. It does not reopen the model or input image. Run the
two stages in separate processes with matching `VTA_BACKEND` values:

```bash
# Export the model's actual VTA workloads during normal deployment.
VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$APP/run.py" \
  --target vta,llvm --simulator fsim \
  --export-workloads "$APP/build/workloads.json"

# Search all occurrences in FSIM and retain every successful native schedule.
VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$APP/tune.py" \
  --workloads "$APP/build/workloads.json" --workload -1 \
  --simulator fsim --trial-batch 100 --min-successful 20 \
  --timeout 60 --output-logs "$APP/tune/vta_64mac/fsim.tmp"

# Measure only those FSIM candidates in TSIM and choose minimum cycles per layer.
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$APP/tune.py" \
  --workloads "$APP/build/workloads.json" --workload -1 \
  --simulator tsim --input-logs "$APP/tune/vta_64mac/fsim.tmp" \
  --timeout 120 --output-logs "$APP/tune/vta_64mac/best.log"

# Replay the selected schedules in normal model deployment.
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$APP/run.py" \
  --target vta,llvm --simulator tsim \
  --schedule "$APP/tune/vta_64mac/best.log"
```

`--workload -1` selects all VTA occurrences; a non-negative value selects one
occurrence, for example `--workload 0`. FSIM defaults are
`--trial-batch 100`, `--min-successful 20`, and `--timeout 60`; TSIM defaults
to `--timeout 120` and requires `--input-logs`. FSIM output contains grouped
successful candidates in native AutoTVM format and must not be passed directly
to deployment. TSIM output contains one cycle-minimum schedule for each selected
occurrence and can be passed to `run.py --schedule`. Both logs have same-stem
JSON metadata.

FSIM continues after a candidate compile error, output mismatch, timeout, or
native worker crash. For each occurrence it reports `trials`, correctness-
verified `successes`, the requested `quota`, and `termination` (`quota_reached`
or `space_exhausted`). Reaching the end of the configuration space with at
least one successful candidate publishes the candidates even when the quota is
short. If no candidate succeeds, tuning fails and keeps the previously saved
files. Workload/configuration errors and compiler/runtime initialization
failures stop the stage. TSIM reports how many input candidates it attempted
and measured successfully; failed candidates do not enter the selected log.

Tuning saves a raw `config.json` snapshot and `config.sha256` beside the
published schedules. Replacing all occurrences can replace results under a
changed configuration; updating one occurrence requires matching existing
identities and preserves other occurrences. Re-tuning a selected occurrence
invalidates its previous best while retaining other best selections. Files are
staged and validated before publication, and ordinary write failures restore
the previous validated set. Do not run concurrent writers in one tune folder.
