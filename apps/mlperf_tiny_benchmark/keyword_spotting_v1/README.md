# MLPerf Tiny Keyword Spotting v1 deployment

This fixed-purpose application imports the committed MLPerf Tiny v1.4
Keyword Spotting reference model, prepares its quantized MFCC input, and
builds a CPU reference graph plus a VTA-partitioned graph. It compares the
two output tensors for twelve committed WAV samples and requires positive
accelerator activity. It is a deployment check; it does not report MLPerf
accuracy, performance, energy, or submission results.

## Model and samples

The model is an immutable copy of
`.envs/tiny-v1.4/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite`.
The model checksum and MLPerf Tiny v1.4 provenance are recorded in
`model/README.md`. The application owns the WAV inputs under `samples/`; their
source-relative provenance and SHA-256 checksums are recorded in
`samples/manifest.json`. Runtime execution never reads `.envs`.

The fixed numeric label order is:

```text
0 Down       1 Go       2 Left    3 No       4 Off       5 On
6 Right      7 Stop     8 Up      9 Yes     10 Silence  11 Unknown
```

## Prerequisites and build

Run commands from the repository root with the pinned environment:

```bash
# Use the existing project environment at .envs/tvm-vta-env; do not recreate it
# during verification.
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python -m pytest \
  vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1/tests
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" --backend all
```

`libvta_fsim` is needed for FSIM. TSIM additionally requires the hardware
library from `libvta_hw`, a valid `VTA_CONFIG_FILE`, and a fresh process. Build
outputs are written below this application’s ignored `build/` directory.

The active contract is the shared absolute `VTA_CONFIG_FILE` plus
`VTA_BACKEND=fsim|tsim`. The build script uses `--backend fsim|tsim|all` and
this runner uses `--simulator host|fsim|tsim`; HOST is CPU reference execution,
while `fsim` and `tsim` must match `VTA_BACKEND`. `TARGET=sim`, `TARGET=tsim`,
and `--target libvta_*` are retired. FPGA backends such as `pynq` and `zcu104`
remain deferred.

## Run

HOST uses the reference graph only and does not initialize a simulator:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1/run.py \
  --simulator host --host-codegen llvm
```

FSIM with the complete LLVM/C matrix:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1/run.py \
  --simulator fsim --host-codegen all
```

TSIM with the complete LLVM/C matrix, in a fresh process:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1/run.py \
  --simulator tsim --host-codegen all
```

The default is LLVM FSIM. Use `--host-codegen c` for one C-host deployment and
`--output-dir PATH` to choose another generated-artifact directory. A
successful single run reports twelve comparisons; each FSIM matrix entry must
have positive GEMM, weight-load, and output-store counters, while each TSIM
entry must have a positive integer `cycle_count`.

## Complete-fusion two-stage tuning

`tune.py` extracts each routed KWS Conv fusion from the prepared graph,
including its per-channel bias, right shift, clipping, and cast. Occurrences
remain separate even when they share a workload. Seed artifacts and durable
search state are written under `tune/` and `build/two_stage_tuning/`.

Run the seed search and deployment in separate processes using the same
absolute geometry file. Full search requires a passing seed deployment report;
`--resume-manifest` resumes its matching durable run and `--replay-manifest`
validates standalone exported artifacts.

```bash
MODEL=keyword_spotting_v1
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/vta/apps/mlperf_tiny_benchmark/$MODEL" \
  ./.envs/tvm-vta-env/bin/python \
  "vta/apps/mlperf_tiny_benchmark/$MODEL/tune/tune.py" --seed --all
```

## AutoTVM schedule tuning

The shared tuner extracts supported VTA convolution/dense task families from
the prepared KWS graph and records unsupported VTA task families in the JSON
sidecar. FSIM and TSIM produce separate logs and sidecars under
`vta/apps/mlperf_tiny_benchmark/build/autotvm/`. Add `--trials-per-task 1` for
a bounded workflow check:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model keyword_spotting_v1 --backend fsim --trials-per-task 1

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model keyword_spotting_v1 --backend tsim --trials-per-task 1
```

Pass the matching `--autotvm-log` and `--autotvm-sidecar` to `run.py` with the
same backend to replay history-best during mixed-graph compilation. Run each
backend in a fresh process. Replay validates model, backend, config, log hash,
and task coverage. MFCC preprocessing and the 12-sample, 12-label output
contract remain in place; TSIM reports simulator `cycle_count`.
