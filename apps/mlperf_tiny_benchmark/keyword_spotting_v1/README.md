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
./scripts/setup_tvm_vta_env.sh
./.envs/tvm-vta-env/bin/python -m pytest \
  vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1/tests
bash scripts/build_vta_lib.sh --target libtvm-vta-ext
bash scripts/build_vta_lib.sh --target libvta_fsim
bash scripts/build_vta_lib.sh --target libvta_hw
```

`libvta_fsim` is needed for FSIM. TSIM additionally requires the hardware
library from `libvta_hw`, a valid `VTA_CONFIG_FILE`, and a fresh process. Build
outputs are written below this application’s ignored `build/` directory.

## Run

HOST uses the reference graph only and does not initialize a simulator:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_config.json" \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1/run.py \
  --simulator host --host-codegen llvm
```

FSIM with the complete LLVM/C matrix:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_config.json" \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1/run.py \
  --simulator fsim --host-codegen all
```

TSIM with the complete LLVM/C matrix, in a fresh process:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/tsim_sample.json" \
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
