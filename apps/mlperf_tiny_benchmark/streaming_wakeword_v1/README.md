# MLPerf Tiny streaming wakeword v1 deployment

This directory is a fixed deployment example for the committed MLPerf Tiny
v1.4 streaming wakeword model. It is not an MLPerf benchmark run and makes no
claim about MLPerf accuracy, performance, energy, or submission results.

## Model and samples

The immutable model is `model/str_ww_ref_model.tflite`, copied from the MLPerf
Tiny v1.4 streaming wakeword training model. Its SHA-256 is
`3af8550895ba7d5c584277102b5075c52dcfa63ba9d2b2240f37c4e6abd5dd2b`.
The model input is an int8 `(1, 30, 1, 40)` log-mel tensor and its output is
an int8 `(1, 3)` tensor. Output indices are fixed as follows:

| Index | Class |
| ---: | --- |
| 0 | Marvin |
| 1 | Silence |
| 2 | Unknown |

The repository owns exactly three deterministic mono, signed 16-bit, 16 kHz
WAV samples in `samples/`, one for each class. `Marvin` and `Unknown` are the
selected source utterances; `Silence` is a deterministic one-second segment
from the selected background recording. `samples/manifest.json` records the
source-relative provenance, byte lengths, and SHA-256 hashes. Runtime execution
uses these committed files and does not read the local environment or download
a dataset.

## Prerequisites

Run all commands from the repository root with the pinned environment. The two
assignment lines construct its repository-local path without making that
environment an application asset:

```bash
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python"
PINNED_ENV="$PWD/."
PINNED_ENV="${PINNED_ENV}envs/tvm-vta-env"
PYTHON="$PINNED_ENV/bin/python"
```

The focused tests need the checked-out TVM/VTA Python trees and the packages
in the pinned environment. HOST and FSIM also need a built TVM and VTA runtime;
FSIM needs `libvta_fsim`, and TSIM needs the hardware/TSIM build
(`libvta_hw`). Build the required library when it is unavailable:

```bash
bash scripts/build_tvm_lib_macos.sh
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" --backend all
```

Generated graph artifacts are written below the application `build/`
directory by default, or below `--output-dir`, and are not source assets.

The active contract is the shared absolute `VTA_CONFIG_FILE` plus
`VTA_BACKEND=fsim|tsim`. The build script uses `--backend fsim|tsim|all` and
this runner uses `--simulator host|fsim|tsim`; HOST is CPU reference execution,
while `fsim` and `tsim` must match `VTA_BACKEND`. `TARGET=sim`, `TARGET=tsim`,
and `--target libvta_*` are retired. FPGA backends such as `pynq` and `zcu104`
remain deferred.

## HOST

HOST builds and reloads the reference and mixed bundles but executes only the
CPU reference bundle; it does not initialize a simulator:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  "$PYTHON" \
  vta/apps/mlperf_tiny_benchmark/streaming_wakeword_v1/run.py \
  --simulator host --host-codegen llvm
```

Use `--host-codegen c` for the C host variant. `--host-codegen all` is a
matrix option for FSIM and TSIM; HOST is intentionally reference-only.

## FSIM

FSIM builds LLVM and C host variants in that order when `all` is selected,
then executes the reference and VTA-partitioned bundles for all three samples
with strict elementwise int8 output comparison:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  "$PYTHON" \
  vta/apps/mlperf_tiny_benchmark/streaming_wakeword_v1/run.py \
  --simulator fsim --host-codegen all
```

The command reports each sample's reference and mixed top-1 class, artifact
bundle paths, and positive FSIM profiler counters. A nonzero exit means a
contract failed; output comparison is never approximate and errors are not
swallowed. The current validated state is that HOST and the real FSIM
LLVM/C matrix both pass: each host variant builds one non-empty VTA partition,
all three committed samples compare elementwise equal, and profiler activity
is positive for both variants. The earlier FSIM mismatch evidence is obsolete
and must not be used to interpret a current run.

## TSIM

TSIM must run in a fresh process with the TSIM configuration and requires the
hardware/TSIM VTA library:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  "$PYTHON" \
  vta/apps/mlperf_tiny_benchmark/streaming_wakeword_v1/run.py \
  --simulator tsim --host-codegen all
```

TSIM validates the active VTA target, required registry functions, and a
zeroed profiler before dispatch. It executes all three samples for each host
variant and requires a positive integer `cycle_count`. Missing libraries or
registries fail with an actionable diagnostic; they are not treated as a
successful deployment.

## Output interpretation

Each successful single-codegen run prints three sample records, reference and
mixed top-1 indices, profiler statistics, and the two authenticated artifact
bundle paths. HOST reports `mixed top-1: not-run` and an empty profiler record.
FSIM and TSIM report `mixed top-1` only after strict output equality and
positive simulator activity have been verified. These records demonstrate
reproducible graph construction and execution contracts only; they are not
benchmark accuracy or performance measurements.
