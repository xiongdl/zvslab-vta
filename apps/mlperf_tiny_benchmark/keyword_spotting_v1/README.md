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

## Actual-deployment schedule tuning

`run.py` is the only deployment command. It accepts `--schedule PATH` for a
native AutoTVM `.log` paired with automatically loaded same-stem `.json`
metadata. Omitting the option or passing `none` uses the default schedule.
Partial snapshots are supported: provided configurations apply to their exact
prepared VTA occurrences; uncovered occurrences use defaults and are listed
in the output. Model, geometry, compute, workload/configuration, and record
identities are validated. `--deployment-report PATH` writes provenance,
coverage, output checks, and measurement evidence. On TSIM,
`--validate-schedule-evidence` requires complete measured coverage and passes
only when the strict per-layer cycle-alignment gate succeeds.

The deployment continues to preprocess its twelve committed WAV files through
the existing quantized MFCC path, compares each mixed output exactly against
the CPU reference, and requires positive simulator activity. These are
execution checks, not an accuracy or MLPerf score claim.

```bash
MODEL_DIR=vta/apps/mlperf_tiny_benchmark/keyword_spotting_v1
export MODEL_DIR
export VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps"

VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator fsim --schedule none

VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator tsim --schedule /path/to/snapshot.log \
  --deployment-report /tmp/kws-deployment.json
```

## Tune actual deployment occurrences

The model's `tune.py` tunes the actual prepared VTA occurrences with shared
lowering. Measure all default schedules on TSIM first, then apply the seed
through `run.py` with the strict evidence option. Search requires that complete
passing report:

```bash
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" --seed --all
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator tsim \
  --schedule "$MODEL_DIR/build/actual_compute_tuning/seed/seed.log" \
  --validate-schedule-evidence \
  --deployment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"

VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" \
  --all --alignment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"
```

The seed log and metadata are stored under `build/actual_compute_tuning/seed/`.
Search tests 100 distinct configs per batch until 20 successful candidates per
occurrence or exhaustion, with 60-second FSIM and 120-second TSIM candidate
timeouts. Use `--workload-index N` or `--max-workloads N` to bound scope.
Resume with `--resume-manifest PATH`, the same alignment report, and the same
model/geometry/compute/options; state lives under
`build/actual_compute_tuning/<run-id>/`.

Search exports `best.log` with same-stem metadata when successful TSIM results
exist. Explicit exports require the resume manifest and `--output-log PATH`:
use `--export-candidate CANDIDATE --workload-index OCCURRENCE` for one ledger
entry, or `--export-best` for the best successful schedules. The resulting
partial or complete snapshot is applied by `run.py --schedule PATH`.
Compatible historical manifests can be converted with
`common.schedule.migrate_legacy_full_fusion` after checking actual compute,
model/geometry identity, valid configurations, native record hashes, and the
single-call TSIM measurement protocol. Migration creates no measurements.
