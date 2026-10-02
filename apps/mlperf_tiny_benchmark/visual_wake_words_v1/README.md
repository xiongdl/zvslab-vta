# MLPerf Tiny VWW HOST deployment

This fixed-purpose application imports the committed floating MLPerf Tiny v1.4
Visual Wake Words model, applies the documented TVM quantization policy once, and builds
both a pure host reference and a mixed VTA Graph Executor artifact. It reloads
the host libraries, compares their output tensors within the fixed `1e-6`
absolute/relative tolerance for the ten committed JPEG samples, and requires
positive simulator activity. The matrix mode builds
LLVM and C variants below separate `llvm-fsim/` and `c-fsim/` (FSIM) or
`llvm-tsim/` and `c-tsim/` (TSIM) bundle roots.

Importing `vta` loads and validates the compiler target extension. The mixed
branch explicitly applies `vta.relay.partition_for_vta()` once, then uses
`vta.relay.plan_devices_for_vta()` to constrain host operators to CPU and
outlined VTA functions to `ext_dev`. The resulting CPU/VTA target map is passed
to `relay.build`, which inserts compiler-owned device copies and produces the
standard mixed runtime module without post-build graph JSON edits. The plan
keeps the thirteen depthwise convolutions on CPU and the twelve deterministic
VTA regions on `ext_dev`.

The VTA compiler target is activated only as the lowering context; the target
map returned by the planner remains the canonical `ext_dev -device=vta`
target paired with the selected LLVM or C host target:

```python
with vta.build_config():
    mixed_factory = relay.build(plan.module, target=plan.targets)
```

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
  vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1/run.py
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
  vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1/run.py \
  --simulator fsim --host-codegen all
```

Complete TSIM matrix (fresh process):

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1/run.py \
  --simulator tsim --host-codegen all
```

Successful output reports thirteen deterministic VTA regions, ten bounded output
comparisons per host, and positive simulator counters. FSIM validates GEMM,
weight-load, and output-store counters; TSIM validates its supported
`cycle_count` counter only; FSIM-only counters are not required for TSIM.
Missing libraries, unexpected model or routing
structure, output differences, wrong configuration, and absent accelerator
activity cause a nonzero exit. TSIM initialization and hardware loading remain
lazy until all four bundles have been built, exported, and reloaded.

## Actual-deployment schedules and tuning

`run.py` is the only deployment entry point. Omitting `--schedule` or passing
`--schedule none` uses default schedules. `--schedule PATH` loads one native
AutoTVM `.log` and its same-stem `.json` metadata automatically. The metadata
binds schedules to the model, actual prepared computation, geometry,
occurrences, workloads/configurations, and native record hashes. Partial
snapshots are allowed; uncovered occurrences remain on default schedules and
the runner reports this coverage. Invalid identities or records fail before
execution.

The graph contains thirteen deterministic single-convolution VTA regions and
keeps thirteen depthwise convolutions on CPU. The ten committed JPEG samples
are compared against the pure HOST reference with the fixed `1e-6`
absolute/relative tolerance and exact classification agreement. The
`--deployment-report PATH` option records model/schedule provenance, coverage,
outputs, and measured evidence. On TSIM,
`--validate-schedule-evidence` requires complete measured coverage and the
strict per-layer cycle-alignment gate. Standard correctness execution still
checks all ten committed samples; schedule evidence measures one performance
sample. `cycle_count` is simulator activity, not FPGA latency or MLPerf
performance.

```bash
MODEL_DIR=vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1
export MODEL_DIR
export VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps"

VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator fsim --schedule none

VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator tsim --schedule /path/to/snapshot.log \
  --deployment-report /tmp/vww-deployment.json
```

## Tune actual deployment occurrences

`tune.py` captures the real prepared VTA occurrences and tunes them through
shared lowering. The seed measures each default schedule on TSIM and writes
`build/actual_compute_tuning/seed/seed.log` with same-stem metadata. Deploy it
with the strict TSIM evidence gate; the passing report is a prerequisite for
search:

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

Search defaults to 100 unique configurations per batch, 20 successful
schedules per occurrence, and 60/120-second FSIM/TSIM candidate timeouts; it
stops at the quota or when the valid configuration space is exhausted. Use
`--workload-index N` or `--max-workloads N` for bounded search. Resume with
`--resume-manifest PATH` and the unchanged alignment report and tuning
identity. The manifest, occurrence ledgers, failures, and intermediate logs are
kept under `build/actual_compute_tuning/<run-id>/`.

When successful TSIM candidates exist, search writes a `best.log` and its
same-stem metadata. To export a specific candidate, pass the matching resume
manifest and `--output-log PATH` with
`--export-candidate CANDIDATE --workload-index OCCURRENCE`; use `--export-best`
for the best successful candidate of each selected occurrence. A partial
candidate is a deployable snapshot and uncovered regions use defaults. Candidate
provenance records whether it has TSIM measurements. Historical compatible
complete-fusion manifests can be migrated with
`common.schedule.migrate_legacy_full_fusion`; it validates the current model,
geometry, occurrence computations, schedule records, and single-call TSIM
protocol without inventing measurements.
