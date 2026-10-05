# MLPerf Tiny VTA applications

Each model application has a deployment entry point, `run.py`, and a tuning
entry point, `tune.py`. Their exact command interfaces are model-specific.
`image_classification_v1` also provides a Makefile and documents its standalone
single-image and FSIM/TSIM workflow in its own README. The other applications
document their own host and schedule interfaces below. Generic per-operator
AutoTVM tuning and separate model deployment commands have been retired.

The six applications are `image_classification_v1`,
`image_classification_v2`, `anomaly_detection_v1`, `keyword_spotting_v1`,
`streaming_wakeword_v1`, and `visual_wake_words_v1`. Each model README records
its committed samples and output checks.

## Prerequisites and default run

Use the existing `.envs/tvm-vta-env`, initialized `tvm/` and `vta/`
submodules, built TVM/VTA libraries, and the shared absolute geometry file.
Select the same backend in `VTA_BACKEND` and `--simulator`. For example:

```bash
export VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps"

VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator fsim
```

Omitting `--schedule` or passing `--schedule none` compiles with the default
schedule. Default runs do no schedule search. `--host-codegen` accepts
`llvm`, `c`, or `all`; the default is `llvm`. FSIM runs the functional
simulator. TSIM reports its supported `cycle_count` in simulator cycles; this
is not FPGA latency or an official MLPerf score.

## One schedule input

`--schedule PATH` accepts one native AutoTVM log and automatically reads the
same-stem JSON metadata, for example `candidate.log` with `candidate.json`.
The metadata binds the records to model, prepared computation, VTA geometry,
occurrence, workload, configuration, and native record hashes. A missing,
damaged, or mismatched pair fails validation before execution. Partial snapshots
are allowed: selected occurrences use the supplied configs, while every
uncovered occurrence uses its normal default; the runner reports this coverage.

```bash
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator tsim --schedule /path/to/best.log \
  --deployment-report /tmp/v2-deployment.json
```

`--validate-schedule-evidence` requires measured complete schedule coverage and
the strict per-layer cycle alignment checks. The deployment report records
model/geometry/schedule identities, selected and default coverage, output
checks, and any measured evidence. Model README files describe the exact
sample and performance gates.

## Seed, search, resume, and export for the other models

Tune the actual deployment occurrences with one of the other models' `tune.py`
scripts. A TSIM seed measures the default
schedule at each occurrence, exports a complete schedule log/metadata pair,
and is then run once through `run.py --validate-schedule-evidence` to create
the alignment report required by search:

```bash
MODEL=image_classification_v2
MODEL_DIR="vta/apps/mlperf_tiny_benchmark/$MODEL"
export MODEL_DIR

VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" --seed --all
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator tsim \
  --schedule "$MODEL_DIR/build/actual_compute_tuning/seed/seed.log" \
  --validate-schedule-evidence \
  --deployment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"

VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" \
  --all --alignment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"
```

Search starts from actual occurrences captured from the normally prepared
deployment graph. It tests distinct valid schedule configurations on FSIM and
measures successful candidates on TSIM using one counted invocation after an
excluded warmup (`tsim_single_call_v1`, units `cycles`). Default search options
are `--trial-batch 100`, `--min-successful 20`, `--fsim-timeout 60`, and
`--tsim-timeout 120` seconds per candidate. A task stops when its successful
quota is reached or its valid configuration space is exhausted. Use
`--workload-index N` or `--max-workloads N` for bounded occurrence selection.
The search ledger and resume manifest live below the model's ignored
`build/actual_compute_tuning/<run-id>/` directory. Failed trials are recorded.

Resume with the matching manifest and unchanged model, geometry, compute,
search options, and seed/alignment identities:

```bash
VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" \
  --all --alignment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json" \
  --resume-manifest "$MODEL_DIR/build/actual_compute_tuning/<run-id>/resume-manifest.json"
```

The search exports `best.log` and its same-stem metadata when each selected
occurrence has a successful TSIM result. Export any ledger candidate or the
best successful candidates explicitly with `--resume-manifest`,
`--workload-index` as needed, `--export-candidate N` or `--export-best`, and
`--output-log PATH`. A candidate export preserves measurement status; an
unmeasured or failed candidate is not represented as measured. Candidate and
best exports are ordinary deployable schedule snapshots and can be supplied
to `run.py --schedule PATH`. A one-occurrence export is a valid partial
snapshot.

Model flows that still support migration can migrate compatible historical full-fusion manifests to the
current actual-compute snapshot format with the common
`common.schedule.migrate_legacy_full_fusion` helper. Migration validates the
real model computation, geometry, schedule configuration, native records, and
historical single-call TSIM measurement protocol; it does not invent new
measurements. The standalone image_classification_v1 tuning flow uses its
current FSIM and TSIM logs and does not use these historical manifests.

## Cleanup

Inspect generated files first, then choose a category and model explicitly:

```bash
./.envs/tvm-vta-env/bin/python scripts/clean_mlperf_tiny.py \
  --model all --cache --tuning-runs --dry-run
```

`--cache` selects recognized compiler/debug output; `--tuning-runs` selects
generated ledgers, logs, and checkpoints. Remove `--dry-run` to delete those
recognized, untracked files. Unknown files, tracked files, models, samples,
saved seed/optimal snapshots, and evidence reports are retained. Removing
tuning runs discards resume state. See `scripts/README.md` for prerequisites
and the complete cleanup contract.
