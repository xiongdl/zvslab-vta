# MLPerf Tiny ResNet-8 Large deployment

This application prepares the committed floating MLPerf Tiny v1.4 ResNet-8
Large model and builds a pure-host reference plus a mixed VTA graph. The mixed
graph contains eight VTA regions and five HOST convolutions. Deployment
compares exact output tensors and top-1 classifications over ten committed
PNG samples ordered by numeric labels 0 through 9, and checks simulator
activity. This is an execution-equivalence example; it does not report an
official MLPerf score, energy, or FPGA latency.

## Prerequisites

Use the existing `.envs/tvm-vta-env`, initialized `tvm/` and `vta/`
submodules, built TVM/VTA libraries, and the shared absolute geometry:

```bash
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" --backend all
```

Run from the repository root. `VTA_BACKEND` must match `--simulator`; HOST is
the CPU reference path, not another VTA backend.

## Default and scheduled runs

```bash
export VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps"

VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator fsim

VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator tsim --schedule none
```

Omitting `--schedule` or passing `none` uses the normal default schedule.
`--schedule PATH` accepts one native AutoTVM log and automatically loads the
same-stem JSON metadata. The metadata binds it to this model, actual prepared
computation, VTA geometry, layer occurrence, workload/configuration, and
native record hashes. Partial snapshots are supported: selected occurrences
use the supplied configurations and uncovered occurrences use defaults; the
runner prints that coverage. Invalid or mismatched metadata fails before
execution.

`--host-codegen` accepts `llvm`, `c`, or `all` (default `llvm`);
`--output-dir PATH` selects generated artifacts. `--deployment-report PATH`
writes schedule provenance, occurrence coverage, output checks, and measured
evidence. On TSIM, `--validate-schedule-evidence` requires complete measured
coverage and strict per-layer cycle alignment, and requires a deployment
report. The correctness set is all ten samples. The performance check uses
one sample and one counted TSIM invocation after an excluded warmup; reported
`cycle_count` values are simulator cycles.

## Tune actual deployment schedules

The model's `tune.py` extracts the actual prepared VTA occurrences and tunes
their captured schedule spaces through shared lowering. It searches the real
deployment computation, including the existing bias, shift, clip, and cast
operations. Start with a TSIM seed for every occurrence, then apply it via
`run.py` to create the required alignment evidence:

```bash
MODEL_DIR=vta/apps/mlperf_tiny_benchmark/image_classification_v2

VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" --seed --all
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator tsim \
  --schedule "$MODEL_DIR/build/actual_compute_tuning/seed/seed.log" \
  --validate-schedule-evidence \
  --deployment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"

VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" \
  --all --alignment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"
```

Seed output is a native `.log` plus same-stem `.json` metadata under
`build/actual_compute_tuning/seed/`. It measures the default configuration at
each actual occurrence. Search requires the passing complete alignment
report. It tests 100 distinct configurations per batch until 20 successful
configurations per occurrence are measured or the valid space is exhausted.
Candidates run first on FSIM and successful candidates are measured on TSIM
with one counted invocation after an excluded warmup. Defaults are 60 seconds
per FSIM candidate and 120 seconds per TSIM candidate. `--workload-index N`
and `--max-workloads N` select bounded search scopes; bounded results are
marked incomplete.

Search ledger, failures, and resume manifest are written under
`build/actual_compute_tuning/<run-id>/`. Resume with `--resume-manifest PATH`,
the same seed report, and unchanged model, geometry, computation, selected
occurrences, and search options. Successful searches also write `best.log`
and matching metadata. Explicit exports require a resume manifest and
`--output-log PATH`; use `--export-candidate CANDIDATE` with `--workload-index OCCURRENCE` to
export one ledger candidate, or `--export-best` to export the best successful
TSIM candidate for each selected occurrence. Candidate and best files use the
same `run.py --schedule PATH` interface. An unmeasured candidate remains
identified as unmeasured, and a single-occurrence export is a valid partial
snapshot.

Compatible historical complete-fusion manifests may be converted with
`common.schedule.migrate_legacy_full_fusion` against the actual prepared
graph. Migration checks complete occurrence coverage, model/geometry/compute
identity, valid schedule configurations, native record hashes, and the
historical single-call TSIM measurement protocol. Incompatible or incomplete
artifacts fail; migration creates no measurements.

The committed `tune/REPORT-FULL.md` and `tune/optimal/20261001T035246.904010Z/`
preserve an earlier complete-fusion search. It measured 21,226,413 baseline
cycles and 1,360,351 selected cycles on its one-sample performance input;
the selected graph passed all ten correctness samples and all eight strict
occurrence cycle checks. These historical measurements used the former
complete-fusion task scope and are not fresh measurements of the actual-compute
search described above.
