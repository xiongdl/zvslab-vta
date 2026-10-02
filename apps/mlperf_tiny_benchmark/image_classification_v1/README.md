# MLPerf Tiny ResNet-8 deployment

This application prepares the committed MLPerf Tiny ResNet-8 model once and
builds a pure-host reference plus a mixed VTA graph. The standard deployment
compares exact outputs on the ten committed PNG samples, ordered by numeric
labels 0 through 9, and checks accelerator activity. Its graph has eight VTA
regions. It is an execution-equivalence example; it does not report an
official MLPerf score, energy, or FPGA latency.

## Prerequisites

Use the existing `.envs/tvm-vta-env`, initialized `tvm/` and `vta/`
submodules, built TVM/VTA libraries, and the shared geometry:

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
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/run.py \
  --simulator fsim

VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v1/run.py \
  --simulator tsim --schedule none
```

Omitting `--schedule` or passing `none` uses the normal default schedule.
`--schedule PATH` takes one native AutoTVM log; its same-stem JSON metadata is
loaded automatically. Metadata validates the model, actual prepared
computation, geometry, occurrence/workload/configuration identities, and
record hashes. A partial snapshot is valid: included occurrences use its
selected configuration and uncovered occurrences use defaults. The runner
prints selected/default coverage. Mismatched or damaged artifacts fail before
execution.

`--host-codegen` accepts `llvm`, `c`, or `all` (default `llvm`). The runner
supports `--output-dir PATH`, `--deployment-report PATH`, and
`--validate-schedule-evidence`. The evidence flag is for TSIM and requires a
deployment report; it checks complete measured schedule coverage and strict
per-layer cycle alignment. The report includes schedule provenance, coverage,
output checks, and measured evidence. The ten committed samples remain the
correctness set; performance measurement uses one sample and one counted TSIM
invocation after an excluded warmup. TSIM `cycle_count` values are simulator
cycles.

## Tune actual deployment schedules

The model's `tune.py` captures actual prepared layer occurrences and searches
their real VTA schedule spaces through shared lowering. There is no separate
fusion deployment command. First export a TSIM measurement of every default
schedule and validate it through the runner:

```bash
MODEL_DIR=vta/apps/mlperf_tiny_benchmark/image_classification_v1

VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" --seed --all
VTA_BACKEND=tsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/run.py" \
  --simulator tsim \
  --schedule "$MODEL_DIR/build/actual_compute_tuning/seed/seed.log" \
  --validate-schedule-evidence \
  --deployment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"
```

Then search with that passing alignment report:

```bash
VTA_BACKEND=fsim ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune.py" \
  --all --alignment-report "$MODEL_DIR/build/actual_compute_tuning/seed/deployment.json"
```

Search defaults are a batch of 100 distinct configurations, 20 successful
configurations per occurrence, and 60/120-second FSIM/TSIM timeouts per
candidate. It stops at the quota or when the valid space is exhausted. Use
`--workload-index N` or `--max-workloads N` for bounded selection. Resume with
`--resume-manifest PATH` and the same alignment report and search identity;
state is saved under `build/actual_compute_tuning/<run-id>/`. Bounded runs are
marked incomplete.

When successful TSIM results exist, the search writes `best.log` and matching
metadata in its run directory. For an explicit snapshot export, supply the
resume manifest and `--output-log PATH`, then choose `--export-best` or
`--export-candidate CANDIDATE --workload-index OCCURRENCE`. Candidate and best snapshots both
load through `run.py --schedule PATH`; a one-occurrence candidate is a partial
schedule and the remaining occurrences use defaults. Every snapshot records
its model/geometry/compute identity and whether the candidate was measured.

Compatible historical complete-fusion manifests can be converted by
`common.schedule.migrate_legacy_full_fusion` after validation against this
model's real prepared computation. Migration requires complete matching
occurrences, valid configurations and native records, and the historical
single-call TSIM protocol. It does not invent measurements.

The committed `tune/REPORT-C4-FULL.md` and `tune/optimal/c4-full/` preserve an
earlier complete-fusion search. Its ten-sample uninstrumented TSIM result was
36,328,640 baseline cycles and 2,591,430 selected cycles (14.0188x fewer);
all ten outputs matched HOST and all eight occurrence cycle checks passed.
Those historical measurements used the former complete-fusion task scope and
are not fresh measurements of the actual-compute search described above.
