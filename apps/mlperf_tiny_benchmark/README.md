# MLPerf Tiny VTA AutoTVM workflow

`autotvm_tuner.py` tunes the VTA tasks extracted from the six applications in
this directory. The model selector `all` has a fixed order:

1. `image_classification_v1`
2. `image_classification_v2`
3. `anomaly_detection_v1`
4. `keyword_spotting_v1`
5. `streaming_wakeword_v1`
6. `visual_wake_words_v1`

Each invocation targets one backend and uses the shared
`vta/config/vta_64mac.json` geometry. Run FSIM and TSIM in separate processes
with matching `VTA_BACKEND` values. A bounded aggregate smoke run is:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model all --backend fsim --trials-per-task 1

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model all --backend tsim --trials-per-task 1
```

Omit `--trials-per-task` to search each supported task's full configuration
space. `--timeout` overrides the default per-measurement timeout. The tuner
completes models sequentially and atomically updates an aggregate JSON summary
after each model. The summary reports the backend/configuration identity,
task/trial counts, supported and unsupported task templates, each model's
status, and the log/sidecar paths. A failed model remains marked failed while
later models continue; the command returns nonzero if any model failed.

To resume, pass the summary path printed by the earlier command:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/autotvm_tuner.py \
  --model all --backend tsim --trials-per-task 1 \
  --resume-summary <autotvm-all-tsim-summary.json>
```

Resume accepts only the same backend, geometry hash, and tuning options. For
each previously completed model, it validates the model hash and log/sidecar
pair before reusing it. Invalid or incomplete pairs are tuned again. Each
model/backend run owns distinct files under the ignored
`build/autotvm/` directory; no model can inherit another model's log.

## Replay and correctness

Replay a model with the matching backend-specific native log and JSON sidecar.
The model runner validates the pair before applying history-best and then uses
its existing sample, CPU/VTA routing, and output checks. For example:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1/run.py \
  --simulator tsim \
  --autotvm-log <visual_wake_words_v1-tsim.log> \
  --autotvm-sidecar <visual_wake_words_v1-tsim.json>
```

The same options are available on the V1, V2, anomaly, keyword spotting, and
streaming wakeword runners. Each run directory's README describes its sample
contract, runtime mode, and backend-specific CLI options. In particular:

- image classification V1 compares baseline and tuned outputs and prints both
  TSIM cycle counts;
- image classification V2, anomaly, keyword spotting, streaming wakeword, and
  VWW report their existing output comparisons and simulator counters while
  applying the matching model/backend log;
- task reports list unsupported VTA templates explicitly; unsupported work is
  not presented as tuned;
- TSIM `cycle_count` is a simulator measurement, not FPGA latency or an
  official MLPerf result. A speedup is claimed only when a tuned/baseline
  comparison demonstrates lower cycles under the same conditions.

Per-model native logs and sidecars are independent for FSIM and TSIM. FSIM
measurement costs must not be interpreted as TSIM cycle counts.

## Per-layer useful-MAC utilization estimates

`mac_utilization.py` associates each extracted VTA Conv/Dense graph occurrence
with the best successful cycle cost for its matching workload in a validated
TSIM AutoTVM log. For each occurrence it reports
`logical_MACs / (best_successful_isolated_TSIM_task_cycles * peak_MACs_per_cycle)`.
AutoTVM records FLOPs, so the script divides by two for logical MACs. Peak
throughput comes from the geometry as
`2**LOG_BATCH * 2**LOG_BLOCK * 2**LOG_BLOCK` MAC/cycle; the checked-in
`vta_64mac.json` yields 64 MAC/cycle. The report includes both a ratio and a
percentage, units, config hash, workload identity, paired log/sidecar identity,
unsupported task coverage, and row counts.

This is an isolated AutoTVM task estimate associated with a layer occurrence.
TSIM does not provide per-layer full-model cycle profiling here, so this value
must not be read as that occurrence's measured cost in full-model execution.
Unsupported task templates are listed in the JSON summary and receive no
fabricated utilization. Repeated layer occurrences remain separate CSV rows,
even when they share a workload and its selected task cost.

The commands require the existing `.envs/tvm-vta-env`, built TVM/VTA TSIM
libraries, and a validated TSIM log/sidecar pair. Their default output is under
the ignored `build/autotvm/mac-utilization/` directory:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/mac_utilization.py \
  --model image_classification_v1 --backend tsim \
  --log vta/apps/mlperf_tiny_benchmark/build/autotvm/<v1-tsim-log>.log \
  --sidecar vta/apps/mlperf_tiny_benchmark/build/autotvm/<v1-tsim-sidecar>.json

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/mac_utilization.py \
  --model all --backend tsim \
  --summary vta/apps/mlperf_tiny_benchmark/build/autotvm/<autotvm-all-tsim-summary>.json
```

The six-model command validates the aggregate summary, every model's sidecar,
and each native log before writing reports. It produces a CSV with one row per
supported occurrence and a JSON summary; stable output names are derived from
the validated artifact identities. Set `--output-dir PATH` to write elsewhere.

## Complete-fusion tuning for the remaining four models

The model-local two-stage adapters cover `anomaly_detection_v1`,
`keyword_spotting_v1`, `streaming_wakeword_v1`, and `visual_wake_words_v1`.
Run one model at a time with its exact directory name below. Seed, full search,
resume, replay, and deployment use separate FSIM/TSIM processes with the shared
absolute geometry. Search defaults to 100 distinct FSIM configurations per
batch until each occurrence has 20 successful schedules or its space is
exhausted. A full search requires that model's passing seed deployment report.

Set the selected model and repository-local Python paths:

```bash
MODEL=anomaly_detection_v1 # or keyword_spotting_v1, streaming_wakeword_v1, visual_wake_words_v1
MODEL_DIR="vta/apps/mlperf_tiny_benchmark/$MODEL"
export MODEL MODEL_DIR
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$PWD/vta/apps/mlperf_tiny_benchmark:$PWD/$MODEL_DIR"
```

Find the generated seed run directory after the seed command and pass its
`best-manifest.json` to the one-sample alignment deployment. Only after the
report passes should the full search run:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune/tune.py" --seed --all

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
  ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune/deployment.py" \
  --best-manifest "$MODEL_DIR/tune/seed/<seed-run-id>/best-manifest.json" \
  --output "$MODEL_DIR/tune/deployment-seed.json"

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune/tune.py" --all \
  --alignment-report "$MODEL_DIR/tune/deployment-seed.json"
```

If an approved full search is interrupted, resume with its run manifest and
the same FSIM identity/options. Replay the self-contained optimal manifest in
a fresh TSIM process; this validates selected records and real lowering
without the intermediate search build. Then deploy the selected manifest once
on the model's committed representative sample and calculate measured
per-occurrence and whole-model MAC utilization:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
  ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune/tune.py" \
  --resume-manifest "$MODEL_DIR/build/two_stage_tuning/<run-id>/manifest.json"

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
  ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune/tune.py" \
  --replay-manifest "$MODEL_DIR/tune/optimal/<run-id>/best-manifest.json"

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
  ./.envs/tvm-vta-env/bin/python "$MODEL_DIR/tune/deployment.py" \
  --best-manifest "$MODEL_DIR/tune/optimal/<run-id>/best-manifest.json" \
  --output "$MODEL_DIR/tune/deployment-full.json"

./.envs/tvm-vta-env/bin/python scripts/mac_utilization.py \
  --deployment-report "$MODEL_DIR/tune/deployment-full.json" \
  --output-json "$MODEL_DIR/tune/mac-utilization-full.json"
```

The matching calculator CSV is written beside the JSON. The committed
per-model `tune/optimal/` manifest, deployment report and MAC CSV/JSON are
self-contained final evidence. See the initiative's `RESULTS.md` for the
four-model occurrence table, search counts, sample hashes, cycle comparisons
and report links.
