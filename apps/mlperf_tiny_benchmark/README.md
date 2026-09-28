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
