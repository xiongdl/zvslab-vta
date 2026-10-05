# MLPerf Tiny VTA applications

The benchmark contains six model applications. `image_classification_v1` is the
read-only deployment template. `image_classification_v2` is the first migrated
standalone deployment and tuning workflow; the other four applications retain
their existing interfaces until their migration checkpoints.

## Image classification V2

See [the application guide](image_classification_v2/README.md) for prerequisites,
all deployment targets, workload export, FSIM and TSIM tuning, schedule replay,
reports, failure cases, cleanup, and a complete manual acceptance procedure.
The app owns its Python modules and Make orchestration. From the repository root:

```bash
APP=vta/apps/mlperf_tiny_benchmark/image_classification_v2
make -C "$APP" deploy TARGET=llvm
make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
  EXPORT_WORKLOADS=build/workloads.json
make -C "$APP" tune-fsim WORKLOADS=build/workloads.json WORKLOAD=0 \
  TRIAL_BATCH=1 MIN_SUCCESSFUL=1
make -C "$APP" tune-tsim WORKLOADS=build/workloads.json \
  INPUT_LOGS=tune/vta_64mac/fsim.tmp WORKLOAD=0
make -C "$APP" clean
```

CPU deployment uses TVM without initializing VTA. VTA deployment uses one
selected host code generator and a matching `VTA_BACKEND`, simulator, and
absolute geometry configuration. Workload export and schedule replay require
real VTA partitions. The model supports one 32×32 RGB input, float32 NHWC, and
reports the predicted CIFAR-10 class with all ten raw scores.

## Remaining applications

Each application directory retains its model, samples, license, and a README
with its current model and invocation notes:

- [Image classification V1](image_classification_v1/README.md)
- [Visual wake words V1](visual_wake_words_v1/README.md)
- [Anomaly detection V1](anomaly_detection_v1/README.md)
- [Keyword spotting V1](keyword_spotting_v1/README.md)
- [Streaming wakeword V1](streaming_wakeword_v1/README.md)

The shared `common/` utilities and legacy tuning entry points remain while
other application consumers are migrated. The maintained cleanup command is
documented in [`scripts/README.md`](../../../scripts/README.md).
