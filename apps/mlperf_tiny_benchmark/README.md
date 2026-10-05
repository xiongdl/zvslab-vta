# MLPerf Tiny VTA applications

The benchmark contains six model applications. `image_classification_v1` is
the read-only deployment template. `image_classification_v2` and
`visual_wake_words_v1` own standalone deployment and tuning workflows; each
keeps its model-specific preparation local to that application. The remaining
three applications still use their existing interfaces while their migration
checkpoints are pending.

## Standalone applications

- [Image classification V2](image_classification_v2/README.md) deploys a
  32×32 RGB input and reports ten raw scores.
- [Visual Wake Words V1](visual_wake_words_v1/README.md) deploys one normalized
  96×96 RGB image and reports the person/non-person class with two scores.

Both applications support the selected targets `c`, `llvm`, `vta,c`, and
`vta,llvm`, workload export, separate FSIM/TSIM tuning, schedule replay,
integrity-checked reports and safe local cleanup. Their READMEs contain
independent prerequisites, commands, expected results and full manual
acceptance procedures.

## Existing application interfaces

The other applications retain their current invocation notes and model assets
until their migration tasks complete:

- [Image classification V1](image_classification_v1/README.md)
- [Anomaly detection V1](anomaly_detection_v1/README.md)
- [Keyword spotting V1](keyword_spotting_v1/README.md)
- [Streaming wakeword V1](streaming_wakeword_v1/README.md)

Repository setup and simulator build instructions remain in
[`scripts/README.md`](../../../scripts/README.md).
