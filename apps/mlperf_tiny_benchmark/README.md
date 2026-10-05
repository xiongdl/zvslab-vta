# MLPerf Tiny VTA applications

The benchmark contains six model applications. `image_classification_v1` is
the read-only deployment template. `image_classification_v2`,
`visual_wake_words_v1`, and `keyword_spotting_v1` own standalone deployment
workflows with model preparation kept local to each application. The other two
applications still use their existing interfaces while their migration
checkpoints are pending.

## Standalone applications

- [Image classification V2](image_classification_v2/README.md) deploys a
  32×32 RGB input and reports ten raw scores.
- [Visual Wake Words V1](visual_wake_words_v1/README.md) deploys one normalized
  96×96 RGB image and reports the person/non-person class with two scores.
- [Keyword Spotting V1](keyword_spotting_v1/README.md) deploys one mono WAV sample
  and reports one of 12 labels with raw int8 scores. Its preserved QNN graph
  currently has zero real VTA partitions and documents the CPU fallback.

Image classification V2 and Visual Wake Words support selected deployment,
real workload export, separate FSIM/TSIM tuning, schedule replay,
integrity-checked reports and safe local cleanup. Keyword Spotting uses the
same deployment command shape, but its preserved int8 graph currently has no
real VTA partitions; its README documents CPU fallback and rejected tuning
requests. Each standalone application README contains independent prerequisites,
commands, expected results and a full manual acceptance procedure.

## Existing application interfaces

The remaining applications retain their current interfaces and model assets
until their migration tasks complete:

- [Image classification V1](image_classification_v1/README.md)
- [Anomaly detection V1](anomaly_detection_v1/README.md)
- [Streaming wakeword V1](streaming_wakeword_v1/README.md)

Repository setup and simulator build instructions remain in
[`scripts/README.md`](../../../scripts/README.md).
