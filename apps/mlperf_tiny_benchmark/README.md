# MLPerf Tiny VTA applications

The benchmark contains five standalone migrated applications and a separate
read-only deployment template. `image_classification_v2`,
`visual_wake_words_v1`, and `keyword_spotting_v1` own standalone deployment
workflows with model preparation kept local to each application. Anomaly
Detection V1 and Streaming Wakeword V1 now follow the standalone selected
deployment and tuning workflow.

## Standalone applications

- [Image classification V2](image_classification_v2/README.md) deploys a
  32×32 RGB input and reports ten raw scores.
- [Visual Wake Words V1](visual_wake_words_v1/README.md) deploys one normalized
  96×96 RGB image and reports the person/non-person class with two scores.
- [Keyword Spotting V1](keyword_spotting_v1/README.md) deploys one mono WAV sample
  and reports one of 12 labels with raw int8 scores. Its preserved QNN graph
  currently has zero real VTA partitions and documents the CPU fallback.
- [Streaming Wakeword V1](streaming_wakeword_v1/README.md) deploys one mono WAV
  sample and reports the Marvin/Silence/Unknown index with raw int8 scores. Its
  exact canonicalized QNN graph currently has zero real VTA partitions and
  documents the CPU fallback and rejected tuning path.
- [Anomaly Detection V1](anomaly_detection_v1/README.md) selects the first
  log-mel feature vector from one WAV, runs one reconstruction, and reports its
  MSE with actual CPU/VTA placement and tuning evidence.

Image classification V2, Visual Wake Words, and Anomaly Detection support
selected deployment, real workload export, separate FSIM/TSIM tuning, schedule
replay, integrity-checked reports and safe local cleanup. Keyword Spotting and
Streaming Wakeword use the same command shape, while their preserved int8 graphs
currently have no real VTA partitions; each README explains CPU fallback and
rejected tuning requests. Each standalone application README contains
independent prerequisites, commands, expected results and a full manual
acceptance procedure.

Repository setup and simulator build instructions remain in
[`scripts/README.md`](../../../scripts/README.md).
