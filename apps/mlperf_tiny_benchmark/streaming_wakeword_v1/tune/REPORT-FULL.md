# Streaming Wakeword V1 full-search result

The selected configuration was measured with AutoTVM TSIM and applied to one
committed WAV through the normal Streaming Wakeword deployment. The deployed
VTA occurrence passed the HOST output check and the inclusive 10% cycle gate.

The full FSIM space contains 240 distinct configurations. Search attempted all
240 and found 12 successful schedules, so it exhausted the space below the
20-success target. All 12 successes have successful AutoTVM TSIM measurements;
the minimum is config index 11 at 9,871 cycles. The selected schedule was
replayed from its exported native record without intermediate build files.

| Occurrence | Symbol | Logical MACs | AutoTVM TSIM | Deployment TSIM | Difference | MAC utilization |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `tvmgen_mlperf_streaming_wakeword_vta_main_0` | 143,360 | 9,871 | 9,872 | 0.010131% | 22.6904% |

The deployed sample is `marvin-00176480_nohash_0.wav`
(SHA-256 `b95e103110b89a0d4dff88023edd537a92834f565cb8e3f38b16f725f3d58451`).
It uses one 16,000-sample window, produces 30 feature frames, and makes one
stateless model invocation. HOST correctness passed. Ordinary and debug
full-model counters both measured 9,872 cycles. The untuned baseline measured
174,633 cycles, for a 17.6897x cycle speedup. Whole-model tuned utilization is
22.6904%; baseline utilization is 1.2827%.

The FSIM log retains 228 candidate failures (8 fold/build assertions and 220
other schedule/build/runtime failures). It also retains one sandbox-denied
localhost RPC tracker startup event separately as infrastructure evidence;
resuming that same run completed the search and all TSIM measurements. The
attempted and failed run state remains under the ignored model `build/` tree.

Machine-readable deployment and MAC results are in `deployment-full.json`,
`mac-utilization-full.json`, and `mac-utilization-full.csv`. The selected
schedule artifacts are in `optimal/20261001T205934.834798Z/`.
