# AD V1 full-search and deployment report

Run: `20261001T174411.872234Z`
Model SHA-256: `c66636f4d7f8af8b10518e7be750a22c9d8d46ec97326b40b0d94c097e0aad9b`
Geometry: `vta/config/vta_64mac.json` (SHA-256 `23b338eacdf5747610d90fd17296e3d0d4236ce416191b7c1cfc597cd67991fa`, peak 64 MAC/cycle)
Seed alignment report: `deployment-seed.json`, all nine occurrences passed at 0% difference.

## Full search

The full search used the passing seed report, 100 distinct FSIM trials per batch, a minimum of 20 successful schedules, and 60/120 second FSIM/TSIM candidate timeouts. The run manifest is under the ignored intermediate directory `../build/two_stage_tuning/20261001T174411.872234Z/manifest.json`. Each exported selected record is self-contained under `optimal/20261001T174411.872234Z/`; standalone replay validated all nine records without relying on that build directory.

| Occurrence | FSIM attempts / space | Successful FSIM schedules | AutoTVM TSIM measurements | Search result | Selected AutoTVM cycles |
| ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 100 / 100 | 32 | 32 | 20-success quota | 2,691 |
| 1 | 100 / 100 | 32 | 32 | 20-success quota | 2,691 |
| 2 | 100 / 100 | 32 | 32 | 20-success quota | 2,691 |
| 3 | 20 / 20 | 5 | 5 | Configuration space exhausted | 437 |
| 4 | 20 / 20 | 6 | 6 | Configuration space exhausted | 516 |
| 5 | 100 / 100 | 32 | 32 | 20-success quota | 2,691 |
| 6 | 100 / 100 | 32 | 32 | 20-success quota | 2,691 |
| 7 | 100 / 100 | 32 | 32 | 20-success quota | 2,691 |
| 8 | 100 / 200 | 37 | 37 | 20-success quota | 11,889 |

All 9 selected entries equal the minimum positive TSIM cycle result among the successful, deployment-lowerable schedules for their occurrence. Across the 9 full-search states, 500 candidate failures were retained separately: 397 copy-pattern lowering failures and 103 other TVM build/measurement failures. A local RPC tracker permission failure from the initial sandboxed attempt was also preserved as infrastructure evidence for each occurrence; the durable full-search run resumed with local RPC access and completed every state. These infrastructure records are not included in FSIM candidate counts.

## Selected real deployment

The selected manifest `optimal/20261001T174411.872234Z/best-manifest.json` passed its standalone replay. Deployment used one committed sample, `normal_id_01_00000000.wav` (SHA-256 `0385da04d6cf8c1f9d0df775f98fda55409a71890c02ed53bb5d2c66171f6828`), and its first deterministic feature window (index 0 of 196). Existing HOST-reference correctness passed. The report records `sample_count=1` and one counted operator/model invocation.

| Occurrence | Symbol | Logical MACs | AutoTVM TSIM cycles | Deployment TSIM cycles | Difference | MAC utilization |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `tvmgen_mlperf_anomaly_vta_main_0` | 16,384 | 2,691 | 2,691 | 0% | 9.5132% |
| 1 | `tvmgen_mlperf_anomaly_vta_main_1` | 16,384 | 2,691 | 2,691 | 0% | 9.5132% |
| 2 | `tvmgen_mlperf_anomaly_vta_main_2` | 16,384 | 2,691 | 2,691 | 0% | 9.5132% |
| 3 | `tvmgen_mlperf_anomaly_vta_main_3` | 1,024 | 437 | 437 | 0% | 3.6613% |
| 4 | `tvmgen_mlperf_anomaly_vta_main_4` | 1,024 | 516 | 516 | 0% | 3.1008% |
| 5 | `tvmgen_mlperf_anomaly_vta_main_5` | 16,384 | 2,691 | 2,691 | 0% | 9.5132% |
| 6 | `tvmgen_mlperf_anomaly_vta_main_6` | 16,384 | 2,691 | 2,691 | 0% | 9.5132% |
| 7 | `tvmgen_mlperf_anomaly_vta_main_7` | 16,384 | 2,691 | 2,691 | 0% | 9.5132% |
| 8 | `tvmgen_mlperf_anomaly_vta_main_8` | 81,920 | 11,889 | 11,889 | 0% | 10.7663% |

All deployed occurrences pass the inclusive 10% gate. The ordinary and debug full-model TSIM runs both measured 28,988 cycles. The measured untuned baseline was 188,773 cycles; the tuned complete deployment was 28,988 cycles (6.5121x speedup). Whole-model useful-MAC utilization was 1.5087% at baseline and 9.8248% when tuned. These are measured complete-model counts; host operations are excluded from the VTA logical MAC total.

Logical MAC counts come from each prepared Conv fusion's actual input/output and weight shapes, excluding padding-only work; repeated symbols remain separate occurrences. Per-occurrence utilization uses `logical_MACs / (deployment_cycles * 64)`. The CSV and calculator JSON contain the derivations and identity hashes: `mac-utilization-full.csv` and `mac-utilization-full.json`. The complete versioned deployment evidence is `deployment-full.json`.
