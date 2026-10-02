# Visual Wake Words V1 full-search result

The C9 full search completed all 13 VTA fusion occurrences with the approved 100-configuration FSIM batches, 20-success quota, 60-second FSIM timeout, and 120-second TSIM timeout. It attempted 1,700 unique configurations, found 401 successful FSIM schedules, and measured all 401 on TSIM. Every occurrence met the success quota; the least successful occurrence had 22 measured schedules. Search evidence and selected native records are in [`optimal/20261001T220401.514781Z/best-manifest.json`](optimal/20261001T220401.514781Z/best-manifest.json).

The selected schedules were applied to one committed image through the normal VWW V1 preprocessing and HOST reference check. The deployment passed for sample `00-non-person-000000000009.jpg` (SHA-256 `d8f0e1e6e7635f189ab52e3e98aef1f7d734814a1fbe41fdb2c5ff8cbfc6dcfc`). It made one stateless model invocation. All 13 operator cycle comparisons passed the inclusive 10% gate; the largest difference was 0.004879% at occurrence 12. Ordinary and debug full-model counters both measured 224,115 cycles. The untuned baseline measured 6,862,109 cycles, a 30.6187x cycle speedup.

Per-occurrence logical MACs are derived from the deployed Conv tensor arithmetic, excluding padding-only work. Useful-MAC utilization is `logical_MACs / (deployment_TSIM_cycles * 64)` for one invocation; HOST work is excluded. The reported deployment cycles and selected AutoTVM cycles are separate measured quantities.

| Occurrence | Symbol | Logical MACs | AutoTVM TSIM | Deployment TSIM | Difference | MAC utilization |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `tvmgen_mlperf_vww_vta_main_0` | 294,912 | 36,524 | 36,525 | 0.002738% | 12.6160% |
| 1 | `tvmgen_mlperf_vww_vta_main_1` | 294,912 | 18,149 | 18,149 | 0.000000% | 25.3898% |
| 2 | `tvmgen_mlperf_vww_vta_main_2` | 589,824 | 28,713 | 28,714 | 0.003483% | 32.0958% |
| 3 | `tvmgen_mlperf_vww_vta_main_3` | 294,912 | 11,827 | 11,827 | 0.000000% | 38.9617% |
| 4 | `tvmgen_mlperf_vww_vta_main_4` | 589,824 | 17,386 | 17,386 | 0.000000% | 53.0082% |
| 5 | `tvmgen_mlperf_vww_vta_main_5` | 294,912 | 8,976 | 8,976 | 0.000000% | 51.3369% |
| 6 | `tvmgen_mlperf_vww_vta_main_6` | 589,824 | 15,462 | 15,462 | 0.000000% | 59.6042% |
| 7 | `tvmgen_mlperf_vww_vta_main_7` | 589,824 | 13,408 | 13,408 | 0.000000% | 68.7351% |
| 8 | `tvmgen_mlperf_vww_vta_main_8` | 589,824 | 14,389 | 14,389 | 0.000000% | 64.0489% |
| 9 | `tvmgen_mlperf_vww_vta_main_9` | 589,824 | 15,737 | 15,737 | 0.000000% | 58.5626% |
| 10 | `tvmgen_mlperf_vww_vta_main_10` | 589,824 | 15,214 | 15,214 | 0.000000% | 60.5758% |
| 11 | `tvmgen_mlperf_vww_vta_main_11` | 294,912 | 7,829 | 7,829 | 0.000000% | 58.8581% |
| 12 | `tvmgen_mlperf_vww_vta_main_12` | 589,824 | 20,498 | 20,499 | 0.004879% | 44.9583% |

Whole-model tuned MAC utilization is 43.1778%; baseline utilization is 1.4102%. Baseline and tuned whole-model counts come from complete-model deployment counters; per-operator cycles are not summed to substitute for those measurements.

Machine-readable deployment and MAC evidence is in [`deployment-full.json`](deployment-full.json), [`mac-utilization-full.json`](mac-utilization-full.json), and [`mac-utilization-full.csv`](mac-utilization-full.csv). The reports bind the selected manifest, model, geometry, sample, and single-call measurement protocol.
