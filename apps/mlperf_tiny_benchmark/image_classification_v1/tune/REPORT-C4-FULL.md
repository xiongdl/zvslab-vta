# IC V1 full-search MAC utilization report

The run measured all eight real model fusion occurrences using adaptive FSIM
search. Each workload used 100-trial increments until at least 20 distinct
successful schedules were found. Every FSIM success was measured on TSIM with
one counted invocation and no warmup. FSIM and TSIM measurement timeouts were
60 and 120 seconds. No configuration space was exhausted.

| Occurrence | FSIM trials | FSIM successes / TSIM measurements | Selected config | Selected TSIM cycles | Real deployed cycles | Cycle difference | MACs / invocation | MAC utilization |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 24 / 24 | 309 | 51,299 | 54,229 | 5.7116% | 2,359,296 | 67.9784% |
| 1 | 200 | 21 / 21 | 105 | 54,144 | 56,301 | 3.9838% | 2,359,296 | 65.4766% |
| 2 | 100 | 25 / 25 | 64 | 14,338 | 15,573 | 8.6135% | 131,072 | 13.1510% |
| 3 | 300 | 29 / 29 | 221 | 26,955 | 28,411 | 5.4016% | 1,179,648 | 64.8763% |
| 4 | 300 | 25 / 25 | 243 | 44,451 | 45,907 | 3.2755% | 2,359,296 | 80.3015% |
| 5 | 200 | 39 / 39 | 111 | 7,980 | 8,547 | 7.1053% | 131,072 | 23.9616% |
| 6 | 200 | 21 / 21 | 446 | 22,961 | 23,689 | 3.1706% | 1,179,648 | 77.8083% |
| 7 | 400 | 25 / 25 | 653 | 41,752 | 42,480 | 1.7436% | 2,359,296 | 86.7797% |

The search logged 1,691 candidate-level FSIM failures across the eight
workloads (attempts minus successful schedules); no infrastructure/RPC errors
and no TSIM candidate failures occurred. Failed FSIM candidate details remain
in the per-workload JSON and native logs under `../build/two_stage_tuning/`.

The first real-deployment comparison found that the minimum-cycle choices for
occurrences 2 and 5 were outside the 10% acceptance threshold: 11,507 TSIM vs
12,961 deployed cycles for occurrence 2 (12.6358%), and 7,097 vs 7,833 cycles
for occurrence 5 (10.3706%). The full TSIM candidate records were already
available. Config 64 (14,338 TSIM cycles) and config 111 (7,980 cycles) were
then applied to the real graph and both passed the limit. The final manifest
records these validated measured candidates; raw state retains the original
minimum-cycle observations and every rejected FSIM candidate.

All ten committed sample outputs exactly match the pure HOST reference. The
instrumented debug graph and ordinary graph report the same single-sample
full-model cycles. The summed selected VTA occurrence cycles are 275,137 per
invocation, equal to the 275,137 full-model cycles, for a zero-cycle residual.
Whole-model counts use ten uninstrumented invocations and include no warmup:

| Measure | Baseline | Tuned |
|---|---:|---:|
| Full-model TSIM cycles (10 samples) | 38,757,180 | 2,751,370 |
| Logical MACs (10 samples) | 120,586,240 | 120,586,240 |
| Whole-model MAC utilization | 4.8614% | 68.4808% |

The tuned run reduces full-model cycles by 14.0865× and raises measured
whole-model utilization by 63.6193 percentage points. Utilization uses actual
whole-model cycles, not the sum of operator cycles. Host work is excluded from
the VTA MAC numerator. The model identity remains metadata so the calculator
can consume other models with the same versioned deployment-report contract.

The final versioned data is in `deployment-c4-full.json` and
`mac-utilization-c4-full.json`. The selected configurations and native records
are under `optimal/c4-full/`. Per-trial build/runtime exceptions, counts and
resume state are under `../build/two_stage_tuning/c4-full/`; the final full
search has no TSIM candidate failures or RPC infrastructure failures. Failed
FSIM configurations are isolated trial failures and did not stop later trials.
