# IC V1 full-search MAC utilization report

The complete search covered all eight IC V1 VTA fusion occurrences. Each
workload used 100-trial FSIM increments until at least 20 distinct schedules
passed or its search space was exhausted; every distinct FSIM success was
measured on TSIM. FSIM and TSIM candidate timeouts were 60 and 120 seconds.
None of the search spaces was exhausted, and no FSIM RPC infrastructure or
TSIM candidate errors occurred.

| Occurrence | FSIM trials | FSIM successes / TSIM measurements | Selected config | TSIM cycles | Deployed cycles | Difference | MACs / invocation | MAC utilization |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 24 / 24 | 309 | 51,299 | 51,299 | 0.0000% | 2,359,296 | 71.8610% |
| 1 | 200 | 21 / 21 | 106 | 53,124 | 53,125 | 0.0019% | 2,359,296 | 69.3911% |
| 2 | 100 | 25 / 25 | 209 | 11,507 | 11,503 | 0.0348% | 131,072 | 17.8041% |
| 3 | 300 | 29 / 29 | 221 | 26,955 | 26,955 | 0.0000% | 1,179,648 | 68.3806% |
| 4 | 300 | 25 / 25 | 243 | 44,451 | 44,451 | 0.0000% | 2,359,296 | 82.9318% |
| 5 | 200 | 39 / 39 | 301 | 7,097 | 7,097 | 0.0000% | 131,072 | 28.8573% |
| 6 | 200 | 21 / 21 | 446 | 22,961 | 22,961 | 0.0000% | 1,179,648 | 80.2752% |
| 7 | 400 | 25 / 25 | 653 | 41,752 | 41,752 | 0.0000% | 2,359,296 | 88.2928% |

The initial threshold failure came from the deployment lowering of a scalar
bias: `_pack_output_constant` expanded the scalar into a packed tensor, causing
two extra VTA DMA loads in each real fused graph. The AutoTVM task represented
the same arithmetic as a scalar ALU immediate. Preserving rank-0 scalar
constants in the deployment graph removes those unnecessary loads without
changing the model's arithmetic, tensor layout, or candidate selection rule.
The selected configurations are now programmatically exported by the unchanged
minimum-positive-TSIM-cycle selector; the minimum configurations for
occurrences 2 and 5 are 209 and 301, not the slower temporary choices 64 and
111 from the earlier report. Their real deployment differences are 0.0348%
and 0.0000%, respectively. All eight occurrences are within the 10% limit.

The exported manifest reuses the already complete FSIM and single-call TSIM
records from the full search. No tuning task or measurement protocol changed,
so the recorded candidates remain valid; deployment was remeasured after
correcting the lowering. All ten committed sample outputs match the pure HOST
reference. Debug-profile and ordinary full-model TSIM cycle counts match.
Summed selected VTA cycles equal the uninstrumented tuned full-model cycle
count, with zero residual overhead.

Whole-model utilization uses the same ten uninstrumented invocations for MACs
and cycles; it does not substitute a sum of isolated operator cycles.

| Measure | Baseline | Tuned |
|---|---:|---:|
| Full-model TSIM cycles (10 samples) | 36,328,640 | 2,591,430 |
| Logical MACs (10 samples) | 120,586,240 | 120,586,240 |
| Whole-model MAC utilization | 5.1864% | 72.7073% |

The tuned model uses 14.0188× fewer TSIM cycles and gains 67.5209 percentage
points in measured whole-model MAC utilization. Host work is excluded from the
VTA MAC numerator; the model identity remains metadata so the calculator stays
model-independent.

The versioned deployment and utilization data are in `deployment-c4-full.json`
and `mac-utilization-c4-full.json`. Selected configurations and self-contained
native records are under `optimal/c4-full/`. Candidate failures, counts and
resume state remain under `../build/two_stage_tuning/c4-full/`.
