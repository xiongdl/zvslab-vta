# Keyword spotting v1 full-search result

The selected KWS V1 schedules passed a real one-sample TSIM deployment. The
run used the committed sample `down-00176480_nohash_0.wav`, existing WAV-to-MFCC
preprocessing, and the VTA `vta_64mac.json` geometry. HOST-reference correctness
passed. All four occurrence cycle differences are below 0.005%, and ordinary
and debug full-model counters agree at 99,884 cycles.

| Occurrence | Symbol | MACs/invocation | AutoTVM TSIM cycles | Deployment TSIM cycles | Difference | MAC utilization |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `tvmgen_mlperf_kws_vta_main_0` | 512,000 | 24,970 | 24,971 | 0.0040% | 32.0372% |
| 1 | `tvmgen_mlperf_kws_vta_main_1` | 512,000 | 24,970 | 24,971 | 0.0040% | 32.0372% |
| 2 | `tvmgen_mlperf_kws_vta_main_2` | 512,000 | 24,970 | 24,971 | 0.0040% | 32.0372% |
| 3 | `tvmgen_mlperf_kws_vta_main_3` | 512,000 | 24,970 | 24,971 | 0.0040% | 32.0372% |

Each operator MAC utilization is calculated from its real deployment cycles
and the geometry peak of 64 MACs/cycle. The measured untuned baseline is
2,354,444 full-model cycles; tuned deployment is 99,884 cycles, for a 23.5718x
cycle speedup. Whole-model utilization is 1.3591% at baseline and 32.0372%
after tuning.

The full search is recorded in `tune/optimal/20261001T192416.141960Z/` and
used 100-candidate FSIM batches with a target of 20 successful schedules per
occurrence. All four valid 384-configuration spaces were evaluated; every
occurrence produced 21 successful FSIM schedules and 21 successful TSIM
measurements. The exported schedule is the minimum positive TSIM-cycle
schedule among its successful candidates. See
`tune/deployment-full.json` for identity-bound deployment evidence and
`tune/mac-utilization-full.json` or `.csv` for the complete per-occurrence MAC
calculation.
