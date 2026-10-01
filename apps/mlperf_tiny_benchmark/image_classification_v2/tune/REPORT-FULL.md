# IC V2 full-search deployment report

The complete default search covered all eight IC V2 VTA fusion occurrences.
FSIM used 100-trial batches and reached the 20-success quota for every
occurrence without exhausting any valid configuration space. Each distinct
successful FSIM schedule was measured on TSIM with 60-second FSIM and
120-second TSIM candidate timeouts. TSIM costs are one counted invocation with
warmup excluded.

| Occurrence | FSIM attempts | FSIM successes / TSIM measurements | Selected config | AutoTVM cycles | Deployed cycles | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 23 / 23 | 314 | 267,819 | 267,819 | 0.000000% |
| 1 | 300 | 28 / 28 | 309 | 267,565 | 267,565 | 0.000000% |
| 2 | 200 | 32 / 32 | 171 | 38,273 | 38,274 | 0.002613% |
| 3 | 300 | 25 / 25 | 471 | 136,801 | 136,801 | 0.000000% |
| 4 | 500 | 23 / 23 | 418 | 250,389 | 250,389 | 0.000000% |
| 5 | 188 | 35 / 35 | 270 | 30,634 | 30,641 | 0.022850% |
| 6 | 400 | 23 / 23 | 589 | 126,893 | 126,893 | 0.000000% |
| 7 | 500 | 20 / 20 | 591 | 241,969 | 241,969 | 0.000000% |

All eight strict checks passed using `10 * abs(deployed - AutoTVM) < AutoTVM`.
The largest difference was 0.022850% for occurrence 5. The one-sample
performance gate used `00-airplane.png`; debug and ordinary complete-run
counters both measured 1,360,351 TSIM cycles. The untuned baseline measured
21,226,413 cycles on that same sample. The tuned run used 15.60 times fewer
full-model cycles.

Only after the performance gate passed, the selected configurations ran on all
ten committed samples for correctness. All ten outputs passed. The ten-sample
phase did not repeat performance profiling. Per-node measurements execute one
graph-resident invocation after clearing the profiler; warmup is excluded.

The resumed run retains its initial FSIM RPC tracker startup errors from the
first restricted attempt (`PermissionError: [Errno 1] Operation not permitted`
on local bind). They remain classified as infrastructure errors in the durable
state and historical worker-failure list, separate from candidate failures.
The authorized resumed run completed all eight FSIM workloads and all TSIM
measurements. Candidate-specific schedule/build/runtime failures remain in the
search state and were not counted as successes. The final manifest reports
`FULL_SEARCH` and is independently replayable without intermediate build
files.

Self-contained selected records and the best manifest are in
`optimal/20261001T035246.904010Z/`. Deployment measurements and identity hashes
are in `deployment-full.json`; candidate logs, failures and resume data remain
under `../build/two_stage_tuning/20261001T035246.904010Z/`.

The model-independent MAC utilization command currently rejects the V2
deployment report as an unsupported artifact kind, so no MAC utilization JSON
is published. This does not affect the per-occurrence cycle gate.
