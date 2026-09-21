# MLPerf Tiny VWW HOST deployment

This fixed-purpose application imports the committed floating MLPerf Tiny v1.4
Visual Wake Words model, applies the documented TVM quantization policy once, and builds
both a pure host reference and a mixed VTA Graph Executor artifact. It reloads
the host libraries, compares their output tensors within the fixed `1e-6`
absolute/relative tolerance for the ten committed JPEG samples, and requires
positive simulator activity. The matrix mode builds
LLVM and C variants below separate `llvm-fsim/` and `c-fsim/` (FSIM) or
`llvm-tsim/` and `c-tsim/` (TSIM) bundle roots.

Importing `vta` loads and validates the compiler target extension. The mixed
branch explicitly applies `vta.relay.partition_for_vta()` once, then uses
`vta.relay.plan_devices_for_vta()` to constrain host operators to CPU and
outlined VTA functions to `ext_dev`. The resulting CPU/VTA target map is passed
to `relay.build`, which inserts compiler-owned device copies and produces the
standard mixed runtime module without post-build graph JSON edits. The plan
keeps the thirteen depthwise convolutions on CPU and the twelve deterministic
VTA regions on `ext_dev`.

The VTA compiler target is activated only as the lowering context; the target
map returned by the planner remains the canonical `ext_dev -device=vta`
target paired with the selected LLVM or C host target:

```python
with vta.build_config():
    mixed_factory = relay.build(plan.module, target=plan.targets)
```

The application is an execution-equivalence example. It does not report model
accuracy, performance, energy, or MLPerf submission results.

## Prerequisites

From the repository root, prepare the pinned Python environment and build the
compiler extension and simulator libraries:

```bash
# Use the existing project environment at .envs/tvm-vta-env; do not recreate it
# during verification.
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" --backend all
```

The active contract is the shared absolute `VTA_CONFIG_FILE` plus
`VTA_BACKEND=fsim|tsim`. The build script uses `--backend fsim|tsim|all` and
this runner uses `--simulator fsim|tsim`; the values must match. The CPU
reference branch is part of the FSIM matrix and is not a separate VTA backend.
`TARGET=sim`, `TARGET=tsim`, and `--target libvta_*` are retired; use the
shared geometry file and explicit backend selectors. FPGA backends such as
`pynq` and `zcu104` remain deferred.

## Run

From the repository root:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1/run.py
```

The optional `--output-dir PATH` changes only the generated-artifact location.
The default `build/` directory is ignored by the repository. The default CLI
mode is LLVM FSIM; pass `--host-codegen c` for a single C-host run or
`--host-codegen all` to build and execute the complete ordered LLVM/C matrix.
Pass `--simulator tsim` with `VTA_CONFIG_FILE` set to the shared
`vta/config/vta_64mac.json` and `VTA_BACKEND=tsim` for the Verilated hardware
model. Matrix bundles
are published as `<host>-<simulator>/{reference,mixed}/`, with each directory
containing its Graph JSON, parameters, DSO, manifest, and generated host
source. A partial export is removed if the build fails.

FSIM matrix:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=fsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1/run.py \
  --simulator fsim --host-codegen all
```

Complete TSIM matrix (fresh process):

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1/run.py \
  --simulator tsim --host-codegen all
```

Successful output reports twelve deterministic VTA regions, ten bounded output
comparisons per host, and positive simulator counters. FSIM validates GEMM,
weight-load, and output-store counters; TSIM validates its supported
`cycle_count` counter only; FSIM-only counters are not required for TSIM.
Missing libraries, unexpected model or routing
structure, output differences, wrong configuration, and absent accelerator
activity cause a nonzero exit. TSIM initialization and hardware loading remain
lazy until all four bundles have been built, exported, and reloaded.
