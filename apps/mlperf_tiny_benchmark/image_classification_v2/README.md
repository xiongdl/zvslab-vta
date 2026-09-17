# MLPerf Tiny ResNet-8 Large HOST deployment

This fixed-purpose application imports the committed floating MLPerf Tiny v1.4
ResNet-8 Large model, applies the documented TVM quantization policy once, and builds
both a pure host reference and a mixed VTA Graph Executor artifact. It reloads
the host libraries, compares their output tensors exactly for the ten committed
PNG samples, and requires positive simulator activity. The matrix mode builds
LLVM and C variants below separate `llvm-fsim/` and `c-fsim/` (FSIM) or
`llvm-tsim/` and `c-tsim/` (TSIM) bundle roots.

The mixed artifacts use the `resnet8_large` identity and contain exactly four
single-convolution VTA regions (`tvmgen_mlperf_resnet_large_vta_main_0` through
`_3`) plus five HOST convolutions.

Importing `vta` loads and validates the compiler target extension. The mixed
branch explicitly applies `vta.relay.partition_for_vta()` once, then passes
`tvm.target.Target("vta")` to `relay.build`; unsupported operators remain in the
LLVM host portion of the same standard runtime module.

The application is an execution-equivalence example. It does not report model
accuracy, performance, energy, or MLPerf submission results.

## Prerequisites

From the repository root, prepare the pinned Python environment and build the
compiler extension and simulator libraries:

```bash
./scripts/setup_tvm_vta_env.sh
./scripts/build_vta_lib.sh --target libtvm-vta-ext
./scripts/build_vta_lib.sh --target libvta_fsim
./scripts/build_vta_lib.sh --target libvta_hw
```

## Run

From the repository root:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_config.json" \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py
```

The optional `--output-dir PATH` changes only the generated-artifact location.
The default `build/` directory is ignored by the repository. The default CLI
mode is LLVM FSIM; pass `--host-codegen c` for a single C-host run or
`--host-codegen all` to build and execute the complete ordered LLVM/C matrix.
Pass `--simulator tsim` with `VTA_CONFIG_FILE` set to
`vta/config/tsim_sample.json` for the Verilated hardware model. Matrix bundles
are published as `<host>-<simulator>/{reference,mixed}/`, with each directory
containing its Graph JSON, parameters, DSO, manifest, and generated host
source. A partial export is removed if the build fails.

FSIM matrix:

```bash
VTA_CONFIG_FILE="$PWD/vta/config/vta_config.json" \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator fsim --host-codegen all
```

Complete TSIM matrix (fresh process):

```bash
VTA_CONFIG_FILE="$PWD/vta/config/tsim_sample.json" \
PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python \
  vta/apps/mlperf_tiny_benchmark/image_classification_v2/run.py \
  --simulator tsim --host-codegen all
```

Successful output reports four deterministic VTA regions and five HOST convolutions, ten exact output
comparisons per host, and positive simulator counters. FSIM validates GEMM,
weight-load, and output-store counters; TSIM validates its supported
`cycle_count` counter only. Missing libraries, unexpected model or routing
structure, output differences, wrong configuration, and absent accelerator
activity cause a nonzero exit. TSIM initialization and hardware loading remain
lazy until all four bundles have been built, exported, and reloaded.
