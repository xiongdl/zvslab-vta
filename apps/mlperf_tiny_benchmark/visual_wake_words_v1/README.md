# Visual Wake Words V1

This application deploys one Visual Wake Words image at a time and tunes only
the real VTA computation in the prepared model. It owns its model preparation,
deployment, workload snapshots, isolated FSIM/TSIM tuning, schedule validation,
reports, and Make orchestration. The committed floating-point model expects one
96×96 RGB image, decoded as float32 NHWC and divided by 255. TVM applies the
fixed `global_scale=8`, `skip_conv_layers=[0]` quantization policy. The result
is class `0` (`non_person`) or `1` (`person`) with both raw scores. This is a
deployment and tuning example; classification accuracy is not an acceptance
gate.

The current TVM/VTA partitioner produces real convolution partitions for this
model. Deployment reports actual placement and measurements. If a future
runtime produces no real partitions, VTA targets truthfully fall back to CPU,
report zero VTA coverage and N/A cycles, and workload export or schedule replay
fails before publishing files.

## Prerequisites

Run from the repository root. Initialize `tvm/` and `vta/`, use the pinned
`.envs/tvm-vta-env` environment, and build the TVM libraries and both VTA
simulators using `scripts/README.md`. The shared geometry file is
`$PWD/vta/config/vta_64mac.json`. Do not install packages for this application.

The direct CLI uses the same pinned interpreter and the app-local Python
package:

```bash
export APP="$PWD/vta/apps/mlperf_tiny_benchmark/visual_wake_words_v1"
export PYTHONPATH="$PWD/tvm/python:$PWD/vta/python:$APP"
export CONFIG="$PWD/vta/config/vta_64mac.json"
export PYTHON="$PWD/.envs/tvm-vta-env/bin/python"
```

The default model is `model/vww_96_float.tflite`; the default image is
`samples/00-non-person-000000000009.jpg`. Optional model/input/output paths are
accepted explicitly, but custom models must match the supported one-input,
one-output VWW tensor and operator topology.

## Manual acceptance

Run these steps from the repository root. Successful deployment prints the
predicted VWW class and two raw scores. The report records model and input
hashes, selected target, CPU/VTA placement, schedule coverage, logical MACs,
and measurements available from the selected simulator.

1. Run the app-owned tests, including asset/model contracts, CPU startup,
   graph bundle integrity, CLI and Make boundaries, workload snapshots, tuning,
   and cleanup:

   ```bash
   VTA_CONFIG_FILE="$CONFIG" VTA_BACKEND=fsim PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" -m pytest -q "$APP/tests"
   ```

2. Confirm both CPU targets work without VTA configuration or a simulator:

   ```bash
   env -u VTA_BACKEND -u VTA_CONFIG_FILE PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target c --deployment-report "$APP/build/c.md"
   env -u VTA_BACKEND -u VTA_CONFIG_FILE PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target llvm --deployment-report "$APP/build/llvm.md"
   ```

   Each command must report one class and two scores; its report shows CPU
   placement and N/A cycles. `--target c --export-workloads ...` must fail
   before runtime startup because CPU deployment cannot export VTA workloads.

3. Check both VTA host code generators under each backend using one image:

   ```bash
   for backend in fsim tsim; do
     for host in c llvm; do
       VTA_BACKEND="$backend" VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
         "$PYTHON" "$APP/deploy.py" --target "vta,$host" --simulator "$backend" \
         --deployment-report "$APP/build/${backend}-${host}.md"
     done
   done
   ```

   Each report must show real VTA partitions and CPU fallback operators.
   FSIM counters must show accelerator activity and cycle fields remain N/A.
   TSIM reports positive whole-graph and per-partition cycles. CPU and VTA
   prepared outputs must agree within the app test's float tolerance. A
   simulator/backend mismatch must fail before model compilation.

4. Export a workload snapshot from the default-schedule FSIM deployment. Then
   move the model and image out of reach for tuning; the tuning commands below
   consume only the exported workload file:

   ```bash
   VTA_BACKEND=fsim VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target vta,llvm --simulator fsim \
     --export-workloads "$APP/build/workloads.json"
   ```

   The snapshot must contain model id/hash, input hash and preprocessing,
   quantization policy, geometry/config hash, compatibility data, real Relay
   occurrences, captured activations, and an integrity seal. Editing a payload
   or loading another model's snapshot must be rejected.

5. Run a bounded FSIM candidate search and TSIM selection for occurrence zero:

   ```bash
   VTA_BACKEND=fsim VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/tune.py" --workloads "$APP/build/workloads.json" \
     --workload 0 --simulator fsim --trial-batch 1 --min-successful 1 \
     --output-logs "$APP/tune/vta_64mac/fsim.tmp"
   VTA_BACKEND=tsim VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/tune.py" --workloads "$APP/build/workloads.json" \
     --workload 0 --simulator tsim --input-logs "$APP/tune/vta_64mac/fsim.tmp" \
     --output-logs "$APP/tune/vta_64mac/best.log"
   ```

   FSIM logs contain candidates, not deployable winners. TSIM selects the
   lowest successful cycle count and publishes `best.log`, matching
   `best.json`, `config.json`, and `config.sha256`. If a candidate cannot be
   compiled or measured, it is counted as a failure; no-success or failed
   publication must preserve any prior valid schedule files.

6. Replay the TSIM-selected schedule and check output/report coverage:

   ```bash
   VTA_BACKEND=tsim VTA_CONFIG_FILE="$CONFIG" PYTHONPATH="$PYTHONPATH" \
     "$PYTHON" "$APP/deploy.py" --target vta,llvm --simulator tsim \
     --schedule "$APP/tune/vta_64mac/best.log" \
     --deployment-report "$APP/build/replay.md"
   ```

   The report marks occurrence zero selected and all other real occurrences as
   default. Selected output must agree with the prepared CPU output. Measured
   replay cycles and selected-occurrence AutoTVM cycles must align strictly
   within 10 percent under the same layer measurement protocol.

7. Confirm the Make workflow routes paths safely, bounds tuning, completes the
   full FSIM→TSIM sequence without redeploying a winner, and cleans only local
   build output and Python caches:

   ```bash
   make -C "$APP" deploy TARGET=llvm REPORT="$APP/build/make-cpu.md"
   make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=fsim \
     EXPORT_WORKLOADS="$APP/build/make-workloads.json"
   make -C "$APP" tune-fsim WORKLOADS="$APP/build/make-workloads.json" \
     WORKLOAD=0 TRIAL_BATCH=1 MIN_SUCCESSFUL=1
   make -C "$APP" tune-tsim WORKLOADS="$APP/build/make-workloads.json" \
     INPUT_LOGS="$APP/tune/vta_64mac/fsim.tmp" WORKLOAD=0
   make -C "$APP" deploy TARGET=vta,llvm SIMULATOR=tsim \
     SCHEDULE="$APP/tune/vta_64mac/best.log" REPORT="$APP/build/make-replay.md"
   make -C "$APP" tune WORKLOAD=0 TRIAL_BATCH=1 MIN_SUCCESSFUL=1
   mkdir -p "$APP/build/custom input path"
   cp "$APP/model/vww_96_float.tflite" "$APP/build/custom input path/model.tflite"
   cp "$APP/samples/00-non-person-000000000009.jpg" "$APP/build/custom input path/input.jpg"
   "$PYTHON" "$APP/deploy.py" --target llvm \
     --model "$APP/build/custom input path/model.tflite" \
     --input "$APP/build/custom input path/input.jpg" \
     --output-dir "$APP/build/custom input path/direct bundle" \
     --deployment-report "$APP/build/custom input path/direct report.md"
   make -C "$APP" deploy TARGET=llvm \
     MODEL="$APP/build/custom input path/model.tflite" \
     INPUT="$APP/build/custom input path/input.jpg" \
     OUTPUT_DIR="$APP/build/custom input path/make bundle" \
     REPORT="$APP/build/custom input path/make report.md"
   make -C "$APP" clean
   ```

   After clean, local `build/` and Python caches are gone, while the model,
   samples, license, and persistent `tune/vta_64mac/` schedule/config files
   remain. Both custom-path commands must create their reports and bundles
   despite spaces in the paths. Running clean again succeeds. The full tune
   target exports when needed, runs FSIM then TSIM, and stops after selection.

## Direct CLI and Make contract

`deploy.py` accepts `--target c|llvm|vta,c|vta,llvm`, `--simulator fsim|tsim`,
`--model`, `--input`, `--schedule`, `--output-dir`, `--deployment-report`, and
`--export-workloads`. It compiles only the selected target. CPU startup does
not import VTA. VTA requests require matching `VTA_BACKEND` and `--simulator`,
plus an absolute `VTA_CONFIG_FILE`.

`tune.py` accepts only an exported `--workloads` snapshot. FSIM accepts
`--trial-batch`, `--min-successful`, and `--timeout`; TSIM requires `--input-logs`
and rejects FSIM-only options. `make deploy`, `make tune-fsim`, `make tune-tsim`,
`make tune`, and `make clean` use corresponding variables from the template.
`make clean` preserves all persistent tuning results and source assets.
