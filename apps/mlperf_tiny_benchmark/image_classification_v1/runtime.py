# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Build and execute one selected MLPerf Tiny ResNet-8 target."""

import json
import hashlib
import tempfile
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend import te_compiler
from tvm.contrib import graph_executor

from graph_artifacts import export_graph_bundle
from model_pipeline import load_sample, prepare_model
from deployment_compute import capture_deployment_compute
from deployment import lower_selected_deployment
from schedule import load_schedule_snapshot


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet.tflite"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build"
MODEL_ID = "image_classification_v1"
INPUT_NAME = "input_1"


@dataclass(frozen=True)
class SimulatorSession:
    """Validated simulator registry adaptor for one configured process."""

    label: str
    environment_target: str
    clear_registry: str
    status_registry: str
    required_registries: tuple
    activity_counter: str
    diagnostic: str

    def validate_environment(self):
        # The testing.simulator module eagerly loads its selected native backend
        # at import time. Environment validation must remain a pure selector so
        # a mismatched backend cannot leave two incompatible simulators loaded.
        from vta.backend import normalize_backend

        active_target = normalize_backend(simulator=self.label)
        if active_target != self.environment_target:
            raise RuntimeError(
                f"simulator {self.label!r} requires VTA backend "
                f"{self.environment_target!r}, active target is {active_target!r}"
            )
        return self

    def load(self):
        # Importing this standard module is deliberately the only simulator
        # initialization path.  In particular, TSIM's import performs the
        # global driver load, hardware-module load, and vta.tsim.init call.
        try:
            from vta.testing import simulator
            simulator.load_backend(self.label)
        except Exception as error:
            missing = [
                name
                for name in self.required_registries
                if tvm.get_global_func(name, allow_missing=True) is None
            ]
            detail = (
                f"missing registry functions: {', '.join(missing)}"
                if missing
                else "standard simulator initialization failed"
            )
            raise RuntimeError(
                f"{self.label.upper()} is unavailable; {detail}. Build the required "
                f"libraries with {self.diagnostic}"
            ) from error

        missing = [
            name
            for name in self.required_registries
            if tvm.get_global_func(name, allow_missing=True) is None
        ]
        if missing:
            raise RuntimeError(
                f"{self.label.upper()} is unavailable; missing registry functions: "
                f"{', '.join(missing)}. Build the required libraries with "
                f"{self.diagnostic}"
            )
        return simulator

    def clear_and_validate(self, simulator=None):
        clear = getattr(simulator, "clear_stats", None) if simulator is not None else None
        status = getattr(simulator, "stats", None) if simulator is not None else None
        clear = clear or tvm.get_global_func(self.clear_registry, allow_missing=True)
        status = status or tvm.get_global_func(self.status_registry, allow_missing=True)
        if clear is None or status is None:
            raise RuntimeError(
                f"{self.label.upper()} profiler registry is unavailable; build the "
                f"required libraries with {self.diagnostic}"
            )
        clear()
        stats = self.read_stats(status)
        if self.label == "tsim":
            if stats != {"cycle_count": 0}:
                raise RuntimeError(f"TSIM profiler did not reset to {{'cycle_count': 0}}: {stats}")
        elif any(value != 0 for value in stats.values()):
            raise RuntimeError(f"FSIM profiler did not reset to zero: {stats}")
        return stats

    def read_stats(self, status=None, simulator=None):
        status = status or getattr(simulator, "stats", None)
        status = status or tvm.get_global_func(self.status_registry, allow_missing=True)
        if status is None:
            raise RuntimeError(
                f"{self.label.upper()} profiler status is unavailable; build the "
                f"required libraries with {self.diagnostic}"
            )
        try:
            raw = status()
            stats = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{self.label.upper()} profiler returned malformed counters") from error
        if not isinstance(stats, dict):
            raise RuntimeError(f"{self.label.upper()} profiler counters must be a JSON object")
        return stats

    def validate_activity(self, stats):
        if self.label == "tsim":
            value = stats.get(self.activity_counter)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(
                    f"TSIM profiler counter cycle_count must be a positive integer: {stats}"
                )
            return
        for counter in ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes"):
            if stats.get(counter, 0) <= 0:
                raise RuntimeError(f"FSIM profiler counter {counter} must be positive")


@dataclass(frozen=True)
class LayerMetrics:
    """One logical convolution measurement row in a deployment report."""

    name: str
    device: str
    operation: str
    logical_macs: int
    cycles: int | None
    peak_macs_per_cycle: int | None
    utilization: float | None


@dataclass(frozen=True)
class SelectedDeploymentResult:
    """Result from compiling and running one selected target."""

    target: str
    simulator: str | None
    model_path: Path
    input_path: Path
    model_sha256: str
    input_sha256: str
    schedule: Path | None
    schedule_coverage: tuple
    predicted_class: int
    scores: np.ndarray
    layers: tuple
    whole_cycles: int | None
    profiler_stats: dict | None


def _simulator_session(simulator):
    if simulator == "fsim":
        return SimulatorSession(
            label="fsim",
            environment_target="fsim",
            clear_registry="vta.simulator.profiler_clear",
            status_registry="vta.simulator.profiler_status",
            required_registries=(
                "vta.simulator.profiler_clear",
                "vta.simulator.profiler_status",
            ),
            activity_counter="gemm_counter",
            diagnostic="bash scripts/build_vta_lib.sh --config /absolute/path/to/vta_64mac.json --backend fsim",
        )
    if simulator == "tsim":
        return SimulatorSession(
            label="tsim",
            environment_target="tsim",
            clear_registry="vta.tsim.profiler_clear",
            status_registry="vta.tsim.profiler_status",
            required_registries=(
                "vta.tsim.init",
                "vta.tsim.profiler_clear",
                "vta.tsim.profiler_status",
                "runtime.module.loadfile_vta-tsim",
            ),
            activity_counter="cycle_count",
            diagnostic="bash scripts/build_vta_lib.sh --config /absolute/path/to/vta_64mac.json --backend tsim",
        )
    raise ValueError(f"unsupported simulator {simulator!r}; supported simulators are fsim and tsim")


def _mixed_target(host_codegen):
    import vta

    environment = vta.get_env()
    host = environment.target_host if host_codegen == "llvm" else tvm.target.Target("c")
    return tvm.target.Target("vta", host=host)


@contextmanager
def _selected_snapshot_lowering(compiler, deployment, snapshot, compiler_config):
    """Route Relay's ordinary VTA lowering through occurrence-bound schedules."""
    name = "vta.relay._relay_to_tir"
    previous = tvm.get_global_func(name, allow_missing=True)
    if previous is None:
        raise RuntimeError("VTA Relay-to-TIR callback is unavailable")

    def lower(module):
        return lower_selected_deployment(
            module, deployment, snapshot, compiler, compiler_config
        )

    tvm.register_func(name, lower, override=True)
    try:
        yield
    finally:
        tvm.register_func(name, previous, override=True)


def validate_mixed_symbols(module, expected_symbols):
    """Require every routed function in the reloaded mixed artifact."""
    for symbol in expected_symbols:
        if not module.implements_function(symbol, True):
            raise RuntimeError(f"reloaded mixed artifact is missing VTA symbol {symbol}")


def _load_simulator(simulator):
    session = _simulator_session(simulator)
    session.validate_environment()
    return session, session.load()


def _conv_macs(call):
    """Count logical MACs from inferred Relay dimensions, excluding packing."""
    output = tuple(int(dim) for dim in call.checked_type.shape)
    weight = tuple(int(dim) for dim in call.args[1].checked_type.shape)
    layout = str(call.attrs.kernel_layout)
    axes = {axis: weight[index] for index, axis in enumerate(layout)}
    if len(output) != 4 or len(weight) != 4:
        raise ValueError("ResNet-8 convolution tensors must have rank four")
    # Kernel I already represents input channels per group.
    return output[0] * output[1] * output[2] * output[3] * axes["H"] * axes["W"] * axes["I"]


def _collect_conv_rows(function, name_prefix, device):
    calls = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            if node.op.name == "nn.conv2d":
                calls.append(node)

    relay.analysis.post_order_visit(function.body, visit)
    return [
        (f"{name_prefix}.conv{index}", device, _conv_macs(call))
        for index, call in enumerate(calls)
    ]


def _collect_non_mac_rows(function, name_prefix, device):
    operations = []

    def visit(node):
        if (isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op)
                and node.op.name != "nn.conv2d"):
            operations.append(node.op.name)

    relay.analysis.post_order_visit(function.body, visit)
    return [
        (f"{name_prefix}.{operation}{index}", device, operation)
        for index, operation in enumerate(operations)
    ]


def _selected_layer_metrics(prepared, target, simulator, measured_cycles):
    rows = []
    non_mac_rows = []
    if target.startswith("vta,"):
        for symbol in prepared.routing.symbols:
            function = next(
                function for function in prepared.mixed_module.functions.values()
                if isinstance(function, relay.Function) and function.attrs is not None
                and "Compiler" in function.attrs and function.attrs.get_str("Compiler") == "vta"
                and function.attrs.get_str("global_symbol") == symbol
            )
            rows.extend(_collect_conv_rows(function, symbol, "vta"))
            non_mac_rows.extend(_collect_non_mac_rows(function, symbol, "vta"))
        rows.extend(_collect_conv_rows(prepared.mixed_module["main"], "cpu", "cpu"))
        non_mac_rows.extend(_collect_non_mac_rows(prepared.mixed_module["main"], "cpu", "cpu"))
    else:
        rows.extend(_collect_conv_rows(prepared.quantized_module["main"], "cpu", "cpu"))
        non_mac_rows.extend(_collect_non_mac_rows(prepared.quantized_module["main"], "cpu", "cpu"))

    peak = None
    if target.startswith("vta,"):
        import vta

        env = vta.get_env()
        peak = int(env.BATCH) * int(env.BLOCK_IN) * int(env.BLOCK_OUT)
    metrics = []
    for name, device, macs in rows:
        symbol = name.rsplit(".conv", 1)[0] if device == "vta" else None
        cycles = measured_cycles.get(symbol) if symbol and simulator == "tsim" else None
        metrics.append(LayerMetrics(
            name, device, "nn.conv2d", macs, cycles, peak if device == "vta" else None,
            macs / (cycles * peak) if cycles and peak else None,
        ))
    metrics.extend(
        LayerMetrics(name, device, operation, 0, None, peak if device == "vta" else None, None)
        for name, device, operation in non_mac_rows
    )
    return tuple(metrics)


def _build_selected_factory(module, target, host_codegen, use_vta):
    if use_vta:
        import vta

        if host_codegen == "c":
            with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
                with vta.build_config(config={"tir.disable_vectorize": True}):
                    return relay.build(module, target=_mixed_target("c"))
        with vta.build_config():
            return relay.build(module, target=_mixed_target("llvm"))
    if host_codegen == "c":
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            return relay.build(module, target=target)
    return relay.build(module, target=target)


def _export_pre_schedule_workloads(prepared, compute, image, input_path,
                                  host_codegen, simulator, output_path):
    """Capture true VTA activations from a temporary default-schedule graph."""
    import vta
    from tvm.contrib.debugger import debug_executor
    from workloads import (
        make_document,
        make_layer_record,
        portable_config_space_identity,
        portable_geometry,
        write_workloads,
    )

    compiler = te_compiler.get()
    compiler.clear()
    try:
        factory = _build_selected_factory(
            prepared.mixed_module, _mixed_target(host_codegen), host_codegen, True
        )
    finally:
        compiler.clear()

    with tempfile.TemporaryDirectory(prefix="resnet8-workloads-") as temporary:
        bundle = export_graph_bundle(
            factory, temporary, "default", artifact_name="resnet8-workloads-default",
            artifact_role="mixed", model_sha256=prepared.imported.model_sha256,
            host_codegen=host_codegen, simulator=simulator,
            expected_vta_symbols=prepared.routing.symbols,
        )
        debug = debug_executor.create(bundle.graph_json, bundle.module, tvm.ext_dev(0))
        debug.load_params(bundle.params)
        debug.set_input(INPUT_NAME, image)
        debug._run_per_layer()
        graph_nodes = debug.debug_datum.get_graph_nodes()
        node_outputs = debug.debug_datum.get_output_tensors()
        raw_nodes = json.loads(bundle.graph_json)["nodes"]
        node_indices = {
            node.get("attrs", {}).get("global_symbol"): index
            for index, node in enumerate(graph_nodes)
            if node.get("attrs", {}).get("global_symbol")
        }
        records = []
        for layer in compute.layers:
            node_index = node_indices.get(layer.symbol)
            if node_index is None:
                raise RuntimeError(f"default graph omitted VTA workload {layer.symbol}")
            activation = None
            for source_index, output_index, *_ in raw_nodes[node_index]["inputs"]:
                key = (
                    f"{graph_nodes[source_index]['name']}____topo-index:{source_index}"
                    f"____output-num:{output_index}"
                )
                tensor = node_outputs[key].numpy()
                if (tuple(tensor.shape) == layer.inputs[0].shape
                        and tensor.dtype.name == layer.inputs[0].dtype):
                    activation = tensor
                    break
            if activation is None:
                raise RuntimeError(
                    f"could not capture the Relay input activation for {layer.symbol}"
                )
            records.append(make_layer_record(
                index=layer.occurrence, symbol=layer.symbol, function=layer.function,
                compute_sha256=layer.compute_sha256,
                inputs=tuple(item.shape for item in layer.inputs),
                input_dtypes=tuple(item.dtype for item in layer.inputs),
                output_shape=layer.output.shape, output_dtype=layer.output.dtype,
                activation=activation,
                config_space_identity=portable_config_space_identity(layer),
            ))

    config_path = Path(os.environ["VTA_CONFIG_FILE"]).expanduser().resolve()
    document = make_document(
        model_sha256=prepared.imported.model_sha256,
        input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        config_bytes=config_path.read_bytes(), config_basename=config_path.name,
        geometry=portable_geometry(compute.geometry), tvm_version=tvm.__version__,
        vta_version=getattr(vta, "__version__", "source-tree"), workloads=records,
    )
    return write_workloads(document, output_path)


def run_selected(target="vta,llvm", simulator="fsim", schedule=None,
                 output_dir=DEFAULT_OUTPUT_DIR, model_path=MODEL_PATH,
                 input_path=None, export_workloads=None):
    """Compile and run exactly one target using one image."""
    if target not in {"c", "llvm", "vta,c", "vta,llvm"}:
        raise ValueError(f"unsupported target {target!r}")
    if simulator not in {"fsim", "tsim"}:
        raise ValueError(f"unsupported simulator {simulator!r}")
    use_vta = target.startswith("vta,")
    if export_workloads is not None and not use_vta:
        raise ValueError("--export-workloads requires a target that includes VTA")
    if not use_vta:
        schedule = None
    model_path = Path(model_path).expanduser().resolve()
    input_path = Path(input_path or APP_ROOT / "samples" / "00-airplane.png").expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    image = load_sample(input_path)
    prepared = prepare_model(model_path, use_vta=use_vta)
    host_codegen = target.split(",")[-1]
    module = prepared.mixed_module if use_vta else prepared.quantized_module
    selected = None
    compute = None
    if use_vta:
        import vta
        from deployment_compute import capture_deployment_compute

        compute = capture_deployment_compute(module, MODEL_ID, prepared.imported.model_sha256)
        selected = load_schedule_snapshot(schedule, compute)
        config = vta.relay.transform.VTACompilerConfig.from_env(vta.get_env())
        compiler = te_compiler.get()
        if selected and selected.selected:
            compiler.clear()
            try:
                with _selected_snapshot_lowering(compiler, compute, selected, config):
                    factory = _build_selected_factory(module, _mixed_target(host_codegen), host_codegen, True)
            finally:
                compiler.clear()
        else:
            factory = _build_selected_factory(module, _mixed_target(host_codegen), host_codegen, True)
    else:
        factory = _build_selected_factory(
            module, tvm.target.Target(host_codegen), host_codegen, False
        )

    role = "mixed" if use_vta else "reference"
    symbols = prepared.routing.symbols if use_vta else ()
    artifact_name = f"resnet8_{target.replace(',', '_')}_{simulator if use_vta else 'cpu'}"
    bundle = export_graph_bundle(
        factory, output_dir, target.replace(",", "_"), artifact_name=artifact_name,
        artifact_role=role, model_sha256=prepared.imported.model_sha256,
        host_codegen=host_codegen, simulator=simulator if use_vta else "cpu",
        expected_vta_symbols=symbols,
    )
    if use_vta:
        validate_mixed_symbols(bundle.module, symbols)
        session, simulator_module = _load_simulator(simulator)
        if export_workloads is not None:
            exported = _export_pre_schedule_workloads(
                prepared, compute, image, input_path, host_codegen,
                simulator, export_workloads,
            )
            print(f"Workloads exported: {exported}")
    device = tvm.ext_dev(0) if use_vta else tvm.cpu(0)
    graph = graph_executor.create(bundle.graph_json, bundle.module, device)
    graph.load_params(bundle.params)
    graph.set_input(INPUT_NAME, image)
    profiler = None
    cycles = None
    if use_vta:
        session.clear_and_validate(simulator_module)
        graph.run()  # Excluded warmup.
        session.clear_and_validate(simulator_module)
    graph.run()
    scores = graph.get_output(0).numpy()
    if use_vta:
        profiler = session.read_stats(simulator=simulator_module)
        session.validate_activity(profiler)
        if simulator == "tsim":
            cycles = profiler["cycle_count"]

    measured = {}
    if use_vta and simulator == "tsim":
        from tvm.contrib.debugger import debug_executor

        debug = debug_executor.create(bundle.graph_json, bundle.module, device)
        debug.load_params(bundle.params)
        debug.set_input(INPUT_NAME, image)
        debug._run_per_layer()
        nodes = debug.debug_datum.get_graph_nodes()
        node_map = {
            node.get("attrs", {}).get("global_symbol"): index
            for index, node in enumerate(nodes)
            if node.get("attrs", {}).get("global_symbol")
        }
        for layer in compute.layers:
            index = node_map.get(layer.symbol)
            if index is None:
                raise RuntimeError(f"TSIM debug graph omitted VTA symbol {layer.symbol}")
            session.clear_and_validate(simulator_module)
            debug._execute_node(index)
            stats = session.read_stats(simulator=simulator_module)
            session.validate_activity(stats)
            measured[layer.symbol] = stats["cycle_count"]
    layers = _selected_layer_metrics(prepared, target, simulator, measured)
    result = SelectedDeploymentResult(
        target=target, simulator=simulator if use_vta else None,
        model_path=model_path, input_path=input_path,
        model_sha256=prepared.imported.model_sha256,
        input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        schedule=Path(schedule).expanduser().resolve() if use_vta and schedule else None,
        schedule_coverage=tuple(
            (layer.occurrence, layer.symbol, layer.occurrence in selected.selected)
            for layer in compute.layers
        ) if use_vta else (),
        predicted_class=int(np.argmax(scores, axis=1)[0]), scores=scores,
        layers=layers, whole_cycles=cycles, profiler_stats=profiler,
    )
    print(f"Predicted CIFAR-10 class: {result.predicted_class} ({LABELS[result.predicted_class]})")
    print("Raw output scores:", scores[0].tolist())
    return result


LABELS = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def write_deployment_report(result, report_path):
    """Write UTF-8 Markdown with arithmetic and unavailable data stated plainly."""
    report_path = Path(report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    config_hash = "N/A"
    if result.target.startswith("vta,"):
        config_path = Path(__import__("os").environ["VTA_CONFIG_FILE"]).expanduser().resolve()
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    output = [
        "# ResNet-8 deployment report", "", f"- Model: `{result.model_path}`",
        f"- Model SHA-256: `{result.model_sha256}`", f"- Input: `{result.input_path}`",
        f"- Input SHA-256: `{result.input_sha256}`", f"- Target: `{result.target}`",
        f"- Simulator: `{result.simulator or 'N/A (CPU target)'}`",
        f"- Config SHA-256: `{config_hash}`",
        f"- Schedule: `{result.schedule or 'default'}`",
        f"- Schedule coverage: {sum(bool(row[2]) for row in result.schedule_coverage)}/{len(result.schedule_coverage)} VTA layers selected",
        f"- Predicted CIFAR-10 class: {result.predicted_class} ({LABELS[result.predicted_class]})", "",
        "## Raw output scores", "", "```text",
        " ".join(f"{float(value):.8g}" for value in result.scores[0]), "```", "",
        "## Layer measurements", "",
        "| Layer | Device | Operation | Logical MACs | Cycles | Peak MAC/cycle | MAC utilization |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for layer in result.layers:
        cycles = f"{layer.cycles:,}" if layer.cycles is not None else "N/A"
        peak = f"{layer.peak_macs_per_cycle:,}" if layer.peak_macs_per_cycle is not None else "N/A"
        utilization = f"{100 * layer.utilization:.2f}%" if layer.utilization is not None else "N/A"
        output.append(
            f"| {layer.name} | {layer.device} | {layer.operation} | {layer.logical_macs:,} | {cycles} | {peak} | {utilization} |"
        )
    output.extend(["", "## Whole model", ""])
    if result.whole_cycles is None:
        reason = "CPU target" if result.simulator is None else "FSIM does not provide cycle counts"
        output.extend([
            f"- Whole-model cycles: N/A ({reason}).",
            "- Whole-model MAC utilization: N/A (cycle count is unavailable).",
        ])
    else:
        peak = next((row.peak_macs_per_cycle for row in result.layers if row.device == "vta"), None)
        total_macs = sum(row.logical_macs for row in result.layers if row.device == "vta")
        utilization = total_macs / (result.whole_cycles * peak) if peak else None
        layer_cycles = sum(row.cycles or 0 for row in result.layers if row.device == "vta")
        output.extend([
            f"- Whole-model TSIM cycles: {result.whole_cycles:,} (one counted invocation after an excluded warmup).",
            f"- VTA logical MACs: {total_macs:,}; divided by whole-model TSIM cycles.",
            f"- Whole-model VTA MAC utilization: {100 * utilization:.2f}%" if utilization is not None else "- Whole-model VTA MAC utilization: N/A.",
            f"- Sum of measured VTA layer cycles: {layer_cycles:,}; residual against whole-model cycles: {result.whole_cycles - layer_cycles:,}.",
        ])
    if result.profiler_stats:
        output.extend(["", "## Simulator counters", "", "```json", json.dumps(result.profiler_stats, indent=2, sort_keys=True), "```"])
    report_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return report_path
