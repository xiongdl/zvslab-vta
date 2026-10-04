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

"""Build, reload, and execute the fixed MLPerf Tiny HOST deployment."""

import json
import hashlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend import te_compiler
from tvm.contrib import graph_executor

from graph_artifacts import export_graph_bundle
from model_pipeline import MODEL_SHA256, load_sample, prepare_model
from deployment_compute import capture_deployment_compute
from deployment import lower_selected_deployment
from schedule import load_schedule_snapshot


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet.tflite"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build"
MODEL_ID = "image_classification_v1"
INPUT_NAME = "input_1"
REQUIRED_PROFILER_COUNTERS = ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes")
SUPPORTED_HOST_CODEGENS = ("llvm", "c")
DEFAULT_HOST_CODEGEN = "llvm"


@dataclass(frozen=True)
class ReloadedArtifact:
    """One exported and reloaded Graph Executor library."""

    path: Path
    artifact_dir: Path
    graph_json: str
    params: bytes
    module: tvm.runtime.Module
    device: tvm.runtime.Device


@dataclass(frozen=True)
class HostArtifacts:
    """The reference and mixed artifacts for one host code generator."""

    host_codegen: str
    simulator: str
    reference: ReloadedArtifact
    mixed: ReloadedArtifact
    vta_symbols: tuple
    schedule_coverage: tuple = ()
    schedule_config_identities: tuple = ()


@dataclass(frozen=True)
class OutputComparison:
    """Exact result agreement for one committed sample."""

    sample_path: Path
    reference: np.ndarray
    mixed: np.ndarray
    top1: int


@dataclass(frozen=True)
class ExecutionSummary:
    """Ten output comparisons and the resulting accelerator activity."""

    comparisons: tuple
    profiler_stats: dict


@dataclass(frozen=True)
class DeploymentResult:
    """All observable outputs of the fixed deployment flow."""

    prepared: object
    artifacts: HostArtifacts
    execution: ExecutionSummary


@dataclass(frozen=True)
class SimulationMatrixResult:
    """All host variants from one prepared simulator deployment matrix."""

    simulator: str
    prepared: object
    artifacts: tuple
    executions: tuple


# Kept as an import-compatible alias for callers of the original FSIM API.
FsimMatrixResult = SimulationMatrixResult


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
        validate_profiler_stats(stats)


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


def shared_library_suffix():
    """Return the deterministic host DSO suffix."""
    return ".dylib" if sys.platform == "darwin" else ".so"


def committed_sample_paths():
    """Read and validate the fixed ten-sample order from committed metadata."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 10:
        raise RuntimeError("sample manifest must contain exactly ten entries")

    labels = tuple(sample.get("numeric_label") for sample in samples)
    if labels != tuple(range(10)):
        raise RuntimeError(f"sample manifest labels must be ordered 0 through 9, received {labels}")

    paths = []
    for sample in samples:
        filename = sample.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RuntimeError(f"sample manifest contains an invalid filename: {filename!r}")
        path = MANIFEST_PATH.parent / filename
        if not path.is_file():
            raise RuntimeError(f"committed sample is missing: {path}")
        paths.append(path)
    if len(set(paths)) != 10:
        raise RuntimeError("sample manifest filenames must be unique")
    return tuple(paths)


def _validate_host_codegen(host_codegen):
    if host_codegen not in SUPPORTED_HOST_CODEGENS:
        raise ValueError(
            f"unsupported host codegen {host_codegen!r}; supported kinds are llvm and c"
        )
    return host_codegen


def _validate_host_codegens(host_codegens):
    try:
        values = tuple(host_codegens)
    except TypeError as error:
        raise ValueError(
            "host_codegens must be exactly ('llvm', 'c') in that order"
        ) from error
    if values != SUPPORTED_HOST_CODEGENS:
        raise ValueError(
            "host_codegens must be exactly ('llvm', 'c') in that order"
        )
    return values


def _artifact_identity(host_codegen, role):
    _validate_host_codegen(host_codegen)
    if role not in {"reference", "mixed"}:
        raise ValueError(f"unsupported artifact role {role!r}")
    if host_codegen == "llvm":
        return "mlperf_resnet_llvm" if role == "reference" else "mlperf_resnet_vta_llvm"
    return "mlperf_resnet_c" if role == "reference" else "mlperf_resnet_vta_c"


def _matrix_artifact_root(output_dir, host_codegen, simulator="fsim"):
    _validate_host_codegen(host_codegen)
    _simulator_session(simulator)
    return Path(output_dir) / f"{host_codegen}-{simulator}"


def _mixed_target(host_codegen=DEFAULT_HOST_CODEGEN):
    import vta

    _validate_host_codegen(host_codegen)
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


def build_host_artifacts(
    prepared,
    output_dir,
    host_codegen=DEFAULT_HOST_CODEGEN,
    simulator="fsim",
    *,
    schedule=None,
):
    """Build, export, and reload both standard host libraries without simulator loading."""
    import vta

    _validate_host_codegen(host_codegen)
    _simulator_session(simulator).validate_environment()
    if prepared.reference_module is not prepared.quantized_module:
        raise RuntimeError("pure LLVM build must use the exact shared quantized module object")

    output_dir = Path(output_dir)
    if host_codegen == "llvm":
        reference_factory = relay.build(prepared.reference_module, target="llvm")
    else:
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            reference_factory = relay.build(
                prepared.reference_module, target=tvm.target.Target("c")
            )
    schedule_path = None if schedule is None or str(schedule).lower() == "none" else schedule
    model_sha256 = getattr(getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256)
    compute = None
    snapshot = None
    compiler_config = None
    if isinstance(prepared.mixed_module, tvm.IRModule):
        compute = capture_deployment_compute(prepared.mixed_module, MODEL_ID, model_sha256)
        snapshot = load_schedule_snapshot(schedule_path, compute)
        compiler_config = vta.relay.transform.VTACompilerConfig.from_env(vta.get_env())
    elif schedule_path is not None:
        raise TypeError("schedule replay requires the prepared actual Relay deployment module")
    compiler = te_compiler.get()
    replay_enabled = bool(snapshot and snapshot.selected)
    if replay_enabled:
        compiler.clear()
        try:
            with _selected_snapshot_lowering(compiler, compute, snapshot, compiler_config):
                mixed_factory = _build_mixed_factory(prepared, host_codegen)
        finally:
            compiler.clear()
    else:
        mixed_factory = _build_mixed_factory(prepared, host_codegen)

    reference_identity = _artifact_identity(host_codegen, "reference")
    mixed_identity = _artifact_identity(host_codegen, "mixed")

    reference_bundle = export_graph_bundle(
        reference_factory,
        output_dir,
        "reference",
        artifact_name=reference_identity,
        artifact_role="reference",
        model_sha256=getattr(getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256),
        host_codegen=host_codegen,
        simulator=simulator,
        forbidden_vta_symbols=prepared.routing.symbols,
    )
    mixed_bundle = export_graph_bundle(
        mixed_factory,
        output_dir,
        "mixed",
        artifact_name=mixed_identity,
        artifact_role="mixed",
        model_sha256=getattr(getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256),
        host_codegen=host_codegen,
        simulator=simulator,
        expected_vta_symbols=prepared.routing.symbols,
    )

    return HostArtifacts(
        host_codegen=host_codegen,
        simulator=simulator,
        reference=ReloadedArtifact(
            path=reference_bundle.library_path,
            artifact_dir=reference_bundle.artifact_dir,
            graph_json=reference_bundle.graph_json,
            params=reference_bundle.params,
            module=reference_bundle.module,
            device=tvm.cpu(0),
        ),
        mixed=ReloadedArtifact(
            path=mixed_bundle.library_path,
            artifact_dir=mixed_bundle.artifact_dir,
            graph_json=mixed_bundle.graph_json,
            params=mixed_bundle.params,
            module=mixed_bundle.module,
            device=tvm.ext_dev(0),
        ),
        vta_symbols=tuple(prepared.routing.symbols),
        schedule_coverage=tuple(
            (layer.occurrence, layer.symbol, layer.occurrence in snapshot.selected)
            for layer in compute.layers
        ) if compute is not None else tuple(
            (occurrence, symbol, False)
            for occurrence, symbol in enumerate(prepared.routing.symbols)
        ),
        schedule_config_identities=tuple(
            (
                occurrence,
                hashlib.sha256(json.dumps(
                    [config.to_json_dict() for config in selected.configs if config is not None],
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
            )
            for occurrence, selected in sorted(snapshot.selected.items())
        ) if snapshot is not None else (),
    )


def _build_mixed_factory(prepared, host_codegen):
    import vta

    if host_codegen == "c":
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            with vta.build_config(config={"tir.disable_vectorize": True}):
                return relay.build(
                    prepared.mixed_module,
                    target=_mixed_target(
                        *(host_codegen,) if host_codegen != DEFAULT_HOST_CODEGEN else ()
                    ),
                )
    with vta.build_config():
        return relay.build(prepared.mixed_module, target=_mixed_target())


def validate_mixed_symbols(module, expected_symbols):
    """Require every routed function in the reloaded mixed artifact."""
    for symbol in expected_symbols:
        if not module.implements_function(symbol, True):
            raise RuntimeError(f"reloaded mixed artifact is missing VTA symbol {symbol}")


def _run_graph(artifact, input_data):
    runtime = graph_executor.create(artifact.graph_json, artifact.module, artifact.device)
    runtime.load_params(artifact.params)
    runtime.set_input(INPUT_NAME, input_data)
    runtime.run()
    return runtime.get_output(0).numpy()


def compare_outputs(sample_path, reference, mixed):
    """Require exact tensor and classification agreement for one sample."""
    if reference.shape != mixed.shape:
        raise RuntimeError(
            f"{sample_path.name} output shape differs: {reference.shape} != {mixed.shape}"
        )
    if reference.dtype != mixed.dtype:
        raise RuntimeError(
            f"{sample_path.name} output dtype differs: {reference.dtype} != {mixed.dtype}"
        )
    if not np.array_equal(reference, mixed):
        raise RuntimeError(f"{sample_path.name} output is not elementwise equal")

    reference_top1 = int(np.argmax(reference, axis=1)[0])
    mixed_top1 = int(np.argmax(mixed, axis=1)[0])
    if reference_top1 != mixed_top1:
        raise RuntimeError(
            f"{sample_path.name} top-1 differs: {reference_top1} != {mixed_top1}"
        )
    return OutputComparison(
        sample_path=Path(sample_path),
        reference=reference,
        mixed=mixed,
        top1=reference_top1,
    )


def validate_profiler_stats(stats):
    """Require GEMM, weight-load, and output-store activity."""
    for counter in REQUIRED_PROFILER_COUNTERS:
        if stats.get(counter, 0) <= 0:
            raise RuntimeError(f"FSIM profiler counter {counter} must be positive")


def _load_fsim():
    return _simulator_session("fsim").load()


def _load_simulator(simulator):
    session = _simulator_session(simulator)
    session.validate_environment()
    return session, session.load()


def execute_samples(artifacts, sample_paths):
    """Run all pure outputs before loading FSIM and executing the mixed graph."""
    sample_paths = tuple(Path(path) for path in sample_paths)
    expected_paths = committed_sample_paths()
    if sample_paths != expected_paths:
        raise RuntimeError("execution must use the ten committed samples in manifest order")

    inputs = tuple((path, load_sample(path)) for path in sample_paths)
    reference_outputs = tuple(
        (path, _run_graph(artifacts.reference, input_data)) for path, input_data in inputs
    )

    validate_mixed_symbols(artifacts.mixed.module, artifacts.vta_symbols)
    session = _simulator_session("fsim")
    simulator = _load_fsim()
    session.validate_environment()
    cleared_stats = session.clear_and_validate(simulator)

    comparisons = []
    for (path, input_data), (_, reference_output) in zip(inputs, reference_outputs):
        mixed_output = _run_graph(artifacts.mixed, input_data)
        comparisons.append(compare_outputs(path, reference_output, mixed_output))

    profiler_stats = session.read_stats(simulator=simulator)
    session.validate_activity(profiler_stats)
    return ExecutionSummary(comparisons=tuple(comparisons), profiler_stats=dict(profiler_stats))


def _execute_matrix(artifacts, sample_paths, simulator):
    """Run references first, then each mixed graph in independent windows."""
    sample_paths = tuple(Path(path) for path in sample_paths)
    expected_paths = committed_sample_paths()
    if sample_paths != expected_paths:
        raise RuntimeError("execution must use the ten committed samples in manifest order")

    # Inputs are intentionally decoded once and shared by every host variant.
    inputs = tuple((path, load_sample(path)) for path in sample_paths)
    reference_outputs = {}
    baseline = None
    for host_artifacts in artifacts:
        host = host_artifacts.host_codegen
        outputs = tuple(
            (path, _run_graph(host_artifacts.reference, input_data))
            for path, input_data in inputs
        )
        if baseline is None:
            baseline = outputs
        else:
            for (path, expected), (_, actual) in zip(baseline, outputs):
                compare_outputs(path, expected, actual)
        reference_outputs[host] = outputs

    session, simulator_module = _load_simulator(simulator)
    executions = []
    for host_artifacts in artifacts:
        host = host_artifacts.host_codegen
        validate_mixed_symbols(host_artifacts.mixed.module, host_artifacts.vta_symbols)
        try:
            cleared_stats = session.clear_and_validate(simulator_module)
        except RuntimeError as error:
            raise RuntimeError(f"{host} {simulator.upper()} profiler reset failed: {error}") from error

        comparisons = []
        for (path, input_data), (_, reference_output) in zip(
            inputs, reference_outputs[host]
        ):
            try:
                mixed_output = _run_graph(host_artifacts.mixed, input_data)
                comparisons.append(compare_outputs(path, reference_output, mixed_output))
            except Exception as error:
                raise RuntimeError(
                    f"{host} mixed {simulator.upper()} failed for sample {path.name}: {error}"
                ) from error

        profiler_stats = session.read_stats(simulator=simulator_module)
        try:
            session.validate_activity(profiler_stats)
        except RuntimeError as error:
            raise RuntimeError(
                f"{host} mixed {simulator.upper()} profiler validation failed: {error}"
            ) from error
        executions.append(
            ExecutionSummary(comparisons=tuple(comparisons), profiler_stats=dict(profiler_stats))
        )
    return tuple(executions)


def _execute_fsim_matrix(artifacts, sample_paths):
    """Compatibility wrapper for the existing FSIM matrix tests and callers."""
    return _execute_matrix(artifacts, sample_paths, "fsim")


def _deploy_matrix(output_dir, host_codegens, simulator, schedule=None):
    session = _simulator_session(simulator)
    session.validate_environment()
    host_codegens = _validate_host_codegens(host_codegens)
    prepared = prepare_model(MODEL_PATH)
    print(f"VTA partitions: {len(prepared.routing.symbols)}")
    print(f"VTA symbols: {', '.join(prepared.routing.symbols)}")
    print(f"VTA composites: {', '.join(prepared.routing.composite_names)}")
    print(f"Host operators: {', '.join(prepared.routing.host_operator_names)}")
    artifacts = tuple(
        build_host_artifacts(
            prepared,
            _matrix_artifact_root(output_dir, host_codegen, simulator),
            host_codegen=host_codegen,
            simulator=simulator,
            schedule=schedule,
        )
        for host_codegen in host_codegens
    )
    executions = _execute_matrix(artifacts, committed_sample_paths(), simulator)
    return SimulationMatrixResult(
        simulator=session.label, prepared=prepared, artifacts=artifacts, executions=executions
    )


def deploy_fsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS, schedule=None):
    """Build and execute the complete ordered LLVM/C FSIM matrix."""
    return _deploy_matrix(output_dir, host_codegens, "fsim", schedule)


def deploy_tsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS, schedule=None):
    """Build and execute the complete ordered LLVM/C TSIM matrix."""
    return _deploy_matrix(output_dir, host_codegens, "tsim", schedule)


def deploy(output_dir=DEFAULT_OUTPUT_DIR, host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim", schedule=None):
    """Perform the complete fixed HOST deployment and return its evidence."""
    _validate_host_codegen(host_codegen)
    session = _simulator_session(simulator)
    session.validate_environment()
    prepared = prepare_model(MODEL_PATH)
    print(f"VTA partitions: {len(prepared.routing.symbols)}")
    print(f"VTA symbols: {', '.join(prepared.routing.symbols)}")
    print(f"VTA composites: {', '.join(prepared.routing.composite_names)}")
    print(f"Host operators: {', '.join(prepared.routing.host_operator_names)}")
    artifacts = build_host_artifacts(
        prepared, output_dir, host_codegen=host_codegen, simulator=simulator,
        schedule=schedule,
    )
    if simulator == "fsim":
        execution = execute_samples(artifacts, committed_sample_paths())
    else:
        execution = _execute_matrix((artifacts,), committed_sample_paths(), simulator)[0]
    return DeploymentResult(prepared=prepared, artifacts=artifacts, execution=execution)


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


def run_selected(target="vta,llvm", simulator="fsim", schedule=None,
                 output_dir=DEFAULT_OUTPUT_DIR, model_path=MODEL_PATH,
                 input_path=None):
    """Compile and run exactly one target using one image."""
    if target not in {"c", "llvm", "vta,c", "vta,llvm"}:
        raise ValueError(f"unsupported target {target!r}")
    if simulator not in {"fsim", "tsim"}:
        raise ValueError(f"unsupported simulator {simulator!r}")
    use_vta = target.startswith("vta,")
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


def __getattr__(name):
    # Older focused tests and tuning helpers address runtime.vta explicitly.
    # Import it only on demand so CPU CLI startup stays simulator-independent.
    if name == "vta":
        import vta

        return vta
    raise AttributeError(name)


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
