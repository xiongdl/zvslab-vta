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
import vta
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.backend import te_compiler
from vta.relay import plan_devices_for_vta

from common.deployment import lower_selected_deployment, write_json_atomic
from common.deployment_compute import capture_deployment_compute
from common.schedule import load_schedule_snapshot

from graph_artifacts import export_graph_bundle
from model_pipeline import INPUT_DTYPE, INPUT_SHAPE, MODEL_SHA256, load_sample, prepare_model


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "vww_96_float.tflite"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build"
REFERENCE_ARTIFACT_STEM = "mlperf_vww_llvm"
MIXED_ARTIFACT_STEM = "mlperf_vww_vta"
INPUT_NAME = "input_1"
REQUIRED_PROFILER_COUNTERS = ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes")
TSIM_ACTIVITY_COUNTER = "cycle_count"
SUPPORTED_HOST_CODEGENS = ("llvm", "c")
DEFAULT_HOST_CODEGEN = "llvm"
MODEL_ID = "visual_wake_words_v1"


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


@contextmanager
def _selected_snapshot_lowering(compiler, deployment, snapshot, compiler_config):
    """Route normal VTA lowering through the validated actual-layer snapshot."""
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
        # Keep validation side-effect free: importing vta.testing.simulator
        # eagerly loads whichever backend is in VTA_BACKEND before checking
        # this session's requested simulator.
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
            activity_counter=TSIM_ACTIVITY_COUNTER,
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

    labels = tuple(sample.get("label") for sample in samples)
    if labels != (0, 0, 0, 0, 0, 1, 1, 1, 1, 1):
        raise RuntimeError(f"sample manifest labels must be ordered five non-person then five person, received {labels}")

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


def committed_sample_labels():
    """Return the fixed manifest labels in committed sample order."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 10:
        raise RuntimeError("sample manifest must contain exactly ten entries")
    labels = tuple(sample.get("label") for sample in samples)
    expected = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
    if labels != expected:
        raise RuntimeError(f"sample manifest labels must be ordered five non-person then five person, received {labels}")
    return labels


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
        return "mlperf_vww_llvm" if role == "reference" else "mlperf_vww_vta_llvm"
    return "mlperf_vww_c" if role == "reference" else "mlperf_vww_vta_c"


def _matrix_artifact_root(output_dir, host_codegen, simulator="fsim"):
    _validate_host_codegen(host_codegen)
    _simulator_session(simulator)
    return Path(output_dir) / f"{host_codegen}-{simulator}"


def _host_target(host_codegen=DEFAULT_HOST_CODEGEN):
    _validate_host_codegen(host_codegen)
    if host_codegen == "llvm":
        return tvm.target.Target(vta.get_env().target_host)
    return tvm.target.Target("c")


def build_host_artifacts(
    prepared, output_dir, host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim", schedule=None,
):
    """Build reference/mixed bundles using a validated actual-deployment schedule."""
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
    device_plan = plan_devices_for_vta(prepared.mixed_module, _host_target(host_codegen))
    schedule_path = None if schedule is None or str(schedule).lower() == "none" else schedule
    if not isinstance(prepared.mixed_module, tvm.IRModule):
        if schedule_path is not None:
            raise TypeError("schedule replay requires the prepared actual Relay deployment module")
        deployment = snapshot = None
    else:
        deployment = capture_deployment_compute(
            prepared.mixed_module, MODEL_ID,
            getattr(getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256),
        )
        snapshot = load_schedule_snapshot(schedule_path, deployment)
    compiler_config = vta.relay.transform.VTACompilerConfig.from_env(vta.get_env())
    compiler = te_compiler.get()
    if snapshot is not None and snapshot.selected:
        compiler.clear()
        try:
            with _selected_snapshot_lowering(compiler, deployment, snapshot, compiler_config):
                with vta.build_config():
                    mixed_factory = relay.build(device_plan.module, target=device_plan.targets)
        finally:
            compiler.clear()
    else:
        try:
            with vta.build_config():
                mixed_factory = relay.build(device_plan.module, target=device_plan.targets)
        finally:
            compiler.clear()

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

    selected = snapshot.selected if snapshot is not None else {}
    config_identities = tuple(
        (
            occurrence,
            hashlib.sha256(json.dumps(
                [config.to_json_dict() for config in row.configs if config is not None],
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        )
        for occurrence, row in sorted(selected.items())
    )
    coverage = (
        snapshot.coverage(deployment)
        if snapshot is not None
        else tuple((index, symbol, False) for index, symbol in enumerate(prepared.routing.symbols))
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
            device=(tvm.cpu(0), tvm.ext_dev(0)),
        ),
        vta_symbols=tuple(prepared.routing.symbols),
        schedule_coverage=coverage,
        schedule_config_identities=config_identities,
    )


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


def compare_outputs(sample_path, reference, mixed, expected_label=None):
    """Require bounded tensor and exact classification agreement for one sample."""
    if reference.shape != mixed.shape:
        raise RuntimeError(
            f"{sample_path.name} output shape differs: {reference.shape} != {mixed.shape}"
        )
    if reference.dtype != mixed.dtype:
        raise RuntimeError(
            f"{sample_path.name} output dtype differs: {reference.dtype} != {mixed.dtype}"
        )
    try:
        np.testing.assert_allclose(reference, mixed, rtol=1e-6, atol=1e-6)
    except AssertionError as error:
        raise RuntimeError(f"{sample_path.name} output is not within tolerance") from error

    reference_top1 = int(np.argmax(reference, axis=1)[0])
    mixed_top1 = int(np.argmax(mixed, axis=1)[0])
    if reference_top1 != mixed_top1:
        raise RuntimeError(
            f"{sample_path.name} top-1 differs: {reference_top1} != {mixed_top1}"
        )
    if expected_label is not None and reference_top1 != int(expected_label):
        raise RuntimeError(
            f"{sample_path.name} top-1 does not match manifest label: "
            f"{reference_top1} != {int(expected_label)}"
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
    expected_labels = committed_sample_labels()
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
    for (index, ((path, input_data), (_, reference_output))) in enumerate(
        zip(inputs, reference_outputs)
    ):
        mixed_output = _run_graph(artifacts.mixed, input_data)
        comparisons.append(
            compare_outputs(path, reference_output, mixed_output, expected_labels[index])
        )

    profiler_stats = session.read_stats(simulator=simulator)
    session.validate_activity(profiler_stats)
    return ExecutionSummary(comparisons=tuple(comparisons), profiler_stats=dict(profiler_stats))


def _execute_matrix(artifacts, sample_paths, simulator):
    """Run references first, then each mixed graph in independent windows."""
    sample_paths = tuple(Path(path) for path in sample_paths)
    expected_paths = committed_sample_paths()
    expected_labels = committed_sample_labels()
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
            for index, ((path, expected), (_, actual)) in enumerate(zip(baseline, outputs)):
                compare_outputs(path, expected, actual, expected_labels[index])
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
        for index, ((path, input_data), (_, reference_output)) in enumerate(
            zip(inputs, reference_outputs[host])
        ):
            try:
                mixed_output = _run_graph(host_artifacts.mixed, input_data)
                comparisons.append(
                    compare_outputs(path, reference_output, mixed_output, expected_labels[index])
                )
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


def deploy_fsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS,
                       schedule=None):
    """Build and execute the complete ordered LLVM/C FSIM matrix."""
    return _deploy_matrix(output_dir, host_codegens, "fsim", schedule)


def deploy_tsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS,
                       schedule=None):
    """Build and execute the complete ordered LLVM/C TSIM matrix."""
    return _deploy_matrix(output_dir, host_codegens, "tsim", schedule)


def deploy(output_dir=DEFAULT_OUTPUT_DIR, host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim",
           schedule=None):
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


def schedule_coverage_rows(artifacts):
    return [
        {"occurrence": occurrence, "symbol": symbol, "selected": selected}
        for occurrence, symbol, selected in artifacts.schedule_coverage
    ]


def _schedule_measurement_cycles(selected, layer):
    measurement = selected.measurement
    if (measurement.get("backend") != "tsim"
            or measurement.get("protocol") != "tsim_single_call_v1"
            or measurement.get("units") != "cycles"):
        raise ValueError(f"occurrence {layer.occurrence} requires one-call TSIM cycle provenance")
    results = measurement.get("results")
    configs = [
        config for (template, _, _, space), config in zip(layer.config_spaces, selected.configs)
        if template != "add.vta" and len(space) > 1
    ]
    if not isinstance(results, (list, tuple)) or len(results) != len(configs):
        raise ValueError(f"occurrence {layer.occurrence} measurement count is invalid")
    cycles = 0
    for config, result in zip(configs, results):
        costs = result.get("costs") if isinstance(result, dict) else None
        if config is None or not isinstance(costs, (list, tuple)) or len(costs) != 1:
            raise ValueError(f"occurrence {layer.occurrence} must have one measured cycle value")
        value = costs[0]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value <= 0:
            raise ValueError(f"occurrence {layer.occurrence} has invalid measured cycle value")
        cycles += int(value)
    if not configs:
        raise ValueError(f"occurrence {layer.occurrence} has no measured selected config")
    return cycles


def _validate_schedule_evidence(deployment, snapshot):
    if len(snapshot.selected) != len(deployment.layers):
        raise ValueError("schedule evidence requires complete occurrence coverage")
    for layer in deployment.layers:
        selected = snapshot.selected.get(layer.occurrence)
        if selected is None or not selected.measured:
            raise ValueError(
                f"schedule evidence requires measured config for occurrence {layer.occurrence}"
            )


def cycles_within_ten_percent(deployment_cycles, measured_cycles):
    """Preserve the VWW inclusive 10% deployment evidence gate."""
    for label, value in (("deployment", deployment_cycles), ("measured", measured_cycles)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} cycle count must be a positive integer")
    return 10 * abs(deployment_cycles - measured_cycles) <= measured_cycles


class _EvidenceProfileSession:
    def __init__(self, session):
        self.session = session

    def clear_and_validate(self, simulator):
        return self.session.clear_and_validate(simulator)

    def read_stats(self, simulator):
        return self.session.read_stats(simulator=simulator)

    def validate_activity(self, stats):
        return self.session.validate_activity(stats)


def write_deployment_report(result, report_path, *, schedule=None,
                            validate_schedule_evidence=False):
    """Write ten HOST-checked outputs and optional graph-resident TSIM evidence."""
    schedule_path = None if schedule is None or str(schedule).lower() == "none" else Path(schedule)
    deployment = capture_deployment_compute(
        result.prepared.mixed_module, MODEL_ID,
        getattr(getattr(result.prepared, "imported", None), "model_sha256", MODEL_SHA256),
    )
    from common.schedule import _geometry_identity

    config_identities = dict(result.artifacts.schedule_config_identities)
    report = {
        "schema_version": 1,
        "artifact_kind": "vta_deployment_profile_v1",
        "model": MODEL_ID,
        "model_sha256": getattr(getattr(result.prepared, "imported", None), "model_sha256", MODEL_SHA256),
        "geometry_sha256": _geometry_identity(deployment),
        "simulator": result.artifacts.simulator,
        "host_codegen": result.artifacts.host_codegen,
        "schedule": str(schedule_path.resolve()) if schedule_path else None,
        "schedule_log_sha256": (
            hashlib.sha256(schedule_path.read_bytes()).hexdigest() if schedule_path else None
        ),
        "schedule_coverage": schedule_coverage_rows(result.artifacts),
        "selected_config_identities": [
            {"occurrence": occurrence, "sha256": config_identities[occurrence]}
            for occurrence in sorted(config_identities)
        ],
        "sample_count": len(result.execution.comparisons),
        "outputs_passed": sum(item.mixed is not None for item in result.execution.comparisons),
        "samples": [
            {"filename": item.sample_path.name, "reference_top1": item.top1,
             "mixed_top1": None if item.mixed is None else int(item.mixed.argmax(axis=1)[0])}
            for item in result.execution.comparisons
        ],
        "profiler_stats": result.execution.profiler_stats,
        "status": "passed",
    }
    if validate_schedule_evidence:
        if result.artifacts.simulator != "tsim":
            raise ValueError("schedule evidence validation requires --simulator tsim")
        if schedule_path is None:
            raise ValueError("schedule evidence validation requires a measured schedule snapshot")
        snapshot = load_schedule_snapshot(schedule_path, deployment)
        _validate_schedule_evidence(deployment, snapshot)
        if (len(result.execution.comparisons) != len(committed_sample_paths())
                or report["outputs_passed"] != len(committed_sample_paths())):
            raise RuntimeError("schedule evidence requires HOST-checked outputs for all ten committed samples")

        from tvm.contrib.debugger import debug_executor
        from mlperf_tiny_benchmark.deployment_evidence import profile_graph_resident_nodes

        sample_path = committed_sample_paths()[0]
        sample_input = load_sample(sample_path)
        session, simulator = _load_simulator("tsim")
        graph = debug_executor.create(
            result.artifacts.mixed.graph_json,
            result.artifacts.mixed.module,
            result.artifacts.mixed.device,
        )
        graph.load_params(result.artifacts.mixed.params)
        graph.set_input(INPUT_NAME, sample_input)
        session.clear_and_validate(simulator)
        graph._run_per_layer()
        performance_stats = session.read_stats(simulator=simulator)
        session.validate_activity(performance_stats)
        rows = profile_graph_resident_nodes(
            result.artifacts.mixed.graph_json,
            [{"occurrence": layer.occurrence, "symbol": layer.symbol}
             for layer in deployment.layers],
            graph, _EvidenceProfileSession(session), simulator,
        )
        occurrences = []
        for layer, row in zip(deployment.layers, rows):
            measured = _schedule_measurement_cycles(snapshot.selected[layer.occurrence], layer)
            deployed = row["deployment_cycles"]
            passed = cycles_within_ten_percent(deployed, measured)
            occurrences.append({
                "occurrence": layer.occurrence, "symbol": layer.symbol,
                "deployment_cycles": deployed, "autotvm_cycles": measured,
                "relative_cycle_difference": abs(deployed - measured) / measured,
                "graph_node": row["graph_node_name"], "counted_invocations": 1,
                "passed": passed,
            })
            if not passed:
                report.update({"status": "failed", "occurrences": occurrences})
                write_json_atomic(Path(report_path).with_suffix(".failure.json"), report)
                raise ValueError(
                    f"deployment versus selected schedule cycles exceeds 10% at occurrence "
                    f"{layer.occurrence}: deployment={deployed}, measured={measured}"
                )
        report.update({
            "measurement_protocol": "tsim_single_call_v1",
            "performance_sample": sample_path.name,
            "performance_sample_count": 1,
            "performance_stats": performance_stats,
            "occurrences": occurrences,
        })
    write_json_atomic(report_path, report)
    return report
