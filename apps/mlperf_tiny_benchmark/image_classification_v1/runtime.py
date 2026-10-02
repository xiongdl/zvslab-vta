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
import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm
import vta
from tvm import relay
from tvm.relay.backend import te_compiler
from tvm.contrib import graph_executor

from graph_artifacts import export_graph_bundle
from model_pipeline import MODEL_SHA256, load_sample, prepare_model
from common.deployment_compute import capture_deployment_compute
from common.deployment import (
    cycles_within_strict_ten_percent,
    lower_selected_deployment,
    validate_occurrence_rows,
    write_json_atomic,
)
from common.schedule import load_schedule_snapshot


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet.tflite"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build"
REFERENCE_ARTIFACT_STEM = "mlperf_resnet_llvm"
MIXED_ARTIFACT_STEM = "mlperf_resnet_vta"
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


@dataclass(frozen=True)
class AutotvmComparisonResult:
    """Baseline and history-best V1 builds executed on the same ten samples."""

    simulator: str
    prepared: object
    baseline_artifacts: HostArtifacts
    tuned_artifacts: HostArtifacts
    baseline_execution: ExecutionSummary
    tuned_execution: ExecutionSummary


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
    _validate_host_codegen(host_codegen)
    environment = vta.get_env()
    host = environment.target_host if host_codegen == "llvm" else tvm.target.Target("c")
    return tvm.target.Target("vta", host=host)


def _history_best(log_path, sidecar_path, prepared, simulator):
    tuner_path = APP_ROOT.parent / "autotvm_tuner.py"
    spec = importlib.util.spec_from_file_location("mlperf_tiny_autotvm_tuner", tuner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared MLPerf Tiny AutoTVM helper: {tuner_path}")
    tuner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tuner
    spec.loader.exec_module(tuner)

    return tuner.history_best(
        log_path,
        sidecar_path,
        model_id="image_classification_v1",
        model_sha256=getattr(
            getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256
        ),
        backend=simulator,
        config_path=tuner.DEFAULT_CONFIG_PATH,
    )


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
    autotvm_log=None,
    autotvm_sidecar=None,
):
    """Build, export, and reload both standard host libraries without simulator loading."""
    _validate_host_codegen(host_codegen)
    _simulator_session(simulator).validate_environment()
    if schedule is not None and (autotvm_log is not None or autotvm_sidecar is not None):
        raise ValueError("use --schedule or the legacy internal AutoTVM replay arguments")
    if (autotvm_log is None) != (autotvm_sidecar is None):
        raise ValueError("AutoTVM replay requires both --autotvm-log and --autotvm-sidecar")
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
    history_context = (
        _history_best(autotvm_log, autotvm_sidecar, prepared, simulator)
        if autotvm_log is not None
        else nullcontext()
    )
    compiler = te_compiler.get()
    replay_enabled = bool(snapshot and snapshot.selected)
    if autotvm_log is not None:
        with history_context:
            # Relay's TECompiler cache does not include AutoTVM history-best in
            # its key. Clear a previous untuned lowering before replaying a log.
            compiler.clear()
            try:
                mixed_factory = _build_mixed_factory(prepared, host_codegen)
            finally:
                compiler.clear()
    elif replay_enabled:
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


def write_deployment_report(
    result, report_path, *, schedule=None, validate_schedule_evidence=False
):
    """Save V1 sample, output, coverage and optional measured TSIM evidence."""
    if validate_schedule_evidence and result.artifacts.simulator != "tsim":
        raise ValueError("schedule evidence validation requires --simulator tsim")
    schedule_path = None if schedule is None or str(schedule).lower() == "none" else Path(schedule)
    coverage = getattr(result.artifacts, "schedule_coverage", ())
    config_identities = dict(getattr(result.artifacts, "schedule_config_identities", ()))
    report = {
        "schema_version": 1,
        "model": MODEL_ID,
        "model_sha256": getattr(
            getattr(result.prepared, "imported", None), "model_sha256", MODEL_SHA256
        ),
        "simulator": result.artifacts.simulator,
        "host_codegen": result.artifacts.host_codegen,
        "schedule": str(schedule_path.resolve()) if schedule_path else None,
        "schedule_coverage": [
            {"occurrence": occurrence, "symbol": symbol, "selected": selected}
            for occurrence, symbol, selected in coverage
        ],
        "selected_config_identities": [
            {"occurrence": occurrence, "sha256": config_identities[occurrence]}
            for occurrence in sorted(config_identities)
        ],
        "sample_count": len(result.execution.comparisons),
        "outputs_passed": len(result.execution.comparisons),
        "profiler_stats": result.execution.profiler_stats,
        "status": "passed",
    }
    if validate_schedule_evidence:
        if schedule_path is None:
            raise ValueError("schedule evidence validation requires a measured schedule snapshot")
        deployment = capture_deployment_compute(
            result.prepared.mixed_module, MODEL_ID, report["model_sha256"]
        )
        snapshot = load_schedule_snapshot(schedule_path, deployment)
        if len(snapshot.selected) != len(deployment.layers):
            raise ValueError("schedule evidence requires complete occurrence coverage")
        if len(result.execution.comparisons) != 10:
            raise RuntimeError("schedule evidence requires all ten committed output comparisons")

        from tvm.contrib.debugger import debug_executor

        session, simulator = _load_simulator("tsim")
        debug_graph = debug_executor.create(
            result.artifacts.mixed.graph_json,
            result.artifacts.mixed.module,
            result.artifacts.mixed.device,
        )
        debug_graph.load_params(result.artifacts.mixed.params)
        first = result.execution.comparisons[0]
        debug_graph.set_input(INPUT_NAME, load_sample(first.sample_path))
        session.clear_and_validate(simulator)
        debug_graph._run_per_layer()
        full_stats = session.read_stats(simulator=simulator)
        session.validate_activity(full_stats)
        compare_outputs(first.sample_path, first.reference, debug_graph.get_output(0).numpy())
        graph_nodes = debug_graph.debug_datum.get_graph_nodes()
        node_by_symbol = {
            node.get("attrs", {}).get("global_symbol"): index
            for index, node in enumerate(graph_nodes)
            if node.get("attrs", {}).get("global_symbol")
        }
        rows = []
        for layer in deployment.layers:
            selected = snapshot.selected.get(layer.occurrence)
            if selected is None or not selected.measured:
                raise ValueError(f"occurrence {layer.occurrence} lacks measured schedule evidence")
            measurement = selected.measurement
            if (measurement.get("backend") != "tsim"
                    or measurement.get("protocol") != "tsim_single_call_v1"
                    or measurement.get("units") != "cycles"):
                raise ValueError(
                    f"occurrence {layer.occurrence} requires tsim_single_call_v1 cycle provenance"
                )
            costs = [cost for item in measurement["results"] for cost in item["costs"]]
            if len(costs) != 1 or isinstance(costs[0], bool) or int(costs[0]) != costs[0]:
                raise ValueError(f"occurrence {layer.occurrence} has invalid TSIM cycle evidence")
            node_index = node_by_symbol.get(layer.symbol)
            if node_index is None:
                raise ValueError(f"debug deployment graph omitted VTA symbol {layer.symbol}")
            session.clear_and_validate(simulator)
            debug_graph._execute_node(node_index)
            stats = session.read_stats(simulator=simulator)
            session.validate_activity(stats)
            deployed_cycles, measured_cycles = stats["cycle_count"], int(costs[0])
            rows.append({
                "occurrence": layer.occurrence,
                "symbol": layer.symbol,
                "deployment_cycles": deployed_cycles,
                "autotvm_cycles": measured_cycles,
                "difference_percent": 100.0 * abs(deployed_cycles - measured_cycles) / measured_cycles,
                "passed": cycles_within_strict_ten_percent(deployed_cycles, measured_cycles),
                "graph_node": graph_nodes[node_index].get("name"),
            })
        try:
            validate_occurrence_rows(rows, [
                {"occurrence": layer.occurrence, "symbol": layer.symbol}
                for layer in deployment.layers
            ])
        except ValueError:
            report["status"] = "failed"
            report["occurrences"] = rows
            report["performance_sample_count"] = 1
            report["performance_sample"] = first.sample_path.name
            report["performance_stats"] = full_stats
            write_json_atomic(Path(report_path).with_suffix(".failure.json"), report)
            raise
        report.update({
            "geometry_sha256": snapshot.geometry_sha256,
            "schedule_log_sha256": hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
            "measurement_protocol": "tsim_single_call_v1",
            "performance_sample_count": 1,
            "performance_sample": first.sample_path.name,
            "performance_stats": full_stats,
            "occurrences": rows,
        })
    write_json_atomic(report_path, report)
    return report


def deploy_autotvm_comparison(
    log_path,
    sidecar_path,
    output_dir=DEFAULT_OUTPUT_DIR / "autotvm-comparison",
    host_codegen=DEFAULT_HOST_CODEGEN,
    simulator="fsim",
):
    """Build baseline and history-best artifacts, then compare all committed samples."""
    _validate_host_codegen(host_codegen)
    session = _simulator_session(simulator)
    session.validate_environment()
    prepared = prepare_model(MODEL_PATH)
    print(f"VTA partitions: {len(prepared.routing.symbols)}")
    baseline_artifacts = build_host_artifacts(
        prepared,
        Path(output_dir) / "baseline",
        host_codegen=host_codegen,
        simulator=simulator,
    )
    tuned_artifacts = build_host_artifacts(
        prepared,
        Path(output_dir) / "tuned",
        host_codegen=host_codegen,
        simulator=simulator,
        autotvm_log=log_path,
        autotvm_sidecar=sidecar_path,
    )
    baseline_execution, tuned_execution = _execute_matrix(
        (baseline_artifacts, tuned_artifacts), committed_sample_paths(), simulator
    )
    if simulator == "tsim":
        baseline_cycles = baseline_execution.profiler_stats["cycle_count"]
        tuned_cycles = tuned_execution.profiler_stats["cycle_count"]
        if tuned_cycles >= baseline_cycles:
            raise RuntimeError(
                "AutoTVM history-best did not lower TSIM cycle_count: "
                f"baseline={baseline_cycles}, tuned={tuned_cycles}"
            )
    return AutotvmComparisonResult(
        simulator=simulator,
        prepared=prepared,
        baseline_artifacts=baseline_artifacts,
        tuned_artifacts=tuned_artifacts,
        baseline_execution=baseline_execution,
        tuned_execution=tuned_execution,
    )
