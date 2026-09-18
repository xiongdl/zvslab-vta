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
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm
import vta
from tvm import relay
from tvm.contrib import graph_executor
from vta.relay import plan_devices_for_vta

from graph_artifacts import export_graph_bundle
from model_pipeline import MODEL_SHA256, load_sample, prepare_model


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
        active_target = getattr(vta.get_env(), "TARGET", None)
        if active_target != self.environment_target:
            raise RuntimeError(
                f"simulator {self.label!r} requires VTA target "
                f"{self.environment_target!r}, active target is {active_target!r}"
            )
        return self

    def load(self):
        # Importing this standard module is deliberately the only simulator
        # initialization path.  In particular, TSIM's import performs the
        # global driver load, hardware-module load, and vta.tsim.init call.
        try:
            from vta.testing import simulator
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
            environment_target="sim",
            clear_registry="vta.simulator.profiler_clear",
            status_registry="vta.simulator.profiler_status",
            required_registries=(
                "vta.simulator.profiler_clear",
                "vta.simulator.profiler_status",
            ),
            activity_counter="gemm_counter",
            diagnostic="bash scripts/build_vta_lib.sh --target libvta_fsim",
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
            diagnostic="bash scripts/build_vta_lib.sh --target libvta_hw",
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


def _active_vta_target(host_target):
    """Activate the compiler extension while building canonical ext_dev targets."""
    return tvm.target.Target("vta", host=host_target)


def build_host_artifacts(prepared, output_dir, host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim"):
    """Build, export, and reload both standard host libraries without simulator loading."""
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
    # VTA's lower-pass bundle includes CPUAccessRewrite, which is correct for
    # an all-ext_dev VTA module but would rewrite ordinary CPU host functions
    # in this explicit CPU+ext_dev graph.  Keep C host lowering native while
    # still disabling unsupported vectorized C codegen.
    build_config = (
        tvm.transform.PassContext(config={"tir.disable_vectorize": True})
        if host_codegen == "c"
        else vta.build_config()
    )
    with _active_vta_target(device_plan.targets[0]), build_config:
        mixed_factory = relay.build(device_plan.module, target=device_plan.targets)

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
            device=(tvm.cpu(0), tvm.ext_dev(0)),
        ),
        vta_symbols=tuple(prepared.routing.symbols),
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


def _deploy_matrix(output_dir, host_codegens, simulator):
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
        )
        for host_codegen in host_codegens
    )
    executions = _execute_matrix(artifacts, committed_sample_paths(), simulator)
    return SimulationMatrixResult(
        simulator=session.label, prepared=prepared, artifacts=artifacts, executions=executions
    )


def deploy_fsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS):
    """Build and execute the complete ordered LLVM/C FSIM matrix."""
    return _deploy_matrix(output_dir, host_codegens, "fsim")


def deploy_tsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS):
    """Build and execute the complete ordered LLVM/C TSIM matrix."""
    return _deploy_matrix(output_dir, host_codegens, "tsim")


def deploy(output_dir=DEFAULT_OUTPUT_DIR, host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim"):
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
        prepared, output_dir, host_codegen=host_codegen, simulator=simulator
    )
    if simulator == "fsim":
        execution = execute_samples(artifacts, committed_sample_paths())
    else:
        execution = _execute_matrix((artifacts,), committed_sample_paths(), simulator)[0]
    return DeploymentResult(prepared=prepared, artifacts=artifacts, execution=execution)
