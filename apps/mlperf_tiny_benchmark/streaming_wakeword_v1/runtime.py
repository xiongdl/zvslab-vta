"""Build and execute the streaming wakeword graph on HOST or FSIM."""

import hashlib
import json
import math
import re
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tvm
import vta
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.backend import te_compiler
from vta.relay import plan_devices_for_vta

from common.deployment_compute import capture_deployment_compute
from common.deployment import lower_selected_deployment, write_json_atomic
from common.schedule import load_schedule_snapshot

from graph_artifacts import export_graph_bundle, validate_output_root
from model_pipeline import (
    CLIP_FRAMES,
    INPUT_DTYPE,
    INPUT_NAME,
    INPUT_SHAPE,
    MODEL_SHA256,
    OUTPUT_DTYPE,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    load_sample,
    prepare_model,
)


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "str_ww_ref_model.tflite"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build"
SUPPORTED_HOST_CODEGENS = ("llvm", "c")
DEFAULT_HOST_CODEGEN = "llvm"
MODEL_ID = "streaming_wakeword_v1"
SUPPORTED_SIMULATORS = ("host", "fsim", "tsim")
EXPECTED_LABEL_NAMES = ("Marvin", "Silence", "Unknown")
REQUIRED_PROFILER_COUNTERS = ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes")


@dataclass(frozen=True)
class SampleRecord:
    order: int
    path: Path
    filename: str
    label: int
    label_name: str


@dataclass(frozen=True)
class ReloadedArtifact:
    path: Path
    artifact_dir: Path
    graph_json: str
    params: bytes
    module: object
    device: object


@dataclass(frozen=True)
class HostArtifacts:
    host_codegen: str
    simulator: str
    reference: ReloadedArtifact
    mixed: ReloadedArtifact
    vta_symbols: tuple
    schedule_coverage: tuple = ()
    schedule_config_identities: tuple = ()


@dataclass(frozen=True)
class OutputComparison:
    sample_path: Path
    reference: np.ndarray
    mixed: np.ndarray | None
    top1: int


@dataclass(frozen=True)
class ExecutionSummary:
    comparisons: tuple
    profiler_stats: dict
    mode: str = "fsim"
    host_codegen: str = DEFAULT_HOST_CODEGEN


@dataclass(frozen=True)
class DeploymentResult:
    prepared: object
    artifacts: HostArtifacts
    execution: ExecutionSummary


@dataclass(frozen=True)
class SimulationMatrixResult:
    simulator: str
    prepared: object
    artifacts: tuple
    executions: tuple


@dataclass(frozen=True)
class SimulatorSession:
    label: str
    environment_target: str
    clear_registry: str
    status_registry: str
    required_registries: tuple
    activity_counter: str
    diagnostic: str

    def validate_environment(self):
        from vta.testing import simulator as backend_simulator

        active_target = backend_simulator.normalize_backend(simulator=self.label)
        if active_target != self.environment_target:
            raise RuntimeError(
                f"simulator {self.label!r} requires VTA backend {self.environment_target!r}; "
                f"active target is {active_target!r}"
            )
        return self

    def load(self):
        try:
            from vta.testing import simulator
            simulator.load_backend(self.label)
        except Exception as error:
            missing = [
                name for name in self.required_registries
                if tvm.get_global_func(name, allow_missing=True) is None
            ]
            detail = ", ".join(missing) if missing else "simulator initialization failed"
            raise RuntimeError(
                f"{self.label.upper()} is unavailable; missing registry functions: {detail}. "
                f"Build the required library with {self.diagnostic}"
            ) from error
        missing = [
            name for name in self.required_registries
            if tvm.get_global_func(name, allow_missing=True) is None
        ]
        if missing:
            raise RuntimeError(
                f"{self.label.upper()} is unavailable; missing registry functions: {', '.join(missing)}. "
                f"Build the required library with {self.diagnostic}"
            )
        return simulator

    def read_stats(self, status=None, simulator=None):
        status = status or getattr(simulator, "stats", None)
        status = status or tvm.get_global_func(self.status_registry, allow_missing=True)
        if status is None:
            raise RuntimeError(
                f"{self.label.upper()} profiler status is unavailable; "
                f"build the required library with {self.diagnostic}"
            )
        try:
            raw = status() if callable(status) else status
            stats = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{self.label.upper()} profiler returned malformed counters") from error
        if not isinstance(stats, dict):
            raise RuntimeError(f"FSIM profiler returned malformed stats: {stats!r}")
        return dict(stats)

    def clear_and_validate(self, simulator=None):
        clear = getattr(simulator, "clear_stats", None) if simulator is not None else None
        status = getattr(simulator, "stats", None) if simulator is not None else None
        clear = clear or tvm.get_global_func(self.clear_registry, allow_missing=True)
        status = status or tvm.get_global_func(self.status_registry, allow_missing=True)
        if clear is None or status is None:
            raise RuntimeError(
                f"FSIM profiler registry is unavailable; build the required library with {self.diagnostic}"
            )
        try:
            clear()
            stats = self.read_stats(status=status)
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError("FSIM profiler failed to clear counters") from error
        if self.label == "tsim":
            if stats != {"cycle_count": 0}:
                raise RuntimeError(
                    f"TSIM profiler did not reset to {{'cycle_count': 0}}: {stats}"
                )
        else:
            invalid = [
                counter for counter in REQUIRED_PROFILER_COUNTERS
                if counter not in stats or isinstance(stats[counter], bool)
                or not isinstance(stats[counter], (int, float)) or stats[counter] != 0
            ]
            if invalid:
                raise RuntimeError(f"FSIM profiler did not reset counters to zero: {invalid}")
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
            required_registries=("vta.simulator.profiler_clear", "vta.simulator.profiler_status"),
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
    return ".dylib" if sys.platform == "darwin" else ".so"


def _validate_host_codegen(host_codegen):
    if host_codegen not in SUPPORTED_HOST_CODEGENS:
        raise ValueError(f"unsupported host codegen {host_codegen!r}; supported kinds are llvm and c")
    return host_codegen


def _validate_host_codegens(host_codegens):
    try:
        values = tuple(host_codegens)
    except TypeError as error:
        raise ValueError("host_codegens must be exactly ('llvm', 'c') in that order") from error
    if values != SUPPORTED_HOST_CODEGENS:
        raise ValueError("host_codegens must be exactly ('llvm', 'c') in that order")
    return values


def _validate_simulator(simulator):
    if simulator not in SUPPORTED_SIMULATORS:
        raise ValueError("supported simulators are host and fsim")
    return simulator


def _manifest_records(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    samples_dir = manifest_path.parent.resolve(strict=True)
    environment_directory = "." + "envs"
    if environment_directory in manifest_path.parts:
        raise ValueError("manifest must not be read from the environment directory")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sample manifest is not valid UTF-8 JSON") from error
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 3:
        raise ValueError("sample manifest must contain exactly three samples")
    mapping = payload.get("class_mapping")
    if mapping != {"0": "Marvin", "1": "Silence", "2": "Unknown"}:
        raise ValueError("manifest class mapping must be Marvin, Silence, Unknown")
    records = []
    seen = set()
    for expected_order, item in enumerate(samples):
        if not isinstance(item, dict):
            raise ValueError("manifest samples must be objects")
        filename = item.get("filename")
        relative = Path(filename) if isinstance(filename, str) else None
        if (
            not isinstance(filename, str) or not filename or relative.is_absolute()
            or ".." in relative.parts or relative.name != filename
        ):
            raise ValueError(f"manifest has an unsafe filename: {filename!r}")
        if filename in seen:
            raise ValueError("manifest sample filenames must be unique")
        seen.add(filename)
        path = samples_dir / relative
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"manifest sample is missing or symlinked: {path}")
        if path.resolve(strict=True).parent != samples_dir:
            raise ValueError(f"manifest sample path is outside samples directory: {path}")
        if type(item.get("order")) is not int or item["order"] != expected_order:
            raise ValueError("manifest sample order must be zero through two")
        label = item.get("label")
        label_name = item.get("label_name")
        if type(label) is not int or label != expected_order or label_name != EXPECTED_LABEL_NAMES[expected_order]:
            raise ValueError("manifest labels must use the fixed streaming wakeword order")
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"manifest sample sha256 is invalid: {filename!r}")
        sample_bytes = path.read_bytes()
        if item.get("byte_length") != len(sample_bytes):
            raise ValueError(f"manifest sample byte_length mismatch: {filename!r}")
        if hashlib.sha256(sample_bytes).hexdigest() != sha256:
            raise ValueError(f"manifest sample sha256 mismatch: {filename!r}")
        records.append(SampleRecord(expected_order, path, filename, label, label_name))
    return tuple(records)


def committed_sample_records(manifest_path=MANIFEST_PATH):
    return _manifest_records(manifest_path)


def committed_sample_paths(manifest_path=MANIFEST_PATH):
    return tuple(record.path for record in committed_sample_records(manifest_path))


def _artifact_identity(host_codegen, role):
    _validate_host_codegen(host_codegen)
    if role not in {"reference", "mixed"}:
        raise ValueError(f"unsupported artifact role {role!r}")
    return f"mlperf_streaming_wakeword_{host_codegen}_{role}"


def _matrix_artifact_root(output_dir, host_codegen, simulator="fsim"):
    _validate_host_codegen(host_codegen)
    _validate_simulator(simulator)
    return Path(output_dir) / f"{host_codegen}-{simulator}"


def _host_target(host_codegen=DEFAULT_HOST_CODEGEN):
    _validate_host_codegen(host_codegen)
    return tvm.target.Target(vta.get_env().target_host if host_codegen == "llvm" else "c")


def _mixed_target(host_codegen=DEFAULT_HOST_CODEGEN):
    _validate_host_codegen(host_codegen)
    host = vta.get_env().target_host if host_codegen == "llvm" else tvm.target.Target("c")
    return tvm.target.Target("vta", host=host)


def _mixed_build_plan(module, host_codegen):
    if isinstance(module, tvm.IRModule):
        return plan_devices_for_vta(module, _host_target(host_codegen))
    return SimpleNamespace(module=module, targets=_mixed_target(host_codegen))


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


def _model_metadata(prepared):
    return {
        "model_sha256": getattr(getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256),
        "input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "dtype": INPUT_DTYPE},
        "output": {"name": OUTPUT_NAME, "shape": list(OUTPUT_SHAPE), "dtype": OUTPUT_DTYPE},
        "labels": list(EXPECTED_LABEL_NAMES),
        "vta_symbols": list(prepared.routing.symbols),
    }


def build_host_artifacts(prepared, output_dir=DEFAULT_OUTPUT_DIR,
                         host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim",
                         schedule=None):
    """Build and reload reference/mixed bundles using a validated schedule snapshot."""
    _validate_host_codegen(host_codegen)
    _validate_simulator(simulator)
    output_dir = validate_output_root(output_dir)
    if host_codegen == "llvm":
        reference_factory = relay.build(prepared.reference_module, target="llvm")
    else:
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            reference_factory = relay.build(
                prepared.reference_module, target=tvm.target.Target("c")
            )
    device_plan = _mixed_build_plan(prepared.mixed_module, host_codegen)
    build_config = {"tir.disable_vectorize": True} if host_codegen == "c" else {}
    schedule_path = None if schedule is None or str(schedule).lower() == "none" else schedule
    deployment = None
    snapshot = None
    compiler_config = None
    if isinstance(prepared.mixed_module, tvm.IRModule):
        deployment = capture_deployment_compute(
            prepared.mixed_module, MODEL_ID, prepared.imported.model_sha256
        )
        snapshot = load_schedule_snapshot(schedule_path, deployment)
        compiler_config = vta.relay.transform.VTACompilerConfig.from_env(vta.get_env())
    elif schedule_path is not None:
        raise TypeError("schedule replay requires the prepared actual Relay deployment module")
    compiler = te_compiler.get()
    replay_enabled = bool(snapshot and snapshot.selected)
    if replay_enabled:
        compiler.clear()
        try:
            with _selected_snapshot_lowering(compiler, deployment, snapshot, compiler_config):
                with vta.build_config(config=build_config):
                    mixed_factory = relay.build(device_plan.module, target=device_plan.targets)
        finally:
            compiler.clear()
    else:
        try:
            with vta.build_config(config=build_config):
                mixed_factory = relay.build(device_plan.module, target=device_plan.targets)
        finally:
            compiler.clear()
    metadata = _model_metadata(prepared)
    model_sha256 = metadata["model_sha256"]
    reference = export_graph_bundle(
        reference_factory,
        output_dir,
        "reference",
        artifact_name=_artifact_identity(host_codegen, "reference"),
        artifact_role="reference",
        model_sha256=model_sha256,
        host_codegen=host_codegen,
        simulator=simulator,
        forbidden_vta_symbols=prepared.routing.symbols,
        metadata=metadata,
    )
    mixed = export_graph_bundle(
        mixed_factory,
        output_dir,
        "mixed",
        artifact_name=_artifact_identity(host_codegen, "mixed"),
        artifact_role="mixed",
        model_sha256=model_sha256,
        host_codegen=host_codegen,
        simulator=simulator,
        expected_vta_symbols=prepared.routing.symbols,
        metadata=metadata,
    )
    selected_configs = snapshot.selected if snapshot is not None else {}
    config_identities = tuple(
        (
            occurrence,
            hashlib.sha256(json.dumps(
                [config.to_json_dict() for config in selected.configs if config is not None],
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        )
        for occurrence, selected in sorted(selected_configs.items())
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
            reference.library_path, reference.artifact_dir, reference.graph_json,
            reference.params, reference.module, tvm.cpu(0),
        ),
        mixed=ReloadedArtifact(
            mixed.library_path, mixed.artifact_dir, mixed.graph_json,
            mixed.params, mixed.module, (tvm.cpu(0), tvm.ext_dev(0)),
        ),
        vta_symbols=tuple(prepared.routing.symbols),
        schedule_coverage=coverage,
        schedule_config_identities=config_identities,
    )


def validate_mixed_symbols(module, expected_symbols):
    checker = getattr(module, "implements_function", None)
    if not callable(checker):
        raise RuntimeError("reloaded mixed artifact cannot validate VTA symbols")
    missing = [symbol for symbol in expected_symbols if not checker(symbol, True)]
    if missing:
        raise RuntimeError(f"reloaded mixed artifact is missing VTA symbols: {missing}")


def _run_graph(artifact, input_data):
    input_data = np.asarray(input_data)
    if input_data.shape != INPUT_SHAPE or input_data.dtype != np.dtype(INPUT_DTYPE):
        raise RuntimeError(f"input must have shape {INPUT_SHAPE} and dtype {INPUT_DTYPE}")
    runtime = graph_executor.create(artifact.graph_json, artifact.module, artifact.device)
    runtime.load_params(artifact.params)
    runtime.set_input(INPUT_NAME, input_data)
    runtime.run()
    output = np.asarray(runtime.get_output(0).numpy())
    if output.shape != OUTPUT_SHAPE:
        raise RuntimeError(f"output shape must be {OUTPUT_SHAPE}, received {output.shape}")
    if output.dtype != np.dtype(OUTPUT_DTYPE):
        raise RuntimeError(f"output dtype must be {OUTPUT_DTYPE}, received {output.dtype}")
    return output


def compare_outputs(sample_path, reference, mixed):
    sample_path = Path(sample_path)
    reference = np.asarray(reference)
    mixed = np.asarray(mixed)
    for label, output in (("reference", reference), ("mixed", mixed)):
        if output.shape != OUTPUT_SHAPE:
            raise RuntimeError(f"{sample_path.name} {label} output shape is {output.shape}")
        if output.dtype != np.dtype(OUTPUT_DTYPE):
            raise RuntimeError(f"{sample_path.name} {label} output dtype is {output.dtype}")
    if reference.shape != mixed.shape:
        raise RuntimeError(f"{sample_path.name} output shape differs: {reference.shape} != {mixed.shape}")
    if reference.dtype != mixed.dtype:
        raise RuntimeError(f"{sample_path.name} output dtype differs: {reference.dtype} != {mixed.dtype}")
    if not np.array_equal(reference, mixed):
        raise RuntimeError(f"{sample_path.name} output is not elementwise equal")
    reference_top1 = int(np.argmax(reference, axis=1)[0])
    mixed_top1 = int(np.argmax(mixed, axis=1)[0])
    if reference_top1 != mixed_top1:
        raise RuntimeError(f"{sample_path.name} top-1 differs: {reference_top1} != {mixed_top1}")
    return OutputComparison(sample_path, reference, mixed, reference_top1)


def _validate_execution_paths(sample_paths):
    paths = tuple(Path(path) for path in sample_paths)
    if paths != committed_sample_paths():
        raise RuntimeError("execution must use the three committed samples in manifest order")
    return paths


def execute_host(artifacts, sample_paths=None):
    """Execute only the reference graph; HOST never loads a simulator."""
    paths = _validate_execution_paths(sample_paths or committed_sample_paths())
    inputs = tuple((path, load_sample(path)) for path in paths)
    outputs = tuple((path, _run_graph(artifacts.reference, data)) for path, data in inputs)
    comparisons = tuple(
        OutputComparison(path, output, None, int(np.argmax(output, axis=1)[0]))
        for path, output in outputs
    )
    return ExecutionSummary(comparisons, {}, "host", artifacts.host_codegen)


def _load_fsim():
    """Lazy FSIM loading, after all reference outputs have completed."""
    return _simulator_session("fsim").load()


def _load_simulator(simulator):
    session = _simulator_session(simulator)
    session.validate_environment()
    return session, session.load()


def validate_profiler_stats(stats):
    for counter in REQUIRED_PROFILER_COUNTERS:
        value = stats.get(counter)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value)) or value <= 0
        ):
            raise RuntimeError(f"FSIM profiler counter {counter} must be positive")


@contextmanager
def _fsim_context():
    yield tvm.ext_dev(0)


def execute_fsim(artifacts, sample_paths=None):
    """Compare all three samples elementwise and require positive FSIM activity."""
    paths = _validate_execution_paths(sample_paths or committed_sample_paths())
    inputs = tuple((path, load_sample(path)) for path in paths)
    reference_outputs = tuple(
        (path, _run_graph(artifacts.reference, data)) for path, data in inputs
    )
    validate_mixed_symbols(artifacts.mixed.module, artifacts.vta_symbols)
    simulator = _load_fsim()
    session = _simulator_session("fsim")
    session.clear_and_validate(simulator)
    comparisons = []
    with _fsim_context():
        for (path, data), (_, reference_output) in zip(inputs, reference_outputs):
            try:
                mixed_output = _run_graph(artifacts.mixed, data)
                comparisons.append(compare_outputs(path, reference_output, mixed_output))
            except Exception as error:
                raise RuntimeError(f"FSIM mixed execution failed for {path.name}: {error}") from error
    stats = session.read_stats(simulator=simulator)
    session.validate_activity(stats)
    return ExecutionSummary(tuple(comparisons), dict(stats), "fsim", artifacts.host_codegen)


def _execute_matrix(artifacts, sample_paths, simulator="fsim"):
    paths = _validate_execution_paths(sample_paths)
    inputs = tuple((path, load_sample(path)) for path in paths)
    references = {}
    baseline = None
    for host_artifacts in artifacts:
        outputs = tuple((path, _run_graph(host_artifacts.reference, data)) for path, data in inputs)
        if baseline is not None:
            for (path, expected), (_, actual) in zip(baseline, outputs):
                compare_outputs(path, expected, actual)
        baseline = outputs if baseline is None else baseline
        references[host_artifacts.host_codegen] = outputs
    session, simulator_module = _load_simulator(simulator)
    executions = []
    for host_artifacts in artifacts:
        validate_mixed_symbols(host_artifacts.mixed.module, host_artifacts.vta_symbols)
        session.clear_and_validate(simulator_module)
        comparisons = []
        for (path, data), (_, reference_output) in zip(inputs, references[host_artifacts.host_codegen]):
            comparisons.append(compare_outputs(path, reference_output, _run_graph(host_artifacts.mixed, data)))
        stats = session.read_stats(simulator=simulator_module)
        session.validate_activity(stats)
        executions.append(
            ExecutionSummary(tuple(comparisons), dict(stats), simulator, host_artifacts.host_codegen)
        )
    return tuple(executions)


def deploy_fsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS,
                       schedule=None):
    """Build the ordered LLVM/C matrix and execute it on FSIM."""
    output_dir = validate_output_root(output_dir)
    host_codegens = _validate_host_codegens(host_codegens)
    _simulator_session("fsim").validate_environment()
    prepared = prepare_model(MODEL_PATH)
    artifacts = tuple(
        build_host_artifacts(
            prepared,
            _matrix_artifact_root(output_dir, codegen, "fsim"),
            codegen,
            "fsim",
            schedule=schedule,
        )
        for codegen in host_codegens
    )
    executions = _execute_matrix(artifacts, committed_sample_paths(), "fsim")
    return SimulationMatrixResult("fsim", prepared, artifacts, executions)


def deploy_tsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS,
                       schedule=None):
    """Build the ordered LLVM/C matrix and execute it on TSIM."""
    output_dir = validate_output_root(output_dir)
    host_codegens = _validate_host_codegens(host_codegens)
    _simulator_session("tsim").validate_environment()
    prepared = prepare_model(MODEL_PATH)
    artifacts = tuple(
        build_host_artifacts(
            prepared,
            _matrix_artifact_root(output_dir, codegen, "tsim"),
            codegen,
            "tsim",
            schedule=schedule,
        )
        for codegen in host_codegens
    )
    executions = _execute_matrix(artifacts, committed_sample_paths(), "tsim")
    return SimulationMatrixResult("tsim", prepared, artifacts, executions)


def deploy(output_dir=DEFAULT_OUTPUT_DIR, host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim",
           schedule=None):
    """Build one artifact pair and execute HOST reference or FSIM mixed output."""
    output_dir = validate_output_root(output_dir)
    _validate_host_codegen(host_codegen)
    _validate_simulator(simulator)
    if simulator in {"fsim", "tsim"}:
        _simulator_session(simulator).validate_environment()
    prepared = prepare_model(MODEL_PATH)
    paths = committed_sample_paths()
    artifacts = build_host_artifacts(
        prepared, output_dir, host_codegen, simulator,
        schedule=schedule,
    )
    if simulator == "host":
        execution = execute_host(artifacts, paths)
    elif simulator == "fsim":
        execution = execute_fsim(artifacts, paths)
    else:
        execution = _execute_matrix((artifacts,), paths, "tsim")[0]
    return DeploymentResult(prepared, artifacts, execution)


def schedule_coverage_rows(artifacts):
    return [
        {"occurrence": occurrence, "symbol": symbol, "selected": selected}
        for occurrence, symbol, selected in artifacts.schedule_coverage
    ]


def select_deployment_sample():
    """Return the first manifest audio window and its stateless execution contract."""
    records = committed_sample_records()
    if not records or records[0].order != 0:
        raise ValueError("Streaming Wakeword V1 sample manifest must begin at order zero")
    sample = records[0]
    input_data = np.asarray(load_sample(sample.path))
    if input_data.shape != INPUT_SHAPE or input_data.dtype != np.dtype(INPUT_DTYPE):
        raise ValueError("Streaming Wakeword V1 sample input violates the model tensor contract")
    evidence = {
        "sample_count": 1,
        "audio_window_count": 1,
        "audio_window_samples": int(CLIP_FRAMES),
        "feature_frame_count": int(INPUT_SHAPE[1]),
        "model_invocations": 1,
        "state_policy": "stateless_single_invocation",
    }
    return sample, input_data, evidence


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
    """Preserve the streaming wakeword inclusive 10% deployment evidence gate."""
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


def write_deployment_report(result, report_path, *, schedule=None, validate_schedule_evidence=False):
    """Write HOST-checked sample results and optional graph-resident TSIM evidence."""
    schedule_path = None if schedule is None or str(schedule).lower() == "none" else Path(schedule)
    config_identities = dict(result.artifacts.schedule_config_identities)
    report = {
        "schema_version": 1,
        "artifact_kind": "vta_deployment_profile_v1",
        "model": MODEL_ID,
        "model_sha256": result.prepared.imported.model_sha256,
        "simulator": result.artifacts.simulator,
        "host_codegen": result.artifacts.host_codegen,
        "schedule": str(schedule_path.resolve()) if schedule_path else None,
        "schedule_coverage": schedule_coverage_rows(result.artifacts),
        "selected_config_identities": [
            {"occurrence": occurrence, "sha256": config_identities[occurrence]}
            for occurrence in sorted(config_identities)
        ],
        "sample_count": len(result.execution.comparisons),
        "outputs_passed": sum(item.mixed is not None for item in result.execution.comparisons),
        "samples": [
            {"filename": item.sample_path.name, "label": item.sample_path.name,
             "reference_top1": item.top1,
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
        deployment = capture_deployment_compute(
            result.prepared.mixed_module, MODEL_ID, result.prepared.imported.model_sha256
        )
        snapshot = load_schedule_snapshot(schedule_path, deployment)
        _validate_schedule_evidence(deployment, snapshot)
        if len(result.execution.comparisons) != len(committed_sample_records()) or report["outputs_passed"] != len(committed_sample_records()):
            raise RuntimeError("schedule evidence requires HOST-checked outputs for every committed sample")

        from tvm.contrib.debugger import debug_executor
        from common.deployment_evidence import profile_graph_resident_nodes

        sample, input_data, sample_evidence = select_deployment_sample()
        session, simulator = _load_simulator("tsim")
        graph = debug_executor.create(
            result.artifacts.mixed.graph_json,
            result.artifacts.mixed.module,
            result.artifacts.mixed.device,
        )
        graph.load_params(result.artifacts.mixed.params)
        graph.set_input(INPUT_NAME, input_data)
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
            "performance_sample": sample.path.name,
            "performance_sample_count": 1,
            "performance_stats": performance_stats,
            "occurrences": occurrences,
            "sample_contract": sample_evidence,
            "sample_sha256": hashlib.sha256(sample.path.read_bytes()).hexdigest(),
        })
    write_json_atomic(report_path, report)
    return report
