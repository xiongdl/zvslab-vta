"""Build and execute the fixed anomaly autoencoder on HOST or FSIM."""

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm
import vta
from tvm import relay
from tvm.contrib import graph_executor

from graph_artifacts import export_graph_bundle
from model_pipeline import (
    INPUT_DTYPE,
    INPUT_NAME,
    INPUT_SHAPE,
    MODEL_SHA256,
    OUTPUT_DTYPE,
    OUTPUT_SHAPE,
    EXPECTED_VTA_SYMBOLS,
    load_sample,
    prepare_model,
)


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "ad01_fp32.tflite"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
DEFAULT_BUILD_DIR = APP_ROOT / "build"
DEFAULT_OUTPUT_DIR = DEFAULT_BUILD_DIR
SUPPORTED_HOST_CODEGENS = ("llvm", "c")
DEFAULT_HOST_CODEGEN = "llvm"
SUPPORTED_MODES = ("host", "fsim", "tsim")
REQUIRED_PROFILER_COUNTERS = ("gemm_counter", "wgt_load_nbytes", "out_store_nbytes")


@dataclass(frozen=True)
class SampleRecord:
    order: int
    path: Path
    filename: str
    class_name: str
    label: int


@dataclass(frozen=True)
class ReloadedArtifact:
    artifact_dir: Path
    library_path: Path
    graph_json: str
    params: bytes
    module: object
    device: object


@dataclass(frozen=True)
class HostArtifacts:
    host_codegen: str
    mode: str
    reference: ReloadedArtifact
    mixed: ReloadedArtifact
    vta_symbols: tuple


@dataclass(frozen=True)
class SampleResult:
    order: int
    filename: str
    label: int
    class_name: str
    predicted_label: int
    score: float
    input_shape: tuple
    output_shape: tuple
    output_dtype: str
    feature_shape: tuple
    reference_score: float
    mixed_score: float | None


@dataclass(frozen=True)
class ExecutionSummary:
    mode: str
    host_codegen: str
    samples: tuple
    summary: dict
    profiler_stats: dict
    model_metadata: dict


@dataclass(frozen=True)
class DeploymentResult:
    prepared: object
    artifacts: HostArtifacts
    execution: ExecutionSummary


@dataclass(frozen=True)
class SimulatorSession:
    """Validated registry adaptor for one configured VTA simulator process."""

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

    def clear_and_validate(self, simulator):
        clear = getattr(simulator, "clear_stats", None)
        status = getattr(simulator, "stats", None)
        if clear is None or status is None:
            raise RuntimeError(
                f"{self.label.upper()} profiler registry is unavailable; build the "
                f"required libraries with {self.diagnostic}"
            )
        clear()
        stats = self.read_stats(status)
        if self.label == "tsim" and stats != {"cycle_count": 0}:
            raise RuntimeError(f"TSIM profiler did not reset to {{'cycle_count': 0}}: {stats}")
        return stats

    def read_stats(self, status):
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
            value = stats.get("cycle_count")
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(
                    f"TSIM profiler counter cycle_count must be a positive integer: {stats}"
                )
            return
        _validate_profiler_stats(stats)


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
            activity_counter="cycle_count",
            diagnostic="bash scripts/build_vta_lib.sh --target libvta_hw",
        )
    raise ValueError(f"unsupported simulator {simulator!r}; supported simulators are fsim and tsim")


def _load_simulator(simulator):
    session = _simulator_session(simulator).validate_environment()
    return session, session.load()


def _validate_mode(mode):
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported mode {mode!r}; use host, fsim, or tsim")
    return mode


def _validate_host_codegen(host_codegen):
    if host_codegen not in SUPPORTED_HOST_CODEGENS:
        raise ValueError(f"unsupported host codegen {host_codegen!r}; use llvm or c")
    return host_codegen


def _manifest_records(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    if ".envs" in manifest_path.parts:
        raise ValueError("manifest must not be read from .envs")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list) or len(samples) != 10:
        raise ValueError("sample manifest must contain exactly ten samples")
    records = []
    for expected_order, item in enumerate(samples):
        filename = item.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"manifest has an unsafe filename: {filename!r}")
        path = manifest_path.parent / filename
        if ".envs" in path.resolve().parts or not path.is_file():
            raise ValueError(f"manifest sample is missing or unsafe: {path}")
        if item.get("order") != expected_order:
            raise ValueError("manifest sample order must be zero through nine")
        label = int(item.get("label"))
        class_name = item.get("class_name")
        if (label, class_name) not in ((0, "normal"), (1, "anomaly")):
            raise ValueError("manifest labels must use 0=normal and 1=anomaly")
        records.append(SampleRecord(expected_order, path, filename, class_name, label))
    if [item.label for item in records] != [0] * 5 + [1] * 5:
        raise ValueError("manifest order must contain five normal then five anomaly samples")
    return tuple(records)


def committed_sample_records(manifest_path=MANIFEST_PATH):
    """Return the ten committed samples in their authenticated manifest order."""
    return _manifest_records(manifest_path)


def committed_sample_paths(manifest_path=MANIFEST_PATH):
    return tuple(record.path for record in committed_sample_records(manifest_path))


def _artifact_identity(host_codegen, role):
    return f"mlperf_anomaly_{host_codegen}_{role}"


def _mixed_target(host_codegen):
    host = vta.get_env().target_host if host_codegen == "llvm" else tvm.target.Target("c")
    return tvm.target.Target("vta", host=host)


def _model_metadata(prepared):
    return {
        "model_sha256": prepared.imported.model_sha256,
        "input": {"name": INPUT_NAME, "shape": list(INPUT_SHAPE), "dtype": INPUT_DTYPE},
        "output": {"name": "Identity", "shape": list(OUTPUT_SHAPE), "dtype": OUTPUT_DTYPE},
        "score": "mean squared error between each input feature vector and its reconstructed output",
        "feature_width": 640,
        "vta_symbols": list(prepared.routing.symbols),
    }


def build_host_artifacts(prepared, build_dir=DEFAULT_BUILD_DIR, host_codegen=DEFAULT_HOST_CODEGEN,
                         mode="host"):
    """Build and reload reference/mixed artifacts without loading a simulator."""
    _validate_host_codegen(host_codegen)
    _validate_mode(mode)
    if prepared.reference_module is not prepared.quantized_module:
        raise RuntimeError("reference must be the shared quantized module")
    build_root = Path(build_dir) / f"{host_codegen}-{mode}"
    if host_codegen == "c":
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            reference_factory = relay.build(prepared.reference_module, target=tvm.target.Target("c"))
    else:
        reference_factory = relay.build(prepared.reference_module, target="llvm")
    if host_codegen == "c":
        with vta.build_config(config={"tir.disable_vectorize": True}):
            mixed_factory = relay.build(prepared.mixed_module, target=_mixed_target(host_codegen))
    else:
        with vta.build_config():
            mixed_factory = relay.build(prepared.mixed_module, target=_mixed_target(host_codegen))
    metadata = _model_metadata(prepared)
    reference = export_graph_bundle(
        reference_factory, build_root, "reference",
        artifact_name=_artifact_identity(host_codegen, "reference"),
        artifact_role="reference", model_sha256=prepared.imported.model_sha256,
        host_codegen=host_codegen, simulator=mode if mode in ("fsim", "tsim") else "host",
        forbidden_vta_symbols=prepared.routing.symbols, metadata=metadata,
    )
    mixed = export_graph_bundle(
        mixed_factory, build_root, "mixed",
        artifact_name=_artifact_identity(host_codegen, "mixed"),
        artifact_role="mixed", model_sha256=prepared.imported.model_sha256,
        host_codegen=host_codegen, simulator=mode if mode in ("fsim", "tsim") else "host",
        expected_vta_symbols=prepared.routing.symbols, metadata=metadata,
    )
    return HostArtifacts(
        host_codegen=host_codegen,
        mode=mode,
        reference=ReloadedArtifact(reference.artifact_dir, reference.library_path, reference.graph_json, reference.params, reference.module, tvm.cpu(0)),
        mixed=ReloadedArtifact(mixed.artifact_dir, mixed.library_path, mixed.graph_json, mixed.params, mixed.module, tvm.ext_dev(0) if mode in ("fsim", "tsim") else tvm.cpu(0)),
        vta_symbols=tuple(prepared.routing.symbols),
    )


def validate_mixed_symbols(module, expected_symbols):
    for symbol in expected_symbols:
        if not module.implements_function(symbol, True):
            raise RuntimeError(f"reloaded mixed artifact is missing VTA symbol {symbol}")


def _run_graph(artifact, input_data, output=None):
    input_data = np.asarray(input_data)
    if input_data.shape != INPUT_SHAPE or input_data.dtype != np.dtype(INPUT_DTYPE):
        raise RuntimeError(f"input must have shape {INPUT_SHAPE} and dtype {INPUT_DTYPE}")
    if output is None:
        runtime = graph_executor.create(artifact.graph_json, artifact.module, artifact.device)
        runtime.load_params(artifact.params)
        runtime.set_input(INPUT_NAME, input_data)
        runtime.run()
        output = runtime.get_output(0).numpy()
    output = np.asarray(output)
    if output.shape != OUTPUT_SHAPE:
        raise RuntimeError(f"output shape must be {OUTPUT_SHAPE}, received {output.shape}")
    if output.dtype != np.dtype(OUTPUT_DTYPE):
        raise RuntimeError(f"output dtype must be {OUTPUT_DTYPE}, received {output.dtype}")
    return output


def _sample_score(artifact, record):
    features = np.asarray(load_sample(record.path))
    if features.ndim != 2 or features.shape[1] != INPUT_SHAPE[1] or features.dtype != np.float32:
        raise RuntimeError(f"{record.filename} preprocessing must return (N, 640) float32")
    outputs = np.concatenate(tuple(_run_graph(artifact, row[None, :]) for row in features), axis=0)
    if outputs.shape != features.shape:
        raise RuntimeError(f"{record.filename} reconstructed tensor shape differs")
    return float(np.mean((features.astype(np.float32) - outputs) ** 2)), features.shape, outputs.dtype


def _reference_raw(artifact, records):
    raw = []
    for record in records:
        score, shape, dtype = _sample_score(artifact.reference, record)
        raw.append({"order": record.order, "filename": record.filename, "label": record.label,
                    "class_name": record.class_name, "reference_score": score,
                    "input_shape": INPUT_SHAPE, "output_shape": OUTPUT_SHAPE,
                    "feature_shape": shape, "output_dtype": str(dtype)})
    return raw


def _with_predictions(results):
    normal_scores = [item["reference_score"] for item in results if item["label"] == 0]
    threshold = float(max(normal_scores)) if normal_scores else 0.0
    completed = tuple(
        SampleResult(
            order=item["order"], filename=item["filename"], label=item["label"],
            class_name=item["class_name"], predicted_label=int(item["reference_score"] > threshold),
            score=float(item["reference_score"]), input_shape=INPUT_SHAPE,
            output_shape=OUTPUT_SHAPE, output_dtype=item["output_dtype"],
            feature_shape=tuple(item["feature_shape"]),
            reference_score=float(item["reference_score"]), mixed_score=item.get("mixed_score"),
        )
        for item in results
    )
    return completed, threshold


def _summary(samples, threshold):
    return {
        "sample_count": len(samples),
        "normal_count": sum(item.label == 0 for item in samples),
        "anomaly_count": sum(item.label == 1 for item in samples),
        "predicted_anomaly_count": sum(item.predicted_label == 1 for item in samples),
        "anomaly_score_threshold": threshold,
        "score_semantics": "mean squared reconstruction error; higher means less like the autoencoder training distribution",
    }


def execute_host(artifacts, records):
    """Execute only the CPU reference artifact; this path never imports simulator."""
    raw = []
    for record in records:
        score, shape, dtype = _sample_score(artifacts.reference, record)
        raw.append({"order": record.order, "filename": record.filename, "label": record.label,
                    "class_name": record.class_name, "reference_score": score,
                    "input_shape": INPUT_SHAPE, "output_shape": OUTPUT_SHAPE,
                    "feature_shape": shape, "output_dtype": str(dtype)})
    samples, threshold = _with_predictions(raw)
    return ExecutionSummary("host", artifacts.host_codegen, samples, _summary(samples, threshold), {}, {})


def _load_fsim():
    """Load VTA FSIM lazily, only immediately before mixed execution."""
    from vta.testing import simulator
    return simulator


def _validate_profiler_stats(stats):
    for counter in REQUIRED_PROFILER_COUNTERS:
        value = stats.get(counter, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise RuntimeError(f"FSIM profiler counter {counter} must be positive")


@contextmanager
def _fsim_context():
    """Make the VTA backend device explicit around mixed graph execution."""
    yield tvm.ext_dev(0)


def execute_fsim(artifacts, records):
    """Run reference first, then enter the lazy FSIM backend for the mixed graph."""
    raw = []
    for record in records:
        score, shape, dtype = _sample_score(artifacts.reference, record)
        raw.append({"order": record.order, "filename": record.filename, "label": record.label,
                    "class_name": record.class_name, "reference_score": score,
                    "input_shape": INPUT_SHAPE, "output_shape": OUTPUT_SHAPE,
                    "feature_shape": shape, "output_dtype": str(dtype)})
    validate_mixed_symbols(artifacts.mixed.module, artifacts.vta_symbols)
    simulator = _load_fsim()
    simulator.clear_stats()
    with _fsim_context():
        for item, record in zip(raw, records):
            mixed_score, shape, dtype = _sample_score(artifacts.mixed, record)
            if shape != item["feature_shape"] or dtype != np.dtype(item["output_dtype"]):
                raise RuntimeError(f"{record.filename} mixed tensor contract differs")
            if not np.isclose(mixed_score, item["reference_score"], rtol=1e-6, atol=1e-6):
                raise RuntimeError(f"{record.filename} reconstruction score differs")
            item["mixed_score"] = mixed_score
    stats = dict(simulator.stats())
    _validate_profiler_stats(stats)
    samples, threshold = _with_predictions(raw)
    return ExecutionSummary("fsim", artifacts.host_codegen, samples, _summary(samples, threshold), stats, {})


def _execute_tsim_artifact(artifacts, records, raw, session, simulator):
    validate_mixed_symbols(artifacts.mixed.module, artifacts.vta_symbols)
    session.clear_and_validate(simulator)
    for item, record in zip(raw, records):
        mixed_score, shape, dtype = _sample_score(artifacts.mixed, record)
        if shape != item["feature_shape"] or dtype != np.dtype(item["output_dtype"]):
            raise RuntimeError(f"{record.filename} mixed tensor contract differs")
        if not np.isclose(mixed_score, item["reference_score"], rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"{record.filename} reconstruction score differs")
        item["mixed_score"] = mixed_score
    stats = session.read_stats(simulator.stats)
    session.validate_activity(stats)
    samples, threshold = _with_predictions(raw)
    return ExecutionSummary("tsim", artifacts.host_codegen, samples, _summary(samples, threshold), stats, {})


def execute_tsim(artifacts, records):
    """Run one host variant through the lazy VTA TSIM simulator."""
    raw = _reference_raw(artifacts, records)
    session, simulator = _load_simulator("tsim")
    return _execute_tsim_artifact(artifacts, records, raw, session, simulator)


def _execute_tsim_matrix(artifacts, records, reference_raw, session, simulator):
    """Execute LLVM/C mixed graphs with one TSIM load and isolated counter windows."""
    executions = []
    for host_artifacts, raw in zip(artifacts, reference_raw):
        executions.append(
            _execute_tsim_artifact(host_artifacts, records, raw, session, simulator)
        )
    return tuple(executions)


def deploy(build_dir=DEFAULT_BUILD_DIR, host_codegen=DEFAULT_HOST_CODEGEN, mode="host",
           manifest_path=MANIFEST_PATH):
    _validate_mode(mode)
    if mode in ("fsim", "tsim"):
        _simulator_session(mode).validate_environment()
    prepared = prepare_model(MODEL_PATH)
    records = committed_sample_records(manifest_path)
    artifacts = build_host_artifacts(prepared, build_dir, host_codegen, mode)
    if mode == "host":
        execution = execute_host(artifacts, records)
    elif mode == "fsim":
        execution = execute_fsim(artifacts, records)
    else:
        execution = execute_tsim(artifacts, records)
    execution = ExecutionSummary(execution.mode, execution.host_codegen, execution.samples,
                                 execution.summary, execution.profiler_stats, _model_metadata(prepared))
    return DeploymentResult(prepared, artifacts, execution)


def deploy_matrix(build_dir=DEFAULT_BUILD_DIR, mode="fsim", manifest_path=MANIFEST_PATH):
    _validate_mode(mode)
    if mode not in ("fsim", "tsim"):
        raise ValueError("host mode uses one selected host codegen; matrix mode is FSIM or TSIM")
    if mode == "tsim":
        return deploy_tsim_matrix(build_dir, manifest_path)
    _simulator_session(mode).validate_environment()
    prepared = prepare_model(MODEL_PATH)
    records = committed_sample_records(manifest_path)
    artifacts = tuple(build_host_artifacts(prepared, build_dir, codegen, mode) for codegen in SUPPORTED_HOST_CODEGENS)
    executions = []
    for artifact in artifacts:
        execution = execute_fsim(artifact, records)
        executions.append(
            ExecutionSummary(
                execution.mode,
                execution.host_codegen,
                execution.samples,
                execution.summary,
                execution.profiler_stats,
                _model_metadata(prepared),
            )
        )
    return prepared, artifacts, tuple(executions)


def deploy_tsim_matrix(build_dir=DEFAULT_BUILD_DIR, manifest_path=MANIFEST_PATH):
    """Build LLVM/C bundles before one lazy TSIM load and ten-sample execution."""
    session = _simulator_session("tsim").validate_environment()
    prepared = prepare_model(MODEL_PATH)
    records = committed_sample_records(manifest_path)
    artifacts = tuple(
        build_host_artifacts(prepared, build_dir, codegen, "tsim")
        for codegen in SUPPORTED_HOST_CODEGENS
    )
    reference_raw = tuple(_reference_raw(artifact, records) for artifact in artifacts)
    session, simulator = _load_simulator("tsim")
    executions = _execute_tsim_matrix(artifacts, records, reference_raw, session, simulator)
    executions = tuple(
        ExecutionSummary(
            execution.mode,
            execution.host_codegen,
            execution.samples,
            execution.summary,
            execution.profiler_stats,
            _model_metadata(prepared),
        )
        for execution in executions
    )
    return prepared, artifacts, executions
