"""Build and execute the fixed anomaly autoencoder on HOST or FSIM."""

import json
import hashlib
import math
import os
import stat
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
DEFAULT_TSIM_WINDOW_BUDGET = 1
TSIM_WINDOW_BUDGET_ENV = "VTA_ANOMALY_TSIM_WINDOW_BUDGET"
FULL_SCORE_SCOPE = "full_windows"
TSIM_SCORE_SCOPE = "representative_windows"


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
    executed_feature_shape: tuple
    total_window_count: int
    executed_window_count: int
    sampled: bool
    score_scope: str
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
        from vta.testing import simulator as backend_simulator

        active_target = backend_simulator.normalize_backend(simulator=self.label)
        if active_target != self.environment_target:
            raise RuntimeError(
                f"simulator {self.label!r} requires VTA backend "
                f"{self.environment_target!r}, active target is {active_target!r}"
            )
        return self

    def load(self):
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

    def clear_and_validate(self, simulator):
        clear = getattr(simulator, "clear_stats", None)
        status = getattr(simulator, "stats", None)
        if clear is None or status is None:
            raise RuntimeError(
                f"{self.label.upper()} profiler registry is unavailable; build the "
                f"required libraries with {self.diagnostic}"
            )
        try:
            clear()
        except Exception as error:
            raise RuntimeError(f"{self.label.upper()} profiler failed to clear counters") from error
        stats = self.read_stats(status)
        if self.label == "tsim" and stats != {"cycle_count": 0}:
            raise RuntimeError(f"TSIM profiler did not reset to {{'cycle_count': 0}}: {stats}")
        if self.label == "fsim":
            invalid = []
            for counter in REQUIRED_PROFILER_COUNTERS:
                value = stats.get(counter)
                if (
                    counter not in stats
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value != 0
                ):
                    invalid.append(counter)
            if invalid:
                raise RuntimeError(
                    "FSIM profiler did not reset required counters to zero: "
                    f"{invalid}; stats={stats}"
                )
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


def _validate_tsim_window_budget(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("TSIM window budget must be a positive integer")
    return value


def resolve_tsim_window_budget(value=None):
    """Resolve the TSIM-only representative-window budget from CLI or env."""
    if value is not None:
        return _validate_tsim_window_budget(value)
    configured = os.environ.get(TSIM_WINDOW_BUDGET_ENV)
    if configured is None:
        return DEFAULT_TSIM_WINDOW_BUDGET
    try:
        configured_value = int(configured)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{TSIM_WINDOW_BUDGET_ENV} must be a positive integer; received {configured!r}"
        ) from error
    return _validate_tsim_window_budget(configured_value)


def _manifest_records(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    samples_dir = manifest_path.parent.resolve(strict=True)
    if ".envs" in manifest_path.parts or ".envs" in samples_dir.parts:
        raise ValueError("manifest must not be read from .envs")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list) or len(samples) != 10:
        raise ValueError("sample manifest must contain exactly ten samples")
    records = []
    filenames = set()
    resolved_paths = set()
    for expected_order, item in enumerate(samples):
        if not isinstance(item, dict):
            raise ValueError("manifest samples must be objects")
        filename = item.get("filename")
        relative_path = Path(filename) if isinstance(filename, str) else None
        if (
            not isinstance(filename, str)
            or not filename
            or relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(f"manifest has an unsafe filename: {filename!r}")
        source_relative_path = item.get("source_relative_path")
        source_path = (
            Path(source_relative_path) if isinstance(source_relative_path, str) else None
        )
        if (
            not isinstance(source_relative_path, str)
            or not source_relative_path
            or source_path.is_absolute()
            or ".." in source_path.parts
            or source_relative_path != f"test/{filename}"
        ):
            raise ValueError(
                "manifest source_relative_path must be a safe path equal to "
                f"test/{filename!s}: {source_relative_path!r}"
            )
        if filename in filenames:
            raise ValueError("manifest sample filenames must be unique")
        filenames.add(filename)
        path = samples_dir / relative_path
        current = samples_dir
        for part in relative_path.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"manifest sample must not be a symlink: {path}")
        if not path.exists() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"manifest sample is missing or not a regular file: {path}")
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(samples_dir)
        except ValueError as error:
            raise ValueError(f"manifest sample path is outside samples directory: {path}") from error
        if ".envs" in resolved_path.parts:
            raise ValueError("manifest sample must not be read from .envs")
        if resolved_path in resolved_paths:
            raise ValueError("manifest sample paths must be unique")
        resolved_paths.add(resolved_path)

        order = item.get("order")
        if type(order) is not int or order != expected_order:
            raise ValueError("manifest sample order must be zero through nine")
        label = item.get("label")
        class_name = item.get("class_name")
        if type(label) is not int or (label, class_name) not in ((0, "normal"), (1, "anomaly")):
            raise ValueError("manifest labels must use 0=normal and 1=anomaly")
        sha256 = item.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"manifest sample sha256 is invalid: {filename!r}")
        byte_length = item.get("byte_length")
        if type(byte_length) is not int or byte_length < 0:
            raise ValueError(f"manifest sample byte_length is invalid: {filename!r}")
        sample_bytes = path.read_bytes()
        if len(sample_bytes) != byte_length:
            raise ValueError(f"manifest sample byte_length mismatch: {filename!r}")
        if hashlib.sha256(sample_bytes).hexdigest() != sha256:
            raise ValueError(f"manifest sample sha256 mismatch: {filename!r}")
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


def _execution_metadata(prepared, execution):
    metadata = _model_metadata(prepared)
    metadata["score"] = execution.summary["score_semantics"]
    metadata["execution"] = {
        "score_scope": execution.summary["score_scope"],
        "sampled": execution.summary["sampled"],
        "tsim_window_budget": execution.summary["tsim_window_budget"],
        "window_selection": (
            "deterministic evenly spaced representative windows"
            if execution.summary["sampled"]
            else "all feature windows"
        ),
    }
    return metadata


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


def _create_graph_executor(artifact):
    runtime = graph_executor.create(artifact.graph_json, artifact.module, artifact.device)
    runtime.load_params(artifact.params)
    return runtime


def _run_graph(artifact, input_data, output=None, graph_runtime=None):
    input_data = np.asarray(input_data)
    if input_data.shape != INPUT_SHAPE or input_data.dtype != np.dtype(INPUT_DTYPE):
        raise RuntimeError(f"input must have shape {INPUT_SHAPE} and dtype {INPUT_DTYPE}")
    if output is None:
        runtime = graph_runtime if graph_runtime is not None else _create_graph_executor(artifact)
        runtime.set_input(INPUT_NAME, input_data)
        runtime.run()
        output = runtime.get_output(0).numpy()
    output = np.asarray(output)
    if output.shape != OUTPUT_SHAPE:
        raise RuntimeError(f"output shape must be {OUTPUT_SHAPE}, received {output.shape}")
    if output.dtype != np.dtype(OUTPUT_DTYPE):
        raise RuntimeError(f"output dtype must be {OUTPUT_DTYPE}, received {output.dtype}")
    return output


def _score_feature_matrix(artifact, features, reuse_executor=False):
    graph_runtime = _create_graph_executor(artifact) if reuse_executor else None
    outputs = np.concatenate(
        tuple(
            _run_graph(artifact, row[None, :], graph_runtime=graph_runtime)
            if reuse_executor
            else _run_graph(artifact, row[None, :])
            for row in features
        ),
        axis=0,
    )
    if outputs.shape != features.shape:
        raise RuntimeError("reconstructed tensor shape differs from input feature matrix")
    return float(np.mean((features.astype(np.float32) - outputs) ** 2)), outputs.dtype


def _select_tsim_windows(features, window_budget):
    if window_budget is None:
        return features, False, FULL_SCORE_SCOPE
    window_budget = resolve_tsim_window_budget(window_budget)
    total_windows = features.shape[0]
    executed_windows = min(window_budget, total_windows)
    if executed_windows == total_windows:
        return features, False, FULL_SCORE_SCOPE
    indices = np.linspace(0, total_windows - 1, executed_windows, dtype=np.int64)
    return features[indices], True, TSIM_SCORE_SCOPE


def _score_record(artifact, record, window_budget=None, reuse_executor=False):
    features = np.asarray(load_sample(record.path))
    if features.ndim != 2 or features.shape[1] != INPUT_SHAPE[1] or features.dtype != np.float32:
        raise RuntimeError(f"{record.filename} preprocessing must return (N, 640) float32")
    selected, sampled, score_scope = _select_tsim_windows(features, window_budget)
    score, output_dtype = _score_feature_matrix(artifact, selected, reuse_executor=reuse_executor)
    return {
        "order": record.order,
        "filename": record.filename,
        "label": record.label,
        "class_name": record.class_name,
        "reference_score": score,
        "input_shape": INPUT_SHAPE,
        "output_shape": OUTPUT_SHAPE,
        "feature_shape": features.shape,
        "executed_feature_shape": selected.shape,
        "total_window_count": features.shape[0],
        "executed_window_count": selected.shape[0],
        "sampled": sampled,
        "score_scope": score_scope,
        "output_dtype": str(output_dtype),
        "_executed_features": selected,
    }


def _sample_score(artifact, record):
    item = _score_record(artifact, record)
    return item["reference_score"], item["feature_shape"], np.dtype(item["output_dtype"])


def _reference_raw(artifact, records, window_budget=None):
    return [
        _score_record(
            artifact.reference,
            record,
            window_budget=window_budget,
            reuse_executor=window_budget is not None,
        )
        for record in records
    ]


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
            executed_feature_shape=tuple(item["executed_feature_shape"]),
            total_window_count=int(item["total_window_count"]),
            executed_window_count=int(item["executed_window_count"]),
            sampled=bool(item["sampled"]), score_scope=item["score_scope"],
            reference_score=float(item["reference_score"]), mixed_score=item.get("mixed_score"),
        )
        for item in results
    )
    return completed, threshold


def _summary(samples, threshold, tsim_window_budget=None):
    sampled = any(item.sampled for item in samples)
    score_scope = TSIM_SCORE_SCOPE if sampled else FULL_SCORE_SCOPE
    score_semantics = (
        "mean squared reconstruction error over executed representative windows; "
        "not a complete audio-window MSE"
        if sampled
        else "mean squared reconstruction error over all feature windows"
    )
    return {
        "sample_count": len(samples),
        "normal_count": sum(item.label == 0 for item in samples),
        "anomaly_count": sum(item.label == 1 for item in samples),
        "predicted_anomaly_count": sum(item.predicted_label == 1 for item in samples),
        "anomaly_score_threshold": threshold,
        "score_semantics": score_semantics,
        "score_scope": score_scope,
        "tsim_window_budget": tsim_window_budget,
        "sampled": sampled,
    }


def execute_host(artifacts, records):
    """Execute only the CPU reference artifact; this path never imports simulator."""
    raw = [_score_record(artifacts.reference, record) for record in records]
    samples, threshold = _with_predictions(raw)
    return ExecutionSummary("host", artifacts.host_codegen, samples, _summary(samples, threshold), {}, {})


def _load_fsim():
    """Load VTA FSIM lazily, only immediately before mixed execution."""
    from vta.testing import simulator
    return simulator


def _validate_profiler_stats(stats):
    for counter in REQUIRED_PROFILER_COUNTERS:
        value = stats.get(counter, 0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or value <= 0
        ):
            raise RuntimeError(f"FSIM profiler counter {counter} must be positive")


@contextmanager
def _fsim_context():
    """Make the VTA backend device explicit around mixed graph execution."""
    yield tvm.ext_dev(0)


def execute_fsim(artifacts, records):
    """Run reference first, then enter the lazy FSIM backend for the mixed graph."""
    raw = [_score_record(artifacts.reference, record) for record in records]
    validate_mixed_symbols(artifacts.mixed.module, artifacts.vta_symbols)
    simulator = _load_fsim()
    session = _simulator_session("fsim")
    session.clear_and_validate(simulator)
    with _fsim_context():
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
    return ExecutionSummary("fsim", artifacts.host_codegen, samples, _summary(samples, threshold), stats, {})


def _execute_tsim_artifact(artifacts, records, raw, session, simulator, window_budget):
    validate_mixed_symbols(artifacts.mixed.module, artifacts.vta_symbols)
    session.clear_and_validate(simulator)
    for item, record in zip(raw, records):
        mixed_score, dtype = _score_feature_matrix(
            artifacts.mixed, item["_executed_features"], reuse_executor=True
        )
        if item["executed_feature_shape"] != item["_executed_features"].shape:
            raise RuntimeError(f"{record.filename} TSIM feature selection contract differs")
        if dtype != np.dtype(item["output_dtype"]):
            raise RuntimeError(f"{record.filename} mixed tensor contract differs")
        if not np.isclose(mixed_score, item["reference_score"], rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"{record.filename} reconstruction score differs")
        item["mixed_score"] = mixed_score
    stats = session.read_stats(simulator.stats)
    session.validate_activity(stats)
    samples, threshold = _with_predictions(raw)
    return ExecutionSummary(
        "tsim", artifacts.host_codegen, samples,
        _summary(samples, threshold, tsim_window_budget=window_budget), stats, {}
    )


def execute_tsim(artifacts, records, tsim_window_budget=None):
    """Run one host variant through the lazy VTA TSIM simulator."""
    window_budget = resolve_tsim_window_budget(tsim_window_budget)
    raw = _reference_raw(artifacts, records, window_budget=window_budget)
    session, simulator = _load_simulator("tsim")
    return _execute_tsim_artifact(artifacts, records, raw, session, simulator, window_budget)


def _execute_tsim_matrix(artifacts, records, reference_raw, session, simulator, window_budget):
    """Execute LLVM/C mixed graphs with one TSIM load and isolated counter windows."""
    executions = []
    for host_artifacts, raw in zip(artifacts, reference_raw):
        executions.append(
            _execute_tsim_artifact(
                host_artifacts, records, raw, session, simulator, window_budget
            )
        )
    return tuple(executions)


def deploy(build_dir=DEFAULT_BUILD_DIR, host_codegen=DEFAULT_HOST_CODEGEN, mode="host",
           manifest_path=MANIFEST_PATH, tsim_window_budget=None):
    _validate_mode(mode)
    window_budget = None
    if mode in ("fsim", "tsim"):
        _simulator_session(mode).validate_environment()
    if mode == "tsim":
        window_budget = resolve_tsim_window_budget(tsim_window_budget)
    prepared = prepare_model(MODEL_PATH)
    records = committed_sample_records(manifest_path)
    artifacts = build_host_artifacts(prepared, build_dir, host_codegen, mode)
    if mode == "host":
        execution = execute_host(artifacts, records)
    elif mode == "fsim":
        execution = execute_fsim(artifacts, records)
    else:
        execution = execute_tsim(artifacts, records, window_budget)
    execution = ExecutionSummary(execution.mode, execution.host_codegen, execution.samples,
                                 execution.summary, execution.profiler_stats,
                                 _execution_metadata(prepared, execution))
    return DeploymentResult(prepared, artifacts, execution)


def deploy_matrix(build_dir=DEFAULT_BUILD_DIR, mode="fsim", manifest_path=MANIFEST_PATH,
                  tsim_window_budget=None):
    _validate_mode(mode)
    if mode not in ("fsim", "tsim"):
        raise ValueError("host mode uses one selected host codegen; matrix mode is FSIM or TSIM")
    if mode == "tsim":
        return deploy_tsim_matrix(build_dir, manifest_path, tsim_window_budget)
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
                _execution_metadata(prepared, execution),
            )
        )
    return prepared, artifacts, tuple(executions)


def deploy_tsim_matrix(build_dir=DEFAULT_BUILD_DIR, manifest_path=MANIFEST_PATH,
                       tsim_window_budget=None):
    """Build LLVM/C bundles before one lazy TSIM load and sampled execution."""
    session = _simulator_session("tsim").validate_environment()
    window_budget = resolve_tsim_window_budget(tsim_window_budget)
    prepared = prepare_model(MODEL_PATH)
    records = committed_sample_records(manifest_path)
    artifacts = tuple(
        build_host_artifacts(prepared, build_dir, codegen, "tsim")
        for codegen in SUPPORTED_HOST_CODEGENS
    )
    reference_raw = tuple(
        _reference_raw(artifact, records, window_budget=window_budget) for artifact in artifacts
    )
    session, simulator = _load_simulator("tsim")
    executions = _execute_tsim_matrix(
        artifacts, records, reference_raw, session, simulator, window_budget
    )
    executions = tuple(
        ExecutionSummary(
            execution.mode,
            execution.host_codegen,
            execution.samples,
            execution.summary,
            execution.profiler_stats,
            _execution_metadata(prepared, execution),
        )
        for execution in executions
    )
    return prepared, artifacts, executions
