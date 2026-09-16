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

from graph_artifacts import export_graph_bundle
from model_pipeline import MODEL_SHA256, load_sample, prepare_model


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet.tflite"
MANIFEST_PATH = APP_ROOT / "samples" / "manifest.json"
DEFAULT_OUTPUT_DIR = APP_ROOT / "build"
REFERENCE_ARTIFACT_STEM = "mlperf_resnet_llvm"
MIXED_ARTIFACT_STEM = "mlperf_resnet_vta"
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
class FsimMatrixResult:
    """All host variants from one prepared FSIM deployment matrix."""

    prepared: object
    artifacts: tuple
    executions: tuple


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


def _matrix_artifact_root(output_dir, host_codegen):
    _validate_host_codegen(host_codegen)
    return Path(output_dir) / f"{host_codegen}-fsim"


def _mixed_target(host_codegen=DEFAULT_HOST_CODEGEN):
    _validate_host_codegen(host_codegen)
    environment = vta.get_env()
    host = environment.target_host if host_codegen == "llvm" else tvm.target.Target("c")
    return tvm.target.Target("vta", host=host)


def build_host_artifacts(prepared, output_dir, host_codegen=DEFAULT_HOST_CODEGEN, simulator="fsim"):
    """Build, export, and reload both standard host libraries without FSIM."""
    _validate_host_codegen(host_codegen)
    if simulator != "fsim":
        raise ValueError(f"unsupported simulator {simulator!r}; only fsim is implemented")
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
    mixed_target = _mixed_target() if host_codegen == DEFAULT_HOST_CODEGEN else _mixed_target(host_codegen)
    if host_codegen == "c":
        with vta.build_config(config={"tir.disable_vectorize": True}):
            mixed_factory = relay.build(prepared.mixed_module, target=mixed_target)
    else:
        with vta.build_config():
            mixed_factory = relay.build(prepared.mixed_module, target=mixed_target)

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
    from vta.testing import simulator

    if not simulator.enabled():
        raise RuntimeError(
            "VTA FSIM is unavailable; run ./scripts/build_vta_lib.sh --target libvta_fsim"
        )
    return simulator


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
    simulator = _load_fsim()
    simulator.clear_stats()
    cleared_stats = simulator.stats()
    if any(value != 0 for value in cleared_stats.values()):
        raise RuntimeError(f"FSIM profiler did not reset to zero: {cleared_stats}")

    comparisons = []
    for (path, input_data), (_, reference_output) in zip(inputs, reference_outputs):
        mixed_output = _run_graph(artifacts.mixed, input_data)
        comparisons.append(compare_outputs(path, reference_output, mixed_output))

    profiler_stats = simulator.stats()
    validate_profiler_stats(profiler_stats)
    return ExecutionSummary(comparisons=tuple(comparisons), profiler_stats=dict(profiler_stats))


def _execute_fsim_matrix(artifacts, sample_paths):
    """Run reference graphs before FSIM, then each mixed graph independently."""
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

    simulator = _load_fsim()
    executions = []
    for host_artifacts in artifacts:
        host = host_artifacts.host_codegen
        validate_mixed_symbols(host_artifacts.mixed.module, host_artifacts.vta_symbols)
        simulator.clear_stats()
        cleared_stats = simulator.stats()
        if any(value != 0 for value in cleared_stats.values()):
            raise RuntimeError(f"{host} FSIM profiler did not reset to zero: {cleared_stats}")

        comparisons = []
        for (path, input_data), (_, reference_output) in zip(
            inputs, reference_outputs[host]
        ):
            try:
                mixed_output = _run_graph(host_artifacts.mixed, input_data)
                comparisons.append(compare_outputs(path, reference_output, mixed_output))
            except Exception as error:
                raise RuntimeError(f"{host} mixed FSIM failed for sample {path.name}: {error}") from error

        profiler_stats = simulator.stats()
        try:
            validate_profiler_stats(profiler_stats)
        except RuntimeError as error:
            raise RuntimeError(f"{host} mixed FSIM profiler validation failed: {error}") from error
        executions.append(
            ExecutionSummary(comparisons=tuple(comparisons), profiler_stats=dict(profiler_stats))
        )
    return tuple(executions)


def deploy_fsim_matrix(output_dir=DEFAULT_OUTPUT_DIR, host_codegens=SUPPORTED_HOST_CODEGENS):
    """Build and execute the complete ordered LLVM/C FSIM matrix."""
    host_codegens = _validate_host_codegens(host_codegens)
    prepared = prepare_model(MODEL_PATH)
    print(f"VTA partitions: {len(prepared.routing.symbols)}")
    print(f"VTA symbols: {', '.join(prepared.routing.symbols)}")
    print(f"VTA composites: {', '.join(prepared.routing.composite_names)}")
    print(f"Host operators: {', '.join(prepared.routing.host_operator_names)}")

    # Build and reload all four bundles before importing the simulator.
    artifacts = tuple(
        build_host_artifacts(
            prepared,
            _matrix_artifact_root(output_dir, host_codegen),
            host_codegen=host_codegen,
            simulator="fsim",
        )
        for host_codegen in host_codegens
    )
    executions = _execute_fsim_matrix(artifacts, committed_sample_paths())
    return FsimMatrixResult(
        prepared=prepared,
        artifacts=artifacts,
        executions=executions,
    )


def deploy(output_dir=DEFAULT_OUTPUT_DIR, host_codegen=DEFAULT_HOST_CODEGEN):
    """Perform the complete fixed HOST deployment and return its evidence."""
    _validate_host_codegen(host_codegen)
    prepared = prepare_model(MODEL_PATH)
    print(f"VTA partitions: {len(prepared.routing.symbols)}")
    print(f"VTA symbols: {', '.join(prepared.routing.symbols)}")
    print(f"VTA composites: {', '.join(prepared.routing.composite_names)}")
    print(f"Host operators: {', '.join(prepared.routing.host_operator_names)}")
    artifacts = build_host_artifacts(prepared, output_dir, host_codegen=host_codegen)
    execution = execute_samples(artifacts, committed_sample_paths())
    return DeploymentResult(prepared=prepared, artifacts=artifacts, execution=execution)
