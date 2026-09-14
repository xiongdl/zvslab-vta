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
    """The pure LLVM and mixed VTA/LLVM artifacts."""

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


def _mixed_target():
    environment = vta.get_env()
    return tvm.target.Target("vta", host=environment.target_host)


def build_host_artifacts(prepared, output_dir):
    """Build, export, and reload both standard host libraries without FSIM."""
    if prepared.reference_module is not prepared.quantized_module:
        raise RuntimeError("pure LLVM build must use the exact shared quantized module object")

    output_dir = Path(output_dir)
    reference_factory = relay.build(prepared.reference_module, target="llvm")
    with vta.build_config():
        mixed_factory = relay.build(prepared.mixed_module, target=_mixed_target())

    reference_bundle = export_graph_bundle(
        reference_factory,
        output_dir,
        "reference",
        artifact_name=REFERENCE_ARTIFACT_STEM,
        artifact_role="reference",
        model_sha256=getattr(getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256),
        host_codegen="llvm",
        simulator="fsim",
        forbidden_vta_symbols=prepared.routing.symbols,
    )
    mixed_bundle = export_graph_bundle(
        mixed_factory,
        output_dir,
        "mixed",
        artifact_name=MIXED_ARTIFACT_STEM,
        artifact_role="mixed",
        model_sha256=getattr(getattr(prepared, "imported", None), "model_sha256", MODEL_SHA256),
        host_codegen="llvm",
        simulator="fsim",
        expected_vta_symbols=prepared.routing.symbols,
    )

    return HostArtifacts(
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


def deploy(output_dir=DEFAULT_OUTPUT_DIR):
    """Perform the complete fixed HOST deployment and return its evidence."""
    prepared = prepare_model(MODEL_PATH)
    print(f"VTA partitions: {len(prepared.routing.symbols)}")
    print(f"VTA symbols: {', '.join(prepared.routing.symbols)}")
    print(f"VTA composites: {', '.join(prepared.routing.composite_names)}")
    print(f"Host operators: {', '.join(prepared.routing.host_operator_names)}")
    artifacts = build_host_artifacts(prepared, output_dir)
    execution = execute_samples(artifacts, committed_sample_paths())
    return DeploymentResult(prepared=prepared, artifacts=artifacts, execution=execution)
