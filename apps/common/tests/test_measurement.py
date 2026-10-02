"""Candidate isolation and output checks through actual VTA lowering."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tvm


APP_ROOT = Path(__file__).resolve().parents[2] / "mlperf_tiny_benchmark" / "image_classification_v2"
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"


@pytest.fixture(scope="module")
def deployment():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    if getattr(sys.modules.get("vta"), "__file__", None) is None:
        sys.modules.pop("vta", None)
    import vta.relay

    spec = importlib.util.spec_from_file_location("measurement_model_pipeline", APP_ROOT / "model_pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    prepared = pipeline.prepare_model(MODEL_PATH)
    from common.deployment_compute import capture_deployment_compute

    captured = capture_deployment_compute(
        prepared.mixed_module,
        "image_classification_v2",
        prepared.imported.model_sha256,
    )
    return prepared, captured


@pytest.fixture(scope="module")
def real_activation(deployment, tmp_path_factory):
    prepared, captured = deployment
    sys.path.insert(0, str(APP_ROOT))
    spec = importlib.util.spec_from_file_location("measurement_model_runtime", APP_ROOT / "runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)

    artifacts = runtime.build_host_artifacts(
        prepared, tmp_path_factory.mktemp("default-fsim"), host_codegen="llvm", simulator="fsim"
    )
    runtime._load_fsim()
    from tvm.contrib.debugger import debug_executor

    graph_executor = debug_executor.create(
        artifacts.mixed.graph_json, artifacts.mixed.module, artifacts.mixed.device
    )
    graph_executor.load_params(artifacts.mixed.params)
    graph_executor.set_input(runtime.INPUT_NAME, runtime.load_sample(runtime.committed_sample_paths()[0]))
    graph_executor._run_per_layer()

    graph = json.loads(artifacts.mixed.graph_json)
    nodes = graph["nodes"]
    node_index = next(
        index for index, node in enumerate(nodes)
        if node.get("attrs", {}).get("func_name") == captured.layers[0].symbol
    )
    source_index, source_output, _ = nodes[node_index]["inputs"][0]
    output_index = source_output
    output_tensors = graph_executor.debug_datum.get_output_tensors()
    source_name = nodes[source_index]["name"]
    activation = output_tensors[
        f"{source_name}____topo-index:{source_index}____output-num:{output_index}"
    ].numpy()
    expected = output_tensors[
        f"{nodes[node_index]['name']}____topo-index:{node_index}____output-num:0"
    ].numpy()
    return runtime, captured.layers[0], activation, expected


def test_candidate_configs_have_distinct_identity_and_fresh_workers(real_activation):
    from common.measurement import measure_candidate

    _, layer, activation, default_output = real_activation
    config_indices = [[0] * len(layer.config_spaces), [0] * len(layer.config_spaces)]
    conv_index = next(
        index for index, (template, _, _, _) in enumerate(layer.config_spaces)
        if template == "conv2d_packed.vta"
    )
    config_indices[1][conv_index] = 1

    first = measure_candidate(layer, activation, config_indices[0], "fsim")
    second = measure_candidate(layer, activation, config_indices[1], "fsim")

    assert first["config_identity"] != second["config_identity"]
    assert first["worker_pid"] != second["worker_pid"]
    np.testing.assert_array_equal(first["output"], default_output)
    np.testing.assert_array_equal(second["output"], default_output)


def test_invalid_candidate_and_exception_restore_dispatch_context(real_activation):
    from tvm.autotvm.task.dispatcher import DispatchContext

    from common.measurement import _SelectedWorkloads, measure_candidate

    _, layer, activation, _ = real_activation
    with pytest.raises(ValueError, match="outside"):
        measure_candidate(
            layer, activation, [len(space) for _, _, _, space in layer.config_spaces], "fsim"
        )

    before = DispatchContext.current
    with pytest.raises(RuntimeError, match="measurement test"):
        with _SelectedWorkloads(()) as selected:
            assert selected is not None
            raise RuntimeError("measurement test")
    assert DispatchContext.current is before

    with pytest.raises(ValueError, match="activation must have shape"):
        measure_candidate(layer, np.zeros((1,), dtype="int8"), [0] * len(layer.config_spaces), "fsim")


def test_failed_candidate_worker_is_discarded_before_next_candidate(real_activation):
    from common.measurement import measure_candidate

    _, layer, activation, expected = real_activation
    invalid = [0] * len(layer.config_spaces)
    conv_index = next(
        index for index, (template, _, _, _) in enumerate(layer.config_spaces)
        if template == "conv2d_packed.vta"
    )
    invalid[conv_index] = len(layer.config_spaces[conv_index][3]) - 1

    with pytest.raises(RuntimeError, match="candidate worker failed"):
        measure_candidate(layer, activation, invalid, "fsim")

    recovered = measure_candidate(layer, activation, [0] * len(layer.config_spaces), "fsim")
    np.testing.assert_array_equal(recovered["output"], expected)
