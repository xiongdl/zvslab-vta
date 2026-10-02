"""One-call TSIM evidence for an actual IC V2 layer candidate."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[2] / "mlperf_tiny_benchmark" / "image_classification_v2"
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"


@pytest.fixture(scope="module")
def actual_layer_activation(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    sys.path.insert(0, str(APP_ROOT))
    if getattr(sys.modules.get("vta"), "__file__", None) is None:
        sys.modules.pop("vta", None)
    import vta.relay

    pipeline_spec = importlib.util.spec_from_file_location("tsim_model_pipeline", APP_ROOT / "model_pipeline.py")
    pipeline = importlib.util.module_from_spec(pipeline_spec)
    sys.modules[pipeline_spec.name] = pipeline
    pipeline_spec.loader.exec_module(pipeline)
    runtime_spec = importlib.util.spec_from_file_location("tsim_model_runtime", APP_ROOT / "runtime.py")
    runtime = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime
    runtime_spec.loader.exec_module(runtime)

    prepared = pipeline.prepare_model(MODEL_PATH)
    from common.deployment_compute import capture_deployment_compute

    deployment = capture_deployment_compute(
        prepared.mixed_module, "image_classification_v2", prepared.imported.model_sha256
    )
    artifacts = runtime.build_host_artifacts(
        prepared,
        tmp_path_factory.mktemp("default-tsim"),
        host_codegen="llvm",
        simulator="tsim",
    )
    runtime._load_simulator("tsim")
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
        if node.get("attrs", {}).get("func_name") == deployment.layers[0].symbol
    )
    source_index, source_output, _ = nodes[node_index]["inputs"][0]
    outputs = graph_executor.debug_datum.get_output_tensors()
    activation = outputs[
        f"{nodes[source_index]['name']}____topo-index:{source_index}____output-num:{source_output}"
    ].numpy()
    expected = outputs[
        f"{nodes[node_index]['name']}____topo-index:{node_index}____output-num:0"
    ].numpy()
    return deployment.layers[0], activation, expected


def test_actual_layer_candidate_uses_one_counted_tsim_invocation(actual_layer_activation):
    from common.measurement import measure_candidate

    layer, activation, expected = actual_layer_activation
    result = measure_candidate(layer, activation, [0] * len(layer.config_spaces), "tsim")

    assert isinstance(result["cycles"], int) and result["cycles"] > 0
    assert result["protocol"] == {
        "name": "tsim_single_call",
        "version": 1,
        "counted_invocations": 1,
        "warmup_excluded": True,
    }
    np.testing.assert_array_equal(result["output"], expected)


def test_invalid_tsim_candidate_is_rejected_before_worker(actual_layer_activation):
    from common.measurement import measure_candidate

    layer, activation, _ = actual_layer_activation
    indices = [0] * len(layer.config_spaces)
    indices[0] = len(layer.config_spaces[0][3])
    with pytest.raises(ValueError, match="outside"):
        measure_candidate(layer, activation, indices, "tsim")
