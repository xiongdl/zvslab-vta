"""Actual prepared-graph coverage tests for shared complete VTA fusions."""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from tvm import relay

APP_ROOT = Path(__file__).resolve().parents[1]
FUSED_TASKS_PATH = APP_ROOT / "fused_tasks.py"
_SPEC = importlib.util.spec_from_file_location("mlperf_tiny_fused_tasks", FUSED_TASKS_PATH)
fused_tasks = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fused_tasks
_SPEC.loader.exec_module(fused_tasks)
MODEL_OCCURRENCES = {
    "anomaly_detection_v1": 9,
    "keyword_spotting_v1": 4,
    "streaming_wakeword_v1": 1,
    "visual_wake_words_v1": 13,
}


def _prepare_model(model):
    app = APP_ROOT / model
    previous_path = list(sys.path)
    for name in ("runtime", "model_pipeline", "graph_artifacts"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(app))
    try:
        runtime = importlib.import_module("runtime")
        return runtime.prepare_model(runtime.MODEL_PATH)
    finally:
        for name in ("runtime", "model_pipeline", "graph_artifacts"):
            sys.modules.pop(name, None)
        sys.path[:] = previous_path


@pytest.mark.parametrize("model,expected_count", MODEL_OCCURRENCES.items())
def test_extracts_complete_fusions_from_actual_prepared_graph(model, expected_count):
    prepared = _prepare_model(model)
    identities = fused_tasks.extract_fused_identities(prepared)

    assert len(identities) == expected_count
    assert tuple(identity.symbol for identity in identities) == prepared.routing.symbols
    assert tuple(identity.occurrence for identity in identities) == tuple(range(expected_count))
    assert all(identity.conv_workload[0] == fused_tasks.TASK_NAME for identity in identities)
    assert len({identity.sha256 for identity in identities}) == expected_count
    assert all(identity.output_dtype == "int8" for identity in identities)
    assert all(-128 <= identity.clip_min < identity.clip_max <= 127 for identity in identities)
    assert all(identity.shift >= 0 for identity in identities)
    assert all(len(identity.conv_workload[1][1]) == 6 for identity in identities)
    assert all(identity.conv_workload[1][2] == "int8" for identity in identities)
    assert all(identity.conv_workload[2][2] == "int8" for identity in identities)
    assert all(identity.conv_workload[6].startswith("NCHW") for identity in identities)
    assert all(len(identity.bias_values) == (1 if not identity.bias_shape else identity.bias_shape[0])
               for identity in identities)

    # The pipeline routing report includes host-side operators explicitly;
    # extraction must preserve it outside the VTA occurrence list.
    inventory = fused_tasks.host_inventory(prepared)
    if inventory["host_convolution_count"] is not None:
        assert inventory["host_convolution_count"] >= 0
    assert inventory["host_operator_names"] == tuple(prepared.routing.host_operator_names)
    if inventory["host_dense_count"] is not None:
        assert inventory["host_dense_count"] == 0

    repeated = fused_tasks.extract_fused_identities(prepared)
    assert [item.canonical_json() for item in repeated] == [
        item.canonical_json() for item in identities
    ]


def test_unsupported_fusion_arithmetic_is_rejected_with_symbol():
    value = relay.var("value", shape=(1,), dtype="int8")
    with pytest.raises(ValueError, match="test_symbol.*expected final cast"):
        fused_tasks._extract_composite(relay.nn.relu(value), "test_symbol", 0)


def test_identity_round_trip_preserves_tensor_bias_constants():
    prepared = _prepare_model("keyword_spotting_v1")
    identities = fused_tasks.extract_fused_identities(prepared)
    identity = next(item for item in identities if item.bias_shape)

    decoded = fused_tasks.FusedOperatorIdentity.from_json(identity.canonical_json())

    assert decoded == identity
    assert len(identity.bias_values) == identity.bias_shape[0]
    assert identity.bias_axis == 3
    assert identity.bias_dtype == "int32"


def test_actual_tensor_bias_task_instantiates_complete_template():
    import vta

    prepared = _prepare_model("keyword_spotting_v1")
    identity = fused_tasks.extract_fused_identities(prepared)[0]
    task = fused_tasks.create_task(identity, vta.get_env().target)

    schedule, tensors = task.instantiate(task.config_space.get(0))

    assert schedule is not None
    assert len(tensors) == 4  # result, data, kernel, and actual vector bias input


def test_blocked_nchw_vector_bias_indexes_channel_outer_and_inner_lanes():
    index = fused_tasks._physical_bias_index(
        "NCHW1n8c", 3, (0, 3, 0, 4, 0, 7), (1, 8, 25, 5, 1, 8), 64
    )

    assert index == 31
