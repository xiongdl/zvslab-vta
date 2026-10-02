"""Integrity and occurrence coverage for deployable AutoTVM snapshots."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2] / "mlperf_tiny_benchmark" / "image_classification_v2"
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet_large_float.tflite"


@pytest.fixture(scope="module")
def deployment():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    if getattr(sys.modules.get("vta"), "__file__", None) is None:
        sys.modules.pop("vta", None)
    import vta.relay

    spec = importlib.util.spec_from_file_location("schedule_artifact_model_pipeline", APP_ROOT / "model_pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    prepared = pipeline.prepare_model(MODEL_PATH)

    from common.deployment_compute import capture_deployment_compute

    return capture_deployment_compute(
        prepared.mixed_module,
        "image_classification_v2",
        prepared.imported.model_sha256,
    )


def _indices(layer, conv_index=0):
    result = []
    for template, _, _, space in layer.config_spaces:
        result.append(conv_index if template == "conv2d_packed.vta" else 0)
    return result


def test_no_input_and_none_select_defaults(deployment):
    from common.schedule import load_schedule_snapshot

    assert load_schedule_snapshot(None, deployment).selected == {}
    assert load_schedule_snapshot("none", deployment).selected == {}


def test_partial_snapshot_retains_default_occurrences_and_repeated_workloads(deployment, tmp_path):
    from common.schedule import export_schedule_snapshot, load_schedule_snapshot

    # IC V2 has repeated schedule workloads at distinct occurrences; the
    # snapshot binds configurations to occurrences and can choose separately.
    first = deployment.layers[0]
    repeated = next(
        layer for layer in deployment.layers[1:]
        if tuple((t, repr(w), target) for t, w, target, _ in layer.config_spaces)
        == tuple((t, repr(w), target) for t, w, target, _ in first.config_spaces)
    )
    conv_space = next(space for template, _, _, space in first.config_spaces if template == "conv2d_packed.vta")
    assert len(conv_space) > 1
    path = tmp_path / "partial.log"
    snapshot = export_schedule_snapshot(
        path,
        deployment,
        {first.occurrence: _indices(first, 0), repeated.occurrence: _indices(repeated, 1)},
    )
    reloaded = load_schedule_snapshot(path, deployment)

    assert set(snapshot.selected) == {first.occurrence, repeated.occurrence}
    conv_index = next(i for i, entry in enumerate(first.config_spaces) if entry[0] == "conv2d_packed.vta")
    assert reloaded.selected[first.occurrence].configs[conv_index].to_json_dict() != reloaded.selected[repeated.occurrence].configs[conv_index].to_json_dict()
    coverage = reloaded.coverage(deployment)
    assert sum(is_selected for _, _, is_selected in coverage) == 2
    assert all(is_selected == (occurrence in {first.occurrence, repeated.occurrence}) for occurrence, _, is_selected in coverage)


def test_complete_snapshot_round_trips_native_config_records(deployment, tmp_path):
    from common.schedule import export_schedule_snapshot, load_schedule_snapshot

    selections = {layer.occurrence: _indices(layer, layer.occurrence % 2) for layer in deployment.layers}
    path = tmp_path / "complete.log"
    exported = export_schedule_snapshot(path, deployment, selections)
    loaded = load_schedule_snapshot(path, deployment)

    assert len(loaded.selected) == len(deployment.layers)
    for layer in deployment.layers:
        assert [config.to_json_dict() for config in loaded.selected[layer.occurrence].configs] == [
            layer.config_spaces[index][3].get(selected).to_json_dict()
            for index, selected in enumerate(selections[layer.occurrence])
        ]
        assert loaded.selected[layer.occurrence].measured is False
        assert loaded.selected[layer.occurrence].measurement is None


def test_measured_candidate_carries_backend_protocol_and_units(deployment, tmp_path):
    from common.schedule import export_schedule_snapshot

    layer = deployment.layers[0]
    selection = _indices(layer)
    provenance = {
        "backend": "tsim",
        "protocol": "tsim_single_call_v1",
        "units": "cycles",
        "results": [
            {"costs": [12.0], "error_no": 0, "all_cost": 0.5, "timestamp": 1.0},
            {"costs": [34.0], "error_no": 0, "all_cost": 0.5, "timestamp": 1.0},
        ],
    }
    snapshot = export_schedule_snapshot(
        tmp_path / "measured.log",
        deployment,
        {layer.occurrence: selection},
        measurements={layer.occurrence: provenance},
    )
    assert snapshot.selected[layer.occurrence].measured is True
    assert snapshot.selected[layer.occurrence].measurement == provenance

    with pytest.raises(ValueError, match="units are required"):
        export_schedule_snapshot(
            tmp_path / "bad-provenance.log",
            deployment,
            {layer.occurrence: selection},
            measurements={layer.occurrence: {"backend": "tsim", "protocol": "one-call", "results": provenance["results"]}},
        )


def test_missing_sidecar_swapped_pair_and_tampered_log_fail(deployment, tmp_path):
    from common.schedule import export_schedule_snapshot, load_schedule_snapshot

    layer = deployment.layers[0]
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    export_schedule_snapshot(first, deployment, {layer.occurrence: _indices(layer)})
    export_schedule_snapshot(second, deployment, {layer.occurrence: _indices(layer, 1)})

    (tmp_path / "first.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_schedule_snapshot(first, deployment)
    (tmp_path / "first.json").write_bytes((tmp_path / "second.json").read_bytes())
    with pytest.raises(ValueError, match="hash"):
        load_schedule_snapshot(first, deployment)
    (tmp_path / "first.json").write_bytes((tmp_path / "second.json").read_bytes())
    first.write_bytes(second.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_schedule_snapshot(first, deployment)


def test_duplicate_unknown_and_swapped_occurrence_records_fail(deployment, tmp_path):
    from common.schedule import export_schedule_snapshot, load_schedule_snapshot

    first, second = deployment.layers[:2]
    path = tmp_path / "schedule.log"
    export_schedule_snapshot(path, deployment, {first.occurrence: _indices(first)})
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())

    duplicate = dict(metadata["occurrences"][0])
    metadata["occurrences"].append(duplicate)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="duplicate occurrence"):
        load_schedule_snapshot(path, deployment)

    metadata["occurrences"] = [dict(metadata["occurrences"][0], occurrence=99)]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unknown occurrence"):
        load_schedule_snapshot(path, deployment)

    metadata["occurrences"] = [dict(metadata["occurrences"][0], occurrence=second.occurrence)]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="symbol does not match"):
        load_schedule_snapshot(path, deployment)


def test_invalid_candidate_configs_are_rejected(deployment, tmp_path):
    from common.schedule import export_schedule_snapshot

    layer = deployment.layers[0]
    invalid = _indices(layer)
    invalid[0] = len(layer.config_spaces[0][3])
    with pytest.raises(ValueError, match="outside"):
        export_schedule_snapshot(tmp_path / "invalid.log", deployment, {layer.occurrence: invalid})

    with pytest.raises(ValueError, match="unknown layer occurrences"):
        export_schedule_snapshot(tmp_path / "unknown.log", deployment, {999: [0, 0]})


def test_metadata_identity_tampering_fails_before_schedule_use(deployment, tmp_path):
    from common.schedule import export_schedule_snapshot, load_schedule_snapshot

    layer = deployment.layers[0]
    path = tmp_path / "identity.log"
    export_schedule_snapshot(path, deployment, {layer.occurrence: _indices(layer)})
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata["geometry_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="geometry identity"):
        load_schedule_snapshot(path, deployment)
