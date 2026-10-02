"""Migrate committed complete-fusion schedules against each real deployment."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
APP_ROOT = ROOT / "vta" / "apps" / "mlperf_tiny_benchmark"
ARTIFACTS = {
    "image_classification_v1": "image_classification_v1/tune/optimal/c4-full/best-manifest.json",
    "image_classification_v2": "image_classification_v2/tune/optimal/20261001T035246.904010Z/best-manifest.json",
    "anomaly_detection_v1": "anomaly_detection_v1/tune/optimal/20261001T174411.872234Z/best-manifest.json",
    "keyword_spotting_v1": "keyword_spotting_v1/tune/optimal/20261001T192416.141960Z/best-manifest.json",
    "streaming_wakeword_v1": "streaming_wakeword_v1/tune/optimal/20261001T205934.834798Z/best-manifest.json",
    "visual_wake_words_v1": "visual_wake_words_v1/tune/optimal/20261001T220401.514781Z/best-manifest.json",
}
DEPLOYMENT_REPORTS = {
    "image_classification_v1": "image_classification_v1/tune/deployment-c4-full.json",
    "image_classification_v2": "image_classification_v2/tune/deployment-full.json",
    "anomaly_detection_v1": "anomaly_detection_v1/tune/deployment-full.json",
    "keyword_spotting_v1": "keyword_spotting_v1/tune/deployment-full.json",
    "streaming_wakeword_v1": "streaming_wakeword_v1/tune/deployment-full.json",
    "visual_wake_words_v1": "visual_wake_words_v1/tune/deployment-full.json",
}


def _capture(model_id):
    from mlperf_tiny_benchmark.model_registry import MODEL_PIPELINES

    directory, subdirectory, filename = MODEL_PIPELINES[model_id]
    app = APP_ROOT / directory
    vta_python = str(ROOT / "vta" / "python")
    sys.path.insert(0, vta_python)
    if getattr(sys.modules.get("vta"), "__file__", None) is None:
        sys.modules.pop("vta", None)
    import vta.relay

    for name in ("model_pipeline", "graph_artifacts"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(app))
    try:
        pipeline = importlib.import_module("model_pipeline")
        prepared = pipeline.prepare_model(app / subdirectory / filename)
        from common.deployment_compute import capture_deployment_compute

        return capture_deployment_compute(
            prepared.mixed_module, model_id, prepared.imported.model_sha256
        )
    finally:
        sys.path.pop(0)
        for name in ("model_pipeline", "graph_artifacts"):
            sys.modules.pop(name, None)
        sys.path.remove(vta_python)


@pytest.mark.parametrize("model_id", tuple(ARTIFACTS))
def test_committed_full_fusion_artifacts_migrate_only_for_actual_layer_identities(
    model_id, tmp_path
):
    from common.schedule import load_schedule_snapshot, migrate_legacy_full_fusion

    manifest = APP_ROOT / ARTIFACTS[model_id]
    deployment = _capture(model_id)
    output = tmp_path / f"{model_id}.log"

    snapshot = migrate_legacy_full_fusion(manifest, deployment, output)
    loaded = load_schedule_snapshot(output, deployment)
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

    assert set(snapshot.selected) == set(range(len(deployment.layers)))
    assert set(loaded.selected) == set(snapshot.selected)
    assert all(selected.measured for selected in loaded.selected.values())
    assert metadata["provenance"]["legacy_manifest_sha256"]
    source = json.loads(manifest.read_text(encoding="utf-8"))
    assert {
        (entry["occurrence"], entry["symbol"]) for entry in metadata["occurrences"]
    } == {(layer.occurrence, layer.symbol) for layer in deployment.layers}
    assert len(source["entries"]) == len(metadata["occurrences"])

    report_path = APP_ROOT / DEPLOYMENT_REPORTS[model_id]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_cycles = {row["symbol"]: row["autotvm_cycles"] for row in report["occurrences"]}
    for row in metadata["occurrences"]:
        measured = next(result for result in row["measurement"]["results"] if result is not None)
        assert measured["costs"] == [report_cycles[row["symbol"]]]

    if model_id == "visual_wake_words_v1":
        calculator_path = ROOT / "scripts" / "mac_utilization.py"
        spec = importlib.util.spec_from_file_location("migration_mac_calculator", calculator_path)
        calculator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calculator)
        calculated = calculator.calculate_deployment_report(report)
        calculator_cycles = {
            row["symbol"]: row["autotvm_cycles_per_invocation"]
            for row in calculated["occurrences"]
        }
        assert calculator_cycles == report_cycles


def test_historical_migration_rejects_missing_single_call_protocol(tmp_path):
    from common.schedule import migrate_legacy_full_fusion

    source = APP_ROOT / ARTIFACTS["streaming_wakeword_v1"]
    value = json.loads(source.read_text(encoding="utf-8"))
    value.pop("measurement_protocol")
    missing_protocol = tmp_path / "best-manifest.json"
    missing_protocol.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="single-call TSIM measurement protocol is missing"):
        migrate_legacy_full_fusion(
            missing_protocol, _capture("streaming_wakeword_v1"), tmp_path / "out.log"
        )


def test_historical_migration_rejects_foreign_occurrence_before_reading_records(tmp_path):
    from common.schedule import migrate_legacy_full_fusion

    source = APP_ROOT / ARTIFACTS["streaming_wakeword_v1"]
    value = json.loads(source.read_text(encoding="utf-8"))
    value["entries"][0]["symbol"] = "foreign_layer"
    tampered = tmp_path / "best-manifest.json"
    tampered.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy fusion identity does not match actual layer"):
        migrate_legacy_full_fusion(
            tampered, _capture("streaming_wakeword_v1"), tmp_path / "out.log"
        )


def test_historical_migration_rejects_a_tampered_fusion_constant(tmp_path):
    from common.schedule import migrate_legacy_full_fusion

    source = APP_ROOT / ARTIFACTS["streaming_wakeword_v1"]
    manifest = json.loads(source.read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    result_path = source.parent / entry["result_json"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["fusion_identity"]["shift"] += 1
    (tmp_path / entry["result_json"]).write_text(json.dumps(result), encoding="utf-8")
    tampered = tmp_path / "best-manifest.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy fusion identity does not match actual layer"):
        migrate_legacy_full_fusion(
            tampered, _capture("streaming_wakeword_v1"), tmp_path / "out.log"
        )


def test_historical_migration_rejects_a_config_that_differs_from_native_record(tmp_path):
    from common.schedule import migrate_legacy_full_fusion

    source = APP_ROOT / ARTIFACTS["streaming_wakeword_v1"]
    manifest = json.loads(source.read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    result_path = source.parent / entry["result_json"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["conv_config"]["index"] += 1
    (tmp_path / entry["result_json"]).write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / entry["native_record"]).write_bytes(
        (source.parent / entry["native_record"]).read_bytes()
    )
    tampered = tmp_path / "best-manifest.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="config differs from its native record"):
        migrate_legacy_full_fusion(
            tampered, _capture("streaming_wakeword_v1"), tmp_path / "out.log"
        )


def test_historical_migration_rejects_tampered_native_record_without_touching_source(
    tmp_path
):
    from common.schedule import migrate_legacy_full_fusion

    source = APP_ROOT / ARTIFACTS["streaming_wakeword_v1"]
    source_value = json.loads(source.read_text(encoding="utf-8"))
    temp_manifest = tmp_path / "best-manifest.json"
    entry = source_value["entries"][0]
    original_native = (source.parent / entry["native_record"]).read_bytes()
    (tmp_path / entry["native_record"]).write_bytes(b"tampered source record\n")
    (tmp_path / entry["result_json"]).write_bytes(
        (source.parent / entry["result_json"]).read_bytes()
    )
    temp_manifest.write_text(json.dumps(source_value), encoding="utf-8")

    with pytest.raises(ValueError, match="native record hash mismatch"):
        migrate_legacy_full_fusion(
            temp_manifest, _capture("streaming_wakeword_v1"), tmp_path / "out.log"
        )
    assert (source.parent / entry["native_record"]).read_bytes() == original_native
