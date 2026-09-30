"""Tests for the validated per-layer VTA MAC utilization core."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "mac_utilization.py"


@pytest.fixture(scope="module")
def utilization():
    spec = importlib.util.spec_from_file_location("mlperf_tiny_mac_utilization", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(template, workload, *, error_no=0, costs=(64,)):
    return (
        SimpleNamespace(task=SimpleNamespace(name=template, workload=workload)),
        SimpleNamespace(error_no=error_no, costs=costs),
    )


def test_vta_64mac_geometry_derives_64_macs_per_cycle(utilization, tmp_path):
    config = tmp_path / "vta_64mac.json"
    config.write_text(
        '{"LOG_BATCH": 0, "LOG_BLOCK": 3}\n', encoding="utf-8"
    )

    assert utilization.peak_macs_per_cycle(config) == 64


def test_invalid_geometry_is_rejected(utilization, tmp_path):
    config = tmp_path / "bad.json"
    config.write_text('{"LOG_BATCH": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="LOG_BLOCK"):
        utilization.peak_macs_per_cycle(config)


def test_flops_convert_to_macs_and_useful_utilization(utilization):
    values = utilization.calculate_utilization(2048, 16, 64)

    assert values == {
        "flop_count": 2048,
        "mac_count": 1024,
        "best_trial_cycles": 16,
        "peak_macs_per_cycle": 64,
        "useful_mac_utilization": 1.0,
        "useful_mac_utilization_percent": 100.0,
    }


def test_minimum_successful_integral_cycle_cost_is_selected(utilization):
    workload = ("conv2d_packed.vta", (1, 2, 3))
    workload_id = utilization.workload_sha256(workload)
    records = [
        _record("conv2d_packed.vta", workload, error_no=1, costs=(1,)),
        _record("conv2d_packed.vta", workload, costs=(100, 90)),
        _record("conv2d_packed.vta", workload, costs=(80,)),
    ]

    assert utilization.minimum_successful_cycles(
        records,
        expected_templates={workload_id: "conv2d_packed.vta"},
        expected_workloads={workload_id},
    ) == {workload_id: 80}


def test_repeated_workload_occurrences_emit_independent_rows(utilization):
    workload_id = "a" * 64
    occurrences = [
        {"template": "conv2d_packed.vta", "workload_sha256": workload_id, "flop_count": 2048},
        {"template": "conv2d_packed.vta", "workload_sha256": workload_id, "flop_count": 2048},
    ]

    rows = utilization.rows_for_occurrences(
        "image_classification_v1",
        occurrences,
        {workload_id: 16},
        peak=64,
    )

    assert len(rows) == 2
    assert [row["layer_ordinal"] for row in rows] == [1, 2]
    assert [row["workload_sha256"] for row in rows] == [workload_id, workload_id]
    assert rows[0]["layer_id"] != rows[1]["layer_id"]
    assert rows[0]["best_trial_cycles"] == rows[1]["best_trial_cycles"] == 16


@pytest.mark.parametrize(
    "records,expected_templates,expected_workloads,error",
    [
        (
            [],
            {"a" * 64: "conv2d_packed.vta"},
            {"a" * 64},
            "missing workloads",
        ),
        (
            [_record("dense_packed.vta", ("conv",))],
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f": "conv2d_packed.vta"},
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f"},
            "template mismatch",
        ),
        (
            [_record("conv2d_packed.vta", ("conv",), costs=(0,))],
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f": "conv2d_packed.vta"},
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f"},
            "positive integral",
        ),
        (
            [_record("conv2d_packed.vta", ("conv",), costs=(1.5,))],
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f": "conv2d_packed.vta"},
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f"},
            "positive integral",
        ),
        (
            [_record("conv2d_packed.vta", ("unexpected",))],
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f": "conv2d_packed.vta"},
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f"},
            "unexpected workload",
        ),
        (
            [_record("conv2d_packed.vta", ("conv",), error_no=1, costs=(1,))],
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f": "conv2d_packed.vta"},
            {"a4ffd41ed4dc1628a12961547df60c722504f456260ea2a40c8d0d15762b8c3f"},
            "no successful cycle cost",
        ),
    ],
)
def test_invalid_or_mismatched_trial_records_fail_clearly(
    utilization, records, expected_templates, expected_workloads, error
):
    with pytest.raises(ValueError, match=error):
        utilization.minimum_successful_cycles(
            records,
            expected_templates=expected_templates,
            expected_workloads=expected_workloads,
        )


@pytest.mark.parametrize(
    "flops,cycles,peak",
    [(3, 1, 64), (2048, 0, 64), (2048, 2.5, 64), (2048, 16, 0)],
)
def test_non_integral_flops_or_invalid_throughput_inputs_rejected(
    utilization, flops, cycles, peak
):
    with pytest.raises(ValueError):
        utilization.calculate_utilization(flops, cycles, peak)


def test_unknown_template_and_ambiguous_occurrence_mapping_rejected(utilization):
    with pytest.raises(ValueError, match="unsupported AutoTVM template"):
        utilization.rows_for_occurrences(
            "image_classification_v1",
            [{"template": "depthwise.vta", "workload_sha256": "a" * 64, "flop_count": 10}],
            {"a" * 64: 1},
            peak=64,
        )

    with pytest.raises(ValueError, match="ambiguous workload mapping"):
        utilization.rows_for_occurrences(
            "image_classification_v1",
            [
                {"template": "conv2d_packed.vta", "workload_sha256": "a" * 64, "flop_count": 10},
                {"template": "conv2d_packed.vta", "workload_sha256": "a" * 64, "flop_count": 12},
            ],
            {"a" * 64: 1},
            peak=64,
        )


def test_tsim_log_targets_and_extracted_workload_coverage_are_validated(utilization):
    record = (SimpleNamespace(target="ext_dev -device=vta -model=tsim_1x8"), object())
    utilization._validate_tsim_targets([record])

    fsim_record = (SimpleNamespace(target="ext_dev -device=vta -model=fsim_1x8"), object())
    with pytest.raises(ValueError, match="not a VTA TSIM target"):
        utilization._validate_tsim_targets([fsim_record])

    metadata = {
        "task_workloads": ["a" * 64],
        "task_report": {"supported": [{"template": "conv2d_packed.vta", "workload_sha256": "a" * 64}]},
    }
    occurrences = [{"template": "conv2d_packed.vta", "workload_sha256": "b" * 64}]
    with pytest.raises(ValueError, match="do not match the AutoTVM sidecar"):
        utilization._validate_extracted_coverage(
            metadata,
            occurrences,
            {"supported": [{"template": "conv2d_packed.vta", "workload_sha256": "b" * 64}]},
        )


def test_mac_report_rejects_tsim_sidecar_without_single_call_protocol(
    utilization, tmp_path, monkeypatch
):
    tuner = utilization._load_tuner()
    monkeypatch.setenv("VTA_BACKEND", "tsim")
    config = tmp_path / "vta_64mac.json"
    config.write_text('{"LOG_BATCH": 0, "LOG_BLOCK": 3}\n', encoding="utf-8")
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config))
    log = tmp_path / "legacy-tsim.log"
    log.write_text("legacy accumulated cycle record", encoding="utf-8")
    sidecar = tmp_path / "legacy-tsim.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": tuner.ARTIFACT_SCHEMA_VERSION,
                "model_id": "image_classification_v1",
                "model_sha256": "a" * 64,
                "backend": "tsim",
                "config_path": str(config.resolve()),
                "config_sha256": utilization.sha256_file(config),
                "log_path": str(log.resolve()),
                "log_sha256": utilization.sha256_file(log),
                "tuning_options": {"tuner": "grid_search"},
            }
        ),
        encoding="utf-8",
    )
    prepared = SimpleNamespace(imported=SimpleNamespace(model_sha256="a" * 64))
    monkeypatch.setattr(
        utilization,
        "_load_model_occurrences",
        lambda _model_id: (tuner, prepared, [], {"supported": [], "unsupported": []}),
    )

    with pytest.raises(ValueError, match="rerun a single-call TSIM measurement"):
        utilization.build_rows(
            "image_classification_v1", log, sidecar, config_path=config
        )


def test_cli_requires_mode_specific_artifacts_and_tsim(utilization):
    parser = utilization.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--model", "all", "--backend", "fsim", "--summary", "summary.json"]
        )
    single = parser.parse_args(
        [
            "--model", "image_classification_v1", "--backend", "tsim",
            "--log", "model.log", "--sidecar", "model.json",
        ]
    )
    aggregate = parser.parse_args(
        ["--model", "all", "--backend", "tsim", "--summary", "summary.json"]
    )
    incomplete_single = parser.parse_args(
        ["--model", "image_classification_v1", "--backend", "tsim"]
    )
    with pytest.raises(ValueError, match="requires both --log and --sidecar"):
        utilization._validate_cli_inputs(incomplete_single)
    conflicting = parser.parse_args(
        [
            "--model", "all", "--backend", "tsim", "--summary", "summary.json",
            "--log", "model.log", "--sidecar", "model.json",
        ]
    )
    with pytest.raises(ValueError, match="does not accept --log/--sidecar"):
        utilization._validate_cli_inputs(conflicting)
    assert single.log == Path("model.log")
    assert aggregate.summary == Path("summary.json")


def test_aggregate_requires_complete_successful_tsim_identity_before_outputs(
    utilization, tmp_path, monkeypatch
):
    monkeypatch.setenv("VTA_BACKEND", "tsim")
    models = ["image_classification_v1", "image_classification_v2"]
    summary = tmp_path / "aggregate.json"
    summary.write_text(
        json.dumps(
            {
                "backend": "tsim",
                "config_path": "/geometry.json",
                "config_sha256": "c" * 64,
                "model_order": models,
                "results": {models[0]: {"status": "succeeded", "log_path": "l", "sidecar_path": "s"}},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "geometry.json"
    config.write_text('{"LOG_BATCH": 0, "LOG_BLOCK": 3}\n', encoding="utf-8")
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(utilization, "_model_order", lambda: models)

    with pytest.raises(ValueError, match="missing models"):
        utilization.load_aggregate_inputs(summary, config)
    assert not output_dir.exists()

    args = utilization.build_parser().parse_args(
        [
            "--model", "all", "--backend", "tsim", "--summary", str(summary),
            "--config", str(config), "--output-dir", str(output_dir),
        ]
    )
    with pytest.raises(ValueError, match="missing models"):
        utilization.run_report(args)
    assert not output_dir.exists()


def test_duplicate_task_extraction_restores_task_extract_environment(utilization, monkeypatch):
    class FakeTaskExtractEnv:
        current = None

        def __init__(self, allow_duplicate=False):
            self.allow_duplicate = allow_duplicate
            self.marker = object()

        @staticmethod
        def get(allow_duplicate=False):
            if FakeTaskExtractEnv.current is None:
                FakeTaskExtractEnv.current = FakeTaskExtractEnv(allow_duplicate)
            else:
                FakeTaskExtractEnv.current.allow_duplicate = allow_duplicate
            return FakeTaskExtractEnv.current

    fake_topi = SimpleNamespace(TaskExtractEnv=FakeTaskExtractEnv)
    monkeypatch.setitem(sys.modules, "tvm.autotvm.task.topi_integration", fake_topi)
    prior_get = FakeTaskExtractEnv.__dict__["get"]
    assert FakeTaskExtractEnv.current is None

    def extract_with_repeats_enabled():
        active = FakeTaskExtractEnv.get()
        assert active is FakeTaskExtractEnv.current
        assert active.allow_duplicate is True
        return "tasks"

    assert utilization._extract_tasks_with_duplicates(extract_with_repeats_enabled) == "tasks"

    assert FakeTaskExtractEnv.current is None
    assert FakeTaskExtractEnv.__dict__["get"] is prior_get

    existing = FakeTaskExtractEnv(allow_duplicate=False)
    FakeTaskExtractEnv.current = existing
    prior_marker = existing.marker
    assert utilization._extract_tasks_with_duplicates(lambda: "tasks") == "tasks"
    assert FakeTaskExtractEnv.current is existing
    assert existing.allow_duplicate is False
    assert existing.marker is prior_marker

    with pytest.raises(RuntimeError, match="extract failed"):
        utilization._extract_tasks_with_duplicates(
            lambda: (_ for _ in ()).throw(RuntimeError("extract failed"))
        )
    assert FakeTaskExtractEnv.current is existing
    assert FakeTaskExtractEnv.__dict__["get"] is prior_get
