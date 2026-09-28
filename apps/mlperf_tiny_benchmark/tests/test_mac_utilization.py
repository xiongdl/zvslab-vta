"""Tests for the validated per-layer VTA MAC utilization core."""

import importlib.util
from pathlib import Path
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
