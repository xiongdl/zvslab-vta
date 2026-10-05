"""Contract checks for the six model-local actual-compute tune entry points."""

import importlib.util
import sys
from pathlib import Path

import pytest


APPS_ROOT = Path(__file__).resolve().parents[2] / "mlperf_tiny_benchmark"
MODEL_IDS = (
    "image_classification_v1",
    "anomaly_detection_v1",
    "keyword_spotting_v1",
    "streaming_wakeword_v1",
    "visual_wake_words_v1",
)


def _tuner(model_id):
    path = APPS_ROOT / model_id / "tune.py"
    name = f"actual_tune_contract_{model_id}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_tuner_exposes_seed_search_resume_and_export_contract(model_id):
    module = _tuner(model_id)
    parser = module._parser()
    if model_id == "image_classification_v1":
        fsim = module.validate_args(
            parser.parse_args(
                [
                    "--workloads", "workloads.json", "--workload", "2",
                    "--output-logs", "fsim.tmp",
                ]
            )
        )
        assert fsim.workloads == Path("workloads.json")
        assert fsim.workload == 2
        assert fsim.simulator == "fsim"
        assert fsim.output_logs == Path("fsim.tmp")
        assert fsim.timeout == 60
        assert fsim.trial_batch == 100
        assert fsim.min_successful == 20

        tsim = module.validate_args(
            parser.parse_args(
                [
                    "--workloads", "workloads.json", "--workload", "-1",
                    "--simulator", "tsim", "--input-logs", "fsim.tmp",
                    "--output-logs", "best.log",
                ]
            )
        )
        assert tsim.workload == -1
        assert tsim.simulator == "tsim"
        assert tsim.input_logs == Path("fsim.tmp")
        assert tsim.output_logs == Path("best.log")
        assert tsim.timeout == 120
        assert tsim.trial_batch is None
        assert tsim.min_successful is None
        return

    search = parser.parse_args(["--all", "--alignment-report", "seed-report.json"])

    assert search.all is True
    assert search.alignment_report == Path("seed-report.json")
    assert search.trial_batch == 100
    assert search.min_successful == 20
    assert search.fsim_timeout == 60
    assert search.tsim_timeout == 120

    candidate = parser.parse_args([
        "--resume-manifest", "resume.json",
        "--workload-index", "2",
        "--export-candidate", "3",
        "--output-log", "candidate.log",
    ])
    assert candidate.resume_manifest == Path("resume.json")
    assert candidate.workload_index == 2
    assert candidate.export_candidate == 3
    assert candidate.output_log == Path("candidate.log")

    best = parser.parse_args([
        "--resume-manifest", "resume.json",
        "--export-best",
        "--output-log", "best.log",
    ])
    assert best.export_best is True
    assert best.output_log == Path("best.log")


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_is_an_all_occurrence_measurement_option(model_id):
    parser = _tuner(model_id)._parser()
    if model_id == "image_classification_v1":
        args = parser.parse_args(
            [
                "--workloads", "workloads.json", "--workload", "-1",
                "--output-logs", "fsim.tmp",
            ]
        )
        assert args.workload == -1
        assert args.workloads == Path("workloads.json")
        assert args.output_logs == Path("fsim.tmp")
        return

    args = parser.parse_args(["--seed", "--all", "--output-log", "seed.log"])

    assert args.seed is True
    assert args.all is True
    assert args.output_log == Path("seed.log")
