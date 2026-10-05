"""Contract checks for the read-only deployment template's tuner CLI."""

import importlib.util
import sys
from pathlib import Path


APP = Path(__file__).resolve().parents[2] / "mlperf_tiny_benchmark" / "image_classification_v1"


def _tuner():
    path = APP / "tune.py"
    name = "reference_selected_tune_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_fsim_and_tsim_tune_options_preserve_selected_workload_defaults():
    module = _tuner()
    parser = module._parser()
    fsim = module.validate_args(parser.parse_args([
        "--workloads", "workloads.json", "--workload", "2", "--output-logs", "fsim.tmp",
    ]))
    assert fsim.workloads == Path("workloads.json")
    assert fsim.workload == 2
    assert fsim.simulator == "fsim"
    assert fsim.output_logs == Path("fsim.tmp")
    assert fsim.timeout == 60
    assert fsim.trial_batch == 100
    assert fsim.min_successful == 20

    tsim = module.validate_args(parser.parse_args([
        "--workloads", "workloads.json", "--workload", "-1", "--simulator", "tsim",
        "--input-logs", "fsim.tmp", "--output-logs", "best.log",
    ]))
    assert tsim.workload == -1
    assert tsim.simulator == "tsim"
    assert tsim.input_logs == Path("fsim.tmp")
    assert tsim.output_logs == Path("best.log")
    assert tsim.timeout == 120
    assert tsim.trial_batch is None
    assert tsim.min_successful is None
