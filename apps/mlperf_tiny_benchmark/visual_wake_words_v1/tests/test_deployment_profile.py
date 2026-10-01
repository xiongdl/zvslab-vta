"""VWW selected-config deployment profile checks."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = APP_ROOT / "tune" / "deployment.py"


def _load_deployment():
    spec = importlib.util.spec_from_file_location("vww_v1_deployment_profile", DEPLOYMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cycle_gate_uses_inclusive_ten_percent():
    deployment = _load_deployment()

    assert deployment.compare_cycles(110, 100)["passed"] is True
    assert deployment.compare_cycles(90, 100)["passed"] is True
    with pytest.raises(ValueError, match="exceeds 10%"):
        deployment.compare_cycles(111, 100)
    with pytest.raises(ValueError, match="exceeds 10%"):
        deployment.compare_cycles(89, 100)


def test_deployment_uses_one_manifest_image_and_existing_preprocessing():
    deployment = _load_deployment()
    sample_path = APP_ROOT / "samples" / "00-non-person-000000000009.jpg"
    runtime = SimpleNamespace(
        committed_sample_paths=lambda: (sample_path, sample_path.with_name("01-non-person-000000000025.jpg")),
        committed_sample_labels=lambda: (0, 0),
        load_sample=lambda path: np.zeros((1, 96, 96, 3), dtype=np.float32),
        INPUT_SHAPE=(1, 96, 96, 3),
        INPUT_DTYPE="float32",
    )

    sample, input_data, evidence = deployment.select_deployment_sample(runtime)

    assert sample.path == sample_path
    assert sample.label == 0
    assert input_data.shape == runtime.INPUT_SHAPE
    assert input_data.dtype == np.dtype(runtime.INPUT_DTYPE)
    assert evidence["sample_count"] == evidence["image_count"] == 1
    assert evidence["model_invocations"] == 1
    assert evidence["state_policy"] == "stateless_single_image"


def test_config_entries_are_bound_to_every_occurrence():
    deployment = _load_deployment()
    identities = (SimpleNamespace(occurrence=0, symbol="vta_0", sha256="fusion0"),)
    import hashlib
    import json
    config = {"tile": 4}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    entries = [{"occurrence": 0, "symbol": "vta_0", "fusion_sha256": "fusion0",
                "config": config, "config_sha256": digest}]

    assert deployment.config_entries_by_symbol(identities, entries) == {"vta_0": config}
    entries[0]["symbol"] = "foreign"
    with pytest.raises(ValueError, match="identity mismatch"):
        deployment.config_entries_by_symbol(identities, entries)


def test_deployment_cli_requires_a_manifest_and_report_path():
    deployment = _load_deployment()

    with pytest.raises(SystemExit):
        deployment.main([])
