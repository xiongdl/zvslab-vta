"""KWS V1 deployment evidence adapter checks."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = APP_ROOT / "tune" / "deployment.py"


def _load_deployment():
    spec = importlib.util.spec_from_file_location("kws_v1_deployment", DEPLOYMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cycle_gate_uses_inclusive_ten_percent_boundary():
    deployment = _load_deployment()

    assert deployment.compare_cycles(110, 100)["passed"]
    with pytest.raises(ValueError, match="exceeds 10%"):
        deployment.compare_cycles(111, 100)


def test_config_binding_preserves_each_occurrence_even_for_same_workload():
    deployment = _load_deployment()
    identities = [
        SimpleNamespace(occurrence=index, symbol=f"vta_{index}", sha256=f"fusion-{index}")
        for index in range(2)
    ]
    entries = []
    for identity in identities:
        config = {"index": identity.occurrence}
        import hashlib
        import json

        config_sha = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        entries.append({
            "occurrence": identity.occurrence,
            "symbol": identity.symbol,
            "fusion_sha256": identity.sha256,
            "config": config,
            "config_sha256": config_sha,
        })

    assert deployment.config_entries_by_symbol(identities, entries) == {
        "vta_0": {"index": 0},
        "vta_1": {"index": 1},
    }
    with pytest.raises(ValueError, match="cover every KWS VTA occurrence"):
        deployment.config_entries_by_symbol(identities, entries[:1])


def test_seed_manifest_rejects_foreign_model(tmp_path):
    deployment = _load_deployment()
    manifest = tmp_path / "foreign.json"
    manifest.write_text('{"schema_version": 1, "model": "anomaly_detection_v1"}')

    with pytest.raises(ValueError, match="mismatched selected-schedule manifest"):
        deployment.validate_seed_manifest(manifest, object(), [])
