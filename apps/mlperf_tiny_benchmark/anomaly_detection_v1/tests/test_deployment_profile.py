"""Focused one-sample AD V1 selected-deployment contract tests."""

import importlib.util
import sys
from pathlib import Path

import pytest


DEPLOYMENT = Path(__file__).resolve().parents[1] / "tune" / "deployment.py"


def _load_deployment():
    spec = importlib.util.spec_from_file_location("ad_v1_deployment", DEPLOYMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cycle_gate_is_inclusive_at_exactly_ten_percent():
    deployment = _load_deployment()

    assert deployment.compare_cycles(110, 100)["passed"] is True
    assert deployment.compare_cycles(90, 100)["passed"] is True
    with pytest.raises(ValueError, match="exceeds 10%"):
        deployment.compare_cycles(111, 100)


def test_manifest_rejects_foreign_model_before_dispatch(tmp_path):
    deployment = _load_deployment()
    path = tmp_path / "foreign.json"
    path.write_text('{"schema_version": 1, "model": "keyword_spotting_v1"}')

    with pytest.raises(ValueError, match="mismatched selected-schedule manifest"):
        deployment.validate_seed_manifest(path, object(), [])


def test_sample_selection_preserves_first_representative_window():
    deployment = _load_deployment()
    import numpy as np

    features = np.arange(5 * 640, dtype=np.float32).reshape(5, 640)
    selected, info = deployment.select_representative_window(features)

    assert selected.shape == (1, 640)
    assert selected[0, 0] == features[0, 0]
    assert info == {"window_index": 0, "total_windows": 5, "executed_windows": 1, "sampled": True}


def test_configs_remain_bound_to_each_symbol_and_occurrence():
    deployment = _load_deployment()
    import hashlib
    import json
    from types import SimpleNamespace

    identities = [
        SimpleNamespace(occurrence=0, symbol="ad_vta_0", sha256="a" * 64),
        SimpleNamespace(occurrence=1, symbol="ad_vta_1", sha256="b" * 64),
    ]
    entries = []
    for identity, config in zip(identities, ({"index": 1}, {"index": 1})):
        entries.append({
            "occurrence": identity.occurrence,
            "symbol": identity.symbol,
            "fusion_sha256": identity.sha256,
            "config": config,
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })

    mapped = deployment.config_entries_by_symbol(identities, entries)

    assert mapped == {"ad_vta_0": {"index": 1}, "ad_vta_1": {"index": 1}}
    entries[1]["symbol"] = "ad_vta_0"
    with pytest.raises(ValueError, match="identity mismatch"):
        deployment.config_entries_by_symbol(identities, entries)
