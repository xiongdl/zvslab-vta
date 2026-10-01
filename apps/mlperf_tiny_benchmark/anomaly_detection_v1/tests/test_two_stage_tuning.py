"""Model identity and prepared-graph coverage for AD V1 two-stage tuning."""

import importlib.util
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
ENTRY = APP_ROOT / "tune" / "tune.py"


def _load_entry():
    spec = importlib.util.spec_from_file_location("ad_v1_two_stage_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_extracts_all_deployed_ad_v1_fusions():
    entry = _load_entry()

    prepared, identities, tasks = entry.legacy.prepare_v1_workloads()

    assert len(prepared.routing.symbols) == 9
    assert len(identities) == len(tasks) == 9
    assert [identity.symbol for identity in identities] == list(prepared.routing.symbols)
    assert entry.legacy.shared.MODEL_PIPELINES["anomaly_detection_v1"][0] == "anomaly_detection_v1"
    assert len({identity.sha256 for identity in identities}) == 9


def test_manifest_parser_rejects_foreign_model(tmp_path):
    entry = _load_entry()
    manifest = tmp_path / "foreign.json"
    manifest.write_text('{"schema_version": 1, "model": "image_classification_v1"}')

    with pytest.raises(ValueError, match="mismatched best manifest"):
        entry._replay_manifest(manifest)


def test_seed_and_full_search_modes_are_explicit():
    entry = _load_entry()

    with pytest.raises(SystemExit, match="full search requires --alignment-report"):
        entry.main(["--all"])
    with pytest.raises(SystemExit, match="--seed requires --all"):
        entry.main(["--seed", "--workload-index", "0"])
