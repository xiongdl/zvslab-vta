"""Deployment evidence contracts for Streaming Wakeword V1."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = APP_ROOT / "tune" / "deployment.py"


def _load_deployment():
    spec = importlib.util.spec_from_file_location("streaming_ww_v1_deployment", DEPLOYMENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cycle_gate_is_inclusive_at_ten_percent():
    deployment = _load_deployment()

    assert deployment.compare_cycles(110, 100)["passed"] is True
    assert deployment.compare_cycles(90, 100)["passed"] is True
    with pytest.raises(ValueError, match="exceeds 10%"):
        deployment.compare_cycles(111, 100)


def test_deployment_uses_one_manifest_sample_and_one_stateless_audio_window():
    deployment = _load_deployment()
    sample = SimpleNamespace(path=Path("marvin.wav"), filename="marvin.wav", order=0)
    runtime = SimpleNamespace(
        committed_sample_records=lambda: (sample, object(), object()),
        load_sample=lambda path: np.zeros((1, 30, 1, 40), dtype=np.int8),
        INPUT_SHAPE=(1, 30, 1, 40),
        INPUT_DTYPE="int8",
        CLIP_FRAMES=16000,
    )

    selected, input_data, evidence = deployment.select_deployment_sample(runtime)

    assert selected is sample
    assert input_data.shape == (1, 30, 1, 40)
    assert input_data.dtype == np.int8
    assert evidence == {
        "sample_count": 1,
        "audio_window_count": 1,
        "audio_window_samples": 16000,
        "feature_frame_count": 30,
        "model_invocations": 1,
        "state_policy": "stateless_single_invocation",
    }


def test_seed_manifest_rejects_foreign_model(tmp_path):
    deployment = _load_deployment()
    manifest = tmp_path / "foreign.json"
    manifest.write_text('{"schema_version": 1, "model": "keyword_spotting_v1"}')

    with pytest.raises(ValueError, match="mismatched selected-schedule manifest"):
        deployment.validate_seed_manifest(manifest, object(), ())


def test_configs_bind_to_exact_occurrence_and_fusion():
    import hashlib
    import json

    deployment = _load_deployment()
    identity = SimpleNamespace(occurrence=0, symbol="vta_symbol_0", sha256="a" * 64)
    config = {"tile": [1, 2]}
    config_sha = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    entry = {
        "occurrence": 0,
        "symbol": identity.symbol,
        "fusion_sha256": identity.sha256,
        "config": config,
        "config_sha256": config_sha,
    }

    assert deployment.config_entries_by_symbol((identity,), [entry]) == {
        identity.symbol: config
    }
    entry["symbol"] = "foreign_symbol"
    with pytest.raises(ValueError, match="identity mismatch"):
        deployment.config_entries_by_symbol((identity,), [entry])
