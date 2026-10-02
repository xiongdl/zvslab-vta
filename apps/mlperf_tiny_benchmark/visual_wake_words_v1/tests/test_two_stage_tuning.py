"""VWW V1 complete-fusion tuning adapter checks."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
ENTRY = APP_ROOT / "tune" / "tune.py"


def _load_entry():
    spec = importlib.util.spec_from_file_location("vww_v1_two_stage_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_extracts_each_deployed_complete_fusion():
    entry = _load_entry()

    prepared, identities, tasks = entry.legacy.prepare_v1_workloads()

    assert len(prepared.routing.symbols) == 13
    assert len(identities) == len(tasks) == 13
    assert [identity.symbol for identity in identities] == list(prepared.routing.symbols)
    assert [identity.occurrence for identity in identities] == list(range(13))
    assert entry.legacy.shared.MODEL_PIPELINES["visual_wake_words_v1"][0] == "visual_wake_words_v1"
    assert all(identity.bias_dtype and identity.bias_values for identity in identities)
    assert all(identity.shift >= 0 for identity in identities)
    assert all(identity.clip_min < identity.clip_max for identity in identities)
    assert {"nn.avg_pool2d", "nn.dense", "nn.softmax", "reshape"} <= set(
        prepared.routing.host_operator_names
    )
    assert prepared.routing.host_depthwise_count == 13
    assert prepared.imported.model_sha256 == entry.legacy.shared._sha256_file(
        APP_ROOT / "model" / "vww_96_float.tflite"
    )
    assert all(task.name == "mlperf_tiny_fused_conv2d.vta" for task in tasks)
    for identity, task in zip(identities, tasks):
        lowered = entry.legacy.fused.lower_with_fused_config(
            prepared, identity, task.config_space.get(0)
        )
        assert lowered.schedule is not None
        assert int(task.flop) > 0


def test_manifest_parser_rejects_foreign_model(tmp_path):
    entry = _load_entry()
    manifest = tmp_path / "foreign.json"
    manifest.write_text('{"schema_version": 1, "model": "keyword_spotting_v1"}')

    with pytest.raises(ValueError, match="mismatched best manifest"):
        entry._replay_manifest(manifest)


def test_seed_and_full_search_modes_are_explicit():
    entry = _load_entry()

    with pytest.raises(SystemExit, match="full search requires --alignment-report"):
        entry.main(["--all"])
    with pytest.raises(SystemExit, match="full search requires --alignment-report"):
        entry.main(["--workload-index", "0"])
    with pytest.raises(SystemExit, match="--seed requires --all"):
        entry.main(["--seed", "--workload-index", "0"])


@pytest.mark.parametrize("mutation", ["empty", "partial", "incomplete"])
def test_standalone_replay_rejects_incomplete_manifest(tmp_path, mutation):
    entry = _load_entry()
    source = next((APP_ROOT / "tune" / "optimal").glob("*/best-manifest.json"))
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if mutation == "empty":
        manifest["entries"] = []
    elif mutation == "partial":
        manifest["entries"] = manifest["entries"][:-1]
    else:
        manifest["status"] = "incomplete"
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        entry._replay_manifest(path)
