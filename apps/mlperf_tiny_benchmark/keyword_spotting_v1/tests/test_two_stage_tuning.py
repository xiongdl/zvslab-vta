"""KWS V1 complete-fusion tuning adapter checks."""

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
ENTRY = APP_ROOT / "tune" / "tune.py"


def _load_entry():
    spec = importlib.util.spec_from_file_location("kws_v1_two_stage_entry", ENTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adapter_extracts_each_deployed_kws_fusion_with_complete_arithmetic():
    entry = _load_entry()

    prepared, identities, tasks = entry.legacy.prepare_v1_workloads()

    assert len(prepared.routing.symbols) == 4
    assert len(identities) == len(tasks) == 4
    assert [identity.symbol for identity in identities] == list(prepared.routing.symbols)
    assert [identity.occurrence for identity in identities] == list(range(4))
    assert entry.legacy.shared.MODEL_PIPELINES["keyword_spotting_v1"][0] == "keyword_spotting_v1"
    assert len({identity.sha256 for identity in identities}) == 4
    assert all(task.name == "mlperf_tiny_fused_conv2d.vta" for task in tasks)
    lowered = entry.legacy.fused.lower_with_fused_config(
        prepared, identities[0], tasks[0].config_space.get(0)
    )
    assert lowered.schedule is not None


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
    with pytest.raises(SystemExit, match="full search requires --alignment-report"):
        entry.main(["--workload-index", "0"])
    with pytest.raises(SystemExit, match="--seed requires --all"):
        entry.main(["--seed", "--workload-index", "0"])


@pytest.mark.parametrize("backend", ["fsim", "tsim"])
def test_full_worker_requires_valid_alignment_report_before_dispatch(tmp_path, backend):
    entry = _load_entry()
    dispatched = []
    setattr(entry, f"_{backend}_worker", lambda args: dispatched.append(args) or 0)
    run_dir = tmp_path / "run"

    with pytest.raises(SystemExit, match="worker requires a passing --alignment-report"):
        entry.main([
            "--worker-backend", backend, "--workload-index", "0",
            "--run-dir", str(run_dir),
        ])
    assert dispatched == []

    source = APP_ROOT / "tune" / "deployment-seed.json"
    valid = tmp_path / "valid.json"
    valid.write_bytes(source.read_bytes())
    assert entry.main([
        "--worker-backend", backend, "--workload-index", "0",
        "--run-dir", str(run_dir), "--alignment-report", str(valid),
    ]) == 0
    assert len(dispatched) == 1
    assert dispatched[0].alignment_report == valid.resolve()


@pytest.mark.parametrize("mutation", ["malformed", "foreign", "incomplete"])
def test_full_worker_rejects_bad_alignment_before_backend_dispatch(tmp_path, mutation):
    entry = _load_entry()
    source = APP_ROOT / "tune" / "deployment-seed.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    if mutation == "malformed":
        serialized = "not json"
    else:
        if mutation == "foreign":
            report["model_id"] = "anomaly_detection_v1"
        else:
            report["occurrences"].pop()
        serialized = json.dumps(report)
    report_path = tmp_path / f"{mutation}.json"
    report_path.write_text(serialized, encoding="utf-8")
    dispatched = []
    entry._fsim_worker = lambda args: dispatched.append(args) or 0

    with pytest.raises((SystemExit, ValueError)):
        entry.main([
            "--worker-backend", "fsim", "--workload-index", "0",
            "--run-dir", str(tmp_path / "run"),
            "--alignment-report", str(report_path),
        ])
    assert dispatched == []


def test_parent_worker_commands_bind_full_report_and_seed_mode():
    entry = _load_entry()
    report = (APP_ROOT / "tune" / "deployment-seed.json").resolve()
    args = SimpleNamespace(
        trial_batch=100, min_successful=20, fsim_timeout=60, tsim_timeout=120,
        resume_manifest=None, alignment_report=report, seed=False,
    )

    for backend in ("fsim", "tsim"):
        command = entry._worker_command(args, 0, backend, Path("/tmp/run"))
        assert command[command.index("--alignment-report") + 1] == str(report)
        assert "--seed" not in command

    args.seed = True
    args.alignment_report = None
    seed_command = entry._worker_command(args, 0, "fsim", Path("/tmp/seed-run"))
    assert "--seed" in seed_command


def test_valid_full_parent_passes_report_to_both_backend_workers(tmp_path, monkeypatch):
    entry = _load_entry()
    report = (APP_ROOT / "tune" / "deployment-seed.json").resolve()
    commands = []

    def fake_run(command, env, check):
        commands.append((command, env, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", fake_run)
    monkeypatch.setattr(entry, "_export_best_artifacts", lambda *args: tmp_path / "best.json")
    args = SimpleNamespace(
        seed=False, min_successful=20, trial_batch=100, fsim_timeout=60,
        tsim_timeout=120, alignment_report=report, workload_index=0, all=False,
        max_workloads=None, resume_manifest=None, build_dir=tmp_path,
        artifact_dir=None,
    )

    assert entry._run_controller(args) == 0
    assert len(commands) == 2
    assert [item[0][item[0].index("--worker-backend") + 1] for item in commands] == ["fsim", "tsim"]
    for command, env, check in commands:
        assert command[command.index("--alignment-report") + 1] == str(report)
        assert env["VTA_BACKEND"] == command[command.index("--worker-backend") + 1]
        assert check is False


def test_seed_worker_dispatch_remains_an_explicit_mode(tmp_path):
    entry = _load_entry()
    dispatched = []
    entry._fsim_worker = lambda args: dispatched.append(args) or 0

    assert entry.main([
        "--worker-backend", "fsim", "--workload-index", "0",
        "--run-dir", str(tmp_path / "seed-run"), "--seed",
    ]) == 0
    assert len(dispatched) == 1
    assert dispatched[0].seed is True


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
