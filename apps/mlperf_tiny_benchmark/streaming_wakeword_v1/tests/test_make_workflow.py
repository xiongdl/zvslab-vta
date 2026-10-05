"""Make forwards selected deployment and tuning arguments safely."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[3]
VTA_ROOT = APP_ROOT.parents[2]


class MakeWorkflow:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.add_cleanup = self.temporary.cleanup
        self.root = Path(self.temporary.name)
        self.record = self.root / "record.jsonl"
        self.python = self.root / "python tools" / "record's shim"
        self.python.parent.mkdir()
        self.python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['RECORD'], 'a', encoding='utf-8') as stream:\n"
            " stream.write(json.dumps({'args':sys.argv[1:],'backend':os.environ.get('VTA_BACKEND'),"
            "'config':os.environ.get('VTA_CONFIG_FILE'),'pythonpath':os.environ.get('PYTHONPATH')})+'\\n')\n"
            "if os.environ.get('FAIL_ON') and any(os.environ['FAIL_ON'] in arg for arg in sys.argv[1:]):\n"
            " sys.exit(19)\n",
            encoding="utf-8",
        )
        self.python.chmod(0o755)

    def close(self):
        self.add_cleanup()

    def invoke(self, target, *, cwd, **variables):
        env = os.environ.copy()
        env["RECORD"] = str(self.record)
        command = ["make", target, f"PYTHON={self.python}"]
        if cwd != APP_ROOT:
            command[1:1] = ["-C", str(APP_ROOT)]
        command.extend(f"{name}={value}" for name, value in variables.items())
        return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)

    def records(self):
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]


def test_make_deploy_uses_single_target_and_quotes_custom_paths(tmp_path):
    workflow = MakeWorkflow()
    try:
        cpu = workflow.invoke("deploy", cwd=APP_ROOT, TARGET="llvm")
        assert cpu.returncode == 0, cpu.stderr
        [call] = workflow.records()
        assert call["backend"] is None
        assert str(APP_ROOT / "model/str_ww_ref_model.tflite") in call["args"]
        assert str(APP_ROOT / "samples/marvin-00176480_nohash_0.wav") in call["args"]
        assert call["pythonpath"] == f"{REPO_ROOT}/tvm/python:{REPO_ROOT}/vta/python:{APP_ROOT}"

        workflow.record.unlink()
        model = tmp_path / "model's ; file.tflite"
        audio = tmp_path / "audio (one).wav"
        config = tmp_path / "custom geometry.json"
        for path in (model, audio, config):
            path.write_bytes(b"fixture")
        vta = workflow.invoke(
            "deploy", cwd=REPO_ROOT, MODEL=model, INPUT=audio, CONFIG=config,
            TARGET="vta,c", SIMULATOR="tsim", SCHEDULE="dir with space/best.log",
            REPORT="reports/one report.md", EXPORT_WORKLOADS="build/workloads.json",
        )
        assert vta.returncode == 0, vta.stderr
        [call] = workflow.records()
        assert call["backend"] == "tsim"
        assert call["config"] == str(config)
        assert str(model) in call["args"]
        assert str(audio) in call["args"]
        assert str(APP_ROOT / "dir with space/best.log") in call["args"]
        assert str(APP_ROOT / "reports/one report.md") in call["args"]
        assert str(APP_ROOT / "build/workloads.json") in call["args"]
    finally:
        workflow.close()


def test_split_tuning_and_full_tuning_forward_stages_in_order(tmp_path):
    workflow = MakeWorkflow()
    try:
        config = tmp_path / "geometry.json"
        workloads = tmp_path / "workloads.json"
        config.write_text("{}", encoding="utf-8")
        workloads.write_text("{}", encoding="utf-8")
        fsim = workflow.invoke(
            "tune-fsim", cwd=APP_ROOT, CONFIG=config, WORKLOADS=workloads,
            WORKLOAD=0, TRIAL_BATCH=1, MIN_SUCCESSFUL=1, OUTPUT_LOGS=tmp_path / "fsim.tmp",
        )
        assert fsim.returncode == 0, fsim.stderr
        tsim = workflow.invoke(
            "tune-tsim", cwd=APP_ROOT, CONFIG=config, WORKLOADS=workloads,
            INPUT_LOGS=tmp_path / "fsim.tmp", WORKLOAD=0, OUTPUT_LOGS=tmp_path / "best.log",
        )
        assert tsim.returncode == 0, tsim.stderr
        full = workflow.invoke(
            "tune", cwd=APP_ROOT, CONFIG=config, WORKLOADS=workloads, WORKLOAD=0,
            TRIAL_BATCH=1, MIN_SUCCESSFUL=1,
        )
        assert full.returncode == 0, full.stderr

        calls = workflow.records()
        assert [call["backend"] for call in calls] == ["fsim", "tsim", "fsim", "tsim"]
        assert calls[0]["args"][0] == str(APP_ROOT / "tune.py")
        assert str(tmp_path / "fsim.tmp") in calls[0]["args"]
        assert str(tmp_path / "fsim.tmp") in calls[1]["args"]
        assert str(tmp_path / "best.log") in calls[1]["args"]
        assert all(str(workloads) in call["args"] for call in calls[2:])
    finally:
        workflow.close()


def test_saved_schedules_are_visible_while_build_output_is_ignored():
    def ignored(path):
        return subprocess.run(
            ["git", "-C", str(VTA_ROOT), "check-ignore", "-q", str(path)],
            capture_output=True,
        ).returncode == 0

    tune = APP_ROOT / "tune/vta_64mac"
    assert not ignored(tune / "fsim.tmp")
    assert not ignored(tune / "best.log")
    assert not ignored(tune / "best.json")
    assert ignored(APP_ROOT / "build/workloads.json")


def test_clean_is_idempotent_and_keeps_persistent_files_and_external_output(tmp_path):
    app = tmp_path / "repo/vta/apps/mlperf_tiny_benchmark/streaming_wakeword_v1"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(APP_ROOT / "Makefile", app / "Makefile")
    shutil.copy2(APP_ROOT / "scripts/make_tasks.sh", app / "scripts/make_tasks.sh")
    external = tmp_path / "custom-output"
    external.mkdir()
    sentinel = external / "sentinel.bin"
    sentinel.write_bytes(b"keep")
    (app / "build/generated.bin").parent.mkdir(parents=True)
    (app / "build/generated.bin").write_bytes(b"drop")
    persistent = app / "tune/config/best.log"
    persistent.parent.mkdir(parents=True)
    persistent.write_bytes(b"keep schedule")
    for cache in (app / "__pycache__", app / "python/__pycache__", app / "tests/__pycache__"):
        cache.mkdir(parents=True)
        (cache / "module.pyc").write_bytes(b"drop cache")

    for _ in range(2):
        result = subprocess.run(
            ["make", "clean", f"OUTPUT_DIR={external}", "PYTHON=/missing/python"],
            cwd=app, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert not (app / "build").exists()
        assert not (app / "__pycache__").exists()
        assert not (app / "python/__pycache__").exists()
        assert not (app / "tests/__pycache__").exists()
        assert persistent.read_bytes() == b"keep schedule"
        assert sentinel.read_bytes() == b"keep"
