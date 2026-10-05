"""Make entrypoints pass safe, ordered arguments to the application CLIs."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


APP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = APP_DIR.parents[3]
VTA_DIR = APP_DIR.parents[2]


class MakeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.record = self.root / "record.jsonl"
        self.python = self.root / "python tools" / "record's shim"
        self.python.parent.mkdir()
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['RECORD'], 'a', encoding='utf-8') as out:\n"
            " out.write(json.dumps({'args': sys.argv[1:], 'backend': os.environ.get('VTA_BACKEND'), 'config': os.environ.get('VTA_CONFIG_FILE'), 'pythonpath': os.environ.get('PYTHONPATH')}) + '\\n')\n"
            "if os.environ.get('FAIL_ON') and any(os.environ['FAIL_ON'] in arg for arg in sys.argv[1:]):\n"
            " sys.exit(19)\n",
            encoding="utf-8",
        )
        self.python.chmod(0o755)

    def invoke(self, target, *, cwd, **variables):
        env = os.environ.copy()
        env["RECORD"] = str(self.record)
        if hasattr(self, "fail_on"):
            env["FAIL_ON"] = self.fail_on
        command = ["make", target, f"PYTHON={self.python}"]
        if cwd != APP_DIR:
            command[1:1] = ["-C", str(APP_DIR)]
        command.extend(f"{name}={value}" for name, value in variables.items())
        return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)

    def records(self):
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]

    def test_deploy_uses_default_paths_and_cpu_skips_backend(self):
        result = self.invoke("deploy", cwd=APP_DIR, TARGET="llvm")
        self.assertEqual(result.returncode, 0, result.stderr)
        [call] = self.records()
        self.assertIn(str(APP_DIR / "deploy.py"), call["args"])
        self.assertEqual(call["backend"], None)
        self.assertIn("--target", call["args"])
        self.assertIn("llvm", call["args"])
        self.assertIn(str(APP_DIR / "model" / "pretrainedResnet.tflite"), call["args"])
        self.assertIn(str(APP_DIR / "samples" / "00-airplane.png"), call["args"])
        self.assertEqual(call["pythonpath"],
                         f"{REPO_DIR}/tvm/python:{REPO_DIR}/vta/python:{APP_DIR}")

    def test_vta_deploy_quotes_explicit_paths_and_selects_backend(self):
        model = self.root / "model's ; file.tflite"
        image = self.root / "image (one).png"
        config = self.root / "custom geometry.json"
        for path in (model, image, config):
            path.write_bytes(b"fixture")
        result = self.invoke(
            "deploy", cwd=REPO_DIR, MODEL=model, INPUT=image, CONFIG=config,
            TARGET="vta,c", SIMULATOR="tsim", SCHEDULE="dir with space/best.log",
            REPORT="reports/one report.md", EXPORT_WORKLOADS="build/workloads.json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        [call] = self.records()
        args = call["args"]
        self.assertEqual(call["backend"], "tsim")
        self.assertEqual(call["config"], str(config))
        self.assertIn(str(model), args)
        self.assertIn(str(image), args)
        self.assertIn(str(APP_DIR / "dir with space" / "best.log"), args)
        self.assertIn(str(APP_DIR / "reports" / "one report.md"), args)
        self.assertIn(str(APP_DIR / "build" / "workloads.json"), args)

    def test_split_tuning_and_full_tuning_have_expected_order_and_backend(self):
        workload = self.root / "captured workloads.json"
        workload.write_text("{}", encoding="utf-8")
        candidates = self.root / "candidates.tmp"
        best = self.root / "best.log"
        fsim = self.invoke(
            "tune-fsim", cwd=APP_DIR, WORKLOADS=workload, WORKLOAD=0,
            TRIAL_BATCH=1, MIN_SUCCESSFUL=1, TIMEOUT=60, OUTPUT_LOGS=candidates,
        )
        self.assertEqual(fsim.returncode, 0, fsim.stderr)
        tsim = self.invoke(
            "tune-tsim", cwd=APP_DIR, WORKLOADS=workload, INPUT_LOGS=candidates,
            WORKLOAD=0, TIMEOUT=120, OUTPUT_LOGS=best,
        )
        self.assertEqual(tsim.returncode, 0, tsim.stderr)
        full = self.invoke(
            "tune", cwd=APP_DIR, WORKLOADS=workload, WORKLOAD=0,
            TRIAL_BATCH=1, MIN_SUCCESSFUL=1,
        )
        self.assertEqual(full.returncode, 0, full.stderr)
        calls = self.records()
        self.assertEqual([call["backend"] for call in calls], ["fsim", "tsim", "fsim", "tsim"])
        self.assertEqual(calls[0]["args"][0], str(APP_DIR / "tune.py"))
        self.assertIn(str(candidates), calls[0]["args"])
        self.assertIn(str(candidates), calls[1]["args"])
        self.assertIn(str(best), calls[1]["args"])
        self.assertIn(str(workload), calls[2]["args"])
        self.assertIn(str(workload), calls[3]["args"])
        self.assertTrue(all("/apps/common" not in call["pythonpath"] for call in calls))

    def test_full_tune_exports_then_runs_both_stages_without_deploying_winner(self):
        output_dir = self.root / "intermediate files"
        result = self.invoke(
            "tune", cwd=APP_DIR, OUTPUT_DIR=output_dir, WORKLOAD=0,
            TRIAL_BATCH=1, MIN_SUCCESSFUL=1, FSIM_TIMEOUT=7, TSIM_TIMEOUT=9,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.records()
        self.assertEqual(len(calls), 3)
        export, fsim, tsim = calls
        self.assertEqual([call["backend"] for call in calls], ["fsim", "fsim", "tsim"])
        self.assertIn(str(APP_DIR / "deploy.py"), export["args"])
        self.assertIn("--export-workloads", export["args"])
        self.assertIn(str(output_dir / "workloads.json"), export["args"])
        self.assertIn(str(APP_DIR / "tune" / "vta_64mac" / "fsim.tmp"), fsim["args"])
        self.assertIn("7", fsim["args"])
        self.assertIn(str(APP_DIR / "tune" / "vta_64mac" / "best.log"), tsim["args"])
        self.assertIn("9", tsim["args"])
        self.assertNotIn("--schedule", [arg for call in calls for arg in call["args"]])

    def test_split_targets_report_missing_required_files_before_launch(self):
        result = self.invoke("tune-tsim", cwd=APP_DIR)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires WORKLOADS=PATH", result.stderr)
        self.assertFalse(self.record.exists())

    def test_full_tune_stops_when_workload_export_fails(self):
        self.fail_on = "deploy.py"
        result = self.invoke("tune", cwd=APP_DIR, OUTPUT_DIR=self.root / "failed run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error 19", result.stderr)
        calls = self.records()
        self.assertEqual(len(calls), 1)
        self.assertIn("--export-workloads", calls[0]["args"])

    def test_saved_tuning_files_are_visible_and_build_intermediates_ignored(self):
        def ignored(path):
            return subprocess.run(
                ["git", "-C", str(VTA_DIR), "check-ignore", "-q", str(path)],
                text=True, capture_output=True,
            ).returncode == 0

        tune_dir = APP_DIR / "tune" / "vta_64mac"
        self.assertFalse(ignored(tune_dir / "fsim.tmp"))
        self.assertFalse(ignored(tune_dir / "best.log"))
        self.assertFalse(ignored(tune_dir / "best.json"))
        self.assertTrue(ignored(APP_DIR / "build" / "workloads.json"))

    def test_make_c_relative_paths_resolve_from_make_working_directory(self):
        result = self.invoke(
            "deploy", cwd=REPO_DIR, TARGET="llvm", INPUT="assets/a ; b.png",
            OUTPUT_DIR="build with space",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        [call] = self.records()
        args = call["args"]
        self.assertIn(str(APP_DIR / "assets" / "a ; b.png"), args)
        self.assertIn(str(APP_DIR / "build with space"), args)


if __name__ == "__main__":
    unittest.main()
