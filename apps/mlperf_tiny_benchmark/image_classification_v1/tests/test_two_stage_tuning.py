"""Independent workloads-driven FSIM and TSIM tuning contracts."""

import importlib.util
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    spec = importlib.util.spec_from_file_location(f"ic_v1_{name}", APP_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def actual_workloads(tmp_path_factory):
    root = APP_ROOT.parents[3]
    python = root / ".envs" / "tvm-vta-env" / "bin" / "python"
    path = tmp_path_factory.mktemp("actual-resnet8")
    workloads = path / "workloads.json"
    env = os.environ.copy()
    env.update({
        "VTA_CONFIG_FILE": str(root / "vta" / "config" / "vta_64mac.json"),
        "VTA_BACKEND": "fsim",
        "PYTHONPATH": os.pathsep.join((
            str(root / "tvm" / "python"), str(root / "vta" / "python")
        )),
    })
    completed = subprocess.run([
        str(python), str(APP_ROOT / "run.py"), "--target", "vta,llvm",
        "--simulator", "fsim", "--export-workloads", str(workloads),
        "--output-dir", str(path / "export"),
    ], cwd=root, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert workloads.is_file()
    return root, python, env, workloads, path


def test_tune_cli_exposes_only_workloads_stage_contract():
    tune = _load("tune")
    args = tune._parser().parse_args([
        "--workloads", "build/workloads.json", "--workload", "-1",
        "--simulator", "fsim", "--trial-batch", "2",
        "--min-successful", "1", "--timeout", "7", "--output-logs", "tune/fsim.tmp",
    ])
    assert args.workloads == Path("build/workloads.json")
    assert args.workload == -1
    assert args.simulator == "fsim"
    assert args.trial_batch == 2
    assert args.min_successful == 1
    assert args.timeout == 7
    assert args.output_logs == Path("tune/fsim.tmp")

    tsim = tune._parser().parse_args([
        "--workloads", "build/workloads.json", "--simulator", "tsim",
        "--input-logs", "tune/fsim.tmp", "--output-logs", "tune/best.log",
    ])
    tune.validate_args(tsim)
    assert tsim.workload == -1
    assert tsim.timeout == 120
    assert tsim.input_logs == Path("tune/fsim.tmp")


@pytest.mark.parametrize("removed", [
    "--seed", "--all", "--workload-index", "--max-workloads",
    "--resume-manifest", "--alignment-report", "--export-candidate",
    "--export-best", "--output-log", "--fsim-timeout", "--tsim-timeout",
])
def test_retired_tune_flags_are_rejected(removed):
    tune = _load("tune")
    with pytest.raises(SystemExit):
        tune._parser().parse_args([removed])


def test_fsim_rejects_tsim_input_and_tsim_rejects_search_controls():
    tune = _load("tune")
    common = ["--workloads", "workloads.json", "--output-logs", "out.log"]
    with pytest.raises(ValueError, match="only valid for TSIM"):
        tune.validate_args(tune._parser().parse_args(common + [
            "--simulator", "fsim", "--input-logs", "in.tmp",
        ]))
    with pytest.raises(ValueError, match="only valid for FSIM"):
        tune.validate_args(tune._parser().parse_args(common + [
            "--simulator", "tsim", "--input-logs", "in.tmp", "--trial-batch", "2",
        ]))


def test_tune_uses_workload_snapshot_without_loading_runtime(monkeypatch, tmp_path):
    tune = _load("tune")
    observed = {}

    class Snapshot:
        layers = (object(), object())

    monkeypatch.setenv("VTA_BACKEND", "fsim")
    def load(path):
        observed["path"] = path
        return Snapshot()

    monkeypatch.setattr(tune, "load_workloads", load)
    assert not hasattr(tune, "_load_runtime")
    monkeypatch.setattr(
        tune, "run_fsim",
        lambda args, snapshot: (observed.update(run=(args, snapshot)) or {1: True}),
    )
    args = tune._parser().parse_args([
        "--workloads", str(tmp_path / "workloads.json"), "--simulator", "fsim",
        "--output-logs", str(tmp_path / "candidates.tmp"), "--workload", "1",
        "--trial-batch", "1", "--min-successful", "1",
    ])
    assert tune.main(args) == {1: True}
    assert observed["path"] == args.workloads
    assert observed["run"][1].layers == Snapshot.layers


def test_zero_successful_fsim_candidates_publish_nothing(monkeypatch, tmp_path):
    tune = _load("tune")
    measurement = importlib.import_module("measurement")

    class Config:
        def valid(self):
            return True

        def to_json_dict(self):
            return {"tile": 0}

    class Space:
        def __len__(self):
            return 2

        def get(self, index):
            return Config()

    layer = type("Layer", (), {
        "occurrence": 0, "config_spaces": (("conv2d_packed.vta", ("conv2d_packed.vta",), "vta", Space()),),
    })()
    workload = type("Workload", (), {
        "index": 0, "symbol": "vta0", "compute_sha256": "c" * 64,
        "config_space_identity": "s" * 64, "activation": object(),
    })()
    config_bytes = b'{"geometry":1}\n'
    snapshot = type("Snapshot", (), {
        "layers": (workload,), "model_sha256": "m" * 64,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "geometry_sha256": "g" * 64, "config_bytes": config_bytes,
    })()
    workload_file = tmp_path / "workloads.json"
    workload_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tune, "_capture_layers", lambda snapshot: ((layer,), None))
    monkeypatch.setattr(measurement, "measure_candidate", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("bounded")))
    args = tune._parser().parse_args([
        "--workloads", str(workload_file), "--simulator", "fsim",
        "--trial-batch", "1", "--min-successful", "1",
        "--output-logs", str(tmp_path / "tune" / "fsim.tmp"),
    ])
    tune.validate_args(args)
    with pytest.raises(RuntimeError, match="no successful candidate"):
        tune.run_fsim(args, snapshot)
    assert not (tmp_path / "tune" / "fsim.tmp").exists()
    assert not (tmp_path / "tune" / "fsim.json").exists()
    assert not (tmp_path / "tune" / "config.json").exists()


@pytest.mark.parametrize("backend", ["fsim", "tsim"])
def test_native_worker_exit_aborts_tuning_stage(monkeypatch, tmp_path, backend):
    tune = _load("tune")
    import measurement
    import tuning
    from types import SimpleNamespace

    config_bytes = b'{"geometry":1}\n'
    workload_file = tmp_path / "workloads.json"
    workload_file.write_text("{}", encoding="utf-8")
    layer = SimpleNamespace(
        occurrence=0, index=0, symbol="vta0", activation=object(),
        compute_sha256="c" * 64, config_space_identity="s" * 64,
        config_spaces=(),
    )
    snapshot = SimpleNamespace(
        layers=(layer,), model_sha256="m" * 64,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        geometry_sha256="g" * 64, config_bytes=config_bytes,
    )
    monkeypatch.setattr(tune, "_capture_layers", lambda _: ((layer,), None))
    monkeypatch.setattr(tune, "_select_layers", lambda _, __: [layer])
    monkeypatch.setattr(
        measurement, "measure_candidate",
        lambda *args: (_ for _ in ()).throw(
            measurement.MeasurementInfrastructureError("worker exited without a result")
        ),
    )
    output = tmp_path / "tune" / ("fsim.tmp" if backend == "fsim" else "best.log")
    command = [
        "--workloads", str(workload_file), "--workload", "0",
        "--simulator", backend, "--output-logs", str(output),
    ]
    if backend == "fsim":
        command.extend(["--trial-batch", "1", "--min-successful", "1"])
        monkeypatch.setattr(tuning, "candidate_indices", lambda *args, **kwargs: iter([[0]]))
        monkeypatch.setattr(tuning, "configs_for_indices", lambda *args: [])
    else:
        candidates = tmp_path / "input.tmp"
        candidates.write_text("candidate", encoding="utf-8")
        candidates.with_suffix(".json").write_text("{}", encoding="utf-8")
        command.extend(["--input-logs", str(candidates)])
        monkeypatch.setattr(tuning, "decode_candidate_log", lambda *args, **kwargs: ({
            "occurrence": 0, "symbol": "vta0", "candidate_id": "d" * 64,
            "configs": [], "config_identity": "i" * 64,
        },))
        monkeypatch.setattr(tune, "_candidate_record_indices", lambda *args: [0])
    args = tune._parser().parse_args(command)

    runner = tune.run_fsim if backend == "fsim" else tune.run_tsim
    with pytest.raises(
        measurement.MeasurementInfrastructureError, match="worker exited without a result"
    ):
        runner(args, snapshot)
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_candidate_log_groups_multiple_native_records_by_occurrence_and_candidate(monkeypatch):
    tuning = _load("tuning")
    monkeypatch.setattr(tuning.autotvm.record, "decode", lambda row: object())
    configs = []
    candidate_b = hashlib.sha256(f"0:{tuning.candidate_identity(configs)}".encode()).hexdigest()
    candidate_a = hashlib.sha256(f"0:{tuning.candidate_identity(configs)}".encode()).hexdigest()
    # The second group uses a distinct config payload and therefore a distinct ID.
    configs_a = [{"template": "packed", "config": {"tile": 1}}]
    candidate_a = hashlib.sha256(f"0:{tuning.candidate_identity(configs_a)}".encode()).hexdigest()
    measurement = {"backend": "fsim", "protocol": "relay_cpu_exact_output_v1",
                   "units": "seconds", "cost": 0.1, "timestamp": 1.0}
    candidates = [
        {"occurrence": 0, "candidate_id": candidate_b, "records": ["row-b"],
         "config_identity": "b" * 64, "configs": configs, "backend": "fsim",
         "measurement": measurement, "output_verified": True},
        {"occurrence": 0, "candidate_id": candidate_a, "records": ["row-a1", "row-a2"],
         "config_identity": "a" * 64, "configs": configs_a, "backend": "fsim",
         "measurement": measurement, "output_verified": True},
    ]
    raw = tuning.encode_candidate_log(candidates, {
        "model_sha256": "m" * 64, "config_sha256": "c" * 64,
        "geometry_sha256": "g" * 64, "workloads_sha256": "w" * 64,
    })
    loaded = tuning.decode_candidate_log(raw.log_bytes, raw.sidecar_bytes, expected={
        "model_sha256": "m" * 64, "config_sha256": "c" * 64,
        "geometry_sha256": "g" * 64, "workloads_sha256": "w" * 64,
    })
    assert [(row["occurrence"], row["candidate_id"]) for row in loaded] == [
        (0, candidate_b), (0, candidate_a)
    ]
    assert [len(row["records"]) for row in loaded] == [1, 2]

    sidecar = json.loads(raw.sidecar_bytes)
    sidecar["log_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        tuning.decode_candidate_log(raw.log_bytes, json.dumps(sidecar).encode(), expected=None)


def test_candidate_selection_uses_lowest_cycles_then_config_identity():
    tuning = _load("tuning")
    selected = tuning.select_minimum_cycles([
        {"cycles": 40, "config_identity": "c" * 64, "item": "slow"},
        {"cycles": 20, "config_identity": "b" * 64, "item": "tie-b"},
        {"cycles": 20, "config_identity": "a" * 64, "item": "tie-a"},
    ])
    assert selected["item"] == "tie-a"


def test_single_occurrence_candidate_merge_keeps_other_workloads(tmp_path):
    tuning = _load("tuning")
    original = [
        {"occurrence": 0, "candidate_id": "old-0", "records": [{"record": "0"}]},
        {"occurrence": 1, "candidate_id": "old-1", "records": [{"record": "1"}]},
    ]
    update = [
        {"occurrence": 0, "candidate_id": "new-0", "records": [{"record": "2"}]},
    ]
    merged = tuning.merge_candidate_groups(original, update, selected={0}, replace_all=False)
    assert [(row["occurrence"], row["candidate_id"]) for row in merged] == [
        (0, "new-0"), (1, "old-1")
    ]
    assert tuning.merge_candidate_groups(original, update, selected={0}, replace_all=True) == update


def test_staged_publication_rolls_back_all_files_on_replace_failure(monkeypatch, tmp_path):
    publication = _load("publication")
    files = {tmp_path / "first": b"old-a", tmp_path / "second": b"old-b"}
    for path, value in files.items():
        path.write_bytes(value)
    replace = publication.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replace failure")
        replace(source, target)

    monkeypatch.setattr(publication.os, "replace", fail_second)
    with pytest.raises(OSError, match="injected"):
        publication.publish_file_set({path: b"new" for path in files})
    assert {path: path.read_bytes() for path in files} == files


def test_config_snapshot_uses_exact_bytes_and_rejects_corruption(tmp_path):
    publication = _load("publication")
    config = b'{"TARGET":"vta","BLOCK_IN":16}\n'
    digest = hashlib.sha256(config).hexdigest()
    config_path, checksum_path = publication.config_files(tmp_path)
    config_path.write_bytes(config)
    checksum_path.write_text(digest + "\n", encoding="ascii")
    assert publication.validate_config_snapshot(tmp_path, config, digest)
    with pytest.raises(ValueError, match="different VTA config"):
        publication.validate_config_snapshot(tmp_path, config + b" ", hashlib.sha256(config + b" ").hexdigest())
    assert publication.validate_config_snapshot(
        tmp_path, config + b" ", hashlib.sha256(config + b" ").hexdigest(), allow_mismatch=True
    )

    checksum_path.write_text("0" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="checksum is corrupt"):
        publication.validate_config_snapshot(tmp_path, config, digest, allow_mismatch=True)


def test_schedule_snapshot_rejects_a_different_raw_geometry_config(tmp_path, monkeypatch):
    schedule = _load("schedule")
    config_path = tmp_path / "vta.json"
    config_path.write_bytes(b'{"TARGET":"vta"}\n')
    monkeypatch.setenv("VTA_CONFIG_FILE", str(config_path))
    path = tmp_path / "best.log"
    path.write_bytes(b"")
    metadata = {
        "schema_version": schedule.SCHEMA_VERSION,
        "model_id": "image_classification_v1",
        "model_sha256": "m" * 64,
        "config_sha256": "0" * 64,
        "geometry_sha256": schedule._geometry_identity(
            type("Deployment", (), {"geometry": {"batch": 1}})()
        ),
        "log_sha256": hashlib.sha256(b"").hexdigest(),
        "occurrences": [],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    deployment = type("Deployment", (), {
        "model_id": "image_classification_v1", "model_sha256": "m" * 64,
        "geometry": {"batch": 1}, "layers": (),
    })()
    with pytest.raises(ValueError, match="config content hash"):
        schedule.load_schedule_snapshot(path, deployment)


def test_loaded_schedule_measurements_expand_default_fallback_slots():
    tune = _load("tune")
    layer = type("Layer", (), {
        "occurrence": 3,
        "config_spaces": (("add.vta", ("add.vta",), "vta", [0]),
                          ("conv2d_packed.vta", ("conv2d_packed.vta",), "vta", [0, 1])),
    })()
    measurement = {
        "backend": "tsim", "protocol": "tsim_single_call_v1", "units": "cycles",
        "results": [{"costs": [12], "error_no": 0, "all_cost": 0.0, "timestamp": 1.0}],
    }
    expanded = tune._measurement_for_export(layer, measurement)
    assert expanded["results"] == [None, measurement["results"][0]]
    with pytest.raises(ValueError, match="extra measurement"):
        tune._measurement_for_export(layer, {**measurement, "results": measurement["results"] * 2})


def test_tuning_writer_lock_rejects_a_second_writer(tmp_path):
    publication = _load("publication")
    with publication.writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another tuning writer"):
            with publication.writer_lock(tmp_path):
                pass
    assert not (tmp_path / ".tune.lock").exists()


def test_cycle_alignment_gate_is_strictly_less_than_ten_percent():
    deployment = _load("deployment")
    assert deployment.cycles_within_strict_ten_percent(109, 100)
    assert not deployment.cycles_within_strict_ten_percent(110, 100)
    with pytest.raises(ValueError, match="positive integer"):
        deployment.cycles_within_strict_ten_percent(0, 100)


@pytest.mark.parametrize("workload", [0, -1], ids=["one-occurrence", "all-occurrences"])
def test_real_fsim_tsim_schedule_replay_and_cycle_alignment(actual_workloads, workload):
    root, python, base_env, workloads, output_root = actual_workloads
    from deployment import cycles_within_strict_ten_percent

    tag = "one" if workload == 0 else "all"
    result_dir = output_root / tag
    candidates = result_dir / "fsim.tmp"
    best = result_dir / "best.log"
    for backend, extra, output in (
        ("fsim", ["--trial-batch", "1", "--min-successful", "1", "--timeout", "120"], candidates),
        ("tsim", ["--input-logs", str(candidates), "--timeout", "120"], best),
    ):
        env = dict(base_env, VTA_BACKEND=backend)
        command = [
            str(python), str(APP_ROOT / "tune.py"), "--workloads", str(workloads),
            "--workload", str(workload), "--simulator", backend,
            *extra, "--output-logs", str(output),
        ]
        completed = subprocess.run(
            command, cwd=output_root, env=env, capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr

    report = result_dir / "deployment.md"
    env = dict(base_env, VTA_BACKEND="tsim")
    completed = subprocess.run([
        str(python), str(APP_ROOT / "run.py"), "--target", "vta,llvm",
        "--simulator", "tsim", "--schedule", str(best),
        "--deployment-report", str(report), "--output-dir", str(result_dir / "replay"),
    ], cwd=output_root, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Predicted CIFAR-10 class: 8 (ship)" in completed.stdout

    metadata = json.loads(best.with_suffix(".json").read_text(encoding="utf-8"))
    rows = {
        row["occurrence"]: row
        for row in metadata["occurrences"]
    }
    report_cycles = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| tvmgen_mlperf_resnet_vta_main_") or ".conv0 |" not in line:
            continue
        fields = [part.strip() for part in line.strip("|").split("|")]
        occurrence = int(fields[0].split("_main_")[1].split(".")[0])
        report_cycles[occurrence] = int(fields[4].replace(",", ""))
    assert set(rows).issubset(report_cycles)
    if workload == -1:
        assert set(rows) == set(report_cycles)
    for occurrence, row in rows.items():
        standalone = int(row["measurement"]["results"][0]["costs"][0])
        assert cycles_within_strict_ten_percent(report_cycles[occurrence], standalone)
