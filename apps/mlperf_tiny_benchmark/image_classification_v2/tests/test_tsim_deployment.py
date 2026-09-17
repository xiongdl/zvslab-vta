# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Focused TSIM adaptor, lazy-initialization, and matrix-contract tests."""

import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = APP_ROOT / "runtime.py"
RUN_PATH = APP_ROOT / "run.py"


def _load_runtime():
    spec = importlib.util.spec_from_file_location("mlperf_resnet_tsim_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def deployment_runtime():
    return _load_runtime()


def test_cli_preserves_v1_options_and_defaults(deployment_runtime):
    assert RUN_PATH.is_file(), f"missing Task 8 implementation: {RUN_PATH}"
    spec = importlib.util.spec_from_file_location("mlperf_resnet_large_run", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(APP_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    assert module._parser().parse_args([]).output_dir == str(deployment_runtime.DEFAULT_OUTPUT_DIR)
    assert module._parser().parse_args([]).host_codegen == "llvm"
    assert module._parser().parse_args([]).simulator == "fsim"
    assert module._parser().parse_args(["--output-dir", "out"]).output_dir == "out"
    assert module._parser().parse_args(["--host-codegen", "c"]).host_codegen == "c"
    assert module._parser().parse_args(["--host-codegen", "all"]).host_codegen == "all"
    assert module._parser().parse_args(["--simulator", "tsim"]).simulator == "tsim"


def test_tsim_mapping_is_explicit_and_never_uses_fsim_enabled(deployment_runtime):
    session = deployment_runtime._simulator_session("tsim")
    assert session.environment_target == "tsim"
    assert session.clear_registry == "vta.tsim.profiler_clear"
    assert session.status_registry == "vta.tsim.profiler_status"
    assert session.required_registries == (
        "vta.tsim.init",
        "vta.tsim.profiler_clear",
        "vta.tsim.profiler_status",
        "runtime.module.loadfile_vta-tsim",
    )
    assert session.activity_counter == "cycle_count"
    assert "--target libvta_hw" in session.diagnostic
    assert "enabled" not in deployment_runtime.SimulatorSession.load.__code__.co_names


def test_tsim_rejects_wrong_environment_before_model_preparation(deployment_runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="sim"))
    monkeypatch.setattr(
        deployment_runtime,
        "prepare_model",
        lambda *_: pytest.fail("model preparation must not start for a target mismatch"),
    )
    with pytest.raises(RuntimeError, match="requires VTA target 'tsim'"):
        deployment_runtime.deploy_tsim_matrix(tmp_path)


def test_tsim_missing_registry_reports_build_command(deployment_runtime, monkeypatch):
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim"))
    monkeypatch.setattr(
        deployment_runtime.tvm,
        "get_global_func",
        lambda name, allow_missing=False: None if name == "vta.tsim.init" else object(),
    )
    with pytest.raises(RuntimeError, match=r"vta\.tsim\.init.*libvta_hw"):
        deployment_runtime._simulator_session("tsim").load()


def test_tsim_stats_require_exact_zero_reset_and_positive_integer_cycles(deployment_runtime):
    session = deployment_runtime._simulator_session("tsim")
    assert session.read_stats(lambda: '{"cycle_count": 0}') == {"cycle_count": 0}
    session.validate_activity({"cycle_count": 1})

    with pytest.raises(RuntimeError, match="cycle_count"):
        session.clear_and_validate(
            SimpleNamespace(clear_stats=lambda: None, stats=lambda: {"cycle_count": 1})
        )
    for value in (0, -1, True, 1.5, "1"):
        with pytest.raises(RuntimeError, match="positive integer"):
            session.validate_activity({"cycle_count": value})
    with pytest.raises(RuntimeError, match="malformed"):
        session.read_stats(lambda: "not-json")


def test_tsim_root_and_public_matrix_alias(deployment_runtime):
    assert deployment_runtime._matrix_artifact_root(Path("out"), "llvm", "tsim") == Path(
        "out/llvm-tsim"
    )
    assert deployment_runtime._matrix_artifact_root(Path("out"), "c", "tsim") == Path(
        "out/c-tsim"
    )
    assert deployment_runtime.FsimMatrixResult is deployment_runtime.SimulationMatrixResult


def test_tsim_matrix_builds_all_hosts_before_single_lazy_load(deployment_runtime, monkeypatch, tmp_path):
    prepared = SimpleNamespace(
        routing=SimpleNamespace(
            symbols=tuple("s%d" % i for i in range(4)),
            composite_names=(),
            host_operator_names=(),
        )
    )
    events = []
    artifacts = []

    monkeypatch.setattr(deployment_runtime, "prepare_model", lambda *_: prepared)
    monkeypatch.setattr(deployment_runtime, "committed_sample_paths", lambda: ())
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim"))

    def fake_build(*args, **kwargs):
        events.append(("build", kwargs["host_codegen"], kwargs["simulator"]))
        artifact = SimpleNamespace(
            host_codegen=kwargs["host_codegen"],
            mixed=SimpleNamespace(module=object()),
            reference=SimpleNamespace(module=object()),
            vta_symbols=prepared.routing.symbols,
        )
        artifacts.append(artifact)
        return artifact

    class FakeSimulator:
        def __init__(self):
            self.after_clear = False

        def clear_stats(self):
            events.append(("clear",))
            self.after_clear = True

        def stats(self):
            events.append(("stats",))
            if self.after_clear:
                self.after_clear = False
                return '{"cycle_count": 0}'
            return '{"cycle_count": 1}'

    simulator = FakeSimulator()
    monkeypatch.setattr(deployment_runtime, "build_host_artifacts", fake_build)
    monkeypatch.setattr(deployment_runtime, "_load_simulator", lambda label: (deployment_runtime._simulator_session(label), simulator))
    monkeypatch.setattr(deployment_runtime, "_run_graph", lambda *_: object())
    monkeypatch.setattr(deployment_runtime, "load_sample", lambda *_: object())
    monkeypatch.setattr(deployment_runtime, "validate_mixed_symbols", lambda *_: None)

    result = deployment_runtime.deploy_tsim_matrix(tmp_path)
    assert [event for event in events if event[0] == "build"] == [
        ("build", "llvm", "tsim"),
        ("build", "c", "tsim"),
    ]
    assert len(result.artifacts) == 2


def test_tsim_matrix_result_records_simulator_and_is_frozen(deployment_runtime, monkeypatch, tmp_path):
    prepared = SimpleNamespace(
        routing=SimpleNamespace(symbols=(), composite_names=(), host_operator_names=())
    )
    monkeypatch.setattr(deployment_runtime, "prepare_model", lambda *_: prepared)
    monkeypatch.setattr(deployment_runtime, "committed_sample_paths", lambda: ())
    monkeypatch.setattr(deployment_runtime.vta, "get_env", lambda: SimpleNamespace(TARGET="tsim"))
    monkeypatch.setattr(deployment_runtime, "build_host_artifacts", lambda *args, **kwargs: ())
    monkeypatch.setattr(deployment_runtime, "_execute_matrix", lambda *args: ())

    result = deployment_runtime.deploy_tsim_matrix(tmp_path)

    assert result.simulator == "tsim"
    with pytest.raises(FrozenInstanceError):
        result.simulator = "fsim"


def test_end_to_end_tsim_matrix_with_reloaded_graph_bundles(deployment_runtime, tmp_path):
    """Run the complete ten-sample TSIM matrix under tsim_sample.json."""
    result = deployment_runtime.deploy_tsim_matrix(tmp_path)
    assert len(result.prepared.routing.symbols) == 4
    assert len(result.artifacts) == 2
    assert all(len(execution.comparisons) == 10 for execution in result.executions)
    assert all(
        isinstance(execution.profiler_stats["cycle_count"], int)
        and execution.profiler_stats["cycle_count"] > 0
        for execution in result.executions
    )

    for host_artifacts in result.artifacts:
        assert host_artifacts.reference.artifact_dir.parent.name == f"{host_artifacts.host_codegen}-tsim"
        assert host_artifacts.mixed.artifact_dir.parent.name == f"{host_artifacts.host_codegen}-tsim"
        for artifact in (host_artifacts.reference, host_artifacts.mixed):
            manifest = json.loads(
                (artifact.artifact_dir / "manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["simulator"] == "tsim"
            assert manifest["host_codegen"] == host_artifacts.host_codegen
            assert artifact.path.is_file()
            assert (artifact.artifact_dir / "graph.json").is_file()
            assert (artifact.artifact_dir / "params.bin").is_file()
            assert any(entry["path"] for entry in manifest["sources"])
