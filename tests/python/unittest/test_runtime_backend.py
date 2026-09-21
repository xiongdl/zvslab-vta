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

"""Focused tests for explicit MLPerf Tiny FSIM/TSIM runtime selection."""

import json
from pathlib import Path

import pytest

from vta import environment
from vta.testing import simulator


VTA_ROOT = Path(__file__).resolve().parents[3]


def test_backend_selection_uses_explicit_values_or_environment(monkeypatch):
    monkeypatch.setenv("VTA_BACKEND", "fsim")

    assert simulator.normalize_backend() == "fsim"
    with pytest.raises(ValueError, match="backend mismatch"):
        simulator.normalize_backend(simulator="tsim")


def test_backend_mismatch_is_rejected_before_loading(monkeypatch):
    monkeypatch.setenv("VTA_BACKEND", "fsim")

    with pytest.raises(ValueError, match="backend mismatch"):
        simulator.normalize_backend(backend="fsim", simulator="tsim")


def test_unknown_backend_reports_canonical_values(monkeypatch):
    monkeypatch.delenv("VTA_BACKEND", raising=False)

    with pytest.raises(ValueError, match=r"fsim.*tsim"):
        simulator.normalize_backend()


def test_fsim_missing_library_diagnostic_names_selected_library(monkeypatch):
    monkeypatch.setattr(simulator, "_loaded_libraries", {})
    monkeypatch.setattr(simulator, "find_libvta", lambda name, optional=False: [])

    with pytest.raises(RuntimeError, match=r"FSIM.*libvta_fsim"):
        simulator.load_backend("fsim")


def test_tsim_missing_library_diagnostic_names_both_required_libraries(monkeypatch):
    monkeypatch.setenv("VTA_BACKEND", "tsim")
    monkeypatch.setattr(simulator, "_loaded_libraries", {})
    monkeypatch.setattr(simulator, "find_libvta", lambda name, optional=False: [])

    with pytest.raises(RuntimeError, match=r"TSIM.*libvta_tsim.*libvta_hw"):
        simulator.load_backend("tsim")


def test_registry_diagnostic_identifies_backend_and_missing_registry(monkeypatch):
    monkeypatch.setattr(simulator, "_loaded_libraries", {})
    monkeypatch.setattr(
        simulator,
        "find_libvta",
        lambda name, optional=False: [f"/tmp/{name}.dylib"],
    )
    monkeypatch.setattr(simulator.ctypes, "CDLL", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        simulator.tvm,
        "get_global_func",
        lambda name, allow_missing=False: None,
    )

    with pytest.raises(RuntimeError, match=r"FSIM.*registries.*libvta_fsim"):
        simulator.load_backend("fsim")


def test_legacy_target_config_is_rejected_by_environment_loader(tmp_path, monkeypatch):
    config_path = tmp_path / "legacy.json"
    config_path.write_text(
        json.dumps(
            {
                "TARGET": "sim",
                "HW_VER": "0.0.2",
                "LOG_INP_WIDTH": 3,
                "LOG_WGT_WIDTH": 3,
                "LOG_ACC_WIDTH": 5,
                "LOG_BATCH": 0,
                "LOG_BLOCK": 3,
                "LOG_UOP_BUFF_SIZE": 12,
                "LOG_INP_BUFF_SIZE": 13,
                "LOG_WGT_BUFF_SIZE": 14,
                "LOG_ACC_BUFF_SIZE": 15,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(environment, "get_vta_config_path", lambda: str(config_path))

    with pytest.raises(ValueError, match=r"Legacy TARGET.*VTA_BACKEND"):
        environment._init_env()


def test_benchmark_runtime_sources_do_not_compare_environment_target():
    runtime_paths = sorted((VTA_ROOT / "apps" / "mlperf_tiny_benchmark").glob("*/runtime.py"))

    assert runtime_paths
    for runtime_path in runtime_paths:
        source = runtime_path.read_text(encoding="utf-8")
        assert "vta.get_env().TARGET" not in source
        assert "--target libvta_" not in source
