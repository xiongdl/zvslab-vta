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

"""Focused tests for the VTA geometry/backend configuration boundary."""

import importlib.util
import json
from pathlib import Path

import pytest


VTA_ROOT = Path(__file__).resolve().parents[3]
CONFIG_TOOL_PATH = VTA_ROOT / "config" / "vta_config.py"


def _load_config_tool():
    spec = importlib.util.spec_from_file_location("vta_config_tool", CONFIG_TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def geometry_config():
    return {
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


@pytest.mark.parametrize("backend", ["fsim", "tsim"])
def test_supported_backends_normalize_to_themselves(backend):
    config_tool = _load_config_tool()

    assert config_tool.normalize_backend(backend) == backend


@pytest.mark.parametrize("backend", ["sim", "unknown", "", "FSIM"])
def test_unknown_backend_fails_with_supported_values(backend):
    config_tool = _load_config_tool()

    with pytest.raises(ValueError, match=r"VTA_BACKEND.*fsim.*tsim"):
        config_tool.normalize_backend(backend)


def test_missing_backend_fails_with_explicit_environment_variable(monkeypatch):
    config_tool = _load_config_tool()
    monkeypatch.delenv("VTA_BACKEND", raising=False)

    with pytest.raises(ValueError, match=r"VTA_BACKEND.*fsim.*tsim"):
        config_tool.normalize_backend()


@pytest.mark.parametrize("target", ["sim", "tsim"])
def test_legacy_simulator_target_is_rejected_with_migration_error(
    tmp_path, geometry_config, target
):
    config_tool = _load_config_tool()
    config_path = tmp_path / "legacy.json"
    geometry_config["TARGET"] = target
    config_path.write_text(json.dumps(geometry_config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"TARGET=.*VTA_BACKEND"):
        config_tool.load_geometry_config(config_path, backend="fsim")


def test_backend_does_not_change_geometry_abi_fingerprint(geometry_config):
    config_tool = _load_config_tool()

    fsim_definitions = config_tool.abi_definitions(geometry_config, backend="fsim")
    tsim_definitions = config_tool.abi_definitions(geometry_config, backend="tsim")

    assert fsim_definitions == tsim_definitions
    assert config_tool.abi_fingerprint(fsim_definitions) == config_tool.abi_fingerprint(
        tsim_definitions
    )
