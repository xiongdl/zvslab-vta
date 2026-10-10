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

"""CLI and propagation contracts for the standalone VTA build entry point."""

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_vta_lib.sh"
VTA_CMAKE = PROJECT_ROOT / "vta" / "CMakeLists.txt"


def _run_build(*args):
    return subprocess.run(
        ["bash", str(BUILD_SCRIPT), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_exposes_only_explicit_config_and_backend_selection():
    result = _run_build("--help")

    assert result.returncode == 0
    assert "--config ABS_PATH" in result.stdout
    assert "--backend BACKEND" in result.stdout
    assert "fsim, tsim, or all" in result.stdout
    assert "--target" not in result.stdout


def test_legacy_target_interface_fails_with_migration_message():
    result = _run_build("--target", "libvta_fsim")

    assert result.returncode != 0
    diagnostic = result.stdout + result.stderr
    assert "--target" in diagnostic
    assert "--config" in diagnostic
    assert "--backend fsim|tsim|all" in diagnostic


def test_unknown_backend_fails_before_toolchain_checks(tmp_path):
    config_path = tmp_path / "geometry.json"
    config_path.write_text("{}", encoding="utf-8")

    result = _run_build("--config", str(config_path), "--backend", "unknown")

    assert result.returncode != 0
    assert "--backend must be fsim, tsim, or all" in (result.stdout + result.stderr)


def test_config_path_is_forwarded_to_cmake_and_hardware_generation():
    script = BUILD_SCRIPT.read_text(encoding="utf-8")
    cmake = VTA_CMAKE.read_text(encoding="utf-8")

    assert '-DVTA_CONFIG_FILE="${config_file}"' in script
    assert '"VTA_CONFIG_FILE=${config_file}"' in script
    assert '"${VTA_CONFIG_FILE}"' in cmake
    assert '"${VTA_CONFIG_FILE}"' in cmake
    assert "vta_config.json" not in script
    assert "tsim_sample.json" not in script


def test_shell_script_is_syntactically_valid():
    result = subprocess.run(
        ["bash", "-n", str(BUILD_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_chisel_geometry_switch_uses_selected_config_even_when_older(tmp_path):
    """Changing config paths must not reuse a newer previous geometry target."""
    import json
    import os
    import sys
    import time
    base = json.loads((PROJECT_ROOT / 'vta/config/vta_64mac.json').read_text())
    configs = []
    for block_out in (3, 4):
        path = tmp_path / f'bo{block_out}.json'
        path.write_text(json.dumps(dict(base, LOG_BLOCK_IN=3, LOG_BLOCK_OUT=block_out)))
        os.utime(path, (time.time() - 100, time.time() - 100))
        configs.append(path)
    properties = tmp_path / 'geometry.properties'
    for config, expected in zip(configs, (8, 16)):
        subprocess.run(['make', '-C', str(PROJECT_ROOT / 'vta/hardware/chisel'),
                        f'VTA_PATH={PROJECT_ROOT / "vta"}', f'TVM_PATH={PROJECT_ROOT / "tvm"}',
                        f'PYTHON={sys.executable}', f'VTA_CONFIG_FILE={config}',
                        f'CHISEL_CONFIG_FILE={properties}', str(properties)], check=True, capture_output=True, text=True)
        values = dict(line.split('=', 1) for line in properties.read_text().splitlines() if '=' in line and not line.startswith('#'))
        assert int(values['BLOCK_OUT']) == expected
