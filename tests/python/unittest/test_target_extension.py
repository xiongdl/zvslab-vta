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

"""Isolated-process contracts for the native VTA target extension."""

import shutil
import sys
from pathlib import Path

import pytest

from byoc_utils import run_isolated_python


VTA_ROOT = Path(__file__).resolve().parents[3]
BUILD_COMMAND = (
    "./scripts/build_vta_lib.sh --config /absolute/path/to/vta_64mac.json --backend fsim"
)


def _extension_name():
    if sys.platform.startswith("win32"):
        return "libtvm-vta-ext.dll"
    if sys.platform.startswith("darwin"):
        return "libtvm-vta-ext.dylib"
    return "libtvm-vta-ext.so"


def _isolated_vta_environment(vta_root):
    config_dir = vta_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VTA_ROOT / "config" / "pkg_config.py", config_dir / "pkg_config.py")
    return {
        "VTA_CONFIG_FILE": str(VTA_ROOT / "config" / "vta_64mac.json"),
        "VTA_BACKEND": "fsim",
        "VTA_PATH": str(vta_root),
    }


def test_full_import_registers_one_native_vta_target_idempotently():
    extension_path = VTA_ROOT / "build" / _extension_name()
    result = run_isolated_python(
        f"""
        import ctypes
        import importlib
        import os

        import tvm

        extension_path = {str(extension_path)!r}
        extension_loads = []
        real_cdll = ctypes.CDLL

        def tracking_cdll(path, *args, **kwargs):
            if os.path.abspath(os.fspath(path)) == extension_path:
                extension_loads.append(kwargs.get("mode"))
            return real_cdll(path, *args, **kwargs)

        ctypes.CDLL = tracking_cdll

        legacy_compiler_global = "relay.ext." + "vta"
        assert "vta" not in tvm.target.Target.list_kinds()
        assert tvm.get_global_func(legacy_compiler_global, allow_missing=True) is None

        import vta

        assert extension_loads == [getattr(ctypes, "RTLD_GLOBAL", 0)]
        assert tvm.target.Target.list_kinds().count("vta") == 1
        first = tvm.target.Target("vta")
        assert first.kind.name == "vta"
        assert first.get_target_device_type() == tvm.runtime.Device.kDLExtDev
        first_relay_to_tir = first.get_kind_attr("RelayToTIR")
        first_tir_to_runtime = first.get_kind_attr("TIRToRuntime")
        assert isinstance(first_relay_to_tir, tvm.transform.ModulePass)
        assert isinstance(first_tir_to_runtime, tvm.runtime.PackedFunc)
        first_kind_handle = first.kind.handle.value
        first_relay_to_tir_handle = first_relay_to_tir.handle.value
        first_tir_to_runtime_handle = first_tir_to_runtime.handle.value
        assert tvm.get_global_func(legacy_compiler_global, allow_missing=True) is None

        importlib.reload(vta)

        assert extension_loads == [getattr(ctypes, "RTLD_GLOBAL", 0)]
        assert tvm.target.Target.list_kinds().count("vta") == 1
        second = tvm.target.Target("vta")
        assert second.kind.handle.value == first_kind_handle
        second_relay_to_tir = second.get_kind_attr("RelayToTIR")
        second_tir_to_runtime = second.get_kind_attr("TIRToRuntime")
        assert isinstance(second_relay_to_tir, tvm.transform.ModulePass)
        assert isinstance(second_tir_to_runtime, tvm.runtime.PackedFunc)
        assert second_relay_to_tir.handle.value == first_relay_to_tir_handle
        assert second_tir_to_runtime.handle.value == first_tir_to_runtime_handle
        assert tvm.get_global_func(legacy_compiler_global, allow_missing=True) is None
        """
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing_hook", ["RelayToTIR", "TIRToRuntime"])
def test_full_import_identifies_each_missing_required_hook(missing_hook):
    extension_path = VTA_ROOT / "build" / _extension_name()
    result = run_isolated_python(
        f"""
        import ctypes
        import os

        import tvm

        extension_path = {str(extension_path)!r}
        missing_hook = {missing_hook!r}
        real_cdll = ctypes.CDLL
        real_get_kind_attr = tvm.target.Target.get_kind_attr

        def tracking_cdll(path, *args, **kwargs):
            extension = real_cdll(path, *args, **kwargs)
            if os.path.abspath(os.fspath(path)) == extension_path:
                def incomplete_get_kind_attr(target, name):
                    if target.kind.name == "vta" and name == missing_hook:
                        return None
                    return real_get_kind_attr(target, name)

                tvm.target.Target.get_kind_attr = incomplete_get_kind_attr
            return extension

        ctypes.CDLL = tracking_cdll

        try:
            import vta
        except ImportError as err:
            message = str(err)
            assert extension_path in message
            assert f"missing required {{missing_hook}} hook" in message
            assert {BUILD_COMMAND!r} in message
        else:
            raise AssertionError(f"missing {{missing_hook}} hook did not fail import")
        """
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("wrong_hook", "replacement_hook", "expected_type"),
    [
        ("RelayToTIR", "TIRToRuntime", "tvm.transform.ModulePass"),
        ("TIRToRuntime", "RelayToTIR", "tvm.runtime.PackedFunc"),
    ],
)
def test_full_import_identifies_each_incompatible_hook_type(
    wrong_hook, replacement_hook, expected_type
):
    extension_path = VTA_ROOT / "build" / _extension_name()
    result = run_isolated_python(
        f"""
        import ctypes
        import os

        import tvm

        extension_path = {str(extension_path)!r}
        wrong_hook = {wrong_hook!r}
        replacement_hook = {replacement_hook!r}
        expected_type = {expected_type!r}
        real_cdll = ctypes.CDLL
        real_get_kind_attr = tvm.target.Target.get_kind_attr

        def tracking_cdll(path, *args, **kwargs):
            extension = real_cdll(path, *args, **kwargs)
            if os.path.abspath(os.fspath(path)) == extension_path:
                target = tvm.target.Target("vta")
                incompatible_hook = real_get_kind_attr(target, replacement_hook)

                def incompatible_get_kind_attr(candidate, name):
                    if candidate.kind.name == "vta" and name == wrong_hook:
                        return incompatible_hook
                    return real_get_kind_attr(candidate, name)

                tvm.target.Target.get_kind_attr = incompatible_get_kind_attr
            return extension

        ctypes.CDLL = tracking_cdll

        try:
            import vta
        except ImportError as err:
            message = str(err)
            assert extension_path in message
            assert f"{{wrong_hook}} hook has incompatible type" in message
            assert expected_type in message
            assert {BUILD_COMMAND!r} in message
        else:
            raise AssertionError(f"incompatible {{wrong_hook}} hook did not fail import")
        """
    )

    assert result.returncode == 0, result.stderr


def test_runtime_only_import_does_not_search_for_or_load_compiler_extension(tmp_path):
    extension_name = _extension_name()
    result = run_isolated_python(
        f"""
        import ctypes
        import os

        import tvm._ffi.base

        extension_name = {extension_name!r}
        searched_paths = []
        loaded_paths = []
        real_exists = os.path.exists
        real_cdll = ctypes.CDLL

        def tracking_exists(path):
            if extension_name in os.fspath(path):
                searched_paths.append(os.fspath(path))
            return real_exists(path)

        def tracking_cdll(path, *args, **kwargs):
            if extension_name in os.fspath(path):
                loaded_paths.append(os.fspath(path))
            return real_cdll(path, *args, **kwargs)

        os.path.exists = tracking_exists
        ctypes.CDLL = tracking_cdll
        tvm._ffi.base._RUNTIME_ONLY = True

        import vta

        assert searched_paths == []
        assert loaded_paths == []
        assert "vta" not in tvm.target.Target.list_kinds()
        """,
        env=_isolated_vta_environment(tmp_path),
    )

    assert result.returncode == 0, result.stderr


def test_full_import_reports_actionable_missing_extension(tmp_path):
    extension_path = tmp_path / "build" / _extension_name()
    result = run_isolated_python(
        f"""
        import tvm

        try:
            import vta
        except ImportError as err:
            message = str(err)
            assert {str(extension_path)!r} in message
            assert {BUILD_COMMAND!r} in message
        else:
            raise AssertionError("missing VTA target extension did not fail the import")
        """,
        env=_isolated_vta_environment(tmp_path),
    )

    assert result.returncode == 0, result.stderr


def test_full_import_preserves_unloadable_extension_error(tmp_path):
    extension_path = tmp_path / "build" / _extension_name()
    extension_path.parent.mkdir()
    extension_path.write_bytes(b"invalid shared library")
    loader_error = "synthetic unresolved TVM symbol"
    result = run_isolated_python(
        f"""
        import ctypes
        import os

        import tvm

        extension_path = {str(extension_path)!r}
        real_cdll = ctypes.CDLL

        def failing_cdll(path, *args, **kwargs):
            if os.path.abspath(os.fspath(path)) == extension_path:
                raise OSError({loader_error!r})
            return real_cdll(path, *args, **kwargs)

        ctypes.CDLL = failing_cdll

        try:
            import vta
        except ImportError as err:
            message = str(err)
            assert extension_path in message
            assert {loader_error!r} in message
            assert {BUILD_COMMAND!r} in message
        else:
            raise AssertionError("unloadable VTA target extension did not fail the import")
        """,
        env=_isolated_vta_environment(tmp_path),
    )

    assert result.returncode == 0, result.stderr
