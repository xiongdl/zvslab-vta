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
"""Explicit FSIM/TSIM backend loading and registry diagnostics."""

import ctypes
import json

import tvm

from ..backend import (
    SUPPORTED_BACKENDS,
    is_simulator_backend,
    normalize_backend,
)
from ..libinfo import find_libvta


BACKEND_LIBRARIES = {
    "fsim": ("libvta_fsim",),
    "tsim": ("libvta_tsim", "libvta_hw"),
}
BACKEND_REGISTRIES = {
    "fsim": (
        "vta.simulator.profiler_clear",
        "vta.simulator.profiler_status",
    ),
    "tsim": (
        "vta.tsim.init",
        "vta.tsim.profiler_clear",
        "vta.tsim.profiler_status",
        "runtime.module.loadfile_vta-tsim",
    ),
}
_loaded_libraries = {}


def _missing_libraries(backend):
    return [
        library
        for library in BACKEND_LIBRARIES[backend]
        if not find_libvta(library, optional=True)
    ]


def validate_backend_registries(backend=None, simulator=None):
    """Return missing registries for the selected explicit backend."""
    selected = normalize_backend(backend, simulator)
    return tuple(
        name
        for name in BACKEND_REGISTRIES[selected]
        if tvm.get_global_func(name, allow_missing=True) is None
    )


def load_backend(backend=None, simulator=None):
    """Load the selected backend libraries and initialize TSIM hardware."""
    selected = normalize_backend(backend, simulator)
    if selected in _loaded_libraries:
        return _loaded_libraries[selected]

    missing = _missing_libraries(selected)
    if missing:
        required = ", ".join(BACKEND_LIBRARIES[selected])
        raise RuntimeError(
            f"{selected.upper()} backend requires {required}; missing library "
            f"{', '.join(missing)}. Build the selected backend with "
            f"scripts/build_vta_lib.sh --config /absolute/path/to/vta_64mac.json "
            f"--backend {selected}"
        )

    paths = []
    handles = []
    try:
        for library in BACKEND_LIBRARIES[selected]:
            path = find_libvta(library)[0]
            handle = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            paths.append(path)
            handles.append(handle)
        if selected == "tsim":
            init = tvm.get_global_func("vta.tsim.init", allow_missing=True)
            if init is None:
                raise RuntimeError(
                    "TSIM backend library loaded but registry vta.tsim.init is missing; "
                    "libvta_tsim/libvta_hw are mismatched"
                )
            hardware = tvm.runtime.load_module(paths[1], "vta-tsim")
            init(hardware)
    except (OSError, RuntimeError) as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(
            f"{selected.upper()} backend library loading failed for {paths}: {error}"
        ) from error

    missing_registries = validate_backend_registries(selected)
    if missing_registries:
        raise RuntimeError(
            f"{selected.upper()} backend libraries are loaded but required registries "
            f"are missing: {', '.join(missing_registries)}; expected libraries: "
            f"{', '.join(BACKEND_LIBRARIES[selected])}"
        )
    _loaded_libraries[selected] = tuple(handles)
    return _loaded_libraries[selected]


def _load_sw(backend=None):
    """Compatibility-free internal loader used by explicit backend callers."""
    return load_backend(backend)


def enabled(backend=None):
    selected = normalize_backend(backend)
    return bool(validate_backend_registries(selected) == ())


def clear_stats(backend=None):
    selected = normalize_backend(backend)
    registry = (
        "vta.simulator.profiler_clear"
        if selected == "fsim"
        else "vta.tsim.profiler_clear"
    )
    function = tvm.get_global_func(registry, allow_missing=True)
    if function is None:
        raise RuntimeError(
            f"{selected.upper()} profiler registry is unavailable; required libraries: "
            f"{', '.join(BACKEND_LIBRARIES[selected])}"
        )
    function()


def stats(backend=None):
    selected = normalize_backend(backend)
    registry = (
        "vta.simulator.profiler_status"
        if selected == "fsim"
        else "vta.tsim.profiler_status"
    )
    function = tvm.get_global_func(registry, allow_missing=True)
    if function is None:
        raise RuntimeError(
            f"{selected.upper()} profiler registry is unavailable; required libraries: "
            f"{', '.join(BACKEND_LIBRARIES[selected])}"
        )
    raw = function()
    return json.loads(raw) if isinstance(raw, str) else raw


DEBUG_SKIP_EXEC = 1


def debug_mode(flag):
    tvm.get_global_func("vta.simulator.profiler_debug_mode")(flag)


# Importing this module is safe without a selector, but an explicit
# VTA_BACKEND eagerly validates its libraries for callers that use this module
# directly. Benchmark adapters still import it lazily after selecting a backend.
try:
    LIBS = load_backend()
except (RuntimeError, ValueError):
    LIBS = ()
