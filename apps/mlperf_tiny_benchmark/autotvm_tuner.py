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

"""AutoTVM support for simulator-backed MLPerf Tiny schedules."""

import contextlib
import json
import os
import time

import tvm
from tvm import autotvm
from tvm.autotvm.measure import MeasureErrorNo, MeasureInput, MeasureResult
from tvm.autotvm.measure.measure_methods import (
    LocalRunner,
    RPCRunner,
    check_remote,
    default_module_loader,
)


PROFILER_REGISTRIES = {
    "fsim": ("vta.simulator.profiler_clear", "vta.simulator.profiler_status"),
    "tsim": ("vta.tsim.profiler_clear", "vta.tsim.profiler_status"),
}


def validate_backend(backend):
    """Require an explicit backend selector matching the requested simulator."""
    if backend not in PROFILER_REGISTRIES:
        raise ValueError(f"unsupported backend {backend!r}; expected fsim or tsim")
    active_backend = os.environ.get("VTA_BACKEND")
    if active_backend is None:
        raise ValueError("VTA_BACKEND must explicitly select fsim or tsim")
    if active_backend != backend:
        raise ValueError(
            f"simulator backend mismatch: requested {backend!r}, "
            f"VTA_BACKEND is {active_backend!r}"
        )
    return backend


def load_simulator_backend(backend):
    """Load the selected simulator and fail with its missing-library diagnostics."""
    validate_backend(backend)
    from vta.testing import simulator

    simulator.load_backend(backend)
    missing = [
        name
        for name in simulator.BACKEND_REGISTRIES[backend]
        if tvm.get_global_func(name, allow_missing=True) is None
    ]
    if missing:
        raise RuntimeError(
            f"{backend.upper()} profiler/runtime registries are unavailable: "
            f"{', '.join(missing)}"
        )


def simulator_server_libraries(backend):
    """Return the runtime libraries to load in AutoTVM's local RPC workers."""
    import vta
    from vta.libinfo import _find_compiler_extension, find_libvta

    names = ("libvta_fsim",) if backend == "fsim" else ("libvta_tsim",)
    return ":".join([_find_compiler_extension()] + [find_libvta(name)[0] for name in names])


def tsim_hardware_library():
    """Return the built TSIM hardware module path."""
    from vta.libinfo import find_libvta

    return find_libvta("libvta_hw")[0]


class SimulatorLocalRunner(LocalRunner):
    """Local RPC runner that initializes VTA only on the loopback server."""

    def __init__(self, backend, **kwargs):
        self.backend = backend
        super().__init__(**kwargs)

    def set_task(self, task):
        from tvm.rpc.server import Server
        from tvm.rpc.tracker import Tracker

        self.task = task
        tracker = Tracker(host="127.0.0.1", port=9000, port_end=10000, silent=True)
        device_key = f"$local$device${tracker.port}"
        server = Server(
            host="127.0.0.1",
            port=9000,
            port_end=10000,
            key=device_key,
            load_library=simulator_server_libraries(self.backend),
            silent=True,
            tracker_addr=("127.0.0.1", tracker.port),
        )
        self.tracker = tracker
        self.server = server
        self.key = device_key
        self.host = "127.0.0.1"
        self.port = tracker.port
        if check_remote(tvm.target.Target("ext_dev"), self.key, self.host, self.port):
            return server, tracker
        raise RuntimeError("VTA ext_dev device is unavailable in the local simulator RPC server")

    def run(self, measure_inputs, build_results):
        runtime_inputs = [
            MeasureInput(tvm.target.Target("ext_dev"), measure_input.task, measure_input.config)
            for measure_input in measure_inputs
        ]
        return super().run(runtime_inputs, build_results)


class ProfilerModuleLoader:
    """Wrap AutoTVM's local RPC loader with per-candidate profiler handling."""

    def __init__(self, backend, base_loader=None, collected=None):
        self.backend = backend
        self.base_loader = base_loader or default_module_loader()
        self.collected = collected if collected is not None else {}

    @contextlib.contextmanager
    def __call__(self, remote_kwargs, build_result):
        clear_registry, status_registry = PROFILER_REGISTRIES[self.backend]
        filename = build_result.filename
        with self.base_loader(remote_kwargs, build_result) as (remote, module):
            try:
                if self.backend == "tsim":
                    hardware_path = tsim_hardware_library()
                    hardware_name = os.path.basename(hardware_path)
                    remote.upload(hardware_path)
                    load_hardware = remote.get_function("runtime.module.loadfile_vta-tsim")
                    hardware_module = load_hardware(hardware_name)
                    remote.get_function("vta.tsim.init")(hardware_module)
                    remote.remove(hardware_name)
                clear = remote.get_function(clear_registry)
                status = remote.get_function(status_registry)
            except Exception as error:
                libraries = "libvta_tsim and libvta_hw" if self.backend == "tsim" else "libvta_fsim"
                raise RuntimeError(
                    f"{self.backend.upper()} AutoTVM profiler registries are unavailable "
                    f"in the local runner; build {libraries} for the selected config"
                ) from error
            clear()
            reset_raw = status()
            reset_stats = json.loads(reset_raw) if isinstance(reset_raw, str) else reset_raw
            if self.backend == "tsim":
                if reset_stats != {"cycle_count": 0}:
                    raise RuntimeError(
                        f"TSIM profiler did not reset before candidate measurement: {reset_stats}"
                    )
            elif any(value != 0 for value in reset_stats.values()):
                raise RuntimeError(
                    f"FSIM profiler did not reset before candidate measurement: {reset_stats}"
                )
            try:
                yield remote, module
            finally:
                if self.backend == "tsim":
                    try:
                        raw = status()
                        stats = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception as error:
                        raise RuntimeError("TSIM AutoTVM profiler status could not be read") from error
                    self.collected[filename] = stats


def tsim_cycle_cost(stats):
    """Validate and return the TSIM cycle counter as the AutoTVM trial cost."""
    value = stats.get("cycle_count") if isinstance(stats, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"TSIM profiler counter cycle_count must be a positive integer: {stats}"
        )
    return value


class TSIMLocalRunner(SimulatorLocalRunner):
    """Use native TSIM cycle counts instead of local RPC wall-clock durations."""

    def __init__(self, **kwargs):
        self.cycle_stats = {}
        loader = ProfilerModuleLoader("tsim", collected=self.cycle_stats)
        super().__init__("tsim", module_loader=loader, **kwargs)

    def run(self, measure_inputs, build_results):
        results = super().run(measure_inputs, build_results)
        converted = []
        for build_result, result in zip(build_results, results):
            if result.error_no != MeasureErrorNo.NO_ERROR:
                converted.append(result)
                continue
            try:
                cost = tsim_cycle_cost(self.cycle_stats[build_result.filename])
            except (KeyError, RuntimeError) as error:
                converted.append(
                    MeasureResult(
                        (str(error),), MeasureErrorNo.RUNTIME_DEVICE, result.all_cost, time.time()
                    )
                )
                continue
            converted.append(
                MeasureResult((cost,), MeasureErrorNo.NO_ERROR, result.all_cost, result.timestamp)
            )
        return converted


def create_runner(backend, timeout=60, number=1, repeat=1, cooldown_interval=0):
    """Create an isolated backend runner with cycle-based TSIM costs."""
    validate_backend(backend)
    load_simulator_backend(backend)
    runner_options = {
        "timeout": timeout,
        "number": number,
        "repeat": repeat,
        "cooldown_interval": cooldown_interval,
    }
    if backend == "tsim":
        return TSIMLocalRunner(**runner_options)
    return SimulatorLocalRunner("fsim", module_loader=ProfilerModuleLoader("fsim"), **runner_options)


def measure_option(backend, **runner_options):
    """Build local schedules and measure them using the selected simulator."""
    return autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=1),
        runner=create_runner(backend, **runner_options),
    )
