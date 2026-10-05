"""Isolated measurement of one actual outlined VTA deployment layer."""

import hashlib
import json
import math
import multiprocessing
import os
import queue
import re
import time
import traceback

import numpy as np


DEFAULT_TIMEOUT_SECONDS = {"fsim": 60, "tsim": 120}


class MeasurementInfrastructureError(RuntimeError):
    """The worker could not initialize the requested compiler/backend."""


def _selection(layer, config_indices):
    tunable = [
        (position, entry)
        for position, entry in enumerate(layer.config_spaces)
        if entry[0] != "add.vta" and len(entry[3]) > 1
    ]
    if len(config_indices) == len(layer.config_spaces):
        indexed = [(entry, config_indices[position]) for position, entry in tunable]
    elif len(config_indices) == len(tunable):
        indexed = [(entry, index) for (_, entry), index in zip(tunable, config_indices)]
    else:
        raise ValueError(
            f"candidate requires {len(tunable)} tunable config indices, got {len(config_indices)}"
        )
    selected = []
    for (template, workload, target, space), index in indexed:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("config indices must be integers")
        if index < 0 or index >= len(space):
            raise ValueError(f"config index {index} is outside {template} space [0, {len(space)})")
        entity = space.get(index)
        if not entity.valid():
            raise ValueError(f"config index {index} is invalid for {template}")
        selected.append((template, workload, target, entity))
    identity = [
        {"template": template, "workload": repr(workload),
         "target": re.sub(r"(?<=-model=)(?:fsim|tsim)_", "sim_", str(target)),
         "config": entity.to_json_dict()}
        for template, workload, target, entity in selected
    ]
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(selected), digest


from .autotvm_dispatch import config_bindings_context


def _single_layer_module(function):
    import tvm
    from tvm import relay

    symbol = function.attrs.get_str("global_symbol")
    global_var = relay.GlobalVar(symbol)
    module = tvm.IRModule()
    module[global_var] = function
    main = relay.Function(
        function.params,
        relay.Call(global_var, list(function.params)),
    )
    module["main"] = main
    return relay.transform.InferType()(module)


def _evaluate_candidate(function_json, activation, config_indices, backend, candidate_started):
    # Select the worker's backend before importing VTA modules so every
    # process-local environment lookup observes its explicit simulator.
    if backend not in DEFAULT_TIMEOUT_SECONDS:
        raise ValueError(f"unsupported backend {backend!r}; expected fsim or tsim")
    os.environ["VTA_BACKEND"] = backend
    try:
        import tvm
        import vta
        import vta.relay
        from tvm import relay
        from tvm.contrib import graph_executor
        from tvm.relay.backend import te_compiler
        from vta.relay import transform
        from vta.testing import simulator

        function = tvm.ir.load_json(function_json)
        compiler_config = transform.VTACompilerConfig.from_env(vta.get_env())
        from .vta_workload import _capture_layer

        layer = _capture_layer(0, function, compiler_config)
        simulator.load_backend(backend)
    except BaseException as error:
        raise MeasurementInfrastructureError(
            f"could not initialize {backend.upper()} compiler/runtime: {error}"
        ) from error
    candidate_started.set()
    selected, identity = _selection(layer, config_indices)
    compiler = te_compiler.get()
    compiler.clear()
    try:
        bindings = ((target, workload, entity) for _, workload, target, entity in selected)
        with config_bindings_context(bindings), vta.build_config():
            factory = relay.build(
                _single_layer_module(function),
                target=tvm.target.Target("vta", host=compiler_config.host_target),
            )
    finally:
        compiler.clear()

    device = tvm.device(compiler_config.device_type, 0)
    executor = graph_executor.create(factory.get_graph_json(), factory.get_lib(), device)
    executor.load_params(tvm.runtime.save_param_dict(factory.get_params()))
    input_name = function.params[0].name_hint
    activation = np.asarray(activation)
    expected = layer.inputs[0]
    if tuple(activation.shape) != expected.shape or str(activation.dtype) != expected.dtype:
        raise ValueError(
            f"activation must have shape {expected.shape} and dtype {expected.dtype}; "
            f"got {activation.shape} and {activation.dtype}"
        )
    executor.set_input(input_name, tvm.nd.array(activation, device))
    start = time.monotonic()
    executor.run()
    duration_seconds = max(time.monotonic() - start, 1e-9)
    output = executor.get_output(0).numpy()
    cycles = None
    protocol = None
    if backend == "tsim":
        simulator.clear_stats(backend)
        executor.run()
        stats = simulator.stats(backend)
        cycles = stats.get("cycle_count")
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise RuntimeError(f"TSIM must report positive cycle_count for one call, got {stats}")
        protocol = {
            "name": "tsim_single_call",
            "version": 1,
            "counted_invocations": 1,
            "warmup_excluded": True,
        }
        output = executor.get_output(0).numpy()

    reference_function = relay.Function(function.params, function.body)
    reference_module = tvm.IRModule.from_expr(reference_function)
    reference = relay.create_executor(
        "debug", mod=reference_module, device=tvm.cpu(), target="llvm"
    ).evaluate()(*[tvm.nd.array(activation)])
    reference = reference.numpy()
    if not np.array_equal(reference, output):
        raise RuntimeError("candidate layer output differs from the actual Relay function")
    return {
        "output": output,
        "cycles": cycles,
        "protocol": protocol,
        "config_identity": identity,
        "worker_pid": os.getpid(),
        "timestamp": time.time(),
        "duration_seconds": duration_seconds,
    }


def _worker(function_json, activation, config_indices, backend, result_queue, candidate_started):
    try:
        result = _evaluate_candidate(
            function_json, activation, config_indices, backend, candidate_started
        )
        result_queue.put((True, False, result))
    except BaseException as error:
        result_queue.put((False, isinstance(error, MeasurementInfrastructureError), traceback.format_exc()))


def measure_candidate(layer, activation, config_indices, backend, timeout=None):
    """Build and run one candidate in a fresh process with isolated TVM caches.

    One warmup is excluded from TSIM counters; after clearing the simulator,
    exactly one counted invocation supplies the returned native cycle count.
    """
    if backend not in DEFAULT_TIMEOUT_SECONDS:
        raise ValueError(f"unsupported backend {backend!r}; expected fsim or tsim")
    timeout = DEFAULT_TIMEOUT_SECONDS[backend] if timeout is None else timeout
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be finite positive seconds")
    try:
        finite_timeout = math.isfinite(timeout)
    except OverflowError:
        finite_timeout = False
    if not finite_timeout or timeout <= 0:
        raise ValueError("timeout must be finite positive seconds")
    _, _ = _selection(layer, config_indices)
    activation = np.asarray(activation)
    expected = layer.inputs[0]
    if tuple(activation.shape) != expected.shape or str(activation.dtype) != expected.dtype:
        raise ValueError(
            f"activation must have shape {expected.shape} and dtype {expected.dtype}; "
            f"got {activation.shape} and {activation.dtype}"
        )
    import tvm

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    candidate_started = context.Event()
    process = context.Process(
        target=_worker,
        args=(
            tvm.ir.save_json(layer.function), activation, list(config_indices), backend,
            result_queue, candidate_started,
        ),
    )
    started = False
    try:
        process.start()
        started = True
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if candidate_started.is_set():
                    raise TimeoutError(
                        f"{backend.upper()} candidate worker exceeded {timeout} seconds"
                    )
                raise MeasurementInfrastructureError(
                    f"{backend.upper()} worker timed out during compiler/runtime initialization"
                )
            try:
                succeeded, infrastructure, value = result_queue.get(timeout=min(0.1, remaining))
                break
            except queue.Empty as error:
                if process.is_alive():
                    continue
                try:
                    succeeded, infrastructure, value = result_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    message = (
                        f"{backend.upper()} candidate worker exited without returning a result "
                        f"(status {process.exitcode})"
                    )
                    if candidate_started.is_set():
                        raise RuntimeError(message) from error
                    raise MeasurementInfrastructureError(message) from error
            except Exception as error:
                raise MeasurementInfrastructureError(
                    f"{backend.upper()} candidate worker result wait failed: {error}"
                ) from error
        process.join(timeout=5)
        if process.is_alive():
            raise MeasurementInfrastructureError("candidate worker did not exit after returning its result")
        if process.exitcode != 0:
            message = f"candidate worker exited abnormally with status {process.exitcode}"
            if candidate_started.is_set():
                raise RuntimeError(message)
            raise MeasurementInfrastructureError(message)
        if not succeeded:
            if infrastructure:
                raise MeasurementInfrastructureError(
                    f"candidate worker infrastructure failure:\n{value}"
                )
            raise RuntimeError(f"candidate worker failed:\n{value}")
        return value
    finally:
        if started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
            process.join()
        result_queue.close()
        result_queue.join_thread()
