"""Isolated one-candidate FSIM and TSIM measurements for IC V1 tuning."""

import shutil

import autotvm_tuner as shared


FSIM_TIMEOUT_SECONDS = 60
TSIM_TIMEOUT_SECONDS = 120


def backend_timeout(backend, timeout=None):
    """Return the IC V1 timeout for one backend, with a validated override."""
    defaults = {"fsim": FSIM_TIMEOUT_SECONDS, "tsim": TSIM_TIMEOUT_SECONDS}
    if backend not in defaults:
        raise ValueError(f"unsupported backend {backend!r}; expected fsim or tsim")
    if timeout is None:
        return defaults[backend]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer number of seconds")
    return timeout


def create_runner(backend, timeout=None, **runner_options):
    """Create one VTA local RPC runner using this backend's IC V1 timeout."""
    return shared.create_runner(
        backend, timeout=backend_timeout(backend, timeout), **runner_options
    )


def _close_builder(builder):
    failures = []
    executor = getattr(builder, "executor", None)
    if executor is not None:
        for worker in tuple(getattr(executor, "_worker_map", {}).values()):
            try:
                shared._terminate_owned_worker(worker)
            except Exception as error:
                failures.append(f"worker: {type(error).__name__}: {error}")
        threadpool = getattr(executor, "_threadpool", None)
        if threadpool is not None:
            try:
                threadpool.shutdown(wait=True, cancel_futures=True)
            except Exception as error:
                failures.append(f"executor: {type(error).__name__}: {error}")
        if not failures:
            builder.executor = None
    tmp_dir = getattr(builder, "tmp_dir", None)
    if tmp_dir:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError as error:
            failures.append(f"build directory {tmp_dir}: {error}")
    if failures:
        raise shared.SimulatorInfrastructureError("; ".join(failures))


def measure_candidate(task, config, backend, timeout=None):
    """Measure one configuration with fresh builder and RPC processes.

    The result retains AutoTVM's native error code and raw costs. A failed
    candidate therefore cannot leave a server or measurement worker for the
    next candidate to reuse.
    """
    selected_timeout = backend_timeout(backend, timeout)
    measure_input = shared.MeasureInput(task.target, task, config)
    option = shared.measure_option(
        backend,
        timeout=selected_timeout,
        number=1,
        repeat=1,
        min_repeat_ms=0,
        cooldown_interval=0,
    )
    runner = option["runner"]
    builder = option["builder"]
    try:
        measure_batch = shared.autotvm.measure.create_measure_batch(task, option)
        results = measure_batch([measure_input])
        if len(results) != 1:
            raise shared.SimulatorInfrastructureError(
                f"{backend.upper()} local RPC returned {len(results)} results for one candidate"
            )
        return results[0]
    finally:
        cleanup_errors = []
        try:
            runner.close()
        except Exception as error:
            cleanup_errors.append(f"runner cleanup: {type(error).__name__}: {error}")
        try:
            _close_builder(builder)
        except Exception as error:
            cleanup_errors.append(f"builder cleanup: {type(error).__name__}: {error}")
        if cleanup_errors:
            raise shared.SimulatorInfrastructureError("; ".join(cleanup_errors))
