"""Focused checks for actual-compute candidate validation and dispatch."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _load_local(name):
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    spec = importlib.util.spec_from_file_location(f"ic_v1_{name}", APP_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_measurement_rejects_invalid_backend_timeout_and_activation():
    measurement = _load_local("measurement")
    layer = SimpleNamespace(
        inputs=(SimpleNamespace(shape=(1, 2), dtype="int8"),), config_spaces=()
    )
    activation = np.zeros((1, 2), dtype="int8")

    for backend, timeout in (
        ("host", None), ("fsim", 0), ("tsim", True), ("fsim", float("nan")),
        ("fsim", float("inf")),
    ):
        with pytest.raises(ValueError):
            measurement.measure_candidate(layer, activation, [], backend, timeout)

    with pytest.raises(ValueError, match="activation must have shape"):
        measurement.measure_candidate(
            layer, np.zeros((2, 2), dtype="int8"), [], "fsim"
        )


@pytest.mark.parametrize("timeout", [float("nan"), float("inf")])
def test_tune_cli_rejects_non_finite_timeout(timeout):
    tune = _load_local("tune")
    args = tune._parser().parse_args([
        "--workloads", "workloads.json", "--timeout", str(timeout),
        "--output-logs", "out.tmp",
    ])

    with pytest.raises(ValueError, match="finite positive"):
        tune.validate_args(args)


def test_measure_candidate_stops_worker_when_queue_wait_fails(monkeypatch):
    import types
    measurement = _load_local("measurement")

    layer = SimpleNamespace(
        function=object(), inputs=(SimpleNamespace(shape=(1, 2), dtype="int8"),),
        config_spaces=(),
    )
    activation = np.zeros((1, 2), dtype="int8")

    class BrokenQueue:
        closed = False
        joined = False

        def get(self, timeout):
            raise OSError("queue transport failed")

        def close(self):
            self.closed = True

        def join_thread(self):
            self.joined = True

    class Worker:
        alive = False
        terminated = False
        joined = False

        def __init__(self, target, args):
            self.target, self.args = target, args
            self.exitcode = None

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            self.joined = True

    class SpawnContext:
        def __init__(self):
            self.queue = BrokenQueue()
            self.worker = None

        def Queue(self, maxsize):
            return self.queue

        def Process(self, target, args):
            self.worker = Worker(target, args)
            return self.worker

    context = SpawnContext()
    monkeypatch.setattr(measurement.multiprocessing, "get_context", lambda name: context)
    monkeypatch.setitem(sys.modules, "tvm", types.SimpleNamespace(
        ir=types.SimpleNamespace(save_json=lambda function: "function-json")
    ))

    with pytest.raises(measurement.MeasurementInfrastructureError, match="result wait failed: queue transport failed"):
        measurement.measure_candidate(layer, activation, [], "fsim", 5)

    assert context.worker.terminated
    assert context.worker.joined
    assert context.queue.closed
    assert context.queue.joined


@pytest.mark.parametrize("backend", ["fsim", "tsim"])
def test_measure_candidate_detects_worker_exit_before_candidate_timeout(monkeypatch, backend):
    import queue
    import types

    measurement = _load_local("measurement")

    class EmptyQueue:
        def get(self, timeout):
            raise queue.Empty

        def close(self):
            pass

        def join_thread(self):
            pass

    class AbortedWorker:
        exitcode = -6

        def __init__(self, target, args):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

    class SpawnContext:
        def Queue(self, maxsize):
            return EmptyQueue()

        def Process(self, target, args):
            return AbortedWorker(target, args)

    monkeypatch.setattr(measurement.multiprocessing, "get_context", lambda name: SpawnContext())
    monkeypatch.setitem(
        sys.modules, "tvm", types.SimpleNamespace(ir=types.SimpleNamespace(save_json=lambda fn: "ir"))
    )
    layer = SimpleNamespace(
        function=object(), inputs=(SimpleNamespace(shape=(1, 2), dtype="int8"),), config_spaces=()
    )

    with pytest.raises(measurement.MeasurementInfrastructureError, match="worker exited without returning a result.*-6"):
        measurement.measure_candidate(layer, np.zeros((1, 2), dtype="int8"), [], backend, 5)


@pytest.mark.parametrize(
    "infrastructure",
    [False, True],
)
def test_measure_candidate_decodes_worker_failure_envelope(monkeypatch, infrastructure):
    import types

    measurement = _load_local("measurement")
    if infrastructure:
        worker_error = measurement.MeasurementInfrastructureError("compiler initialization failed")
        expected_error = measurement.MeasurementInfrastructureError
        expected_message = "candidate worker infrastructure failure"
    else:
        worker_error = ValueError("invalid candidate")
        expected_error = RuntimeError
        expected_message = "candidate worker failed"

    class ResultQueue:
        def __init__(self):
            self.value = None

        def put(self, value):
            self.value = value

        def get(self, timeout):
            return self.value

        def close(self):
            pass

        def join_thread(self):
            pass

    class Worker:
        exitcode = 0

        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.alive = False

        def start(self):
            self.target(*self.args)

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive = False

    class SpawnContext:
        def Queue(self, maxsize):
            self.queue = ResultQueue()
            return self.queue

        def Process(self, target, args):
            return Worker(target, args)

    monkeypatch.setattr(measurement.multiprocessing, "get_context", lambda name: SpawnContext())
    monkeypatch.setattr(
        measurement,
        "_evaluate_candidate",
        lambda *args: (_ for _ in ()).throw(worker_error),
    )
    monkeypatch.setitem(
        sys.modules, "tvm", types.SimpleNamespace(ir=types.SimpleNamespace(save_json=lambda fn: "ir"))
    )
    layer = SimpleNamespace(
        function=object(), inputs=(SimpleNamespace(shape=(1, 2), dtype="int8"),), config_spaces=()
    )

    with pytest.raises(expected_error, match=expected_message):
        measurement.measure_candidate(layer, np.zeros((1, 2), dtype="int8"), [], "fsim", 5)


def test_deployment_and_measurement_share_occurrence_config_dispatch():
    import tvm
    from tvm import autotvm

    dispatch = _load_local("dispatch")
    target = tvm.target.Target("llvm")
    workload = ("conv2d_packed.vta", 1, 2, 3)
    first = object()
    second = object()
    fallback = object()

    with dispatch.config_bindings_context(((target, workload, fallback),)):
        with dispatch.config_bindings_context(((target, workload, first),)):
            assert autotvm.DispatchContext.current.query(target, workload) is first
        assert autotvm.DispatchContext.current.query(target, workload) is fallback

        with dispatch.config_space_context(
            (("conv2d_packed.vta", workload, target, object()),), (second,)
        ):
            assert autotvm.DispatchContext.current.query(target, workload) is second
        assert autotvm.DispatchContext.current.query(target, workload) is fallback

    with pytest.raises(ValueError, match="duplicate AutoTVM"):
        with dispatch.config_bindings_context(
            ((target, workload, first), (target, workload, second))
        ):
            pass


def test_app_runtime_and_tuner_import_without_apps_directory():
    import os
    import subprocess

    app_path = str(APP_ROOT)
    repository_root = APP_ROOT.parents[3]
    tvm_path = str(repository_root / "tvm" / "python")
    vta_path = str(repository_root / "vta" / "python")
    env = os.environ.copy()
    env.update({
        "VTA_CONFIG_FILE": str(repository_root / "vta" / "config" / "vta_64mac.json"),
        "VTA_BACKEND": "fsim",
        "PYTHONPATH": os.pathsep.join((tvm_path, vta_path)),
    })
    code = (
        "import sys; "
        f"sys.path.insert(0, {app_path!r}); "
        "import runtime, tune; "
        "assert not any(name == 'common' or name.startswith('common.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
