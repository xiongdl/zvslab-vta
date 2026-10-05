"""Candidate measurement failures remain isolated to their worker process."""

import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


APP_ROOT = Path(__file__).resolve().parents[1]


def _abort_after_candidate_start(function_json, activation, config_indices, backend, result_queue, phase):
    phase.set()
    os.abort()


def test_aborted_candidate_worker_is_reported_and_reaped(monkeypatch):
    import importlib
    import types

    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    measurement = importlib.import_module("python.measurement")
    context = multiprocessing.get_context("spawn")
    workers = []

    class TrackingContext:
        def Event(self):
            return context.Event()

        def Queue(self, maxsize):
            return context.Queue(maxsize=maxsize)

        def Process(self, target, args):
            worker = context.Process(target=target, args=args)
            workers.append(worker)
            return worker

    monkeypatch.setattr(measurement.multiprocessing, "get_context", lambda _name: TrackingContext())
    monkeypatch.setattr(measurement, "_worker", _abort_after_candidate_start)
    monkeypatch.setitem(
        sys.modules,
        "tvm",
        types.SimpleNamespace(ir=SimpleNamespace(save_json=lambda _function: "relay-ir")),
    )
    layer = SimpleNamespace(
        function=object(),
        inputs=(SimpleNamespace(shape=(1, 2), dtype="int8"),),
        config_spaces=(),
    )

    with pytest.raises(RuntimeError, match="candidate worker exited without returning a result"):
        measurement.measure_candidate(layer, np.zeros((1, 2), dtype=np.int8), [], "fsim", 5)

    assert len(workers) == 1
    assert workers[0].exitcode == -6
    assert not workers[0].is_alive()
    assert workers[0] not in multiprocessing.active_children()
