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

    for backend, timeout in (("host", None), ("fsim", 0), ("tsim", True)):
        with pytest.raises(ValueError):
            measurement.measure_candidate(layer, activation, [], backend, timeout)

    with pytest.raises(ValueError, match="activation must have shape"):
        measurement.measure_candidate(
            layer, np.zeros((2, 2), dtype="int8"), [], "fsim"
        )


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
