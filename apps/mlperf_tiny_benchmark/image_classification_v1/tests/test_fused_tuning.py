"""Semantic and builder checks for the complete IC V1 AutoTVM task."""

import importlib.util
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm import autotvm, relay
from tvm.autotvm.measure import MeasureInput


APP_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = APP_ROOT / "fused_tasks.py"
MODEL_PATH = APP_ROOT / "model" / "pretrainedResnet.tflite"


def _load_module():
    registered = sys.modules.get("fused_tasks")
    if registered is not None and Path(registered.__file__).resolve() == TASKS_PATH.resolve():
        return registered
    spec = importlib.util.spec_from_file_location("fused_tasks", TASKS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fused():
    return _load_module()


@pytest.fixture(scope="module")
def prepared():
    sys.path.insert(0, str(APP_ROOT.parent))
    import autotvm_tuner

    pipeline = autotvm_tuner._load_model_pipeline("image_classification_v1")
    return pipeline.prepare_model(MODEL_PATH)


def _outlined_vta_functions(prepared):
    return [
        prepared.mixed_module[gv]
        for gv in prepared.mixed_module.get_global_vars()
        if prepared.mixed_module[gv].attrs
        and "Compiler" in prepared.mixed_module[gv].attrs
        and prepared.mixed_module[gv].attrs["Compiler"] == "vta"
    ]


def _relay_reference(func, values):
    reference = relay.Function(func.params, func.body)
    mod = tvm.IRModule.from_expr(reference)
    executable = relay.create_executor("debug", mod=mod, device=tvm.cpu(), target="llvm").evaluate()
    return executable(tvm.nd.array(values)).numpy()


def test_extracts_complete_ordered_fusion_and_occurrence_identity(fused, prepared):
    identities = fused.extract_fused_identities(prepared)

    assert len(identities) == 8
    assert [item.occurrence for item in identities] == list(range(8))
    assert [item.symbol for item in identities] == [
        function.attrs.get_str("global_symbol") for function in _outlined_vta_functions(prepared)
    ]
    assert all((item.bias, item.shift, item.clip_min, item.clip_max, item.output_dtype)
               == (64, 7, -127, 127, "int8") for item in identities[:2])
    assert identities[0].conv_workload == identities[1].conv_workload
    assert identities[0].sha256 != identities[1].sha256


def test_identity_roundtrips_and_distinguishes_postprocessing(fused, prepared):
    identity = fused.extract_fused_identities(prepared)[0]
    restored = fused.FusedConvIdentity.from_json(identity.canonical_json())
    changed = replace(identity, bias=identity.bias + 1)

    assert restored == identity
    assert restored.task_args() == identity.task_args()
    assert changed.sha256 != identity.sha256
    assert changed.conv_workload == identity.conv_workload


def test_real_fusion_reference_covers_negative_and_saturated_results(prepared, fused):
    function = _outlined_vta_functions(prepared)[0]
    identity = fused.extract_fused_identities(prepared)[0]
    rng = np.random.default_rng(7)
    values = rng.integers(-127, 128, size=tuple(int(x) for x in function.params[0].checked_type.shape),
                          dtype="int8")
    actual = _relay_reference(function, values)
    composite = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, relay.Function):
            if node.op.attrs and "Composite" in node.op.attrs:
                composite.append(node)

    relay.analysis.post_order_visit(function.body, visit)
    conv = composite[0].op.body.args[0].args[0].args[0].args[0]
    conv = relay.bind(conv, {composite[0].op.params[0]: composite[0].args[0]})
    accumulator = _relay_reference(relay.Function(function.params, conv), values)
    expected = np.clip((accumulator.astype("int32") + identity.bias) >> identity.shift,
                       identity.clip_min, identity.clip_max).astype(identity.output_dtype)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int8
    assert np.any(actual < 0)
    assert np.any(actual == -127) or np.any(actual == 127)
    # These constants are read from the real function, so this assertion ties the
    # numerical reference to the selected fused task rather than fixed script data.
    assert identity.shift == 7
    assert identity.bias == 64


def test_fused_tasks_reconstruct_in_subprocess(fused, prepared):
    identity = fused.extract_fused_identities(prepared)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(APP_ROOT), *env.get("PYTHONPATH", "").split(os.pathsep)]
    )
    code = (
        "import fused_tasks, vta; "
        f"i=fused_tasks.FusedConvIdentity.from_json({identity.canonical_json()!r}); "
        "t=fused_tasks.create_task(i, 'vta'); "
        "print(t.name, len(t.config_space), t.workload[-2:])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )

    assert result.stdout.startswith(f"{fused.TASK_NAME} ")
    assert identity.symbol in result.stdout


def test_local_builder_compiles_two_configs_and_keeps_fusion_semantics(
    fused, prepared, monkeypatch
):
    import vta

    identity = fused.extract_fused_identities(prepared)[0]
    task = fused.create_task(identity, "vta")
    assert task.name == fused.TASK_NAME
    assert task.flop > 0 and int(task.flop) == task.flop and int(task.flop) % 2 == 0
    configs = [task.config_space.get(0), task.config_space.get(1)]
    builder = autotvm.LocalBuilder(n_parallel=1)
    builder.set_task(task, {})
    try:
        results = [builder.build([MeasureInput(task.target, task, config)])[0] for config in configs]
    finally:
        # Process enumeration is blocked in the managed macOS test sandbox;
        # LocalBuilder's own worker is still terminated by PopenWorker.kill.
        from tvm.contrib import popen_pool

        monkeypatch.setattr(popen_pool, "kill_child_processes", lambda _pid: None)
        del builder

    assert all(hasattr(result, "error") and result.error is None for result in results)
    assert all(result.filename for result in results)
    assert results[0].filename != results[1].filename

    lowered = fused.lower_with_fused_config(prepared, identity, configs[0])
    assert lowered.schedule is not None
