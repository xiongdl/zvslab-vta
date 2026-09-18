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

import json
import re

import numpy as np
import pytest
import tvm
import vta
from tvm import relay, rpc
from tvm.contrib import graph_executor, utils

from byoc_utils import (
    make_adjacent_qnn_conv2d_module,
    make_qnn_conv2d_module,
    make_qnn_conv2d_near_miss_module,
)
from vta.relay import partition_for_vta, plan_devices_for_vta
from vta.relay.transform import lower_vta_function


def _run_graph(factory, device, input_data):
    runtime = graph_executor.GraphModule(factory["default"](device))
    runtime.set_input("data", input_data)
    runtime.run()
    return runtime.get_output(0).numpy()


def _simulator_setup(env):
    if env.TARGET == "sim":
        return (
            "libvta_fsim",
            "vta.simulator.profiler_clear",
            "vta.simulator.profiler_status",
            "./scripts/build_vta_lib.sh --target libvta_fsim",
        )
    if env.TARGET == "tsim":
        return (
            "libvta_tsim + libvta_hw",
            "vta.tsim.profiler_clear",
            "vta.tsim.profiler_status",
            "./scripts/build_vta_lib.sh --target libvta_hw",
        )
    raise RuntimeError(
        "VTA BYOC runtime validation requires sim or tsim, got " f"{env.TARGET}"
    )


def _remote_simulator_stats(remote, status_name):
    status = remote.get_function(status_name)
    return json.loads(status())


def _require_simulator(env):
    from vta.testing import simulator

    library, clear_name, status_name, build_command = _simulator_setup(env)
    if (
        tvm.get_global_func(clear_name, allow_missing=True) is None
        or tvm.get_global_func(status_name, allow_missing=True) is None
    ):
        raise RuntimeError(
            f"VTA {library} is unavailable; run {build_command}"
        )
    return simulator, clear_name, status_name


def _assert_accelerator_activity(env, runtime_stats, expected_out_store_nbytes=None):
    if env.TARGET == "sim":
        assert runtime_stats["gemm_counter"] > 0
        assert runtime_stats["wgt_load_nbytes"] > 0
        assert runtime_stats["out_store_nbytes"] > 0
        if expected_out_store_nbytes is not None:
            assert runtime_stats["out_store_nbytes"] == expected_out_store_nbytes
        return
    if env.TARGET == "tsim":
        assert runtime_stats["cycle_count"] > 0
        return
    raise RuntimeError(
        "VTA BYOC runtime validation requires sim or tsim, got " f"{env.TARGET}"
    )


def _require_runtime_symbol(module, symbol):
    if not module.implements_function(symbol, True):
        raise RuntimeError(f"loaded VTA artifact does not implement {symbol}")


def _test_exported_approved_graph_executes(bias_kind, **fixture_overrides):
    env = vta.get_env()
    mod = make_qnn_conv2d_module(env, bias_kind=bias_kind, **fixture_overrides)
    input_shape = tuple(int(dim) for dim in mod["main"].params[0].checked_type.shape)
    input_data = ((np.arange(np.prod(input_shape)) % 17) - 8).reshape(input_shape)
    input_data = input_data.astype(env.inp_dtype)

    reference_factory = relay.build(mod, target="llvm")
    expected = _run_graph(reference_factory, tvm.cpu(0), input_data)

    partitioned = partition_for_vta(mod, mod_name="runtime")
    external_functions = [
        function
        for function in partitioned.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
    ]
    assert len(external_functions) == 1
    external = external_functions[0]
    symbol = external.attrs.get_str("global_symbol")
    assert len(external.params) == 1
    assert "abs" in partitioned["main"].astext(show_meta_data=False)
    assert "transpose" in partitioned["main"].astext(show_meta_data=False)

    legacy_compiler_global = "relay.ext." + "vta"
    assert tvm.get_global_func(legacy_compiler_global, allow_missing=True) is None
    with vta.build_config():
        factory = relay.build(
            partitioned,
            target=tvm.target.Target("vta", host=env.target_host),
        )

    graph = json.loads(factory.get_graph_json())
    graph_inputs = [graph["nodes"][index]["name"] for index in graph["arg_nodes"]]
    assert graph_inputs == ["data"]

    artifact_dir = utils.tempdir()
    artifact_name = f"vta_byoc_runtime_{bias_kind or 'no_bias'}.tar"
    artifact_path = artifact_dir.relpath(artifact_name)
    factory.export_library(artifact_path)

    simulator, clear_name, status_name = _require_simulator(env)
    simulator.clear_stats()
    assert all(counter == 0 for counter in simulator.stats().values())
    remote = rpc.LocalSession()
    remote.upload(artifact_path)
    loaded = remote.load_module(artifact_name)
    _require_runtime_symbol(loaded, symbol)

    remote.get_function(clear_name)()
    runtime = graph_executor.create(factory.get_graph_json(), loaded, remote.ext_dev(0))
    runtime.set_input("data", input_data)
    assert all(counter == 0 for counter in _remote_simulator_stats(remote, status_name).values())
    runtime.run()
    actual = runtime.get_output(0).numpy()
    runtime_stats = _remote_simulator_stats(remote, status_name)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    expected_out_store_nbytes = int(np.prod(expected.shape)) * expected.dtype.itemsize
    _assert_accelerator_activity(env, runtime_stats, expected_out_store_nbytes)
    return expected, runtime_stats


def test_exported_no_bias_graph_executes_on_simulator():
    _test_exported_approved_graph_executes(None)


def test_exported_bias_add_graph_executes_on_simulator():
    _test_exported_approved_graph_executes("bias_add")


def test_exported_broadcast_add_graph_executes_on_simulator():
    _test_exported_approved_graph_executes("add")


@pytest.mark.parametrize(
    ("data_layout", "kernel_layout"),
    [
        pytest.param("NCHW", "OIHW", id="nchw-oihw"),
        pytest.param("NHWC", "HWIO", id="nhwc-hwio"),
    ],
)
@pytest.mark.parametrize(
    ("input_spatial", "output_spatial"),
    [pytest.param(32, 16, id="32-to-16"), pytest.param(16, 8, id="16-to-8")],
)
def test_exported_asymmetric_stride2_graph_matches_host_and_full_abi_store(
    data_layout, kernel_layout, input_spatial, output_spatial
):
    env = vta.get_env()
    output, runtime_stats = _test_exported_approved_graph_executes(
        None,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=(0, 0, 1, 1),
        input_height=input_spatial,
        input_width=input_spatial,
    )

    expected_shape = (
        (env.BATCH, env.BLOCK_OUT, output_spatial, output_spatial)
        if data_layout == "NHWC"
        else (env.BATCH, output_spatial, output_spatial, env.BLOCK_OUT)
    )
    assert output.shape == expected_shape
    assert output.dtype == np.dtype(env.out_dtype)
    assert runtime_stats["out_store_nbytes"] == (
        int(np.prod(expected_shape)) * np.dtype(env.out_dtype).itemsize
    )


def test_near_miss_executes_only_on_host():
    env = vta.get_env()
    mod, _ = make_qnn_conv2d_near_miss_module(env)
    partitioned = partition_for_vta(mod)

    assert not any(
        isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        for function in partitioned.functions.values()
    )

    factory = relay.build(partitioned, target="llvm")
    graph = json.loads(factory.get_graph_json())
    assert set(graph["attrs"]["device_index"][1]) == {tvm.cpu(0).device_type}

    data_shape = tuple(int(dim) for dim in mod["main"].params[0].checked_type.shape)
    weight_shape = tuple(int(dim) for dim in mod["main"].params[1].checked_type.shape)
    data = ((np.arange(np.prod(data_shape)) % 17) - 8).reshape(data_shape)
    weight = ((np.arange(np.prod(weight_shape)) % 5) - 2).reshape(weight_shape)
    runtime = graph_executor.GraphModule(factory["default"](tvm.cpu(0)))
    runtime.set_input("data", data.astype(env.inp_dtype))
    runtime.set_input("weight", weight.astype(env.wgt_dtype))
    runtime.run()
    output = runtime.get_output(0).numpy()

    assert output.shape == (env.BATCH, 8, 8, env.BLOCK_OUT)
    assert output.dtype == np.dtype(env.out_dtype)


def _make_mixed_depthwise_vta_module(env):
    data = relay.var(
        "data", shape=(env.BATCH, 8, 8, env.BLOCK_IN), dtype=env.inp_dtype
    )
    depthwise_kernel = relay.const(
        np.ones((3, 3, env.BLOCK_IN, 1), dtype=env.wgt_dtype)
    )
    depthwise = relay.nn.conv2d(
        relay.abs(data),
        depthwise_kernel,
        channels=env.BLOCK_IN,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWOI",
        groups=env.BLOCK_IN,
        out_dtype=env.acc_dtype,
    )
    depthwise = relay.cast(
        relay.clip(
            relay.right_shift(depthwise, relay.const(1, env.acc_dtype)),
            a_min=-128,
            a_max=127,
        ),
        env.out_dtype,
    )
    base_output = relay.transpose(depthwise, axes=(0, 3, 1, 2))
    base_kernel = relay.const(
        np.ones((env.BLOCK_OUT, env.BLOCK_IN, 3, 3), dtype=env.wgt_dtype)
    )
    base_output = relay.nn.conv2d(
        relay.abs(base_output),
        base_kernel,
        channels=env.BLOCK_OUT,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype=env.acc_dtype,
    )
    base_output = relay.cast(
        relay.clip(
            relay.right_shift(base_output, relay.const(1, env.acc_dtype)),
            a_min=-128,
            a_max=127,
        ),
        env.out_dtype,
    )
    return relay.transform.InferType()(
        tvm.IRModule.from_expr(relay.Function([data], base_output))
    )


def test_plan_devices_for_vta_contract_and_input_immutability():
    env = vta.get_env()
    module = partition_for_vta(make_qnn_conv2d_module(env), mod_name="planner_contract")
    before = module.astext(show_meta_data=True)

    plan = plan_devices_for_vta(module, tvm.target.Target("llvm"))

    assert isinstance(plan.module, tvm.IRModule)
    assert len(plan.targets) == 2
    assert plan.targets[0].kind.name == "llvm"
    assert plan.targets[0].get_target_device_type() == tvm.cpu(0).device_type
    assert plan.targets[1].kind.name == "ext_dev"
    assert plan.targets[1].device_name == "vta"
    assert set(plan.targets[1].keys) == {"vta", "cpu"}
    assert plan.targets[1].host == plan.targets[0]
    assert plan.targets[1].get_target_device_type() == tvm.ext_dev(0).device_type
    assert module.astext(show_meta_data=True) == before
    with pytest.raises((AttributeError, TypeError)):
        plan.module = module
    with pytest.raises(TypeError):
        plan.targets[0] = plan.targets[1]

    planned_text = plan.module["main"].astext(show_meta_data=False)
    assert "VirtualDevice(device_type=12" in planned_text
    assert "VirtualDevice(device_type=1" in planned_text
    assert "device_copy" not in planned_text


def test_plan_devices_for_vta_rejects_invalid_inputs():
    env = vta.get_env()
    module = partition_for_vta(make_qnn_conv2d_module(env), mod_name="planner_invalid")

    with pytest.raises(TypeError, match="module must be a tvm.IRModule"):
        plan_devices_for_vta(None, tvm.target.Target("llvm"))
    with pytest.raises(TypeError, match="host_target must be a tvm.target.Target"):
        plan_devices_for_vta(module, "llvm")
    untyped = tvm.IRModule.from_expr(relay.Function([], relay.const(1, "int8")))
    with pytest.raises(ValueError, match="inferred types"):
        plan_devices_for_vta(untyped, tvm.target.Target("llvm"))
    host_only = relay.transform.InferType()(
        tvm.IRModule.from_expr(relay.Function([], relay.const(1, "int8")))
    )
    with pytest.raises(ValueError, match="outlined VTA function"):
        plan_devices_for_vta(host_only, tvm.target.Target("llvm"))
    with pytest.raises(ValueError, match="llvm or c"):
        plan_devices_for_vta(module, tvm.target.Target("stackvm"))
    with pytest.raises(ValueError, match="nested host"):
        plan_devices_for_vta(
            module,
            tvm.target.Target("llvm", host=tvm.target.Target("llvm")),
        )


def test_plan_devices_for_vta_handles_deep_nested_host_graph():
    """Keep nested host expressions inferable around multiple VTA calls."""
    env = vta.get_env()
    module = make_adjacent_qnn_conv2d_module(env, count=2)
    partitioned = partition_for_vta(module, mod_name="nested_host_planner")

    plan = plan_devices_for_vta(partitioned, tvm.target.Target("llvm"))
    with tvm.target.Target("vta", host=plan.targets[0]), vta.build_config():
        factory = relay.build(plan.module, target=plan.targets)

    graph = json.loads(factory.get_graph_json())
    assert set(graph["attrs"]["device_index"][1]) == {
        tvm.cpu(0).device_type,
        tvm.ext_dev(0).device_type,
    }
    assert any(node["name"] == "__copy" for node in graph["nodes"])


def test_mixed_unpacked_depthwise_host_and_vta_graph_executes_on_simulator():
    env = vta.get_env()
    mod = _make_mixed_depthwise_vta_module(env)
    partitioned = partition_for_vta(mod, mod_name="mixed_depthwise_runtime")
    external_functions = [
        function
        for function in partitioned.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
    ]
    assert len(external_functions) == 1
    symbol = external_functions[0].attrs.get_str("global_symbol")
    main_text = partitioned["main"].astext(show_meta_data=False)
    assert f"groups={env.BLOCK_OUT}" in main_text
    assert external_functions[0].attrs.get_str("Compiler") == "vta"

    input_shape = tuple(int(dim) for dim in mod["main"].params[0].checked_type.shape)
    input_data = ((np.arange(np.prod(input_shape)) % 17) - 8).reshape(input_shape).astype(env.inp_dtype)
    reference_factory = relay.build(mod, target="llvm")
    expected = _run_graph(reference_factory, tvm.cpu(0), input_data)

    plan = plan_devices_for_vta(partitioned, tvm.target.Target(env.target_host))
    with tvm.target.Target("vta", host=plan.targets[0]), vta.build_config():
        factory = relay.build(plan.module, target=plan.targets)
    graph = json.loads(factory.get_graph_json())
    assert symbol in factory.get_graph_json()
    assert set(graph["attrs"]["device_index"][1]) == {
        tvm.cpu(0).device_type,
        tvm.ext_dev(0).device_type,
    }
    assert any(node["name"] == "__copy" for node in graph["nodes"])

    artifact_dir = utils.tempdir()
    artifact_name = "vta_byoc_mixed_depthwise_runtime.tar"
    artifact_path = artifact_dir.relpath(artifact_name)
    factory.export_library(artifact_path)
    simulator, clear_name, status_name = _require_simulator(env)
    simulator.clear_stats()
    remote = rpc.LocalSession()
    remote.upload(artifact_path)
    loaded = remote.load_module(artifact_name)
    _require_runtime_symbol(loaded, symbol)
    remote.get_function(clear_name)()
    runtime = graph_executor.create(
        factory.get_graph_json(), loaded, [remote.cpu(0), remote.ext_dev(0)]
    )
    runtime.load_params(tvm.runtime.save_param_dict(factory.get_params()))
    runtime.set_input("data", input_data)
    runtime.run()
    actual = runtime.get_output(0).numpy()
    runtime_stats = _remote_simulator_stats(remote, status_name)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    _assert_accelerator_activity(env, runtime_stats)


def test_missing_simulator_reports_setup_command(monkeypatch):
    env = vta.get_env()
    _, clear_name, _, build_command = _simulator_setup(env)
    monkeypatch.setattr(tvm, "get_global_func", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match=build_command):
        _require_simulator(env)


def test_unsupported_simulator_target_reports_target_name(monkeypatch):
    class UnsupportedEnvironment:
        TARGET = "pynq"

    with pytest.raises(RuntimeError, match="requires sim or tsim, got pynq"):
        _simulator_setup(UnsupportedEnvironment())


def test_loaded_artifact_must_implement_expected_symbol():
    data = tvm.te.placeholder((1,), name="data")
    output = tvm.te.compute((1,), lambda index: data[index], name="output")
    unrelated = tvm.build(
        tvm.te.create_schedule(output.op),
        [data, output],
        target="llvm",
        name="unrelated",
    )

    with pytest.raises(RuntimeError, match="loaded VTA artifact.*tvmgen_missing_vta_main_0"):
        _require_runtime_symbol(unrelated, "tvmgen_missing_vta_main_0")


def _direct_vta_runtime_module():
    env = vta.get_env()
    partitioned = partition_for_vta(
        make_qnn_conv2d_module(env),
        mod_name="fingerprint_mismatch",
    )
    external = next(
        function
        for function in partitioned.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
    )
    symbol = external.attrs.get_str("global_symbol")
    primfunc = lower_vta_function(external)
    target = tvm.target.Target("vta", host=env.target_host)
    hook = target.get_kind_attr("TIRToRuntime")
    return hook(tvm.IRModule({symbol: primfunc}), target), external, symbol


def test_mismatched_artifact_fails_before_fsim_profiler_activity(tmp_path):
    runtime_module, external, symbol = _direct_vta_runtime_module()
    llvm_source = runtime_module.get_source("ll")
    check_call = re.search(r"(@VTACheckConfig\(i64 )(-?\d+)(\))", llvm_source)
    assert check_call is not None
    actual_value = int(check_call.group(2)) & ((1 << 64) - 1)
    expected_value = actual_value ^ 1
    expected_literal = (
        expected_value if expected_value < (1 << 63) else expected_value - (1 << 64)
    )
    mismatched_source = (
        llvm_source[: check_call.start(2)]
        + str(expected_literal)
        + llvm_source[check_call.end(2) :]
    )
    mismatch_path = tmp_path / "mismatched_vta.ll"
    mismatch_path.write_text(mismatched_source, encoding="utf-8")

    simulator, _, _ = _require_simulator(vta.get_env())
    mismatched_module = tvm.runtime.load_module(str(mismatch_path))
    input_shape = tuple(int(dim) for dim in external.params[0].checked_type.shape)
    output_shape = tuple(int(dim) for dim in external.ret_type.shape)
    input_data = np.arange(np.prod(input_shape), dtype="int8").reshape(input_shape)
    device = tvm.ext_dev(0)
    device_input = tvm.nd.array(input_data, device=device)
    device_output = tvm.nd.empty(output_shape, external.ret_type.dtype, device=device)

    simulator.clear_stats()
    with pytest.raises(tvm.error.TVMError) as error:
        mismatched_module[symbol](device_input, device_output)

    diagnostic = str(error.value).lower()
    assert "{:016x}".format(expected_value) in diagnostic
    assert "{:016x}".format(actual_value) in diagnostic
    assert all(counter == 0 for counter in simulator.stats().values())
