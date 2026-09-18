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

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import tvm
import vta
import vta.relay
from tvm import relay

from byoc_utils import (
    make_qnn_conv2d_module,
    make_qnn_conv2d_near_miss_module,
    run_isolated_python,
)
from vta.relay import COMPILER_NAME, VTACompilerConfig, partition_for_vta


LEGACY_COMPILER_GLOBAL = "relay.ext." + "vta"


def test_compiler_identity_is_stable():
    assert COMPILER_NAME == "vta"


def test_relay_package_exports_public_compiler_surface():
    assert vta.relay.__all__ == [
        "COMPILER_NAME",
        "VTACompilerConfig",
        "partition_for_vta",
        "VTADevicePlan",
        "plan_devices_for_vta",
    ]
    assert partition_for_vta is vta.relay.partition_for_vta


def test_partition_for_vta_has_public_api_documentation():
    documentation = partition_for_vta.__doc__

    assert documentation is not None
    assert "Parameters" in documentation
    assert "Returns" in documentation


def test_compiler_config_captures_active_environment():
    env = vta.get_env()

    config = VTACompilerConfig.from_env(env)

    assert config.batch == env.BATCH
    assert config.block_in == env.BLOCK_IN
    assert config.block_out == env.BLOCK_OUT
    assert config.input_dtype == env.inp_dtype
    assert config.weight_dtype == env.wgt_dtype
    assert config.accumulator_dtype == env.acc_dtype
    assert config.output_dtype == env.out_dtype
    assert config.target == str(env.target)
    assert config.host_target == str(env.target_host)
    assert config.model == env.MODEL
    assert config.device_type == tvm.runtime.Device.kDLExtDev


def test_compiler_config_is_immutable():
    config = VTACompilerConfig.from_env(vta.get_env())

    with pytest.raises(FrozenInstanceError):
        config.batch = 2


def test_compiler_config_is_deterministic_for_the_same_environment():
    env = vta.get_env()

    assert VTACompilerConfig.from_env(env) == VTACompilerConfig.from_env(env)


def test_importing_vta_does_not_register_external_compiler():
    result = run_isolated_python(
        f"""
        import tvm

        assert tvm.get_global_func(
            {LEGACY_COMPILER_GLOBAL!r}, allow_missing=True
        ) is None
        import vta
        assert tvm.get_global_func(
            {LEGACY_COMPILER_GLOBAL!r}, allow_missing=True
        ) is None
        """
    )

    assert result.returncode == 0, result.stderr


def test_compiler_config_rejects_invalid_block_factor():
    env = vta.get_env()
    invalid_env = copy.copy(env)
    invalid_env.BLOCK_IN = 0

    with pytest.raises(ValueError, match="BLOCK_IN must be a positive integer"):
        VTACompilerConfig.from_env(invalid_env)


def test_compiler_config_rejects_non_vta_execution_target():
    env = vta.get_env()

    class InvalidTargetEnvironment:
        target = tvm.target.Target("llvm")

        def __getattr__(self, name):
            return getattr(env, name)

    with pytest.raises(ValueError, match="target must use the ext_dev kind with device=vta"):
        VTACompilerConfig.from_env(InvalidTargetEnvironment())


def test_compiler_config_converts_invalid_host_target_error():
    env = vta.get_env()

    class InvalidHostEnvironment:
        target_host = ""

        def __getattr__(self, name):
            return getattr(env, name)

    with pytest.raises(ValueError, match="target_host must define a valid TVM target"):
        VTACompilerConfig.from_env(InvalidHostEnvironment())


def test_supported_fixture_is_typed_and_deterministic():
    first = make_qnn_conv2d_module(vta.get_env())
    second = make_qnn_conv2d_module(vta.get_env())

    assert isinstance(first, tvm.IRModule)
    assert first["main"].checked_type is not None
    assert tvm.ir.structural_equal(first, second)


def test_near_miss_fixture_names_unsupported_capability():
    mod, capability = make_qnn_conv2d_near_miss_module(vta.get_env())

    assert isinstance(mod, tvm.IRModule)
    assert mod["main"].checked_type is not None
    assert capability == "constant_weights"


def _operator_names(expr):
    names = []

    def visit(node):
        if isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op):
            names.append(node.op.name)

    relay.analysis.post_order_visit(expr, visit)
    return names


def _find_call(expr, operator_name):
    matches = []

    def visit(node):
        if (
            isinstance(node, relay.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == operator_name
        ):
            matches.append(node)

    relay.analysis.post_order_visit(expr, visit)
    assert len(matches) == 1
    return matches[0]


def test_supported_fixture_locks_host_and_candidate_operator_order():
    mod = make_qnn_conv2d_module(vta.get_env())

    assert _operator_names(mod["main"].body) == [
        "abs",
        "nn.conv2d",
        "right_shift",
        "clip",
        "cast",
        "transpose",
    ]


def test_supported_fixture_locks_types_and_constant_ownership():
    env = vta.get_env()
    mod = make_qnn_conv2d_module(env)
    conv = _find_call(mod["main"].body, "nn.conv2d")

    assert len(mod["main"].params) == 1
    assert isinstance(conv.args[1], relay.Constant)
    assert conv.args[0].checked_type.dtype == env.inp_dtype
    assert conv.args[1].checked_type.dtype == env.wgt_dtype
    assert conv.checked_type.dtype == env.acc_dtype
    assert mod["main"].ret_type.dtype == env.out_dtype


def test_near_miss_differs_only_by_constant_weight_capability():
    env = vta.get_env()
    supported = make_qnn_conv2d_module(env)
    near_miss, capability = make_qnn_conv2d_near_miss_module(env)
    weight_shape = (env.BLOCK_OUT, env.BLOCK_IN, 3, 3)
    bound_main = relay.build_module.bind_params_by_name(
        near_miss["main"],
        {"weight": tvm.nd.array(np.ones(weight_shape, dtype=env.wgt_dtype))},
    )
    rebound = relay.transform.InferType()(tvm.IRModule.from_expr(bound_main))

    assert capability == "constant_weights"
    assert tvm.ir.structural_equal(supported, rebound, map_free_vars=True)
