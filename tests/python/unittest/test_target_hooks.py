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

"""Contracts for the native, module-at-a-time VTA target hook."""

import pytest
import tvm
import vta
from tvm import relay

from byoc_utils import make_qnn_conv2d_module
from vta.relay import partition_for_vta


LEGACY_COMPILER_GLOBAL = "relay.ext." + "vta"


def _is_vta_relay_function(function):
    return (
        isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        and function.attrs.get_str("Compiler") == "vta"
    )


def _vta_relay_functions(mod):
    """Return global and nested Relay functions still owned by VTA."""
    found = []
    for global_var, function in mod.functions.items():
        if not isinstance(function, relay.Function):
            continue
        if _is_vta_relay_function(function):
            found.append((global_var.name_hint, function))

        def visit(node):
            if _is_vta_relay_function(node):
                found.append((global_var.name_hint, node))

        relay.analysis.post_order_visit(function.body, visit)
    return found


def _vta_function_template():
    partitioned = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env()), mod_name="target_hooks_template"
    )
    return next(
        function
        for function in partitioned.functions.values()
        if _is_vta_relay_function(function)
    )


def _module_with_global_vta_functions(count, *, malformed_index=None):
    template = _vta_function_template()
    mod = tvm.IRModule()
    global_vars = []
    for index in range(count):
        symbol = f"tvmgen_target_hooks_vta_{index}"
        function = template.with_attr("global_symbol", symbol)
        if index == malformed_index:
            function = function.with_attr("Primitive", 0)
        global_var = relay.GlobalVar(symbol)
        mod[global_var] = function
        global_vars.append(global_var)

    data_type = template.params[0].checked_type
    data = relay.var("data", type_annotation=data_type)
    host_value = relay.abs(data)
    calls = [relay.Call(global_var, [host_value]) for global_var in global_vars]
    result = calls[0] if len(calls) == 1 else relay.Tuple(calls)
    mod["main"] = relay.Function([data], result)

    host_data = relay.var("host_data", type_annotation=data_type)
    mod["host_helper"] = relay.Function([host_data], relay.negative(host_data))
    mod = mod.with_attr("target_hooks_fixture", tvm.runtime.String("preserve"))
    return relay.transform.InferType()(mod), global_vars


def _module_with_nested_vta_function():
    symbol = "tvmgen_target_hooks_nested_vta"
    nested = _vta_function_template().with_attr("global_symbol", symbol)
    data = relay.var("data", type_annotation=nested.params[0].checked_type)
    main = relay.Function([data], relay.Call(nested, [relay.abs(data)]))
    return relay.transform.InferType()(tvm.IRModule.from_expr(main)), symbol


def _module_with_global_and_nested_vta_functions():
    mod, _ = _module_with_global_vta_functions(1)
    nested = _vta_function_template().with_attr(
        "global_symbol", "tvmgen_target_hooks_lower_te_nested_vta"
    )
    main_global = mod.get_global_var("main")
    main = mod[main_global]
    nested_call = relay.Call(nested, [relay.abs(main.params[0])])
    mod.update_func(
        main_global, relay.Function(main.params, relay.Tuple([main.body, nested_call]))
    )
    return relay.transform.InferType()(mod)


def _relay_to_tir_hook():
    hook = tvm.target.Target("vta").get_kind_attr("RelayToTIR")
    assert hook is not None
    assert isinstance(hook, tvm.transform.ModulePass)
    return hook


def test_relay_to_tir_hook_is_typed_and_preserves_module_without_vta_functions():
    data = relay.var("data", shape=(1, 16, 8, 8), dtype=vta.get_env().inp_dtype)
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(relay.abs(data)))

    lowered = _relay_to_tir_hook()(mod)

    tvm.ir.assert_structural_equal(lowered, mod)
    assert tvm.get_global_func(LEGACY_COMPILER_GLOBAL, allow_missing=True) is None


def test_relay_to_tir_rejects_vta_functions_without_active_target():
    mod, _ = _module_with_global_vta_functions(1)

    with pytest.raises(tvm.error.TVMError, match="active vta target"):
        _relay_to_tir_hook()(mod)


@pytest.mark.parametrize("vta_function_count", [1, 3])
@pytest.mark.parametrize("host_kind", ["llvm", "c"])
def test_one_hook_invocation_replaces_every_existing_vta_global_in_place(
    vta_function_count, host_kind
):
    mod, global_vars = _module_with_global_vta_functions(vta_function_count)
    host_helper_before = mod["host_helper"]

    with tvm.target.Target("vta", host=tvm.target.Target(host_kind)):
        lowered = _relay_to_tir_hook()(mod)

    assert _vta_relay_functions(lowered) == []
    assert tvm.ir.structural_equal(lowered["host_helper"], host_helper_before)
    assert str(lowered.attrs["target_hooks_fixture"]) == "preserve"
    for global_var in global_vars:
        updated_global_var = lowered.get_global_var(global_var.name_hint)
        assert updated_global_var.same_as(global_var)
        primfunc = lowered[updated_global_var]
        assert isinstance(primfunc, tvm.tir.PrimFunc)
        assert str(primfunc.attrs["global_symbol"]) == global_var.name_hint
        assert primfunc.attrs["relay_attrs"].get_str("Compiler") == "vta"
    assert tvm.get_global_func(LEGACY_COMPILER_GLOBAL, allow_missing=True) is None


@pytest.mark.parametrize("host_kind", ["llvm", "c"])
def test_relay_to_tir_outlines_and_replaces_nested_vta_function(host_kind):
    mod, symbol = _module_with_nested_vta_function()

    with tvm.target.Target("vta", host=tvm.target.Target(host_kind)):
        lowered = _relay_to_tir_hook()(mod)

    assert _vta_relay_functions(lowered) == []
    assert isinstance(lowered[symbol], tvm.tir.PrimFunc)
    assert tvm.get_global_func(LEGACY_COMPILER_GLOBAL, allow_missing=True) is None


def test_relay_to_tir_validates_all_vta_functions_before_mutating_module():
    mod, global_vars = _module_with_global_vta_functions(2, malformed_index=1)
    before = tvm.ir.save_json(mod)

    with pytest.raises((ValueError, tvm.error.TVMError), match="Primitive=1"):
        _relay_to_tir_hook()(mod)

    assert tvm.ir.save_json(mod) == before
    assert all(
        isinstance(mod[global_var], relay.Function) for global_var in global_vars
    )
    assert tvm.get_global_func(LEGACY_COMPILER_GLOBAL, allow_missing=True) is None


@tvm.instrument.pass_instrument
class _LowerTEBoundary:
    def __init__(self):
        self.visits = 0

    def run_before_pass(self, mod, pass_info):
        if pass_info.name == "LowerTE":
            self.visits += 1
            assert _vta_relay_functions(mod) == []


def test_no_global_or_nested_vta_relay_function_reaches_ordinary_lower_te():
    mod = _module_with_global_and_nested_vta_functions()
    host_target = tvm.target.Target("llvm")
    generic_target = tvm.target.Target("llvm", host=host_target)
    vta_target = tvm.target.Target("vta", host=host_target)
    boundary = _LowerTEBoundary()
    pass_context = tvm.transform.PassContext(instruments=[boundary])
    config = tvm.target.make_compilation_config(
        pass_context, [generic_target, vta_target]
    )
    mod = relay.transform.PlanDevices(config)(mod)
    mod = relay.transform.InferType()(mod)
    lower_te = tvm.get_global_func("relay.tec.LowerTE")

    with pass_context:
        lower_te("target_hooks", config)(mod)

    assert boundary.visits == 1
    assert tvm.get_global_func(LEGACY_COMPILER_GLOBAL, allow_missing=True) is None
