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

"""Explicit CPU/VTA Relay device planning for heterogeneous builds."""

from dataclasses import dataclass

import tvm
from tvm import relay

from .contract import COMPILER_NAME


@dataclass(frozen=True)
class VTADevicePlan:
    """Immutable Relay module and canonical targets for a VTA build."""

    module: tvm.IRModule
    targets: tvm.ir.Array


def _typed_tensor(expr):
    try:
        checked_type = expr.checked_type
    except ValueError:
        return False
    return isinstance(checked_type, relay.TensorType)


def _validate_typed_module(module):
    if not isinstance(module, tvm.IRModule):
        raise TypeError("module must be a tvm.IRModule")
    try:
        main = module["main"]
    except tvm.error.TVMError as err:
        raise ValueError("module must contain a main function") from err
    if not isinstance(main, relay.Function):
        raise ValueError("module main must be a Relay function")
    try:
        main.checked_type
    except ValueError as err:
        raise ValueError("module must have inferred types") from err
    if main.ret_type is None or not all(_typed_tensor(param) for param in main.params):
        raise ValueError("module must have inferred tensor types")
    for function in module.functions.values():
        if not isinstance(function, relay.Function):
            continue
        try:
            function.checked_type
        except ValueError as err:
            raise ValueError("module must have inferred types") from err


def _validate_vta_function(function):
    attrs = function.attrs
    if attrs is None or attrs.get_str("Compiler") != COMPILER_NAME:
        raise ValueError("outlined function must have Compiler='vta'")
    if "Primitive" not in attrs or int(attrs["Primitive"]) != 1:
        raise ValueError("outlined VTA function must have Primitive=1")
    if "global_symbol" not in attrs or not attrs.get_str("global_symbol"):
        raise ValueError("outlined VTA function must have a non-empty global_symbol")
    if not all(_typed_tensor(param) for param in function.params) or not isinstance(
        function.ret_type, relay.TensorType
    ):
        raise ValueError("outlined VTA function must have inferred tensor types")


class _VTACallAnnotator(relay.ExprMutator):
    """Constrain host and outlined VTA calls to their virtual devices."""

    def __init__(self, cpu_device, vta_device, symbols):
        super().__init__()
        self._cpu_device = cpu_device
        self._vta_device = vta_device
        self._symbols = symbols
        self._vta_bindings = set()

    def visit_let(self, let):
        is_vta_call = (
            isinstance(let.value, relay.Call)
            and isinstance(let.value.op, relay.GlobalVar)
            and let.value.op.name_hint in self._symbols
        )
        value = self.visit(let.value)
        if is_vta_call:
            self._vta_bindings.add(let.var.name_hint)
        body = self.visit(let.body)
        return relay.Let(let.var, value, body, let.span)

    def visit_call(self, call):
        updated = super().visit_call(call)
        if isinstance(updated.op, relay.GlobalVar) and updated.op.name_hint in self._symbols:
            # Constrain the boundary and let Relay's device planner materialize
            # the CPU-to-ext_dev transfer.
            args = [
                relay.annotation.on_device(
                    arg,
                    self._vta_device,
                    constrain_result=True,
                    constrain_body=False,
                )
                for arg in updated.args
            ]
            updated = relay.Call(
                updated.op,
                args,
                updated.attrs,
                updated.type_args,
                updated.span,
            )
            return relay.annotation.on_device(
                updated,
                self._vta_device,
                constrain_result=True,
                constrain_body=False,
            )
        args = [
            relay.annotation.on_device(
                arg,
                self._cpu_device,
                constrain_result=True,
                constrain_body=False,
            )
            if isinstance(arg, relay.Var) and arg.name_hint in self._vta_bindings
            else arg
            for arg in updated.args
        ]
        updated = relay.Call(updated.op, args, updated.attrs, updated.type_args, updated.span)
        return relay.annotation.on_device(
            updated,
            self._cpu_device,
            constrain_result=True,
            constrain_body=True,
        )


def _canonical_host_target(host_target):
    if not isinstance(host_target, tvm.target.Target):
        raise TypeError("host_target must be a tvm.target.Target")
    if host_target.host is not None:
        raise ValueError("host_target must not have a nested host target")
    if host_target.get_target_device_type() != tvm.runtime.Device.kDLCPU:
        raise ValueError("host_target must target the CPU device")
    if host_target.kind.name not in ("llvm", "c"):
        raise ValueError("host_target must use the llvm or c target kind")
    return tvm.target.Target(host_target)


def plan_devices_for_vta(module, host_target):
    """Annotate a typed partitioned module and return CPU/VTA build targets.

    The input module is cloned before the main function is annotated. Host
    computation is assigned to the CPU target while calls to outlined VTA
    functions are assigned to ``ext_dev``. Relay's regular device planner then
    inserts any required ``device_copy`` boundaries during compilation.
    """

    _validate_typed_module(module)
    cpu_target = _canonical_host_target(host_target)

    vta_functions = {}
    for global_var, function in module.functions.items():
        if not isinstance(function, relay.Function):
            continue
        attrs = function.attrs
        if attrs is not None and "Compiler" in attrs and attrs.get_str("Compiler") == COMPILER_NAME:
            _validate_vta_function(function)
            vta_functions[global_var.name_hint] = function
    if not vta_functions:
        raise ValueError("module must contain an outlined VTA function")

    # The registered compiler target kind is ``vta``.  Its device type is
    # ext_dev, which is the virtual device used by Relay annotations.
    vta_target = tvm.target.Target("vta", host=cpu_target)
    if vta_target.get_target_device_type() != tvm.runtime.Device.kDLExtDev:
        raise ValueError("VTA target must use the ext_dev device")

    cpu_device = tvm.device(cpu_target.get_target_device_type(), 0)
    vta_device = tvm.device(vta_target.get_target_device_type(), 0)
    vta_virtual_device = tvm.target.VirtualDevice(vta_device)
    planned = relay.transform.ToANormalForm()(module.clone())
    for global_var, function in list(planned.functions.items()):
        if not isinstance(function, relay.Function):
            continue
        attrs = function.attrs
        if attrs is None or "Compiler" not in attrs or attrs.get_str("Compiler") != COMPILER_NAME:
            continue
        params = [
            relay.expr.VarWithFields(param, None, None, vta_virtual_device, None)
            for param in function.params
        ]
        body = relay.bind(function.body, dict(zip(function.params, params)))
        function = relay.Function(
            params,
            body,
            function.ret_type,
            function.type_params,
            function.attrs,
        )
        planned[global_var] = relay.function.FunctionWithFields(
            function,
            params,
            None,
            None,
            None,
            None,
            vta_virtual_device,
            None,
        )
    main = planned["main"]
    body = _VTACallAnnotator(cpu_device, vta_device, frozenset(vta_functions)).visit(main.body)
    body = relay.annotation.on_device(body, cpu_device, constrain_result=True, constrain_body=False)
    planned["main"] = relay.Function(
        main.params,
        body,
        main.ret_type,
        main.type_params,
        main.attrs,
    )
    planned = relay.transform.InferType()(planned)
    targets = tvm.runtime.convert([cpu_target, vta_target])
    return VTADevicePlan(module=planned, targets=targets)


__all__ = ["VTADevicePlan", "plan_devices_for_vta"]
