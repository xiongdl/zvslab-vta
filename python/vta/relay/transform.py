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

"""Function-local Relay legalization and lowering for VTA."""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.backend import te_compiler

from ..build_module import build_config as vta_build_config
from ..environment import get_env
from .contract import COMPILER_NAME, VTACompilerConfig
from .patterns import QNN_CONV2D_COMPOSITE, check_qnn_conv2d


def _is_typed_tensor(expr):
    return isinstance(getattr(expr, "checked_type", None), relay.TensorType)


def _composite_calls(func):
    calls = []

    def visit(node):
        if (
            isinstance(node, relay.Call)
            and isinstance(node.op, relay.Function)
            and node.op.attrs is not None
            and "Composite" in node.op.attrs
        ):
            calls.append(node)

    relay.analysis.post_order_visit(func.body, visit)
    return calls


def _validate_vta_function(func, config=None):
    """Validate an outlined VTA function and return its composite call."""
    if not isinstance(func, relay.Function):
        raise TypeError("func must be a tvm.relay.Function")
    if config is not None and not isinstance(config, VTACompilerConfig):
        raise TypeError("config must be a VTACompilerConfig or None")
    config = config or VTACompilerConfig.from_env(get_env())

    attrs = func.attrs
    if attrs is None or "Compiler" not in attrs or attrs.get_str("Compiler") != COMPILER_NAME:
        raise ValueError("outlined function must have Compiler='vta'")
    if "Primitive" not in attrs or int(attrs["Primitive"]) != 1:
        raise ValueError("outlined VTA function must have Primitive=1")
    if "global_symbol" not in attrs or not attrs.get_str("global_symbol"):
        raise ValueError("outlined VTA function must have a non-empty global_symbol")
    if not all(_is_typed_tensor(param) for param in func.params) or not isinstance(
        func.ret_type, relay.TensorType
    ):
        raise ValueError("outlined VTA function must have inferred tensor types")

    composite_calls = _composite_calls(func)
    if len(composite_calls) != 1:
        raise ValueError("outlined VTA function must contain exactly one composite call")
    composite_call = composite_calls[0]
    composite_name = composite_call.op.attrs.get_str("Composite")
    if composite_name != QNN_CONV2D_COMPOSITE:
        raise ValueError(f"outlined VTA function must contain {QNN_CONV2D_COMPOSITE}")
    if not check_qnn_conv2d(composite_call.op.body, config):
        raise ValueError(
            f"{QNN_CONV2D_COMPOSITE} does not satisfy the active VTA configuration"
        )
    return composite_call


def _static_shape(expr):
    return tuple(int(dim) for dim in expr.checked_type.shape)


def _pack_data(data, shape, data_layout, config):
    if data_layout == "NCHW":
        batch, channels, height, width = shape
        reshaped = relay.reshape(
            data,
            (
                batch // config.batch,
                config.batch,
                channels // config.block_in,
                config.block_in,
                height,
                width,
            ),
        )
        return relay.transpose(reshaped, axes=(0, 2, 4, 5, 1, 3))

    batch, height, width, channels = shape
    reshaped = relay.reshape(
        data,
        (
            batch // config.batch,
            config.batch,
            height,
            width,
            channels // config.block_in,
            config.block_in,
        ),
    )
    return relay.transpose(reshaped, axes=(0, 4, 2, 3, 1, 5))


def _pack_weight(weight, shape, kernel_layout, config):
    if kernel_layout == "OIHW":
        output_channels, input_channels, height, width = shape
        reshaped = relay.reshape(
            weight,
            (
                output_channels // config.block_out,
                config.block_out,
                input_channels // config.block_in,
                config.block_in,
                height,
                width,
            ),
        )
        return relay.transpose(reshaped, axes=(0, 2, 4, 5, 1, 3))

    height, width, input_channels, output_channels = shape
    reshaped = relay.reshape(
        weight,
        (
            height,
            width,
            input_channels // config.block_in,
            config.block_in,
            output_channels // config.block_out,
            config.block_out,
        ),
    )
    return relay.transpose(reshaped, axes=(4, 2, 0, 1, 5, 3))


def _pack_output_constant(constant, output_layout, config):
    values = constant.data.numpy()
    if values.ndim == 0:
        # Keep scalar constants scalar. VTA's ALU can encode scalar arithmetic
        # as an immediate; expanding it to a packed tensor forces an otherwise
        # unnecessary DRAM-to-accumulator DMA during deployment.
        return constant

    channel_vector = values.ndim == 1
    if values.ndim == 1:
        values = values.reshape(values.shape[0], 1, 1)
    elif values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3:
        raise ValueError("VTA output constant must have a broadcast-compatible shape")
    if output_layout == "NHWC" and not channel_vector:
        values = values.transpose(2, 0, 1)
    channels, height, width = values.shape
    if channels <= 0 or channels % config.block_out != 0:
        raise ValueError("VTA output constant channels must be broadcast-compatible")
    values = values.reshape(
        channels // config.block_out, config.block_out, height, width, 1
    ).transpose(0, 2, 3, 4, 1)
    values = np.broadcast_to(
        values,
        (channels // config.block_out, height, width, config.batch, config.block_out),
    )
    return relay.const(values.copy(), dtype=constant.data.dtype)


def _unpack_data(data, output_shape, output_layout):
    axes = (0, 4, 1, 5, 2, 3) if output_layout == "NCHW" else (0, 4, 2, 3, 1, 5)
    transposed = relay.transpose(data, axes=axes)
    return relay.reshape(transposed, output_shape)


def _infer_function(func):
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(func))
    return next(iter(mod.functions.values()))


def legalize_vta_function(func, config=None):
    """Rewrite one outlined VTA function into locally packed Relay."""
    config = config or VTACompilerConfig.from_env(get_env())
    composite_call = _validate_vta_function(func, config)
    composite_body = composite_call.op.body
    clipped = composite_body.args[0]
    shifted = clipped.args[0]
    conv_or_bias = shifted.args[0]
    bias = None
    if isinstance(conv_or_bias.op, tvm.ir.Op) and conv_or_bias.op.name in (
        "nn.bias_add",
        "add",
    ):
        conv2d = conv_or_bias.args[0]
        bias = conv_or_bias.args[1]
    else:
        conv2d = conv_or_bias
    if not isinstance(conv2d.op, tvm.ir.Op) or conv2d.op.name != "nn.conv2d":
        raise ValueError("VTA composite does not contain the expected nn.conv2d")

    data_layout = str(conv2d.attrs.data_layout)
    kernel_layout = str(conv2d.attrs.kernel_layout)
    output_layout = str(conv2d.attrs.out_layout) or data_layout
    packed_data = _pack_data(
        composite_call.args[0],
        _static_shape(composite_call.args[0]),
        data_layout,
        config,
    )
    packed_weight = _pack_weight(
        conv2d.args[1], _static_shape(conv2d.args[1]), kernel_layout, config
    )
    packed_data_layout = f"NCHW{config.batch}n{config.block_in}c"
    packed_output_layout = f"NCHW{config.batch}n{config.block_out}c"
    packed_kernel_layout = f"OIHW{config.block_out}o{config.block_in}i"
    packed_conv2d = relay.nn.conv2d(
        packed_data,
        packed_weight,
        strides=conv2d.attrs.strides,
        padding=conv2d.attrs.padding,
        dilation=conv2d.attrs.dilation,
        groups=conv2d.attrs.groups,
        channels=conv2d.attrs.channels,
        kernel_size=conv2d.attrs.kernel_size,
        data_layout=packed_data_layout,
        kernel_layout=packed_kernel_layout,
        out_layout=packed_output_layout,
        out_dtype=conv2d.attrs.out_dtype,
    )
    packed_accumulator = packed_conv2d
    if bias is not None:
        packed_accumulator = relay.add(
            packed_accumulator, _pack_output_constant(bias, output_layout, config)
        )
    packed_shifted = relay.right_shift(packed_accumulator, shifted.args[1])
    packed_clipped = relay.clip(packed_shifted, clipped.attrs.a_min, clipped.attrs.a_max)
    packed_cast = relay.cast(packed_clipped, composite_body.attrs.dtype)
    unpacked = _unpack_data(packed_cast, _static_shape(composite_body), output_layout)
    legalized = relay.Function(
        func.params,
        unpacked,
        func.ret_type,
        func.type_params,
        func.attrs,
    )
    return _infer_function(legalized)


def _packed_output(legalized):
    reshape = legalized.body
    if not isinstance(reshape, relay.Call) or reshape.op.name != "reshape":
        raise ValueError("legalized VTA function must end with output reshape")
    transpose = reshape.args[0]
    if (
        not isinstance(transpose, relay.Call)
        or transpose.op.name != "transpose"
        or tuple(int(axis) for axis in transpose.attrs.axes)
        not in ((0, 4, 1, 5, 2, 3), (0, 4, 2, 3, 1, 5))
    ):
        raise ValueError("legalized VTA function must contain the approved output unpack")
    return transpose.args[0]


def _has_vta_gemm_tensorization(schedule):
    """Return whether a TE schedule contains the VTA GEMM tensor intrinsic."""
    if schedule is None:
        return False
    for stage in schedule.stages:
        for iter_var_attr in stage.iter_var_attrs.values():
            tensor_intrin = iter_var_attr.tensor_intrin
            if tensor_intrin is not None and tensor_intrin.name == "GEMM":
                return True
    return False


def _packed_core(func, config):
    legalized = legalize_vta_function(func, config)
    packed_output = _packed_output(legalized)
    packed_core = relay.Function(
        legalized.params,
        packed_output,
        packed_output.checked_type,
    ).with_attr("Primitive", 1)
    return _infer_function(packed_core)


def _schedule_packed_core(packed_core, config):
    target = tvm.target.Target(config.target, host=config.host_target)
    with vta_build_config():
        cached = te_compiler.get().lower(packed_core, target, mod_name="vta")
    if cached.schedule is None or not _has_vta_gemm_tensorization(cached.schedule):
        raise ValueError("VTA lowering did not produce a GEMM-tensorized TE schedule")
    return cached


def _lower_to_scheduled_te(func, config=None):
    """Legalize and schedule the packed core of one outlined VTA function."""
    config = config or VTACompilerConfig.from_env(get_env())
    return _schedule_packed_core(_packed_core(func, config), config)


def capture_vta_compute(func, config=None):
    """Return the deployment function's packed compute before schedule choice.

    The function is validated and legalized through the same path used by
    :func:`lower_vta_function`. Non-scalar Relay constants are lifted into
    explicit parameters and returned as bound tensors for standalone capture
    or measurement.
    """
    config = config or VTACompilerConfig.from_env(get_env())
    _validate_vta_function(func, config)
    return _lift_constants(_packed_core(func, config))


def lower_vta_compute(compute, config=None):
    """Schedule and lower captured packed compute with active AutoTVM config."""
    if not isinstance(compute, relay.Function):
        raise TypeError("compute must be a captured Relay function")
    config = config or VTACompilerConfig.from_env(get_env())
    return _schedule_packed_core(compute, config)


class _ConstantLifter(relay.ExprMutator):
    def __init__(self):
        super().__init__()
        self.params = []
        self.values = []

    def visit_constant(self, constant):
        if len(constant.checked_type.shape) == 0:
            return constant
        param = relay.var(f"vta_const_{len(self.params)}", type_annotation=constant.checked_type)
        self.params.append(param)
        self.values.append(constant.data)
        return param


def _lift_constants(func):
    lifter = _ConstantLifter()
    body = lifter.visit(func.body)
    lifted = relay.Function(
        list(func.params) + lifter.params,
        body,
        func.ret_type,
        func.type_params,
        func.attrs,
    )
    return _infer_function(lifted), lifter.values


def _internalize_constants(primfunc, runtime_param_count, values):
    params = list(primfunc.params)
    constant_params = params[runtime_param_count:-1]
    if len(constant_params) != len(values):
        raise ValueError("VTA TIR constant parameters do not match Relay constants")
    body = primfunc.body
    for index in reversed(range(len(values))):
        param = constant_params[index]
        buffer = primfunc.buffer_map[param]
        device_data = tvm.tir.Var(
            f"vta_const_{index}",
            tvm.ir.PointerType(tvm.ir.PrimType(buffer.dtype), "global"),
        )
        host_data = tvm.tir.Var(
            f"vta_const_{index}_host",
            tvm.ir.PointerType(tvm.ir.PrimType(buffer.dtype), "global"),
        )
        device_cpu_data = tvm.tir.Var(
            f"vta_const_{index}_ptr",
            tvm.ir.PointerType(tvm.ir.PrimType(buffer.dtype), "global"),
        )
        element_count = int(np.prod(values[index].shape))
        host_buffer = tvm.tir.decl_buffer(
            (element_count,), buffer.dtype, data=host_data, name=f"vta_const_{index}_host"
        )
        device_cpu_buffer = tvm.tir.decl_buffer(
            (element_count,), buffer.dtype, data=device_cpu_data, name=f"vta_const_{index}"
        )
        builder = tvm.tir.ir_builder.create()
        with builder.for_range(0, element_count, name=f"vta_const_{index}_index") as offset:
            builder.emit(
                tvm.tir.BufferStore(
                    device_cpu_buffer,
                    tvm.tir.BufferLoad(host_buffer, [offset]),
                    [offset],
                )
            )

        body = tvm.tir.stmt_functor.substitute(body, {buffer.data: device_data})
        body = tvm.tir.SeqStmt([builder.get(), body])
        body = tvm.tir.LetStmt(
            device_cpu_data,
            tvm.tir.call_extern(
                "handle", "VTABufferCPUPtr", get_env().dev.command_handle, device_data
            ),
            body,
        )
        body = tvm.tir.Allocate(
            device_data,
            buffer.dtype,
            (element_count,),
            tvm.tir.const(True, "bool"),
            body,
        )
        flattened = tvm.nd.array(values[index].numpy().reshape(-1))
        body = tvm.tir.AllocateConst(
            host_data,
            buffer.dtype,
            (element_count,),
            flattened,
            body,
        )
    kept_params = params[:runtime_param_count] + params[-1:]
    buffer_map = {param: primfunc.buffer_map[param] for param in kept_params}
    return tvm.tir.PrimFunc(
        kept_params,
        body,
        primfunc.ret_type,
        buffer_map,
        primfunc.attrs,
        primfunc.span,
    )


def _restore_unpacked_output(primfunc, output_type, output_layout, config):
    """Make the packed TE output internal and restore the original Relay ABI."""
    packed_param = primfunc.params[-1]
    packed_buffer = primfunc.buffer_map[packed_param]
    packed_data = tvm.tir.Var(
        "packed_output",
        tvm.ir.PointerType(tvm.ir.PrimType(packed_buffer.dtype), "global"),
    )
    packed_cpu_data = tvm.tir.Var(
        "packed_output_ptr",
        tvm.ir.PointerType(tvm.ir.PrimType(packed_buffer.dtype), "global"),
    )
    internal_buffer = tvm.tir.decl_buffer(
        packed_buffer.shape,
        packed_buffer.dtype,
        name="packed_output",
        data=packed_cpu_data,
    )
    body = tvm.tir.stmt_functor.substitute(primfunc.body, {packed_buffer.data: packed_data})

    output_param = tvm.tir.Var("output", "handle")
    output_buffer = tvm.tir.decl_buffer(output_type.shape, output_type.dtype, name="output")
    output_cpu_data = tvm.tir.Var(
        "output_ptr",
        tvm.ir.PointerType(tvm.ir.PrimType(output_buffer.dtype), "global"),
    )
    output_cpu_buffer = tvm.tir.decl_buffer(
        output_buffer.shape,
        output_buffer.dtype,
        name="output",
        data=output_cpu_data,
    )
    builder = tvm.tir.ir_builder.create()
    if output_layout == "NCHW":
        batch, channels, height, width = output_type.shape
    else:
        batch, height, width, channels = output_type.shape
    with builder.for_range(0, batch, name="n") as n:
        with builder.for_range(0, channels, name="c") as c:
            with builder.for_range(0, height, name="h") as h:
                with builder.for_range(0, width, name="w") as w:
                    output_indices = (
                        [n, c, h, w] if output_layout == "NCHW" else [n, h, w, c]
                    )
                    builder.emit(
                        tvm.tir.BufferStore(
                            output_cpu_buffer,
                            tvm.tir.BufferLoad(
                                internal_buffer,
                                [
                                    n // config.batch,
                                    c // config.block_out,
                                    h,
                                    w,
                                    n % config.batch,
                                    c % config.block_out,
                                ],
                            ),
                            output_indices,
                        )
                    )
    unpack = tvm.tir.LetStmt(
        packed_cpu_data,
        tvm.tir.call_extern(
            "handle", "VTABufferCPUPtr", get_env().dev.command_handle, packed_data
        ),
        tvm.tir.LetStmt(
            output_cpu_data,
            tvm.tir.call_extern(
                "handle",
                "VTABufferCPUPtr",
                get_env().dev.command_handle,
                output_buffer.data,
            ),
            builder.get(),
        ),
    )
    body = tvm.tir.Allocate(
        packed_data,
        packed_buffer.dtype,
        packed_buffer.shape,
        tvm.tir.const(True, "bool"),
        tvm.tir.SeqStmt([body, unpack]),
    )
    buffer_map = {param: primfunc.buffer_map[param] for param in primfunc.params[:-1]}
    buffer_map[output_param] = output_buffer
    return tvm.tir.PrimFunc(
        list(primfunc.params[:-1]) + [output_param],
        body,
        primfunc.ret_type,
        buffer_map,
        primfunc.attrs,
        primfunc.span,
    )


def lower_vta_function(func, config=None):
    """Lower one outlined VTA Relay function for the module-level target hook.

    Parameters
    ----------
    func : tvm.relay.Function
        A typed, primitive function produced by :func:`partition_for_vta`.
        The function is revalidated against the active VTA contract before
        legalization and lowering.

    config : VTACompilerConfig, optional
        Explicit compilation configuration.  When omitted, the configuration
        is derived from the active VTA environment.

    Returns
    -------
    primfunc : tvm.tir.PrimFunc
        A single scheduled VTA PrimFunc with the Relay ``global_symbol``, VTA
        target, and original Relay attributes attached.  Constants and packed
        tensors are internal; parameters retain the original unpacked NCHW or
        NHWC ABI in input-then-output order.

    Raises
    ------
    TypeError
        If ``func`` or ``config`` has the wrong Python type.
    ValueError
        If the outlined function violates the VTA contract or lowering does
        not produce exactly one GEMM-tensorized VTA PrimFunc.

    Notes
    -----
    This is an internal boundary used by the native RelayToTIR hook and is
    intentionally not exported from :mod:`vta.relay`.
    """
    config = config or VTACompilerConfig.from_env(get_env())
    composite_call = _validate_vta_function(func, config)
    composite_body = composite_call.op.body
    conv_or_bias = composite_body.args[0].args[0].args[0]
    if isinstance(conv_or_bias.op, tvm.ir.Op) and conv_or_bias.op.name in (
        "nn.bias_add",
        "add",
    ):
        conv2d = conv_or_bias.args[0]
    else:
        conv2d = conv_or_bias
    output_layout = str(conv2d.attrs.out_layout) or str(conv2d.attrs.data_layout)

    packed_core, constants = capture_vta_compute(func, config)
    cached = lower_vta_compute(packed_core, config)
    symbol = func.attrs.get_str("global_symbol")
    scheduled_primfuncs = [
        item for item in cached.funcs.functions.values() if isinstance(item, tvm.tir.PrimFunc)
    ]
    if len(scheduled_primfuncs) != 1:
        raise ValueError("VTA TE lowering must produce exactly one scheduled TIR PrimFunc")
    primfunc = _internalize_constants(scheduled_primfuncs[0], len(func.params), constants)
    primfunc = _restore_unpacked_output(primfunc, func.ret_type, output_layout, config)
    return (
        primfunc
        .with_attr("global_symbol", symbol)
        .with_attr("target", cached.target)
        .with_attr("relay_attrs", func.attrs)
    )


def _is_vta_relay_function(func):
    return (
        isinstance(func, relay.Function)
        and func.attrs is not None
        and "Compiler" in func.attrs
        and func.attrs.get_str("Compiler") == COMPILER_NAME
    )


def _collect_vta_relay_functions(mod):
    """Collect unique global and nested VTA Relay functions."""
    functions = []
    seen_handles = set()

    def collect(node):
        if not _is_vta_relay_function(node):
            return
        handle = node.handle.value
        if handle not in seen_handles:
            seen_handles.add(handle)
            functions.append(node)

    for function in mod.functions.values():
        if not isinstance(function, relay.Function):
            continue
        collect(function)
        relay.analysis.post_order_visit(function.body, collect)
    return functions


def _global_vta_relay_functions(mod):
    return sorted(
        (
            (global_var, function)
            for global_var, function in mod.functions.items()
            if _is_vta_relay_function(function)
        ),
        key=lambda item: item[0].name_hint,
    )


@tvm.register_func("vta.relay._relay_to_tir")
def _relay_to_tir(mod):
    """Lower every VTA Relay function in one module transaction."""
    if not isinstance(mod, tvm.IRModule):
        raise TypeError("mod must be a tvm.IRModule")

    config = VTACompilerConfig.from_env(get_env())
    vta_functions = _collect_vta_relay_functions(mod)
    for function in vta_functions:
        _validate_vta_function(function, config)
    if not vta_functions:
        return mod

    outlined = relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols(
        COMPILER_NAME
    )(mod)
    global_functions = _global_vta_relay_functions(outlined)
    global_handles = {function.handle.value for _, function in global_functions}
    remaining_nested = [
        function
        for function in _collect_vta_relay_functions(outlined)
        if function.handle.value not in global_handles
    ]
    if remaining_nested:
        raise ValueError("all nested Compiler='vta' functions must be directly outlineable")

    lowered_functions = [
        (global_var, lower_vta_function(function, config))
        for global_var, function in global_functions
    ]
    for global_var, primfunc in lowered_functions:
        outlined.update_func(global_var, primfunc)
    return outlined
