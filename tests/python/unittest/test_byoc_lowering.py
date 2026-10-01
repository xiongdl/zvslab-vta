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

from dataclasses import replace

import numpy as np
import pytest
import tvm
import vta
from tvm import relay

from byoc_utils import make_qnn_conv2d_module
from vta.relay import partition_for_vta
from vta.relay.contract import VTACompilerConfig
from vta.relay.transform import (
    _composite_calls,
    _has_vta_gemm_tensorization,
    _lower_to_scheduled_te,
    _pack_output_constant,
    _validate_vta_function,
    legalize_vta_function,
    lower_vta_function,
)
from vta.top.vta_conv2d import conv2d_packed


SUPPORTED_LAYOUTS = [
    pytest.param("NCHW", "OIHW", id="nchw-oihw"),
    pytest.param("NHWC", "HWIO", id="nhwc-hwio"),
]
SUPPORTED_KERNELS = [
    pytest.param((1, 1), (0, 0), id="1x1"),
    pytest.param((3, 3), (1, 1), id="3x3"),
]
SUPPORTED_STRIDES = [pytest.param((1, 1), id="stride1"), pytest.param((2, 2), id="stride2")]
ASYMMETRIC_STRIDE2_SPATIAL_CASES = [
    pytest.param(32, 16, id="32-to-16"),
    pytest.param(16, 8, id="16-to-8"),
]
LEGACY_COMPILER_GLOBAL = "relay.ext." + "vta"


def _partitioned_function(bias_kind=None, **overrides):
    mod = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env(), bias_kind=bias_kind, **overrides),
        mod_name="lowering",
    )
    return next(
        function
        for function in mod.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        and function.attrs.get_str("Compiler") == "vta"
    )


def test_validate_vta_function_accepts_approved_partition():
    external = _partitioned_function()

    composite_call = _validate_vta_function(external)

    assert isinstance(composite_call, relay.Call)
    assert composite_call.op.attrs.get_str("Composite") == "vta.qnn_conv2d"


@pytest.mark.parametrize("invalid_func", [None, tvm.IRModule(), relay.var("data")])
def test_validate_vta_function_rejects_wrong_python_type(invalid_func):
    with pytest.raises(TypeError, match="func must be a tvm.relay.Function"):
        _validate_vta_function(invalid_func)


def test_validate_vta_function_rejects_wrong_config_type():
    with pytest.raises(TypeError, match="config must be a VTACompilerConfig or None"):
        _validate_vta_function(_partitioned_function(), config={})


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("Compiler", "cpu", "Compiler='vta'"),
        ("Primitive", 0, "Primitive=1"),
        ("global_symbol", "", "non-empty global_symbol"),
    ],
)
def test_validate_vta_function_rejects_invalid_external_attribute(attribute, value, message):
    external = _partitioned_function().with_attr(attribute, value)

    with pytest.raises(ValueError, match=message):
        _validate_vta_function(external)


def test_validate_vta_function_rejects_missing_external_attributes():
    external = _partitioned_function()
    unannotated = relay.Function(external.params, external.body, external.ret_type)

    with pytest.raises(ValueError, match="Compiler='vta'"):
        _validate_vta_function(unannotated)


def test_validate_vta_function_rejects_unsupported_composite_name():
    external = _partitioned_function()
    composite_call = external.body
    unsupported = composite_call.op.with_attr("Composite", "vta.unsupported")
    malformed = relay.Function(
        external.params,
        relay.Call(unsupported, composite_call.args),
        external.ret_type,
        attrs=external.attrs,
    )

    with pytest.raises(ValueError, match="vta.qnn_conv2d"):
        _validate_vta_function(malformed)


def test_validate_vta_function_rejects_malformed_composite_body():
    external = _partitioned_function()
    composite_call = external.body
    composite = composite_call.op
    malformed_composite = relay.Function(
        composite.params,
        relay.abs(composite.params[0]),
        attrs=composite.attrs,
    )
    malformed = relay.Function(
        external.params,
        relay.Call(malformed_composite, composite_call.args),
        external.ret_type,
        attrs=external.attrs,
    )
    malformed_mod = relay.transform.InferType()(tvm.IRModule.from_expr(malformed))
    malformed = next(iter(malformed_mod.functions.values()))

    with pytest.raises(ValueError, match="does not satisfy the active VTA configuration"):
        _validate_vta_function(malformed)


def test_validate_vta_function_rejects_config_mismatch():
    external = _partitioned_function()
    config = VTACompilerConfig.from_env(vta.get_env())
    incompatible = replace(config, block_in=config.block_in * 2)

    with pytest.raises(ValueError, match="does not satisfy the active VTA configuration"):
        _validate_vta_function(external, incompatible)


def test_importing_lowering_module_does_not_register_external_compiler():
    assert tvm.get_global_func(LEGACY_COMPILER_GLOBAL, allow_missing=True) is None


def _find_operator_calls(expr, operator_name):
    calls = []

    def visit(node):
        if (
            isinstance(node, relay.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == operator_name
        ):
            calls.append(node)

    relay.analysis.post_order_visit(expr, visit)
    return calls


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
def test_legalize_vta_function_packs_full_convolution_matrix(
    data_layout, kernel_layout, kernel_size, padding, strides
):
    env = vta.get_env()
    legalized = legalize_vta_function(
        _partitioned_function(
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
        )
    )
    conv = _find_operator_calls(legalized.body, "nn.conv2d")[0]

    assert str(conv.attrs.data_layout) == f"NCHW{env.BATCH}n{env.BLOCK_IN}c"
    assert str(conv.attrs.kernel_layout) == f"OIHW{env.BLOCK_OUT}o{env.BLOCK_IN}i"
    assert str(conv.attrs.out_layout) == f"NCHW{env.BATCH}n{env.BLOCK_OUT}c"
    assert tuple(int(dim) for dim in conv.args[0].checked_type.shape) == (
        1,
        1,
        8,
        8,
        env.BATCH,
        env.BLOCK_IN,
    )
    assert tuple(int(dim) for dim in conv.args[1].checked_type.shape) == (
        1,
        1,
        *kernel_size,
        env.BLOCK_OUT,
        env.BLOCK_IN,
    )
    assert tuple(int(value) for value in conv.attrs.strides) == strides


def test_legalize_vta_function_keeps_input_and_output_block_factors_distinct():
    config = VTACompilerConfig.from_env(vta.get_env())
    config = replace(config, block_out=config.block_out // 2)

    legalized = legalize_vta_function(_partitioned_function(), config)
    conv = _find_operator_calls(legalized.body, "nn.conv2d")[0]

    assert str(conv.attrs.data_layout) == f"NCHW{config.batch}n{config.block_in}c"
    assert str(conv.attrs.out_layout) == f"NCHW{config.batch}n{config.block_out}c"
    assert tuple(int(dim) for dim in conv.args[1].checked_type.shape)[-2:] == (
        config.block_out,
        config.block_in,
    )


@pytest.mark.parametrize(
    ("data_layout", "kernel_layout", "data_axes", "weight_axes", "unpack_axes"),
    [
        pytest.param(
            "NCHW",
            "OIHW",
            (0, 2, 4, 5, 1, 3),
            (0, 2, 4, 5, 1, 3),
            (0, 4, 1, 5, 2, 3),
            id="nchw-oihw",
        ),
        pytest.param(
            "NHWC",
            "HWIO",
            (0, 4, 2, 3, 1, 5),
            (4, 2, 0, 1, 5, 3),
            (0, 4, 2, 3, 1, 5),
            id="nhwc-hwio",
        ),
    ],
)
def test_legalize_vta_function_uses_layout_specific_pack_permutations(
    data_layout, kernel_layout, data_axes, weight_axes, unpack_axes
):
    legalized = legalize_vta_function(
        _partitioned_function(data_layout=data_layout, kernel_layout=kernel_layout)
    )
    conv = _find_operator_calls(legalized.body, "nn.conv2d")[0]
    data_transpose = conv.args[0]
    weight_transpose = conv.args[1]

    assert data_transpose.op.name == "transpose"
    assert tuple(int(axis) for axis in data_transpose.attrs.axes) == data_axes
    assert data_transpose.args[0].op.name == "reshape"
    assert weight_transpose.op.name == "transpose"
    assert tuple(int(axis) for axis in weight_transpose.attrs.axes) == weight_axes
    assert weight_transpose.args[0].op.name == "reshape"
    assert isinstance(weight_transpose.args[0].args[0], relay.Constant)
    unpack_transpose = legalized.body.args[0]
    assert unpack_transpose.op.name == "transpose"
    assert tuple(int(axis) for axis in unpack_transpose.attrs.axes) == unpack_axes


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
def test_legalize_vta_function_preserves_original_unpacked_external_abi(
    data_layout, kernel_layout, kernel_size, padding, strides
):
    external = _partitioned_function(
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )
    legalized = legalize_vta_function(external)

    assert tvm.ir.structural_equal(external.params[0].checked_type, legalized.params[0].checked_type)
    assert tvm.ir.structural_equal(external.ret_type, legalized.ret_type)
    assert legalized.body.op.name == "reshape"
    unpack_transpose = legalized.body.args[0]
    assert unpack_transpose.op.name == "transpose"
    assert legalized.attrs.get_str("global_symbol") == external.attrs.get_str("global_symbol")


def test_legalize_vta_function_removes_composite_wrapper_deterministically():
    external = _partitioned_function()

    first = legalize_vta_function(external)
    second = legalize_vta_function(external)

    assert _composite_calls(first) == []
    assert tvm.ir.structural_equal(first, second)


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_legalize_vta_function_preserves_supported_fused_forms(bias_kind):
    legalized = legalize_vta_function(_partitioned_function(bias_kind))

    assert len(_find_operator_calls(legalized.body, "nn.conv2d")) == 1
    assert len(_find_operator_calls(legalized.body, "right_shift")) == 1
    assert len(_find_operator_calls(legalized.body, "clip")) == 1
    assert len(_find_operator_calls(legalized.body, "cast")) == 1
    assert _find_operator_calls(legalized.body, "nn.bias_add") == []
    assert len(_find_operator_calls(legalized.body, "add")) == (bias_kind is not None)


@pytest.mark.parametrize("bias_kind", ["bias_add", "add"])
def test_legalize_vta_function_packs_optional_constant_for_broadcast(bias_kind):
    env = vta.get_env()
    legalized = legalize_vta_function(_partitioned_function(bias_kind))
    packed_add = _find_operator_calls(legalized.body, "add")[0]
    packed_constant = packed_add.args[1]

    assert tuple(int(dim) for dim in packed_constant.checked_type.shape) == (
        env.BLOCK_OUT // env.BLOCK_OUT,
        1,
        1,
        env.BATCH,
        env.BLOCK_OUT,
    )
    assert isinstance(packed_constant, relay.Constant)
    np.testing.assert_array_equal(packed_constant.data.numpy(), 1)


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
def test_legalize_vta_function_preserves_scalar_output_constant_exactly(
    data_layout, kernel_layout
):
    external = _partitioned_function(
        "scalar_add",
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )
    legalized = legalize_vta_function(external)
    packed_add = _find_operator_calls(legalized.body, "add")[0]
    packed_constant = packed_add.args[1]

    assert isinstance(packed_constant, relay.Constant)
    assert tuple(int(dim) for dim in packed_constant.checked_type.shape) == ()
    np.testing.assert_array_equal(packed_constant.data.numpy(), 64)

    input_shape = tuple(int(dim) for dim in external.params[0].checked_type.shape)
    input_data = np.arange(np.prod(input_shape), dtype="int8").reshape(input_shape) % 8
    expected = _evaluate_relay_function(external, input_data)
    actual = _evaluate_relay_function(_canonicalize_packed_convolution(legalized), input_data)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 3)])
@pytest.mark.parametrize("output_layout", ["NCHW", "NHWC"])
def test_pack_output_constant_rejects_non_broadcast_shape(shape, output_layout):
    malformed = relay.const(np.ones(shape, dtype=vta.get_env().acc_dtype))

    with pytest.raises(ValueError, match="broadcast"):
        _pack_output_constant(
            malformed,
            output_layout,
            VTACompilerConfig.from_env(vta.get_env()),
        )


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_legalize_vta_function_preserves_quantization_tail_attributes(bias_kind):
    legalized = legalize_vta_function(_partitioned_function(bias_kind))
    shifted = _find_operator_calls(legalized.body, "right_shift")[0]
    clipped = _find_operator_calls(legalized.body, "clip")[0]
    cast = _find_operator_calls(legalized.body, "cast")[0]

    assert int(shifted.args[1].data.numpy().item()) == 1
    assert float(clipped.attrs.a_min) == -128
    assert float(clipped.attrs.a_max) == 127
    assert str(cast.attrs.dtype) == vta.get_env().out_dtype


def _evaluate_relay_function(func, input_data):
    executable = relay.Function(func.params, func.body, func.ret_type)
    mod = relay.transform.InferType()(tvm.IRModule.from_expr(executable))
    executor = relay.create_executor("debug", mod=mod, device=tvm.cpu(), target="llvm")
    return executor.evaluate()(input_data).numpy()


class _CanonicalizePackedConv2D(relay.ExprMutator):
    """Convert the packed conv to NCHW so LLVM can check Relay semantics."""

    def visit_call(self, call):
        args = [self.visit(arg) for arg in call.args]
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.conv2d":
            data = relay.layout_transform(args[0], str(call.attrs.data_layout), "NCHW")
            weight = relay.layout_transform(args[1], str(call.attrs.kernel_layout), "OIHW")
            conv = relay.nn.conv2d(
                data,
                weight,
                strides=call.attrs.strides,
                padding=call.attrs.padding,
                dilation=call.attrs.dilation,
                groups=call.attrs.groups,
                channels=call.attrs.channels,
                kernel_size=call.attrs.kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype=call.attrs.out_dtype,
            )
            return relay.layout_transform(conv, "NCHW", str(call.attrs.out_layout))
        return relay.Call(call.op, args, call.attrs, call.type_args, call.span)


def _canonicalize_packed_convolution(func):
    body = _CanonicalizePackedConv2D().visit(func.body)
    return relay.Function(func.params, body, func.ret_type)


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_legalized_relay_matches_every_supported_composite_exactly(
    data_layout, kernel_layout, kernel_size, padding, strides, bias_kind
):
    external = _partitioned_function(
        bias_kind,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )
    legalized = legalize_vta_function(external)
    input_shape = tuple(int(dim) for dim in external.params[0].checked_type.shape)
    input_data = np.arange(np.prod(input_shape), dtype="int8").reshape(input_shape) % 8

    expected = _evaluate_relay_function(external, input_data)
    actual = _evaluate_relay_function(_canonicalize_packed_convolution(legalized), input_data)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(
    ("input_spatial", "output_spatial"), ASYMMETRIC_STRIDE2_SPATIAL_CASES
)
def test_legalized_asymmetric_stride2_padding_preserves_exact_relay_values(
    data_layout, kernel_layout, input_spatial, output_spatial
):
    env = vta.get_env()
    external = _partitioned_function(
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=(0, 0, 1, 1),
        input_height=input_spatial,
        input_width=input_spatial,
    )
    legalized = legalize_vta_function(external)
    conv = _find_operator_calls(legalized.body, "nn.conv2d")[0]

    assert tuple(int(value) for value in conv.attrs.padding) == (0, 0, 1, 1)
    assert tuple(int(dim) for dim in conv.checked_type.shape) == (
        1,
        1,
        output_spatial,
        output_spatial,
        env.BATCH,
        env.BLOCK_OUT,
    )

    input_shape = tuple(int(dim) for dim in external.params[0].checked_type.shape)
    input_data = ((np.arange(np.prod(input_shape)) % 17) - 8).reshape(input_shape)
    input_data = input_data.astype(env.inp_dtype)
    expected = _evaluate_relay_function(external, input_data)
    actual = _evaluate_relay_function(_canonicalize_packed_convolution(legalized), input_data)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
def test_lower_to_scheduled_te_uses_packed_output_and_vta_target_across_matrix(
    data_layout, kernel_layout, kernel_size, padding, strides
):
    env = vta.get_env()

    cached = _lower_to_scheduled_te(
        _partitioned_function(
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
        )
    )

    assert cached.schedule is not None
    assert len(cached.inputs) == 1
    assert len(cached.outputs) == 1
    spatial = 8 if strides == (1, 1) else 4
    assert tuple(int(dim) for dim in cached.outputs[0].shape) == (
        1,
        1,
        spatial,
        spatial,
        env.BATCH,
        env.BLOCK_OUT,
    )
    assert cached.outputs[0].dtype == env.out_dtype
    assert cached.target.kind.name == "ext_dev"
    assert "vta" in cached.target.keys
    assert cached.target.host.kind.name == "llvm"
    stage_tags = {stage.op.tag for stage in cached.schedule.stages}
    assert "conv2d_dense" in stage_tags
    assert "elemwise" in stage_tags


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
def test_lower_to_scheduled_te_tensorizes_vta_gemm_across_matrix(
    data_layout, kernel_layout, kernel_size, padding, strides
):
    cached = _lower_to_scheduled_te(
        _partitioned_function(
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
        )
    )

    assert _has_vta_gemm_tensorization(cached.schedule)


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(
    ("input_spatial", "output_spatial"), ASYMMETRIC_STRIDE2_SPATIAL_CASES
)
def test_lower_to_scheduled_te_honors_distinct_padding_before_and_after(
    data_layout, kernel_layout, input_spatial, output_spatial
):
    env = vta.get_env()
    cached = _lower_to_scheduled_te(
        _partitioned_function(
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=(0, 0, 1, 1),
            input_height=input_spatial,
            input_width=input_spatial,
        )
    )

    assert tuple(int(dim) for dim in cached.outputs[0].shape) == (
        1,
        1,
        output_spatial,
        output_spatial,
        env.BATCH,
        env.BLOCK_OUT,
    )
    pad_stages = [stage for stage in cached.schedule.stages if stage.op.name == "pad_data"]
    assert len(pad_stages) == 1
    assert tuple(int(dim) for dim in pad_stages[0].op.output(0).shape) == (
        1,
        1,
        input_spatial + 1,
        input_spatial + 1,
        env.BATCH,
        env.BLOCK_IN,
    )
    assert _has_vta_gemm_tensorization(cached.schedule)


@pytest.mark.parametrize("padding", [(), (0,), (0, 0, 1), (0, 0, 1, 1, 0)])
def test_packed_conv2d_rejects_malformed_padding_geometry(padding):
    env = vta.get_env()
    data = tvm.te.placeholder(
        (1, 1, 16, 16, env.BATCH, env.BLOCK_IN),
        dtype=env.inp_dtype,
        name="data",
    )
    weight = tvm.te.placeholder(
        (1, 1, 3, 3, env.BLOCK_OUT, env.BLOCK_IN),
        dtype=env.wgt_dtype,
        name="weight",
    )

    with tvm.target.Target("vta", host=env.target_host):
        with pytest.raises(ValueError, match="padding.*two or four"):
            conv2d_packed(
                data,
                weight,
                (2, 2),
                padding,
                (1, 1),
                f"NCHW{env.BATCH}n{env.BLOCK_IN}c",
                env.acc_dtype,
            )


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
def test_scalar_output_constant_lowers_to_tensorized_vta_primfunc(
    data_layout, kernel_layout
):
    external = _partitioned_function(
        "scalar_add",
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )

    cached = _lower_to_scheduled_te(external)
    primfunc = lower_vta_function(external)

    assert _has_vta_gemm_tensorization(cached.schedule)
    assert isinstance(primfunc, tvm.tir.PrimFunc)
    assert tvm.tir.analysis.verify_well_formed(primfunc)
    assert str(primfunc.attrs["global_symbol"]) == external.attrs.get_str("global_symbol")


def test_unscheduled_te_graph_has_no_vta_gemm_tensorization():
    cached = _lower_to_scheduled_te(_partitioned_function())
    unscheduled = tvm.te.create_schedule(cached.outputs[0].op)

    assert not _has_vta_gemm_tensorization(unscheduled)


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
def test_lower_vta_function_returns_metadata_preserving_primfunc_across_matrix(
    data_layout, kernel_layout, kernel_size, padding, strides
):
    external = _partitioned_function(
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )

    primfunc = lower_vta_function(external)

    assert isinstance(primfunc, tvm.tir.PrimFunc)
    assert tvm.tir.analysis.verify_well_formed(primfunc)
    assert str(primfunc.attrs["global_symbol"]) == external.attrs.get_str("global_symbol")
    assert primfunc.attrs["target"].kind.name == "ext_dev"
    assert "vta" in primfunc.attrs["target"].keys
    assert primfunc.attrs["target"].host.kind.name == "llvm"
    assert primfunc.attrs["relay_attrs"].get_str("Compiler") == "vta"
    buffers = [primfunc.buffer_map[param] for param in primfunc.params]
    assert len(buffers) == 2
    assert tuple(int(dim) for dim in buffers[0].shape) == tuple(
        int(dim) for dim in external.params[0].checked_type.shape
    )
    assert tuple(int(dim) for dim in buffers[1].shape) == tuple(
        int(dim) for dim in external.ret_type.shape
    )


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_lower_vta_function_is_structurally_deterministic(bias_kind):
    external = _partitioned_function(bias_kind)

    first = lower_vta_function(external)
    second = lower_vta_function(external)

    assert tvm.ir.structural_equal(first, second, map_free_vars=True)


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_lower_vta_function_preserves_vta_intrinsic_and_coprocessor_structure(bias_kind):
    primfunc = lower_vta_function(_partitioned_function(bias_kind))
    call_ops = set()
    attr_keys = set()

    def visit(node):
        if isinstance(node, tvm.tir.Call):
            call_ops.add(str(node.op))
        if isinstance(node, tvm.tir.AttrStmt):
            attr_keys.add(str(node.attr_key))

    tvm.tir.stmt_functor.post_order_visit(primfunc.body, visit)

    assert "Op(tir.vta.uop_push)" in call_ops
    assert "Op(tir.vta.coproc_sync)" in call_ops
    assert "Op(tir.vta.coproc_dep_push)" in call_ops
    assert "Op(tir.vta.coproc_dep_pop)" in call_ops
    assert {"coproc_scope", "coproc_uop_scope"}.issubset(attr_keys)
