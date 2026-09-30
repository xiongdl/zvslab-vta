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

import pytest
import numpy as np
import tvm
import vta
from tvm import relay
from tvm.relay.op.contrib import get_pattern_table

from byoc_utils import (
    make_adjacent_qnn_conv2d_module,
    make_qnn_conv2d_module,
    make_qnn_conv2d_near_miss_module,
)
from vta.relay.contract import VTACompilerConfig
from vta.relay.patterns import QNN_CONV2D_COMPOSITE, check_qnn_conv2d, pattern_table
from vta.relay.partition import _partition_pipeline, partition_for_vta


SUPPORTED_LAYOUTS = [
    pytest.param("NCHW", "OIHW", id="nchw-oihw"),
    pytest.param("NHWC", "HWIO", id="nhwc-hwio"),
]
SUPPORTED_KERNELS = [
    pytest.param((1, 1), (0, 0), id="1x1"),
    pytest.param((3, 3), (1, 1), id="3x3"),
]
SUPPORTED_STRIDES = [pytest.param((1, 1), id="stride1"), pytest.param((2, 2), id="stride2")]


def _merge_composites(mod):
    return relay.transform.MergeComposite(pattern_table())(mod)


def _composite_functions(expr):
    functions = []

    def visit(node):
        if isinstance(node, relay.Function) and node.attrs is not None:
            if node.attrs.get_str("Composite") == QNN_CONV2D_COMPOSITE:
                functions.append(node)

    relay.analysis.post_order_visit(expr, visit)
    return functions


def _root_call(mod):
    return mod["main"].body.args[0]


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


@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_qnn_conv2d_pattern_matches_approved_forms(bias_kind):
    mod = make_qnn_conv2d_module(vta.get_env(), bias_kind=bias_kind)

    merged = _merge_composites(mod)

    assert len(_composite_functions(merged["main"].body)) == 1


def test_qnn_conv2d_pattern_keeps_host_operations_outside_composite():
    mod = make_qnn_conv2d_module(vta.get_env())

    merged = _merge_composites(mod)
    composite = _composite_functions(merged["main"].body)[0]

    assert isinstance(merged["main"].body.op, tvm.ir.Op)
    assert merged["main"].body.op.name == "transpose"
    assert isinstance(composite.body, relay.Call)
    assert composite.body.op.name == "cast"
    assert "abs" not in composite.astext(show_meta_data=False)
    assert "transpose" not in composite.astext(show_meta_data=False)


def test_pattern_module_import_does_not_register_global_table():
    assert get_pattern_table("vta") is None


def test_qnn_conv2d_predicate_rejects_non_constant_weight():
    mod, _ = make_qnn_conv2d_near_miss_module(vta.get_env())

    assert not check_qnn_conv2d(_root_call(mod))


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_dtype": "int16"},
        {"weight_dtype": "int16"},
        {"accumulator_dtype": "int16"},
        {"output_dtype": "int16"},
    ],
)
def test_qnn_conv2d_predicate_rejects_wrong_dtype(overrides):
    mod = make_qnn_conv2d_module(vta.get_env(), **overrides)

    assert not check_qnn_conv2d(
        _root_call(mod), VTACompilerConfig.from_env(vta.get_env())
    )


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
@pytest.mark.parametrize("bias_kind", [None, "bias_add", "add"])
def test_qnn_conv2d_predicate_accepts_full_approved_hardware_matrix(
    data_layout, kernel_layout, kernel_size, padding, strides, bias_kind
):
    mod = make_qnn_conv2d_module(
        vta.get_env(),
        bias_kind=bias_kind,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_layout=data_layout,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )

    assert check_qnn_conv2d(_root_call(mod))


@pytest.mark.parametrize(
    "overrides",
    [
        {"kernel_size": (2, 2), "padding": (0, 0)},
        {"strides": (3, 3)},
        {"strides": (1, 2)},
        {"dilation": (2, 2)},
        {"shift": -1},
        {"shift": 32},
        {"clip_bounds": (-129, 127)},
        {"clip_bounds": (-128, 128)},
        {"input_channels": vta.get_env().BLOCK_IN // 2},
        {"output_channels": vta.get_env().BLOCK_OUT // 2},
        {"data_layout": "NCHW", "kernel_layout": "HWIO"},
        {"data_layout": "NHWC", "kernel_layout": "OIHW"},
        {"data_layout": "NHWC", "kernel_layout": "HWIO", "out_layout": "NCHW"},
        {"groups": 2},
    ],
)
def test_qnn_conv2d_predicate_rejects_unsupported_hardware_boundary(overrides):
    mod = make_qnn_conv2d_module(vta.get_env(), **overrides)

    assert not check_qnn_conv2d(_root_call(mod))


def test_qnn_conv2d_predicate_rejects_dynamic_shape():
    mod = make_qnn_conv2d_module(vta.get_env(), input_height=tvm.tir.Any())

    assert not check_qnn_conv2d(_root_call(mod))


@pytest.mark.parametrize(
    ("kernel_size", "padding"),
    [
        pytest.param((1, 1), (1, 1), id="padded-1x1"),
        pytest.param((3, 3), (0, 0), id="valid-3x3"),
        pytest.param((3, 3), (0, 0, 1, 1), id="asymmetric-3x3"),
    ],
)
def test_qnn_conv2d_predicate_accepts_any_statically_valid_padding(kernel_size, padding):
    mod = make_qnn_conv2d_module(
        vta.get_env(), kernel_size=kernel_size, padding=padding
    )

    assert check_qnn_conv2d(_root_call(mod))


def test_qnn_conv2d_predicate_rejects_batch_not_divisible_by_config():
    mod = make_qnn_conv2d_module(vta.get_env())
    config = VTACompilerConfig.from_env(vta.get_env())

    assert not check_qnn_conv2d(_root_call(mod), replace(config, batch=config.batch + 1))


def _vta_functions(mod):
    return [
        function
        for function in mod.functions.values()
        if isinstance(function, relay.Function)
        and function.attrs is not None
        and "Compiler" in function.attrs
        and function.attrs.get_str("Compiler") == "vta"
    ]


def test_partition_for_vta_returns_typed_module_with_vta_region():
    mod = make_qnn_conv2d_module(vta.get_env())

    partitioned = partition_for_vta(mod)

    assert isinstance(partitioned, tvm.IRModule)
    assert partitioned["main"].checked_type is not None
    assert len(_vta_functions(partitioned)) == 1


def test_partition_for_vta_uses_approved_pass_order():
    pipeline = _partition_pipeline(VTACompilerConfig.from_env(vta.get_env()), "default")

    assert [compiler_pass.info.name for compiler_pass in pipeline.passes] == [
        "InferType",
        "MergeComposite",
        "AnnotateTarget",
        "sequential",
        "InferType",
    ]


def test_partition_for_vta_binds_weight_parameter_before_matching():
    env = vta.get_env()
    mod, _ = make_qnn_conv2d_near_miss_module(env)
    weight_shape = (env.BLOCK_OUT, env.BLOCK_IN, 3, 3)

    partitioned = partition_for_vta(
        mod,
        params={"weight": tvm.nd.array(np.ones(weight_shape, dtype=env.wgt_dtype))},
    )

    assert len(_vta_functions(partitioned)) == 1


@pytest.mark.parametrize("invalid_mod", [None, relay.var("data")])
def test_partition_for_vta_rejects_non_module_input(invalid_mod):
    with pytest.raises(TypeError, match="mod must be a tvm.IRModule"):
        partition_for_vta(invalid_mod)


@pytest.mark.parametrize("invalid_params", [[], "weight", 1])
def test_partition_for_vta_rejects_non_mapping_params(invalid_params):
    with pytest.raises(TypeError, match="params must be a mapping or None"):
        partition_for_vta(tvm.IRModule(), params=invalid_params)


@pytest.mark.parametrize("invalid_name", [None, "", "bad\nname", "bad\x00name"])
def test_partition_for_vta_rejects_invalid_module_name(invalid_name):
    with pytest.raises(ValueError, match="mod_name must be a non-empty string without control"):
        partition_for_vta(tvm.IRModule(), mod_name=invalid_name)


def test_partitioned_vta_function_has_required_attributes_and_types():
    partitioned = partition_for_vta(
        make_qnn_conv2d_module(vta.get_env()), mod_name="fixture"
    )
    external = _vta_functions(partitioned)[0]

    assert int(external.attrs.Primitive) == 1
    assert int(external.attrs.Inline) == 1
    assert external.attrs.get_str("global_symbol") == "tvmgen_fixture_vta_main_0"
    assert external.params[0].checked_type.dtype == vta.get_env().inp_dtype
    assert external.ret_type.dtype == vta.get_env().out_dtype


def test_partition_keeps_host_operations_outside_vta_function():
    partitioned = partition_for_vta(make_qnn_conv2d_module(vta.get_env()))
    external = _vta_functions(partitioned)[0]

    assert _operator_names(partitioned["main"].body) == ["abs", "transpose"]
    assert "abs" not in external.astext(show_meta_data=False)
    assert "transpose" not in external.astext(show_meta_data=False)


def test_partition_owns_convolution_weight_constant():
    partitioned = partition_for_vta(make_qnn_conv2d_module(vta.get_env()))
    external = _vta_functions(partitioned)[0]
    conv = _find_call(external.body, "nn.conv2d")

    assert isinstance(conv.args[1], relay.Constant)


def test_adjacent_candidates_become_separate_single_composite_vta_functions():
    partitioned = partition_for_vta(
        make_adjacent_qnn_conv2d_module(vta.get_env()), mod_name="adjacent"
    )
    external_functions = _vta_functions(partitioned)

    assert len(external_functions) == 2
    assert sorted(function.attrs.get_str("global_symbol") for function in external_functions) == [
        "tvmgen_adjacent_vta_main_0",
        "tvmgen_adjacent_vta_main_1",
    ]
    assert all(len(_composite_functions(function.body)) == 1 for function in external_functions)


def test_partition_symbol_is_deterministic():
    first = partition_for_vta(make_qnn_conv2d_module(vta.get_env()), mod_name="fixture")
    second = partition_for_vta(make_qnn_conv2d_module(vta.get_env()), mod_name="fixture")

    assert tvm.ir.structural_equal(first, second)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"kernel_size": (2, 2), "padding": (0, 0)},
        {"strides": (3, 3)},
        {"input_channels": vta.get_env().BLOCK_IN // 2},
        {"output_channels": vta.get_env().BLOCK_OUT // 2},
        {"data_layout": "NCHW", "kernel_layout": "HWIO"},
        {"data_layout": "NHWC", "kernel_layout": "OIHW"},
    ],
)
def test_near_miss_remains_typed_on_host(overrides):
    if overrides:
        near_miss = make_qnn_conv2d_module(vta.get_env(), **overrides)
    else:
        near_miss, _ = make_qnn_conv2d_near_miss_module(vta.get_env())

    partitioned = partition_for_vta(near_miss)

    assert _vta_functions(partitioned) == []
    assert partitioned["main"].checked_type is not None


@pytest.mark.parametrize(("data_layout", "kernel_layout"), SUPPORTED_LAYOUTS)
@pytest.mark.parametrize(("kernel_size", "padding"), SUPPORTED_KERNELS)
@pytest.mark.parametrize("strides", SUPPORTED_STRIDES)
def test_partition_is_deterministic_across_the_supported_matrix(
    data_layout, kernel_layout, kernel_size, padding, strides
):
    mod = make_qnn_conv2d_module(
        vta.get_env(),
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
    )

    first = partition_for_vta(mod, mod_name="matrix")
    second = partition_for_vta(mod, mod_name="matrix")

    assert tvm.ir.structural_equal(first, second)
    assert len(_vta_functions(first)) == 1


def test_partition_for_vta_is_idempotent():
    once = partition_for_vta(make_qnn_conv2d_module(vta.get_env()))

    twice = partition_for_vta(once)

    assert tvm.ir.structural_equal(once, twice)
