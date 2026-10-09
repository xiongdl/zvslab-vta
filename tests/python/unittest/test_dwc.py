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
"""Native DwC encoding and lowering; numerical FSIM tests are added separately."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import tvm
import vta
from vta.environment import Environment


@pytest.fixture(scope="module")
def runtime_probe(tmp_path_factory):
    root = Path(__file__).resolve().parents[3]
    tvm_root = Path(os.environ["TVM_PATH"])
    build_dir = tmp_path_factory.mktemp("dwc_runtime")
    abi_header = build_dir / "abi_config.h"
    config = os.environ.get("VTA_CONFIG_FILE", str(root / "config/vta_64mac.json"))
    cfg = [sys.executable, str(root / "config/vta_config.py"), "--use-cfg=" + config]
    subprocess.run(cfg + ["--abi-header=" + str(abi_header)], check=True)
    flags = shlex.split(
        subprocess.check_output(cfg + ["--backend-contract", "--defs"], text=True)
    )
    binary = build_dir / "probe"
    command = [
        os.environ.get("CXX", "c++"), "-std=c++17", *flags,
        "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>", "-include", str(abi_header),
        "-I" + str(root / "include"), "-I" + str(tvm_root / "include"),
        "-I" + str(tvm_root / "3rdparty/dlpack/include"),
        "-I" + str(tvm_root / "3rdparty/dmlc-core/include"),
        str(root / "src/runtime/runtime.cc"), str(root / "src/sim/sim_tlpp.cc"),
        str(Path(__file__).with_name("dwc_runtime_probe.cc")),
        "-L" + str(tvm_root / "build"), "-ltvm", "-Wl,-rpath," + str(tvm_root / "build"),
        "-o", str(binary),
    ]
    subprocess.run(command, check=True)
    return binary


@pytest.mark.parametrize("mode,opcode", [(0, 2), (1, 4), (2, 5)])
@pytest.mark.parametrize("serial", [False, True])
def test_runtime_compute_encoding_dependencies_and_reset(runtime_probe, mode, opcode, serial):
    result = subprocess.run(
        [str(runtime_probe), str(mode), str(int(serial))],
        check=True, capture_output=True, text=True, timeout=15,
    )
    rows = [
        [int(x) for x in line.split()[1:]]
        for line in result.stdout.splitlines() if line.startswith("RESULT ")
    ]
    assert [row[0] for row in rows] == [opcode, opcode]
    assert [row[1] for row in rows] == [0, 1]
    expected_factors = (
        [1, 3, 4, 11, 13, 17, 19, 23, 29] if mode != 1
        else [1, 3, 4, 11, 13, 0, 19, 23, 0]
    )
    assert [row[2:11] for row in rows] == [expected_factors] * 2
    if not serial:
        assert rows[0][13:15] == [1, 1]
    if mode == 2:
        assert "DWC" in result.stdout


def test_dwc_runtime_keeps_nine_taps_in_one_uop_range(runtime_probe):
    result = subprocess.run(
        [str(runtime_probe), "3", "0"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    row = [
        int(x) for line in result.stdout.splitlines()
        if line.startswith("RESULT ") for x in line.split()[1:]
    ]
    assert row[:5] == [5, 0, 9, 1, 1]


def _calls(stmt, name):
    found = []

    def visit(node):
        if (
            isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op)
            and node.op.name == name
        ):
            found.append(node)

    tvm.tir.stmt_functor.post_order_visit(stmt, visit)
    return found


def _intrinsic_env(block_in, block_out):
    """Project the specified geometry until Task3 adds asymmetric config parsing.

    PkgConfig currently replaces LOG_BLOCK_IN/OUT with LOG_BLOCK. These tests
    exercise the real intrinsic/lowering consumers, rather than claiming that
    production config loading or a mismatched simulator can execute this shape.
    """
    env = Environment(dict(vta.get_env().cfg_dict))
    env.LOG_BLOCK_IN, env.LOG_BLOCK_OUT = block_in, block_out
    env.BLOCK_IN, env.BLOCK_OUT = 1 << block_in, 1 << block_out
    env.INP_ELEM_BITS = env.BATCH * env.BLOCK_IN * env.INP_WIDTH
    env.WGT_ELEM_BITS = env.BLOCK_OUT * env.BLOCK_IN * env.WGT_WIDTH
    env.ACC_ELEM_BITS = env.BATCH * env.BLOCK_OUT * env.ACC_WIDTH
    env.OUT_ELEM_BITS = env.BATCH * env.BLOCK_OUT * env.OUT_WIDTH
    for prefix in ("INP", "WGT", "ACC", "OUT"):
        setattr(env, prefix + "_ELEM_BYTES", getattr(env, prefix + "_ELEM_BITS") // 8)
    return env


@pytest.mark.parametrize("block_in,block_out", [(3, 3), (3, 4), (4, 3)])
def test_dwc_intrinsic_uses_compute_mode_and_preserves_input_subblocks(block_in, block_out):
    env = _intrinsic_env(block_in, block_out)
    assert (env.BLOCK_IN, env.BLOCK_OUT) == (1 << block_in, 1 << block_out)
    with env:
        intrin = env.dwc
        assert tuple(int(x) for x in intrin.op.input_tensors[0].shape) == (env.BATCH, env.BLOCK_OUT)
        assert tuple(int(x) for x in intrin.op.input_tensors[1].shape) == (env.BLOCK_OUT, 1)
        for stmt, reset in [(intrin.body, 0), (intrin.reduce_init, 1), (intrin.reduce_update, 0)]:
            calls = _calls(stmt, "tir.vta.uop_push")
            assert len(calls) == 1
            assert [int(x) for x in calls[0].args[:2]] == [2, reset]
            queue_ids = []

            def visit_scope(node):
                if isinstance(node, tvm.tir.AttrStmt) and node.attr_key == "coproc_scope":
                    queue_ids.append(int(node.value))

            tvm.tir.stmt_functor.post_order_visit(stmt, visit_scope)
            assert queue_ids == [2]
        call = _calls(intrin.body, "tir.vta.uop_push")[0]
        inp, weight = intrin.buffers[:2]
        # Reverse geometry's second 8-channel subblock must remain addressable.
        addr = tvm.tir.stmt_functor.substitute(call.args[3], {inp.elem_offset: 8})
        assert int(tvm.arith.Analyzer().simplify(addr)) == 1
        # Kernel phase varies within one packed entry without changing its address.
        addr = tvm.tir.stmt_functor.substitute(
            call.args[4], {weight.elem_offset: env.BLOCK_IN - 1}
        )
        assert int(tvm.arith.Analyzer().simplify(addr)) == 0
        assert [int(x) for x in weight.strides] == [env.BLOCK_IN, 1]
        for stmt in [env.mock.dwc.body, env.mock.dwc.reduce_init, env.mock.dwc.reduce_update]:
            assert not _calls(stmt, "tir.vta.uop_push")


@pytest.mark.parametrize("block_in,block_out", [(3, 3), (3, 4), (4, 3)])
def test_dwc_tensorization_lowers_nine_real_kernel_taps(block_in, block_out):
    from tvm import te

    env = _intrinsic_env(block_in, block_out)
    assert (env.BLOCK_IN, env.BLOCK_OUT) == (1 << block_in, 1 << block_out)
    with env:
        data = te.placeholder((9, env.BATCH, env.BLOCK_OUT), env.inp_dtype, name="data")
        weight = te.placeholder(
            ((9 + env.BLOCK_IN - 1) // env.BLOCK_IN, env.BLOCK_OUT, env.BLOCK_IN),
            env.wgt_dtype, name="weight",
        )
        # Bind already-loaded SRAM windows: physical DMA packing is Task3/5.
        inp, wgt = data, weight
        ko = te.reduce_axis((0, 9), name="ko")
        ki = te.reduce_axis((0, 1), name="ki")
        acc = te.compute(
            (env.BATCH, env.BLOCK_OUT),
            lambda b, c: te.sum(
                inp[ko, b, c].astype(env.acc_dtype)
                * wgt[ko // env.BLOCK_IN, c, ko % env.BLOCK_IN + ki].astype(env.acc_dtype),
                axis=[ko, ki],
            ), name="acc",
        )
        out = te.compute(acc.shape, lambda *i: acc(*i).astype(env.out_dtype), name="out")
        schedule = te.create_schedule(out.op)
        schedule[acc].set_scope(env.acc_scope)
        schedule[out].pragma(schedule[out].op.axis[0], env.dma_copy)
        schedule[acc].reorder(ko, *acc.op.axis, ki)
        schedule[acc].unroll(ko)
        schedule[acc].tensorize(acc.op.axis[0], env.dwc)
        binds = {
            data: tvm.tir.decl_buffer(data.shape, data.dtype, scope=env.inp_scope),
            weight: tvm.tir.decl_buffer(
                weight.shape, weight.dtype, scope=env.wgt_scope,
                data_alignment=env.BLOCK_OUT * env.BLOCK_IN,
            ),
        }
        lowered = vta.lower(schedule, [data, weight, out], binds=binds)
        calls = _calls(lowered["main"].body, "tir.vta.uop_push")
        assert len(calls) == 10  # One reset plus nine actual kernel positions.
        assert [int(call.args[0]) for call in calls] == [2] * 10
        assert sum(int(call.args[1]) == 1 for call in calls) == 1
        updates = [call for call in calls if int(call.args[1]) == 0]
        analyzer = tvm.arith.Analyzer()
        expected_inputs = (
            [0, 2, 4, 6, 8, 10, 12, 14, 16] if block_out == 4
            else [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )
        expected_weights = (
            [0, 0, 0, 0, 0, 0, 0, 0, 1] if block_in == 3
            else [0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        assert [int(analyzer.simplify(call.args[3])) for call in updates] == expected_inputs
        assert [int(analyzer.simplify(call.args[4])) for call in updates] == expected_weights
        vta.build(
            schedule, [data, weight, out],
            tvm.target.Target("ext_dev", host=env.target_host),
            binds=binds,
        )


@pytest.mark.parametrize("mode", [4, 5])
def test_existing_modes_still_reject_adjacent_accumulator_writes(runtime_probe, mode):
    result = subprocess.run(
        [str(runtime_probe), str(mode), "0"], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "seq_[i].dst_idx != dst_index" in result.stderr
