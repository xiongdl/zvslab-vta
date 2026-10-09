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
"""Native DwC encoding, lowering, and real-driver numerical execution."""
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
        assert call.args[3].args[1].same_as(inp.data)
        assert call.args[4].args[1].same_as(weight.data)
        assert [int(call.args[index].args[4]) for index in (3, 4)] == [1, 1]
        module = tvm.IRModule({"main": tvm.tir.PrimFunc(
            [buffer.data for buffer in intrin.buffers], intrin.body
        )})
        from vta.transform import LowerDWCAddresses
        call = _calls(LowerDWCAddresses()(module)["main"].body, "tir.vta.uop_push")[0]
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


def _run_address_lowering(env, module):
    """Use the registered DwC address pass before generic vector conversion."""
    lowered = tvm.tir.transform.StorageRewrite()(module)
    with vta.build_config() as context:
        for _, transform in context.config["tir.add_lower_pass"]:
            if transform.info.name == "tir.vta.LowerDWCAddresses":
                lowered = transform(lowered)
    return tvm.tir.transform.LowerDeviceStorageAccessInfo()(lowered)


def _allocated_dwc_probe(env, keep_live):
    from vta.transform import dwc_uop_push

    inp_lanes = env.BATCH * max(env.BLOCK_IN, env.BLOCK_OUT)
    wgt_lanes = env.BLOCK_IN * env.BLOCK_OUT
    a = tvm.tir.decl_buffer((inp_lanes,), "int8", name="a", scope=env.inp_scope)
    b = tvm.tir.decl_buffer((inp_lanes,), "int8", name="b", scope=env.inp_scope)
    w = tvm.tir.decl_buffer((wgt_lanes,), "int8", name="w", scope=env.wgt_scope)
    z = tvm.tir.decl_buffer((wgt_lanes,), "int8", name="z", scope=env.wgt_scope)
    out = tvm.tir.decl_buffer((env.BATCH * env.BLOCK_OUT,), "int32", scope=env.acc_scope)
    nodes = [
        tvm.tir.Evaluate(tvm.tir.call_extern(
            "int32", "load_" + buf.name, buf.access_ptr("w", "int32")
        )) for buf in [a, b, w, z]
    ]
    if not keep_live:
        nodes.append(tvm.tir.Evaluate(dwc_uop_push(env, out, a, w)))
    # Selecting a reverse subblock must retain both its allocation base and +8.
    second_input = tvm.tir.decl_buffer(
        b.shape, b.dtype, data=b.data, scope=env.inp_scope,
        elem_offset=8 if env.BLOCK_IN > env.BLOCK_OUT else 0,
    )
    nodes.append(tvm.tir.Evaluate(dwc_uop_push(env, out, second_input, z)))
    if keep_live:
        nodes.append(tvm.tir.Evaluate(tvm.tir.call_extern(
            "int32", "keep_both_live",
            *[buf.access_ptr("r", "int32") for buf in [a, b, w, z]],
        )))
    body = tvm.tir.SeqStmt(nodes)
    for buf in reversed([a, b, w, z, out]):
        body = tvm.tir.Allocate(
            buf.data, buf.dtype, buf.shape, tvm.tir.const(True, "bool"), body
        )
    return tvm.IRModule({"main": tvm.tir.PrimFunc([], body)})


@pytest.mark.parametrize("block_in,block_out", [(3, 3), (3, 4), (4, 3)])
@pytest.mark.parametrize("keep_live", [False, True], ids=["dwc_read_lifetimes", "nonzero_bases"])
def test_dwc_storage_rewrite_preserves_bases_and_read_lifetimes(block_in, block_out, keep_live):
    env = _intrinsic_env(block_in, block_out)
    with env:
        lowered = _run_address_lowering(env, _allocated_dwc_probe(env, keep_live))["main"].body
        externs = _calls(lowered, "tir.call_extern")
        loads = {
            call.args[0].value: int(call.args[1]) for call in externs
            if call.args[0].value.startswith("load_")
        }
        assert loads == {
            "load_a": 0, "load_b": 2 if block_out == 4 else 1,
            "load_w": 0, "load_z": 1,
        }
        uops = _calls(lowered, "tir.vta.uop_push")
        reads = [[int(call.args[3]), int(call.args[4])] for call in uops]
        second_src = 3 if block_in == 4 else (2 if block_out == 4 else 1)
        assert reads == ([[second_src, 1]] if keep_live else [[0, 0], [second_src, 1]])


@pytest.fixture(scope="module", params=[(0, 3, 3), (0, 4, 3), (1, 3, 3), (0, 3, 4)],
                ids=["production_8x8", "independent_16x8", "independent_batch2", "independent_8x16"])
def fsim_probe(request, tmp_path_factory):
    """Compile matching real FSIM geometry; default uses the production library.

    Independent builds override C macros until asymmetric production configs
    are supported. These are numerical driver tests, not TOP/RTL acceptance.
    """
    root = Path(__file__).resolve().parents[3]
    tvm_root = Path(os.environ["TVM_PATH"])
    geometry = request.param
    build_dir = tmp_path_factory.mktemp("dwc_fsim")
    cfg = [sys.executable, str(root / "config/vta_config.py"),
           "--use-cfg=" + str(root / "config/vta_64mac.json")]
    flags = shlex.split(subprocess.check_output(cfg + ["--backend-contract", "--defs"], text=True))
    overrides = dict(zip(["VTA_LOG_BATCH", "VTA_LOG_BLOCK_IN", "VTA_LOG_BLOCK_OUT"], geometry))
    flags = [flag for flag in flags if flag.split("=")[0][2:] not in overrides]
    flags += ["-D" + key + "=" + str(value) for key, value in overrides.items()]
    binary = build_dir / "probe"
    command = [
        os.environ.get("CXX", "c++"), "-std=c++17", *flags,
        "-DDMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>",
        "-I" + str(root / "include"), "-I" + str(tvm_root / "include"),
        "-I" + str(tvm_root / "3rdparty/dlpack/include"),
        "-I" + str(tvm_root / "3rdparty/dmlc-core/include"),
        str(Path(__file__).with_name("dwc_fsim_probe.cc")),
    ]
    if geometry == (0, 3, 3):
        command += ["-L" + str(root / "build"), "-lvta_fsim",
                    "-Wl,-rpath," + str(root / "build")]
    else:
        command += [str(root / "src/sim/sim_driver.cc"),
                    str(root / "src/sim/sim_tlpp.cc"),
                    str(root / "src/vmem/virtual_memory.cc")]
    command += ["-L" + str(tvm_root / "build"), "-ltvm",
                "-Wl,-rpath," + str(tvm_root / "build"), "-o", str(binary)]
    subprocess.run(command, check=True)
    return binary, tuple(1 << value for value in geometry)


@pytest.mark.parametrize("gemm", [False, True], ids=["dwc", "gemm_regression"])
def test_fsim_signed_nine_taps_restart_reset_and_batch_sharing(fsim_probe, gemm):
    # Detect missing dispatch, unsigned weights, stale/extra consumption,
    # batch-dependent shifts, lost BI16 input halves, and changed GEMM behavior.
    import json

    binary, (batch, block_in, block_out) = fsim_probe
    result = subprocess.run([str(binary), str(int(gemm))],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert f"GEOMETRY {batch} {block_in} {block_out}" in result.stdout
    rows = [[int(x) for x in line.split()[1:]] for line in result.stdout.splitlines()
            if line.startswith("RESULT ")]
    assert len(rows) == 2 * 4 * batch * block_out
    for snapshot, pos, b, c, actual in rows:
        total = 0
        for tap in range(9):
            input_channels = range(block_in) if gemm else [
                c + (block_out if block_in > block_out and pos % 2 else 0)]
            for channel in input_channels:
                inp = 1 + (pos * 11 + tap * 7 + b * 13 + channel * 3) % 31
                if (tap + channel + b) % 2:
                    inp = -inp
                phase = channel if gemm else tap % block_in
                weight_tap = tap // block_in * block_in + phase
                weight = 0
                if weight_tap < 9:
                    weight = 1 + (pos // 2 * 5 + c * 3 + weight_tap * 2) % 13
                    if (weight_tap + c + pos // 2) % 2:
                        weight = -weight
                total += inp * weight
        expected = 2 * total if snapshot else total + 101 + pos * 17 + b * 5 + c
        assert actual == expected, (snapshot, pos, b, c, actual, expected)
    profile = json.loads(result.stdout.split("PROFILE ", 1)[1].split("GEOMETRY", 1)[0])
    assert profile["gemm_counter"] == (108 if gemm else 0)
    assert profile["dwc_counter"] == (0 if gemm else 108)
    assert profile["alu_counter"] == 32
    assert profile["out_store_nbytes"] == 8 * 4 * batch * block_out
