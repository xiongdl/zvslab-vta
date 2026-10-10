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
"""Depthwise TOP contracts against production VTA lowering."""
import pytest
import tvm
from tvm import te
import vta
from vta.environment import Environment


@pytest.mark.parametrize('bi,bo', [(3, 3), (3, 4), (4, 3)])
def test_default_depthwise_lowering(bi, bo):
    from vta.top import depthwise_conv2d_packed, schedule_depthwise_conv2d_packed, dwc_kernel
    cfg = dict(vta.get_env().cfg_dict)
    cfg.update(LOG_BLOCK_IN=bi, LOG_BLOCK_OUT=bo)
    env = Environment(cfg)
    with env:
        data = te.placeholder((1, 25, 5, 64 // env.BLOCK_IN, env.BATCH, env.BLOCK_IN), 'int8', 'data')
        weight = te.placeholder((64 // env.BLOCK_OUT, (9 + env.BLOCK_IN - 1) // env.BLOCK_IN, env.BLOCK_OUT, env.BLOCK_IN), 'int8', 'weight')
        kernel = dwc_kernel(weight, (3, 3))
        with tvm.target.Target('ext_dev'):
            out = depthwise_conv2d_packed(data, kernel, (1, 1), (1, 1), (1, 1), 'int32')
            result = te.compute(out.shape, lambda *i: out[i].astype('int8'), name='result', tag='elemwise')
            schedule = schedule_depthwise_conv2d_packed([result])
        module = vta.lower(schedule, [data, weight, result])
        with vta.build_config(disabled_pass={'tir.CommonSubexprElimTIR'}):
            vta.build(schedule, [data, weight, result], tvm.target.Target('ext_dev', host='llvm'))
        updates = []

        def visit(node):
            if isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and (node.op.name == 'tir.vta.uop_push') and (int(node.args[0]) == 2) and (int(node.args[1]) == 0):
                updates.append(node)
        tvm.tir.stmt_functor.post_order_visit(module['main'].body, visit)
        assert len(updates) == (72 if bi > bo else 9)
        ranges = []

        def scopes(node):
            if isinstance(node, tvm.tir.AttrStmt) and node.attr_key == 'coproc_uop_scope':
                taps = []

                def calls(value):
                    if isinstance(value, tvm.tir.Call) and isinstance(value.op, tvm.ir.Op) and (value.op.name == 'tir.vta.uop_push') and (int(value.args[0]) == 2) and (int(value.args[1]) == 0):
                        taps.append(value)
                tvm.tir.stmt_functor.post_order_visit(node.body, calls)
                if taps:
                    ranges.append(taps)
        tvm.tir.stmt_functor.post_order_visit(module['main'].body, scopes)
        assert ranges and all(len(taps) == 9 for taps in ranges)
        def at_origin(expr):
            variables = []
            tvm.tir.stmt_functor.post_order_visit(
                expr, lambda node: variables.append(node) if isinstance(node, tvm.tir.Var) else None
            )
            return int(tvm.arith.Analyzer().simplify(
                tvm.tir.stmt_functor.substitute(expr, {var: tvm.tir.const(0, var.dtype) for var in variables})
            ))
        offsets = [row * 7 * 8 + col * 8 for row in range(3) for col in range(3)]
        for channel, taps in enumerate(ranges):
            sources = [at_origin(call.args[3]) for call in taps]
            assert sources == [offset + (channel if bi > bo else 0) for offset in offsets]
            weights = [at_origin(call.args[4]) for call in taps]
            assert [value - weights[0] for value in weights] == [tap // env.BLOCK_IN for tap in range(9)]


@pytest.mark.parametrize('stride,padding', [((1, 1), (1, 1)), ((2, 1), (1, 0, 0, 1)), ((1, 1), (0, 0))])
def test_fsim_depthwise_signed_int32(stride, padding):
    import numpy as np
    from vta.top import depthwise_conv2d_packed, schedule_depthwise_conv2d_packed, dwc_kernel
    from vta.top.dwc_layout import pack_dwc_input, pack_dwc_weight
    from vta.testing import simulator
    env = vta.get_env()
    simulator.load_backend('fsim')
    rng = np.random.default_rng(17)
    x = rng.integers(-31, 32, (1, 16, 5, 4), dtype=np.int8)
    k = rng.integers(-12, 13, (16, 1, 3, 3), dtype=np.int8)
    xp = pack_dwc_input(x, env).data
    kp = pack_dwc_weight(k[:, 0], env)
    data = te.placeholder(xp.shape, 'int8', 'data')
    weight = te.placeholder(kp.shape, 'int8', 'weight')
    snapshots = []
    for byte in range(4):
        with tvm.target.Target('ext_dev'):
            acc = depthwise_conv2d_packed(data, dwc_kernel(weight, (3, 3)), stride, padding, (1, 1), 'int32')
            shifted = te.compute(acc.shape, lambda *i: acc[i] >> 8 * byte, name='shifted', tag='elemwise')
            out = te.compute(acc.shape, lambda *i: shifted[i].astype('int8'), name='result', tag='elemwise')
            schedule = schedule_depthwise_conv2d_packed([out])
        with vta.build_config(disabled_pass={'tir.CommonSubexprElimTIR'}):
            mod = vta.build(schedule, [data, weight, out], tvm.target.Target('ext_dev', host='llvm'), name='dwc')
        dev = tvm.ext_dev(0)
        result = tvm.nd.empty(tuple((int(z) for z in out.shape)), 'int8', dev)
        simulator.clear_stats()
        mod(tvm.nd.array(xp, dev), tvm.nd.array(kp, dev), result)
        profile = simulator.stats()
        assert profile['dwc_counter'] == 9 * int(acc.shape[1]) * int(acc.shape[2]) * int(acc.shape[3])
        assert profile['gemm_counter'] == 0
        snapshots.append(result.numpy().view('uint8').astype('uint32'))
    actual = sum((snapshot << 8 * byte for (byte, snapshot) in enumerate(snapshots))).astype('uint32').view('int32')
    (pt, pl, pb, pr) = padding if len(padding) == 4 else (*padding, *padding)
    padded = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    reference = np.zeros(tuple((int(z) for z in acc.shape)), dtype=np.int32)
    for h in range(reference.shape[1]):
        for w in range(reference.shape[2]):
            for c in range(16):
                reference[0, h, w, c // env.BLOCK_OUT, 0, c % env.BLOCK_OUT] = sum((int(padded[0, c, h * stride[0] + kh, w * stride[1] + kw]) * int(k[c, 0, kh, kw]) for kh in range(3) for kw in range(3)))
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize('source,dest', [(1, 3), (3, 1)])
def test_dma_queue_bridge_uses_balanced_compute_tokens(source, dest):
    from vta.transform import BridgeDMAQueueDependencies
    body = tvm.tir.SeqStmt([tvm.tir.Evaluate(tvm.tir.call_intrin('int32', f'tir.vta.coproc_dep_{kind}', source, dest)) for kind in ['push', 'pop']])
    module = BridgeDMAQueueDependencies()(tvm.IRModule({'main': tvm.tir.PrimFunc([], body)}))
    calls = []

    def visit(node):
        if isinstance(node, tvm.tir.Call):
            calls.append((node.op.name, tuple(map(int, node.args))))
    tvm.tir.stmt_functor.post_order_visit(module['main'].body, visit)
    assert calls == [('tir.vta.coproc_dep_push', (source, 2)), ('tir.vta.coproc_dep_pop', (source, 2)), ('tir.vta.coproc_dep_push', (2, dest)), ('tir.vta.coproc_dep_pop', (2, dest))]


def test_real_kws_layer_signed_int32():
    """Same real operands on the currently built geometry/backend, all 8000 outputs."""
    import importlib
    import json
    import os
    import sys
    from pathlib import Path
    import numpy as np
    from vta.top import depthwise_conv2d_packed, schedule_depthwise_conv2d_packed, dwc_kernel
    from vta.top.dwc_layout import pack_dwc_input, pack_dwc_weight
    from vta.testing import simulator
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'apps/mlperf_tiny_benchmark'))
    sample = importlib.import_module('keyword_spotting_v1.python.dwc_sample').extract_dwc_sample()
    env = vta.get_env()
    backend = os.environ['VTA_BACKEND']
    simulator.load_backend(backend)
    assert (env.BLOCK_IN, env.BLOCK_OUT) in [(8, 8), (8, 16), (16, 8)]
    x = sample.activation.transpose(0, 3, 1, 2)
    k = sample.weight[:, :, :, 0].transpose(2, 0, 1)
    # Every physical lane retains its actual channel; neither expanded upper
    # channels nor reverse lower/upper subblocks are replaced with zeros.
    assert np.all(np.any(x != 0, axis=(0, 2, 3)))
    assert np.all(np.any(k != 0, axis=(1, 2)))
    xp, kp = pack_dwc_input(x, env).data, pack_dwc_weight(k, env)
    data, weight = te.placeholder(xp.shape, 'int8', 'data'), te.placeholder(kp.shape, 'int8', 'weight')
    dev = tvm.ext_dev(0)
    snapshots, profiles = [], []
    for byte in range(4):
        with tvm.target.Target('ext_dev'):
            acc = depthwise_conv2d_packed(data, dwc_kernel(weight, (3, 3)), sample.strides, sample.padding, (1, 1), 'int32')
            shifted = te.compute(acc.shape, lambda *i: acc[i] >> (8 * byte), name='shifted', tag='elemwise')
            out = te.compute(acc.shape, lambda *i: shifted[i].astype('int8'), name='result', tag='elemwise')
            schedule = schedule_depthwise_conv2d_packed([out])
        with vta.build_config(debug_flag=env.DEBUG_DUMP_INSN, disabled_pass={'tir.CommonSubexprElimTIR'}):
            mod = vta.build(schedule, [data, weight, out], tvm.target.Target('ext_dev', host='llvm'), name='kws_dwc')
        result = tvm.nd.empty(tuple(map(int, out.shape)), 'int8', dev)
        simulator.clear_stats(backend)
        mod(tvm.nd.array(xp, dev), tvm.nd.array(kp, dev), result)
        profile = simulator.stats(backend)
        if backend == 'fsim':
            assert profile['dwc_counter'] == 9 * 25 * 5 * (64 // env.BLOCK_OUT)
            assert profile['gemm_counter'] == 0
        else:
            assert profile['cycle_count'] > 0
        profiles.append(profile)
        snapshots.append(result.numpy().view('uint8').astype('uint32'))
    actual = sum(snapshot << (8 * byte) for byte, snapshot in enumerate(snapshots)).astype('uint32').view('int32')
    logical = actual.reshape(1, 25, 5, 64)
    np.testing.assert_array_equal(logical, sample.reference)
    artifact_dir = os.environ.get('DWC_ARTIFACTS')
    if artifact_dir:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(artifact_dir) / 'sample-and-result.npz', activation=sample.activation, weight=sample.weight, reference=sample.reference, actual=logical, packed_input=xp, packed_weight=kp)
        import hashlib
        import shutil
        root = Path(__file__).resolve().parents[3]
        sources = [*root.glob('src/**/*.cc'), *root.glob('include/**/*.h'), *root.glob('hardware/chisel/src/main/scala/**/*.scala'), *root.glob('python/vta/**/*.py')]
        fingerprints = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}
        Path(artifact_dir, 'source-fingerprints.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
        if backend == 'tsim':
            for filename in ('Test.DefaultTsimConfig.sv', 'vta_geometry.properties'):
                shutil.copy2(root / 'build/chisel' / filename, Path(artifact_dir) / filename)
    # Flush native instruction dumps before the single-line JSON record.
    import ctypes
    ctypes.CDLL(None).fflush(None)
    print('KWS_ACCEPTANCE ' + json.dumps({'backend': backend, 'geometry': [env.BLOCK_IN, env.BLOCK_OUT], 'hashes': sample.hashes, 'strides': sample.strides, 'padding': sample.padding, 'profiles': profiles, 'outputs': int(logical.size), 'range': [int(logical.min()), int(logical.max())]}, sort_keys=True), flush=True)
