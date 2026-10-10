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
"""Native depthwise convolution on BI-packed activation and weight buffers."""
import tvm
from tvm import te, topi, autotvm
from ..environment import get_env
from ..intrin import dwc


def dwc_kernel(weight, kernel_size):
    """Attach the actual (KH, KW) to Task3's packed four-dimensional weight.

    Entry count includes padding and cannot determine the kernel dimensions.
    This identity carries metadata; the external buffer remains [CB,entries,BO,BI].
    """
    (kh, kw) = map(int, kernel_size)
    env = get_env()
    shape = topi.utils.get_const_tuple(weight.shape)
    entry_shape = ((kh * kw + env.BLOCK_IN - 1) // env.BLOCK_IN,
                   env.BLOCK_OUT, env.BLOCK_IN)
    if kh <= 0 or kw <= 0 or len(shape) != 4 or shape[1:] != entry_shape:
        raise ValueError('invalid packed DwC kernel shape or kernel_size')
    return te.compute(weight.shape, lambda *i: weight[i], name='dwc_kernel', attrs={'kernel_size': [kh, kw]})


@autotvm.register_topi_compute('depthwise_conv2d_packed.vta')
def depthwise_conv2d_packed(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Depth multiplier one; data is [N/B,H,W,C/BI,B,BI]."""
    env = get_env()
    if env.BATCH != 1:
        raise ValueError('the default DwC schedule supports BATCH=1')
    shape = topi.utils.get_const_tuple(data.shape)
    if len(shape) != 6 or shape[-2:] != (env.BATCH, env.BLOCK_IN):
        raise ValueError('expected BI-packed input')
    if tuple(dilation) != (1, 1):
        raise ValueError('DwC supports dilation one')
    if not kernel.op.attrs or 'kernel_size' not in kernel.op.attrs:
        raise ValueError('wrap packed weights with dwc_kernel(weight, (KH,KW))')
    (kh, kw) = map(int, kernel.op.attrs['kernel_size'])
    (sh, sw) = map(int, strides)
    if sh <= 0 or sw <= 0:
        raise ValueError('strides must be positive')
    padding = tuple(map(int, padding))
    if len(padding) == 2:
        (pt, pl) = padding
        (pb, pr) = padding
    elif len(padding) == 4:
        (pt, pl, pb, pr) = padding
    else:
        raise ValueError('padding must contain two or four integers')
    if min(pt, pl, pb, pr) < 0:
        raise ValueError('padding must be nonnegative')
    if out_dtype != env.acc_dtype:
        raise ValueError('DwC accumulation requires int32')
    channels = shape[3] * env.BLOCK_IN
    if channels != int(kernel.shape[0]) * env.BLOCK_OUT:
        raise ValueError('input and kernel channel counts differ')
    if kh * (shape[2] + pl + pr) * channels * env.BATCH * env.INP_WIDTH // 8 > env.INP_BUFF_SIZE:
        raise ValueError('DwC row tile exceeds input SRAM')
    weight_bytes = (int(kernel.shape[0]) * int(kernel.shape[1])
                    * env.BLOCK_OUT * env.BLOCK_IN * env.WGT_WIDTH // 8)
    if weight_bytes > env.WGT_BUFF_SIZE:
        raise ValueError('packed DwC weights exceed weight SRAM')
    # Flatten W and CB so spatial padding is expressed in DMA vector units.
    flat = te.compute(
        (shape[0], shape[1], shape[2] * shape[3], env.BATCH, env.BLOCK_IN),
        lambda n, h, x, b, i: data[n, h, x // shape[3], x % shape[3], b, i],
        name='dwc_flat',
    )
    padded = flat
    if any(padding):
        padded = topi.nn.pad(
            flat, [0, pt, pl * shape[3], 0, 0],
            [0, pb, pr * shape[3], 0, 0], name='dwc_pad',
        )
    # This logical BO view is inlined into the physical BI storage.
    view = te.compute(
        (shape[0], shape[1] + pt + pb, shape[2] + pl + pr,
         channels // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT),
        lambda n, h, w, c, b, i: padded[
            n, h, w * shape[3] + (c * env.BLOCK_OUT + i) // env.BLOCK_IN,
            b, (c * env.BLOCK_OUT + i) % env.BLOCK_IN,
        ],
        name='dwc_view',
    )
    source_weight = kernel.op.input_tensors[0]

    def load_weight(ins, outs):
        irb = tvm.tir.ir_builder.create()
        irb.scope_attr(env.dev.vta_axis, 'coproc_scope', env.dev.get_task_qid(env.dev.QID_LOAD_WGT))
        entries = int(source_weight.shape[0]) * int(source_weight.shape[1])
        irb.emit(tvm.tir.call_extern(
            'int32', 'VTALoadBuffer2D', env.dev.command_handle, ins[0].data,
            0, entries, 1, entries, 0, 0, 0, 0,
            outs[0].access_ptr('r', 'int32'), env.dev.MEM_ID_WGT,
        ))
        return irb.get()
    # An opaque full-buffer stage prevents region inference from trimming
    # the unused tail of the last BI entry, which DMA must transfer in full.
    weight = te.extern(
        source_weight.shape, [source_weight], load_weight,
        name='dwc_weight_load', dtype=source_weight.dtype,
        out_buffers=[tvm.tir.decl_buffer(
            source_weight.shape, source_weight.dtype, name='dwc_weight_sram',
            # Keep the TE default alignment for buffer binding. Physical SRAM
            # allocations retain WGT_ELEM_BITS alignment through MemoryInfo.
            scope=env.wgt_scope,
        )],
    )
    tap = te.reduce_axis((0, kh * kw), 'kernel_tap')
    unit = te.reduce_axis((0, 1), 'tap_unit')
    oh = (shape[1] + pt + pb - kh) // sh + 1
    ow = (shape[2] + pl + pr - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError('DwC kernel exceeds padded input')
    res = te.compute(
        (shape[0], oh, ow, channels // env.BLOCK_OUT, env.BATCH, env.BLOCK_OUT),
        lambda n, h, w, c, b, i: te.sum(
            view[n, h * sh + tap // kw, w * sw + tap % kw, c, b, i].astype(out_dtype)
            * weight[c, tap // env.BLOCK_IN, i,
                     tap % env.BLOCK_IN + unit].astype(out_dtype),
            axis=[tap, unit],
        ),
        name='dwc_acc', tag='dwc_packed',
    )
    return te.compute(res.shape, lambda *i: res[i], name='dwc_output', tag='elemwise')


@autotvm.register_topi_schedule('depthwise_conv2d_packed.vta')
def schedule_depthwise_conv2d_packed(cfg, outs):
    """Use complete input rows and one output/channel block, without tuning.

    Build with ``vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"})``.
    The runtime uop callback ABI accepts one captured int; width/channel fusion
    gives that scalar, while disabling CSE prevents extra captured temporaries.
    Reverse geometry unrolls output channel blocks because legacy TVM's
    tensorization matcher cannot simplify their fused subvector offsets.
    """
    env = get_env()
    output = outs[0]
    s = te.create_schedule(output.op)
    ewise = []

    def find(tensor):
        if tensor.op.tag == 'dwc_packed':
            return tensor
        if tensor is not output:
            ewise.append(tensor)
        return find(tensor.op.input_tensors[0])
    acc = find(output)
    alu_stages = []
    for tensor in ewise:
        if isinstance(tensor.op.body[0], tvm.tir.ProducerLoad):
            s[tensor].compute_inline()
        else:
            s[tensor].set_scope(env.acc_scope)
            alu_stages.append(tensor)
    (view, weight) = acc.op.input_tensors
    padded = view.op.input_tensors[0]
    s[view].compute_inline()
    if padded.op.name == 'dwc_pad':
        s[padded.op.input_tensors[0]].compute_inline()
    if isinstance(padded.op, te.ComputeOp):
        inp = padded
        s[inp].set_scope(env.inp_scope)
    else:
        inp = s.cache_read(padded, env.inp_scope, [view])
    (n, h, w, c, b, i) = s[output].op.axis
    if env.BLOCK_IN > env.BLOCK_OUT:
        s[output].unroll(c)
        position = c
    else:
        position = s[output].fuse(w, c)
    for tensor in alu_stages:
        s[tensor].compute_at(s[output], position)
        s[tensor].pragma(s[tensor].op.axis[0], env.alu)
    s[acc].set_scope(env.acc_scope)
    s[acc].compute_at(s[output], position)
    s[inp].compute_at(s[output], h)
    s[weight].set_scope(env.wgt_scope)
    s[inp].pragma(s[inp].op.axis[0], env.dma_copy)
    axes = s[acc].op.axis
    (tap, unit) = s[acc].op.reduce_axis
    s[acc].reorder(*axes[:-2], tap, *axes[-2:], unit)
    s[acc].unroll(tap)
    s[acc].tensorize(axes[-2], dwc(env, packed=True))
    s[output].pragma(b, env.dma_copy)
    return s
