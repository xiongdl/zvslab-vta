/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package vta.core

import chisel3._
import chisel3.util._
import vta.util.config._

import ISA._

/** MemDecode.
 *
 * Decode memory instructions with a Bundle. This is similar to an union,
 * therefore order matters when declaring fields. These are the instructions
 * decoded with this bundle:
 *   - LUOP
 *   - LWGT
 *   - LINP
 *   - LACC
 *   - SOUT
 */
class MemDecode extends Bundle {
  val xpad_1 = UInt(M_PAD_BITS.W)
  val xpad_0 = UInt(M_PAD_BITS.W)
  val ypad_1 = UInt(M_PAD_BITS.W)
  val ypad_0 = UInt(M_PAD_BITS.W)
  val xstride = UInt(M_STRIDE_BITS.W)
  val xsize = UInt(M_SIZE_BITS.W)
  val ysize = UInt(M_SIZE_BITS.W)
  val empty_0 = UInt(InstructionLayout.memMidPaddingBits.W)
  val dram_offset = UInt(M_DRAM_OFFSET_BITS.W)
  val sram_offset = UInt(M_SRAM_OFFSET_BITS.W)
  val id = UInt(M_ID_BITS.W)
  val push_next = Bool()
  val push_prev = Bool()
  val pop_next = Bool()
  val pop_prev = Bool()
  val op = UInt(OP_BITS.W)
}

/** GemmDecode.
 *
 * Decode GEMM instruction with a Bundle. This is similar to an union,
 * therefore order matters when declaring fields.
 */
class GemmDecode(implicit p: Parameters) extends Bundle {
  val empty_1 = UInt(InstructionLayout.gemmHighPaddingBits(p).W)
  val wgt_1 = UInt(InstructionLayout.wgtIndexBits(p).W)
  val wgt_0 = UInt(InstructionLayout.wgtIndexBits(p).W)
  val inp_1 = UInt(InstructionLayout.inpIndexBits(p).W)
  val inp_0 = UInt(InstructionLayout.inpIndexBits(p).W)
  val acc_1 = UInt(InstructionLayout.accIndexBits(p).W)
  val acc_0 = UInt(InstructionLayout.accIndexBits(p).W)
  val empty_0 = UInt(InstructionLayout.midPaddingBits(p).W)
  val lp_1 = UInt(C_ITER_BITS.W)
  val lp_0 = UInt(C_ITER_BITS.W)
  val uop_end = UInt(InstructionLayout.uopEndBits(p).W)
  val uop_begin = UInt(InstructionLayout.uopIndexBits(p).W)
  val reset = Bool()
  val push_next = Bool()
  val push_prev = Bool()
  val pop_next = Bool()
  val pop_prev = Bool()
  val op = UInt(OP_BITS.W)
}

/** AluDecode.
 *
 * Decode ALU instructions with a Bundle. This is similar to an union,
 * therefore order matters when declaring fields. These are the instructions
 * decoded with this bundle:
 *   - VMIN
 *   - VMAX
 *   - VADD
 *   - VSHX
 */
class AluDecode(implicit p: Parameters) extends Bundle {
  val empty_1 = UInt(InstructionLayout.aluHighPaddingBits(p).W)
  val alu_imm = UInt(C_ALU_IMM_BITS.W)
  val alu_use_imm = Bool()
  val alu_op = UInt(C_ALU_OP_BITS.W)
  val src_1 = UInt(InstructionLayout.accIndexBits(p).W)
  val src_0 = UInt(InstructionLayout.accIndexBits(p).W)
  val dst_1 = UInt(InstructionLayout.accIndexBits(p).W)
  val dst_0 = UInt(InstructionLayout.accIndexBits(p).W)
  val empty_0 = UInt(InstructionLayout.midPaddingBits(p).W)
  val lp_1 = UInt(C_ITER_BITS.W)
  val lp_0 = UInt(C_ITER_BITS.W)
  val uop_end = UInt(InstructionLayout.uopEndBits(p).W)
  val uop_begin = UInt(InstructionLayout.uopIndexBits(p).W)
  val reset = Bool()
  val push_next = Bool()
  val push_prev = Bool()
  val pop_next = Bool()
  val pop_prev = Bool()
  val op = UInt(OP_BITS.W)
}

/** UopDecode.
 *
 * Decode micro-ops (uops).
 */
class UopDecode(implicit p: Parameters) extends Bundle {
  val empty = UInt(InstructionLayout.uopHighPaddingBits(p).W)
  val u2 = UInt(InstructionLayout.uopWgtBits(p).W)
  val u1 = UInt(InstructionLayout.uopSrcBits(p).W)
  val u0 = UInt(InstructionLayout.uopDstBits(p).W)
}

/** FetchDecode.
 *
 * Partial decoding for dispatching instructions to Load, Compute, and Store.
 */
class FetchDecode(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val isLoad = Output(Bool())
    val isCompute = Output(Bool())
    val isStore = Output(Bool())
  })
  // Dispatch is based only on the ISA opcode and subtype fields. The rest of
  // the instruction contains payloads consumed by the downstream decoders.
  val mem = io.inst.asTypeOf(new MemDecode)
  val alu = io.inst.asTypeOf(new AluDecode)
  val taskOpcode = mem.op
  val memId = mem.id
  val aluId = alu.alu_op

  val isLoadOp = taskOpcode === OP_L
  val isStoreOp = taskOpcode === OP_S
  val isGemmOp = taskOpcode === OP_G
  val isFinishOp = taskOpcode === OP_F
  val isAluOp = taskOpcode === OP_A

  val isInputOrWeight = memId === M_ID_I || memId === M_ID_W
  val isUopOrAccumulator = memId === M_ID_U || memId === M_ID_A || memId === M_ID_A_8BIT
  val isOutput = memId === M_ID_O
  val isStoreSync = memId === M_ID_U && mem.xsize === 0.U
  val isSupportedAlu = aluId < ALU_OP_NUM.U

  io.isLoad := isLoadOp && isInputOrWeight
  io.isCompute := (isLoadOp && isUopOrAccumulator) || isGemmOp || isFinishOp ||
    (isAluOp && isSupportedAlu)
  io.isStore := isStoreOp && (isOutput || isStoreSync)
}

/** LoadDecode.
 *
 * Decode dependencies, type and sync for Load module.
 */
class LoadDecode(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val push_next = Output(Bool())
    val pop_next = Output(Bool())
    val isInput = Output(Bool())
    val isWeight = Output(Bool())
    val isSync = Output(Bool())
  })
  val dec = io.inst.asTypeOf(new MemDecode)
  io.push_next := dec.push_next
  io.pop_next := dec.pop_next
  io.isInput := dec.op === OP_L && dec.id === M_ID_I && dec.xsize =/= 0.U
  io.isWeight := dec.op === OP_L && dec.id === M_ID_W && dec.xsize =/= 0.U
  io.isSync := dec.op === OP_L && (dec.id === M_ID_I || dec.id === M_ID_W) && dec.xsize === 0.U
}

/** ComputeDecode.
 *
 * Decode dependencies, type and sync for Compute module.
 */
class ComputeDecode(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val push_next = Output(Bool())
    val push_prev = Output(Bool())
    val pop_next = Output(Bool())
    val pop_prev = Output(Bool())
    val isLoadAcc = Output(Bool())
    val isLoadUop = Output(Bool())
    val isSync = Output(Bool())
    val isAlu = Output(Bool())
    val isGemm = Output(Bool())
    val isFinish = Output(Bool())
  })
  val dec = io.inst.asTypeOf(new MemDecode)
  val alu = io.inst.asTypeOf(new AluDecode)
  io.push_next := dec.push_next
  io.push_prev := dec.push_prev
  io.pop_next := dec.pop_next
  io.pop_prev := dec.pop_prev
  io.isLoadAcc := dec.op === OP_L &&
    (dec.id === M_ID_A || dec.id === M_ID_A_8BIT) && dec.xsize =/= 0.U
  io.isLoadUop := dec.op === OP_L && dec.id === M_ID_U && dec.xsize =/= 0.U
  io.isSync := dec.op === OP_L &&
    (dec.id === M_ID_A || dec.id === M_ID_A_8BIT || dec.id === M_ID_U) && dec.xsize === 0.U
  io.isAlu := dec.op === OP_A && alu.alu_op < ALU_OP_NUM.U
  io.isGemm := dec.op === OP_G
  io.isFinish := dec.op === OP_F
}

/** StoreDecode.
 *
 * Decode dependencies, type and sync for Store module.
 */
class StoreDecode(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(INST_BITS.W))
    val push_prev = Output(Bool())
    val pop_prev = Output(Bool())
    val isStore = Output(Bool())
    val isSync = Output(Bool())
  })
  val dec = io.inst.asTypeOf(new MemDecode)
  io.push_prev := dec.push_prev
  io.pop_prev := dec.pop_prev
  io.isStore := dec.op === OP_S && dec.id === M_ID_O && dec.xsize =/= 0.U
  io.isSync := dec.op === OP_S && dec.xsize === 0.U &&
    (dec.id === M_ID_O || dec.id === M_ID_U)
}
