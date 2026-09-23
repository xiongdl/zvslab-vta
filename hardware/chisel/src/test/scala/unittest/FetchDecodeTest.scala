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

package unittest

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import vta.core._
import vta.util.config._

class AluDecodeProbe(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(128.W))
    val opcode = Output(UInt(3.W))
    val immediate = Output(UInt(16.W))
  })
  val dec = io.inst.asTypeOf(new AluDecode)
  io.opcode := dec.alu_op
  io.immediate := dec.alu_imm
}

class UopDecodeProbe(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(32.W))
    val u0 = Output(UInt(10.W))
    val u1 = Output(UInt(10.W))
    val u2 = Output(UInt(8.W))
  })
  val dec = io.inst.asTypeOf(new UopDecode)
  io.u0 := dec.u0
  io.u1 := dec.u1
  io.u2 := dec.u2
}

class FetchDecodeTest extends AnyFlatSpec with ChiselScalatestTester {
  private implicit val p: Parameters = new GeometryCoreConfig
  private val instBits = 128
  private val taskLoad = 0
  private val taskStore = 1
  private val taskGemm = 2
  private val taskFinish = 3
  private val taskAlu = 4

  private val memIdUop = 0
  private val memIdWeight = 1
  private val memIdInput = 2
  private val memIdAcc = 3
  private val memIdAcc8Bit = 5
  private val memIdOutput = 4

  private val aluMin = 0
  private val aluMax = 1
  private val aluAdd = 2
  private val aluShift = 3
  private val aluMul = 4
  // VTAAluInsn.alu_opcode follows the C ABI layout for vta_64mac.json.
  private val aluOpcodeLsb = 104

  private def withField(inst: BigInt, value: Int, lsb: Int, width: Int): BigInt = {
    val mask = (BigInt(1) << width) - 1
    inst | ((BigInt(value) & mask) << lsb)
  }

  private def instruction(task: Int, fields: (Int, Int, Int)*): BigInt =
    fields.foldLeft(BigInt(task)) { case (inst, (value, lsb, width)) =>
      withField(inst, value, lsb, width)
    }

  private def expectRoute(
    c: FetchDecode,
    inst: BigInt,
    load: Boolean,
    compute: Boolean,
    store: Boolean): Unit = {
    c.io.inst.poke(inst.U(instBits.W))
    c.clock.step()
    c.io.isLoad.expect(load.B)
    c.io.isCompute.expect(compute.B)
    c.io.isStore.expect(store.B)
  }

  behavior of "FetchDecode"

  it should "decode ALU fields at their geometry-derived C ABI positions" in {
    test(new AluDecodeProbe) { c =>
      val inst = instruction(taskAlu, (aluAdd, aluOpcodeLsb, 3), (0x3456, 108, 16))
      c.io.inst.poke(inst.U(instBits.W))
      c.io.opcode.expect(aluAdd.U)
      c.io.immediate.expect(0x3456.U)
    }
  }

  it should "decode uop fields at their geometry-derived C ABI positions" in {
    test(new UopDecodeProbe) { c =>
      val inst = (0x5a << 20) | (0x2aa << 10) | 0x155
      c.io.inst.poke(inst.U(32.W))
      c.io.u0.expect(0x155.U)
      c.io.u1.expect(0x2aa.U)
      c.io.u2.expect(0x5a.U)
    }
  }

  it should "dispatch legal instructions with non-zero payload fields" in {
    test(new FetchDecode) { c =>
      val instructions = Seq(
        instruction(taskLoad, (memIdInput, 7, 3), (0x1234, 80, 16)),
        instruction(taskLoad, (memIdWeight, 7, 3), (0x2345, 80, 16)),
        instruction(taskLoad, (memIdUop, 7, 3), (0x3456, 80, 16)),
        instruction(taskLoad, (memIdAcc, 7, 3), (0x4567, 80, 16)),
        instruction(taskLoad, (memIdAcc8Bit, 7, 3), (0x4a5b, 80, 16)),
        instruction(taskStore, (memIdOutput, 7, 3), (0x5678, 80, 16)),
        instruction(taskGemm, (0x1a2b, 16, 16), (0x345, 96, 11)),
        instruction(taskFinish, (0x2b3c, 16, 16), (0x456, 96, 11)),
        instruction(taskAlu, (aluMin, aluOpcodeLsb, 3), (0x1234, 108, 16)),
        instruction(taskAlu, (aluMax, aluOpcodeLsb, 3), (0x2345, 108, 16)),
        instruction(taskAlu, (aluAdd, aluOpcodeLsb, 3), (0x3456, 108, 16)),
        instruction(taskAlu, (aluShift, aluOpcodeLsb, 3), (0x4567, 108, 16)),
        instruction(taskAlu, (aluMul, aluOpcodeLsb, 3), (0x5678, 108, 16))
      )

      instructions.take(2).foreach { inst =>
        expectRoute(c, inst, load = true, compute = false, store = false)
      }
      instructions.slice(2, 5).foreach { inst =>
        expectRoute(c, inst, load = false, compute = true, store = false)
      }
      expectRoute(c, instructions(5), load = false, compute = false, store = true)
      instructions.drop(6).foreach { inst =>
        expectRoute(c, inst, load = false, compute = true, store = false)
      }
      expectRoute(c, instruction(taskStore, (memIdUop, 7, 3), (0, 80, 16)),
        load = false, compute = false, store = true)
    }
  }

  it should "reject unsupported opcode and memory subtype encodings" in {
    test(new FetchDecode) { c =>
      expectRoute(c, instruction(7, (0x1234, 80, 16)), load = false, compute = false, store = false)
      expectRoute(c, instruction(taskLoad, (6, 7, 3), (0x2345, 80, 16)), load = false, compute = false, store = false)
      expectRoute(c, instruction(taskLoad, (7, 7, 3), (0x3456, 80, 16)), load = false, compute = false, store = false)
      expectRoute(c, instruction(taskStore, (memIdWeight, 7, 3), (0, 80, 16)),
        load = false, compute = false, store = false)
      expectRoute(c,
        instruction(taskAlu, (5, aluOpcodeLsb, 3), (0x4567, 108, 16)),
        load = false, compute = false, store = false)
    }
  }
}
