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

class FetchDecodeTest extends AnyFlatSpec with ChiselScalatestTester {
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
  private val memIdOutput = 4

  private val aluMin = 0
  private val aluMax = 1
  private val aluAdd = 2
  private val aluShift = 3

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

  it should "dispatch legal instructions with non-zero payload fields" in {
    test(new FetchDecode) { c =>
      val instructions = Seq(
        instruction(taskLoad, (memIdInput, 7, 3), (0x1234, 80, 16)),
        instruction(taskLoad, (memIdWeight, 7, 3), (0x2345, 80, 16)),
        instruction(taskLoad, (memIdUop, 7, 3), (0x3456, 80, 16)),
        instruction(taskLoad, (memIdAcc, 7, 3), (0x4567, 80, 16)),
        instruction(taskStore, (memIdOutput, 7, 3), (0x5678, 80, 16)),
        instruction(taskGemm, (0x1a2b, 16, 16), (0x345, 96, 11)),
        instruction(taskFinish, (0x2b3c, 16, 16), (0x456, 96, 11)),
        instruction(taskAlu, (aluMin, 108, 3), (0x1234, 80, 16)),
        instruction(taskAlu, (aluMax, 108, 3), (0x2345, 80, 16)),
        instruction(taskAlu, (aluAdd, 108, 3), (0x3456, 80, 16)),
        instruction(taskAlu, (aluShift, 108, 3), (0x4567, 80, 16))
      )

      instructions.take(2).foreach { inst =>
        expectRoute(c, inst, load = true, compute = false, store = false)
      }
      instructions.slice(2, 4).foreach { inst =>
        expectRoute(c, inst, load = false, compute = true, store = false)
      }
      expectRoute(c, instructions(4), load = false, compute = false, store = true)
      instructions.drop(5).foreach { inst =>
        expectRoute(c, inst, load = false, compute = true, store = false)
      }
    }
  }

  it should "reject unsupported opcode and memory subtype encodings" in {
    test(new FetchDecode) { c =>
      expectRoute(c, instruction(7, (0x1234, 80, 16)), load = false, compute = false, store = false)
      expectRoute(c, instruction(taskLoad, (5, 7, 3), (0x2345, 80, 16)), load = false, compute = false, store = false)
      expectRoute(c, instruction(taskLoad, (7, 7, 3), (0x3456, 80, 16)), load = false, compute = false, store = false)
      expectRoute(c, instruction(taskAlu, (4, 108, 3), (0x4567, 80, 16)), load = false, compute = false, store = false)
    }
  }
}
