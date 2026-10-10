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
import vta.DefaultTsimConfig
import vta.core._
import vta.shell._
import vta.util.config._

class InputDmaBankTest extends AnyFlatSpec with ChiselScalatestTester {
  for ((bi, bo, bus) <- Seq((8, 8, 64), (8, 16, 64), (16, 8, 64), (8, 16, 512))) {
    it should s"route DMA input vectors through BI$bi BO$bo banks on VME$bus" in {
      val base = new DefaultTsimConfig
      implicit val p: Parameters = base.alterPartial {
        case CoreKey => base(CoreKey).copy(blockIn = bi, blockOut = bo,
          inpMemDepth = 8192 / bi, wgtMemDepth = 16384 / (bi * bo),
          accMemDepth = 32768 / (bo * 4), outMemDepth = 32768 / (bo * 4))
        case ShellKey => base(ShellKey).copy(
          memParams = base(ShellKey).memParams.copy(dataBits = bus))
      }
      val vectorCount = if (bi == 8 && bo == 16) 2 else 1
      val laneCount = bus / 8
      val inputElements = vectorCount * bi
      val instruction = BigInt(2) << 7 | // input memory id
        BigInt(1) << 64 | BigInt(vectorCount) << 80 | BigInt(vectorCount) << 96

      test(new TensorLoad("inp")) { c =>
        c.io.start.poke(false.B)
        c.io.inst.poke(instruction.U)
        c.io.baddr.poke(0.U)
        c.io.vme_rd.cmd.ready.poke(true.B)
        c.io.vme_rd.data.valid.poke(false.B)
        c.io.vme_rd.data.bits.data.poke(0.U)
        c.io.vme_rd.data.bits.tag.poke(0.U)
        c.io.vme_rd.data.bits.last.poke(false.B)
        c.io.tensor.rd(0).idx.valid.poke(false.B)
        c.io.tensor.rd(0).idx.bits.poke(0.U)
        c.io.tensor.wr(0).valid.poke(false.B)

        c.io.start.poke(true.B)
        c.clock.step()
        c.io.start.poke(false.B)
        var cycles = 0
        while (!c.io.vme_rd.cmd.valid.peek().litToBoolean && cycles < 20) {
          c.clock.step()
          cycles += 1
        }
        assert(c.io.vme_rd.cmd.valid.peek().litToBoolean)
        val responses = c.io.vme_rd.cmd.bits.len.peek().litValue.toInt + 1
        val responseTag = c.io.vme_rd.cmd.bits.tag.peek().litValue
        c.clock.step() // accept the VME command

        for (beat <- 0 until responses) {
          var word = BigInt(0)
          for (lane <- 0 until laneCount) {
            val element = beat * laneCount + lane
            if (element < inputElements) word |= BigInt(element + 1) << (lane * 8)
          }
          c.io.vme_rd.data.valid.poke(true.B)
          c.io.vme_rd.data.bits.data.poke(word.U)
          c.io.vme_rd.data.bits.tag.poke(responseTag.U)
          c.io.vme_rd.data.bits.last.poke((beat == responses - 1).B)
          c.clock.step()
        }
        c.io.vme_rd.data.valid.poke(false.B)

        var doneCycles = 0
        while (!c.io.done.peek().litToBoolean && doneCycles < 30) {
          c.clock.step()
          doneCycles += 1
        }
        assert(c.io.done.peek().litToBoolean)

        val lanes = math.min(bi, bo)
        val parallel = math.max(bi, bo)
        val ratio = parallel / lanes
        val readCount = if (bi > bo) ratio else 1
        for (index <- 0 until readCount) {
          c.io.tensor.rd(0).idx.valid.poke(true.B)
          c.io.tensor.rd(0).idx.bits.poke(index.U)
          c.clock.step()
          var latency = 0
          while (!c.io.tensor.rd(0).data.valid.peek().litToBoolean && latency < 4) {
            c.clock.step()
            latency += 1
          }
          assert(c.io.tensor.rd(0).data.valid.peek().litToBoolean)
          for (lane <- 0 until parallel) {
            val selected = (index / ratio) * ratio + (index % ratio + lane / lanes) % ratio
            c.io.tensor.rd(0).data.bits(0)(lane).expect((selected * lanes + lane % lanes + 1).U)
          }
        }
      }
    }
  }
}
