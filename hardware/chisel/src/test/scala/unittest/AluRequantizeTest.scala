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
import vta.util.config._

class AluRequantizeTest extends AnyFlatSpec with ChiselScalatestTester {
  implicit private val p: Parameters = new DefaultTsimConfig

  private val WordModulus = BigInt(1) << 32
  private val WordMask = WordModulus - 1
  private val Q31Half = BigInt(1) << 30

  private case class CmsisGolden(x: Int, multiplier: Int, shift: Int, doubleResult: Int, singleResult: Int)

  private lazy val cmsisGoldens: Seq[CmsisGolden] = {
    val stream = Option(getClass.getResourceAsStream("/cmsis_requantize_golden.csv"))
      .getOrElse(throw new IllegalStateException("missing pinned CMSIS-NN golden vectors"))
    val source = scala.io.Source.fromInputStream(stream, "US-ASCII")
    try {
      source.getLines().filter(line => line.nonEmpty && !line.startsWith("#") && !line.startsWith("x,"))
        .map { line =>
          val fields = line.split(",").map(_.toInt)
          require(fields.length == 5, s"invalid CMSIS vector: $line")
          CmsisGolden(fields(0), fields(1), fields(2), fields(3), fields(4))
        }.toVector
    } finally {
      source.close()
    }
  }

  private def signed32(value: BigInt): Int = {
    val word = value & WordMask
    if (word.testBit(31)) (word - WordModulus).toInt else word.toInt
  }

  private def roundQ31(product: BigInt, rounding: Int): Int = {
    val quotient = product >> 31
    val remainder = product - (quotient << 31)
    val increment = rounding match {
      case 0 => false
      case 1 => remainder >= Q31Half
      case 2 => remainder > Q31Half || (remainder == Q31Half && product >= 0)
      case _ => throw new IllegalArgumentException(s"invalid rounding mode $rounding")
    }
    signed32(quotient + (if (increment) 1 else 0))
  }

  private def roundRight(value: Int, shift: Int, rounding: Int): Int = {
    require(shift >= 0 && shift <= 31)
    if (shift == 0) {
      value
    } else {
      val quotient = BigInt(value) >> shift
      val remainder = BigInt(value) - (quotient << shift)
      val half = BigInt(1) << (shift - 1)
      val increment = rounding match {
        case 0 => false
        case 1 => remainder >= half
        case 2 => remainder > half || (remainder == half && value >= 0)
        case _ => throw new IllegalArgumentException(s"invalid rounding mode $rounding")
      }
      signed32(quotient + (if (increment) 1 else 0))
    }
  }

  private def unsigned(value: Int): BigInt = BigInt(value.toLong & 0xffffffffL)

  private def drive(
    c: AluVector,
    opcode: Int,
    rounding: Int,
    lhs: Seq[Int],
    rhs: Seq[Int],
    valid: Boolean = true): Unit = {
    require(lhs.nonEmpty && rhs.nonEmpty)
    c.io.opcode.poke(opcode.U)
    c.io.rounding.poke(rounding.U)
    c.io.acc_a.data.valid.poke(valid.B)
    c.io.acc_b.data.valid.poke(valid.B)
    for (lane <- 0 until c.blockOut) {
      c.io.acc_a.data.bits(0)(lane).poke(unsigned(lhs(lane % lhs.length)).U)
      c.io.acc_b.data.bits(0)(lane).poke(unsigned(rhs(lane % rhs.length)).U)
    }
  }

  private def vectorResult(c: AluVector): Seq[Int] =
    (0 until c.blockOut).map(lane => c.io.acc_y.data.bits(0)(lane).peek().litValue.toInt)

  behavior of "AluRequantize"

  it should "match reproducible CMSIS-NN double- and single-rounding vectors through per-lane RMUL and RSFT" in {
    test(new AluVector) { c =>
      val cases = cmsisGoldens
      assert(cases.size >= c.blockOut, s"need at least ${c.blockOut} fixed CMSIS lanes")
      val modes = Seq(
        (1, 2, (golden: CmsisGolden) => golden.doubleResult),
        (0, 1, (golden: CmsisGolden) => golden.singleResult))
      assert(cases.head.x == 1 && cases.head.multiplier == (1 << 30) && cases.head.shift == -1)
      assert(cases.head.doubleResult == 1 && cases.head.singleResult == 0)
      for (chunk <- cases.grouped(c.blockOut)) {
        val active = chunk.toVector
        val xs = active.map(_.x)
        val multipliers = active.map(_.multiplier)
        val shifts = active.map(g => -g.shift)
        for ((rmulRounding, rsftRounding, cmsisResult) <- modes) {
          drive(c, opcode = 5, rounding = rmulRounding, lhs = xs, rhs = multipliers)
          c.clock.step()
          c.io.acc_y.data.valid.expect(true.B)
          val products = vectorResult(c)
          for (lane <- active.indices) {
            val expected = roundQ31(BigInt(active(lane).x) * BigInt(active(lane).multiplier), rmulRounding)
            assert(products(lane) == expected,
              s"CMSIS RMUL intermediate lane $lane: ${products(lane)} != $expected")
          }

          drive(c, opcode = 6, rounding = rsftRounding, lhs = products, rhs = shifts)
          c.clock.step()
          c.io.acc_y.data.valid.expect(true.B)
          val results = vectorResult(c)
          for (lane <- active.indices) {
            val expected = cmsisResult(active(lane))
            val stageExpected = roundRight(products(lane), shifts(lane), rsftRounding)
            assert(stageExpected == expected,
              s"pinned CMSIS fixture mismatch for $active lane $lane: $stageExpected != $expected")
            assert(results(lane) == expected,
              s"CMSIS requantize lane $lane: ${results(lane)} != $expected")
          }
        }
      }
    }
  }

  it should "round full signed Q31 products for every rounding mode" in {
    test(new AluVector) { c =>
      val values = Seq(Int.MinValue, Int.MinValue + 1, -1073741825, -7, -3, -1, 0,
        1, 3, 7, 1073741825, Int.MaxValue)
      val multipliers = Seq(Int.MinValue, Int.MaxValue, 1879048193, 1 << 30, 1 << 30,
        1 << 30, Int.MaxValue, 1 << 30, 1 << 30, 1 << 30, Int.MaxValue, Int.MaxValue)
      for (rounding <- 0 to 2) {
        drive(c, opcode = 5, rounding = rounding, lhs = values, rhs = multipliers)
        c.clock.step()
        c.io.acc_y.data.valid.expect(true.B)
        val actual = vectorResult(c)
        for (lane <- 0 until c.blockOut) {
          val index = lane % values.length
          val expected = roundQ31(BigInt(values(index)) * BigInt(multipliers(index)), rounding)
          assert(actual(lane) == expected,
            s"RMUL round=$rounding lane=$lane: ${actual(lane)} != $expected")
        }
      }
    }
  }

  it should "use each lane's own shift for signed RSFT ties, zero, and shift 31" in {
    test(new AluVector) { c =>
      val values = Seq(5, 7, -5, -7, 15, -15, Int.MaxValue, Int.MinValue, 1, -1)
      val shifts = Seq(0, 2, 1, 1, 2, 2, 31, 31, 1, 31)
      for (rounding <- 0 to 2) {
        drive(c, opcode = 6, rounding = rounding, lhs = values, rhs = shifts)
        c.clock.step()
        c.io.acc_y.data.valid.expect(true.B)
        val actual = vectorResult(c)
        for (lane <- 0 until c.blockOut) {
          val index = lane % values.length
          val expected = roundRight(values(index), shifts(index), rounding)
          assert(actual(lane) == expected,
            s"RSFT round=$rounding lane=$lane: ${actual(lane)} != $expected")
        }
      }
    }
  }

  it should "carry opcode and rounding with each valid input through one AluReg cycle" in {
    test(new AluReg) { c =>
      val inputs = Seq(
        (5, 1, 1, 1 << 30, true),
        (6, 2, -5, 1, true),
        (5, 0, 1, 1 << 30, true),
        (6, 1, 0, 1, true),
        (4, 0, -12345, 6789, true),
        (6, 1, 123456789, 31, true),
        (4, 0, 99, 3, false),
        (5, 2, -3, 1 << 30, true))
      for ((opcode, rounding, lhs, rhs, valid) <- inputs) {
        c.io.opcode.poke(opcode.U)
        c.io.rounding.poke(rounding.U)
        c.io.a.valid.poke(valid.B)
        c.io.b.valid.poke(valid.B)
        c.io.a.bits.poke(unsigned(lhs).U)
        c.io.b.bits.poke(unsigned(rhs).U)
        c.clock.step()
        c.io.y.valid.expect(valid.B)
        if (valid) {
          val expected = opcode match {
            case 4 => signed32(BigInt(lhs) * BigInt(rhs))
            case 5 => roundQ31(BigInt(lhs) * BigInt(rhs), rounding)
            case 6 => roundRight(lhs, rhs, rounding)
          }
          c.io.y.bits.expect(unsigned(expected).U)
        }
      }
    }
  }

  it should "mark unsupported rounding combinations and RSFT counts illegal" in {
    test(new Alu) { c =>
      c.io.a.poke(1.S)
      c.io.b.poke(1.S)
      c.io.opcode.poke(5.U)
      c.io.rounding.poke(3.U)
      c.io.legal.expect(false.B)
      c.io.opcode.poke(4.U)
      c.io.rounding.poke(1.U)
      c.io.legal.expect(false.B)
      c.io.opcode.poke(6.U)
      c.io.rounding.poke(0.U)
      c.io.b.poke((-1).S)
      c.io.legal.expect(false.B)
      c.io.b.poke(32.S)
      c.io.legal.expect(false.B)
      c.io.b.poke(31.S)
      c.io.legal.expect(true.B)
      c.io.opcode.poke(7.U)
      c.io.legal.expect(false.B)
    }
  }
}
