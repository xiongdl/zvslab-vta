package unittest

import chisel3._
import chiseltest._
import chiseltest.iotesters._
import org.scalatest.flatspec.AnyFlatSpec
import vta.DefaultTsimConfig
import vta.core._
import vta.util.config._

class TensorGemmInputGeometryTester(c: TensorGemmSimple, logicalInputs: Seq[Int], bi: Int, bo: Int)
  extends PeekPokeTester(c) {
  poke(c.io.start, 0)
  poke(c.io.dec.reset, 0)
  poke(c.io.dec.uop_begin, 0)
  poke(c.io.dec.uop_end, 1)
  poke(c.io.dec.lp_0, 1)
  poke(c.io.dec.lp_1, logicalInputs.length)
  poke(c.io.dec.acc_0, 1)
  poke(c.io.dec.acc_1, 1)
  poke(c.io.dec.inp_0, 1)
  poke(c.io.dec.inp_1, 1)
  poke(c.io.dec.wgt_0, 1)
  poke(c.io.dec.wgt_1, 1)
  poke(c.io.uop.data.bits.u0, 0)
  poke(c.io.uop.data.bits.u1, logicalInputs.head)
  poke(c.io.uop.data.bits.u2, 0)

  // Expanded physical input lanes carry conspicuous values. A GEMM dot must
  // consume exactly BI logical lanes from each selected source vector.
  for (j <- c.io.inp.rd(0).data.bits(0).indices) {
    poke(c.io.inp.rd(0).data.bits(0)(j), if (j < bi) j + 1 else 99)
  }
  for (row <- c.io.wgt.rd(0).data.bits; lane <- row) poke(lane, 1)
  for (row <- c.io.acc.rd(0).data.bits; lane <- row) poke(lane, 0)

  class TensorMock(tm: TensorMaster) {
    poke(tm.rd(0).data.valid, 0)
    var valid = peek(tm.rd(0).idx.valid)
    def update() {
      poke(tm.rd(0).data.valid, valid)
      valid = peek(tm.rd(0).idx.valid)
    }
  }
  class UopMock {
    poke(c.io.uop.data.valid, 0)
    var valid = peek(c.io.uop.idx.valid)
    def update() {
      poke(c.io.uop.data.valid, valid)
      valid = peek(c.io.uop.idx.valid)
    }
  }
  val uop = new UopMock
  val inp = new TensorMock(c.io.inp)
  val wgt = new TensorMock(c.io.wgt)
  val acc = new TensorMock(c.io.acc)

  // UopMaster is not a TensorMaster; update its read response separately.
  def tick(): Unit = {
    step(1)
    uop.update(); inp.update(); wgt.update(); acc.update()
  }

  poke(c.io.start, 1)
  var checks = 0
  var cycles = 0
  val expected = bi * (bi + 1) / 2 // each output reads a distinct zeroed accumulator
  while (peek(c.io.done) == 0 && cycles < 200) {
    tick()
    if (peek(c.io.inp.rd(0).idx.valid) == 1) {
      val expectedIdx = BigInt(logicalInputs(checks) * (bi / math.min(bi, bo)))
      expect(c.io.inp.rd(0).idx.bits, expectedIdx)
      checks += 1
    }
    if (peek(c.io.out.wr(0).valid) == 1) {
      for (lane <- 0 until c.io.out.wr(0).bits.data(0).size) {
        expect(c.io.out.wr(0).bits.data(0)(lane), expected)
      }
    }
    poke(c.io.start, 0)
    cycles += 1
  }
  assert(checks == logicalInputs.length, s"expected ${logicalInputs.length} input reads, observed $checks")
  assert(peek(c.io.done) == 1, "GEMM did not finish")
}

class TensorGemmInputGeometryTest extends AnyFlatSpec with ChiselScalatestTester {
  for ((bi, bo) <- Seq((8, 8), (8, 16), (16, 8))) {
    it should s"run GEMM with BI$bi BO$bo logical sources and BI-only dot products" in {
      val base = new DefaultTsimConfig
      implicit val p: Parameters = base.alterPartial {
        case CoreKey => base(CoreKey).copy(blockIn = bi, blockOut = bo,
          inpMemDepth = 64, wgtMemDepth = 256, accMemDepth = 256, outMemDepth = 256)
      }
      val sources = Seq(1, 2)
      test(new TensorGemmSimple()(p)).withAnnotations(Seq(TreadleBackendAnnotation))
        .runPeekPoke(c => new TensorGemmInputGeometryTester(c, sources, bi, bo))
    }
  }
}
