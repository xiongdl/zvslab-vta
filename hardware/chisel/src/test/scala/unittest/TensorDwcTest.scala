package unittest

import chisel3._
import chiseltest._
import chiseltest.iotesters._
import org.scalatest.flatspec.AnyFlatSpec
import vta.DefaultTsimConfig
import vta.core._
import vta.util.config._

class TensorDwcTester[T <: TensorGemmIfc](c: T, bi: Int, bo: Int, batch: Int, sameDst: Boolean = false) extends PeekPokeTester[T](c) {
  poke(c.io.start, 0)
  poke(c.io.dec.op, 5)
  poke(c.io.dec.reset, 0)
  poke(c.io.dec.uop_begin, 0); poke(c.io.dec.uop_end, 9)
  poke(c.io.dec.lp_0, 1); poke(c.io.dec.lp_1, 2)
  poke(c.io.dec.acc_0, 0); poke(c.io.dec.acc_1, if (sameDst) 0 else 1)
  poke(c.io.dec.inp_0, 0); poke(c.io.dec.inp_1, 10)
  poke(c.io.dec.wgt_0, 0); poke(c.io.dec.wgt_1, 0)
  poke(c.io.uop.data.valid, 0)
  poke(c.io.inp.rd(0).data.valid, 0)
  for (r <- c.io.wgt.rd) poke(r.data.valid, 0)
  for (r <- c.io.acc.rd) poke(r.data.valid, 0)
  val memory = Array.fill(2, batch, bo)(BigInt(0))
  def input(src: Int, b: Int, lane: Int): Int = (src % 9 + 1) + b * 2 + lane
  def weight(tap: Int, lane: Int): Int = if (tap % 2 == 0) lane + 1 else -(lane + 1)
  for (resetMode <- Seq(true, false, false)) {
    poke(c.io.dec.reset, if (resetMode) 1 else 0)
    val initial = memory.map(_.map(_.clone()))
    var reads = 0; var writes = 0; var inputs = 0
    poke(c.io.start, 1)
    var cycle = 0
    var sawDone = false
    while (!sawDone && cycle < 300) {
      sawDone = peek(c.io.done) == 1
      val uv = peek(c.io.uop.idx.valid).toInt
      val ui = peek(c.io.uop.idx.bits).toInt
      val iv = peek(c.io.inp.rd(0).idx.valid).toInt
      val ii = peek(c.io.inp.rd(0).idx.bits).toInt
      if (iv == 1) { assert(ii == inputs % 9 + (inputs / 9) * 10); inputs += 1 }
      val w = c.io.wgt.rd.map(r => (peek(r.idx.valid).toInt, peek(r.idx.bits).toInt))
      reads += w.head._1
      val a = c.io.acc.rd.map(r => (peek(r.idx.valid).toInt, peek(r.idx.bits).toInt))
      val ad = a.map { case (_, idx) => memory(idx).map(_.clone()) }
      if (peek(c.io.acc.wr(0).valid) == 1) {
        val dst = peek(c.io.acc.wr(0).bits.idx).toInt
        val tap = writes % 9
        for (g <- c.io.acc.wr.indices; b <- 0 until batch; lane <- c.io.acc.wr(g).bits.data(b).indices) {
          val channel = g * c.io.acc.wr(g).bits.data(b).size + lane
          val range = writes / 9
          val previous = if (sameDst && range == 1) (0 until 9).map(k => input(k, b, channel) * weight(k, channel)).sum else 0
          val expected = if (resetMode) BigInt(0) else initial(dst)(b)(channel) + previous +
            (0 to tap).map(k => input(k + range * 10, b, channel) * weight(k, channel)).sum
          expect(c.io.acc.wr(g).bits.data(b)(lane), expected & BigInt("ffffffff", 16))
          memory(dst)(b)(channel) = expected & BigInt("ffffffff", 16)
        }
        assert(peek(c.io.out.wr(0).valid) == (if (resetMode) 0 else 1))
        writes += 1
      }
      step(1)
      poke(c.io.start, 0)
      poke(c.io.uop.data.valid, uv)
      poke(c.io.uop.data.bits.u0, 0); poke(c.io.uop.data.bits.u1, ui); poke(c.io.uop.data.bits.u2, ui / bi)
      poke(c.io.inp.rd(0).data.valid, iv)
      for (b <- 0 until batch; lane <- c.io.inp.rd(0).data.bits(b).indices)
        poke(c.io.inp.rd(0).data.bits(b)(lane), input(ii, b, lane))
      for (g <- c.io.wgt.rd.indices) {
        poke(c.io.wgt.rd(g).data.valid, w(g)._1)
        for (lane <- c.io.wgt.rd(g).data.bits.indices; k <- 0 until bi) {
          val channel = g * c.io.wgt.rd(g).data.bits.size + lane
          poke(c.io.wgt.rd(g).data.bits(lane)(k), weight(w(g)._2 * bi + k, channel) & 255)
        }
      }
      for (g <- c.io.acc.rd.indices) {
        poke(c.io.acc.rd(g).data.valid, a(g)._1)
        for (b <- 0 until batch; lane <- c.io.acc.rd(g).data.bits(b).indices)
          poke(c.io.acc.rd(g).data.bits(b)(lane), ad(g)(b)(g * c.io.acc.rd(g).data.bits(b).size + lane))
      }
      cycle += 1
    }
    assert(sawDone)
    assert(inputs == 18 && writes == 18, s"reads=$inputs writes=$writes")
    assert(reads == 2 * ((9 + bi - 1) / bi), s"weight reads=$reads")
    step(2)
  }
}

class TensorDwcTest extends AnyFlatSpec with ChiselScalatestTester {
  for ((bi, bo) <- Seq((8, 8), (8, 16), (16, 8)); batch <- Seq(1, 2); simple <- Seq(false, true); sameDst <- Seq(false, true)) {
    it should s"consume nine signed taps and reload ranges for BI$bi BO$bo B$batch simple=$simple sameDst=$sameDst" in {
      val base = new DefaultTsimConfig
      implicit val p: Parameters = base.alterPartial {
        case CoreKey => base(CoreKey).copy(blockIn = bi, blockOut = bo, batch = batch,
          inpMemDepth = 64, wgtMemDepth = 256, accMemDepth = 256, outMemDepth = 256)
      }
      test(if (simple) new TensorGemmSimple()(p) else new TensorGemm()(p)).withAnnotations(Seq(TreadleBackendAnnotation))
        .runPeekPoke(c => new TensorDwcTester(c, bi, bo, batch, sameDst))
    }
  }
}

class DwcWeightBankTest extends AnyFlatSpec with ChiselScalatestTester {
  it should "hold signed low lanes on random valid bubbles and prioritize reset" in {
    test(new DwcWeightBank(16, 8, 8)).withAnnotations(Seq(TreadleBackendAnnotation)) { c =>
      val rng = new scala.util.Random(42)
      c.io.clear.poke(true.B); c.io.consume.poke(false.B); c.io.reload.poke(false.B)
      c.clock.step(); c.io.clear.poke(false.B)
      val packed = (0 until 8).map(k => BigInt((k - 7) & 255) << (8 * k)).reduce(_ | _)
      for (lane <- 0 until 16) c.io.memory(lane).poke(packed.U)
      c.io.reload.poke(true.B); c.io.consume.poke(true.B)
      for (lane <- 0 until 16) c.io.current(lane).expect((-7).S)
      c.clock.step(); c.io.reload.poke(false.B)
      for (tap <- 1 until 8) {
        c.io.consume.poke(false.B)
        for (_ <- 0 until 1 + rng.nextInt(4)) {
          for (lane <- 0 until 16) c.io.current(lane).expect((tap - 7).S)
          c.clock.step()
        }
        c.io.consume.poke(true.B); c.clock.step()
      }
      for (lane <- 0 until 16) c.io.current(lane).expect(0.S)
      c.io.reload.poke(true.B); c.io.clear.poke(true.B); c.clock.step()
      c.io.reload.poke(false.B); c.io.consume.poke(false.B); c.io.clear.poke(false.B)
      for (lane <- 0 until 16) c.io.current(lane).expect(0.S)
    }
  }
}

class DotProductDwcTest extends AnyFlatSpec with ChiselScalatestTester {
  for (bi <- Seq(8, 16)) {
    it should s"keep GEMM latency when selecting multiplier zero for BI$bi" in {
      test(new DotProduct(blockIn = bi)).withAnnotations(Seq(TreadleBackendAnnotation)) { c =>
        for (j <- 0 until bi) { c.io.a(j).poke((j + 1).S); c.io.b(j).poke((-2).S) }
        c.io.dwc.poke(false.B); c.clock.step(2)
        c.io.y.expect((-bi * (bi + 1)).S)
        c.io.dwc.poke(true.B); c.clock.step()
        c.io.y.expect((-bi * (bi + 1)).S)
        c.clock.step(); c.io.y.expect((-2).S)
        c.io.dwc.poke(false.B); c.clock.step(); c.io.y.expect((-2).S)
        c.clock.step(); c.io.y.expect((-bi * (bi + 1)).S)
      }
    }
  }
}

class DwcDispatchHarness(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val inst = Input(UInt(128.W))
    val compute = Output(Bool())
    val gemm = Output(Bool())
    val dependencies = Output(UInt(4.W))
  })
  val fetch = Module(new FetchDecode)
  val compute = Module(new ComputeDecode)
  fetch.io.inst := io.inst; compute.io.inst := io.inst
  io.compute := fetch.io.isCompute
  io.gemm := compute.io.isGemm
  io.dependencies := chisel3.util.Cat(compute.io.push_next, compute.io.push_prev,
    compute.io.pop_next, compute.io.pop_prev)
}

class DwcDispatchTest extends AnyFlatSpec with ChiselScalatestTester {
  it should "route opcode five and preserve every dependency bit" in {
    implicit val p: Parameters = new DefaultTsimConfig
    test(new DwcDispatchHarness).withAnnotations(Seq(TreadleBackendAnnotation)) { c =>
      for (deps <- 0 until 16) {
        c.io.inst.poke((BigInt(deps) << 3 | BigInt(5)).U)
        c.io.compute.expect(true.B); c.io.gemm.expect(true.B)
        c.io.dependencies.expect(deps.U)
      }
    }
  }
}

class TensorDwcSplitTest extends AnyFlatSpec with ChiselScalatestTester {
  for ((bi, bo) <- Seq((8, 8), (8, 16), (16, 8))) {
    it should s"select global input channels in two MVM groups for BI$bi BO$bo" in {
      val base = new DefaultTsimConfig
      implicit val p: Parameters = base.alterPartial {
        case CoreKey => base(CoreKey).copy(blockIn = bi, blockOut = bo, blockOutFactor = 2,
          inpMemDepth = 64, wgtMemDepth = 256, accMemDepth = 256, outMemDepth = 256)
      }
      test(new TensorGemm()(p)).withAnnotations(Seq(TreadleBackendAnnotation))
        .runPeekPoke(c => new TensorDwcTester(c, bi, bo, 1, true))
    }
  }
}
