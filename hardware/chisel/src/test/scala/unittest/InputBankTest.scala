package unittest

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import vta.DefaultTsimConfig
import vta.core._
import vta.shell._
import vta.util.config._

class InputBankTest extends AnyFlatSpec with ChiselScalatestTester {
  for ((bi, bo) <- Seq((8,8),(8,16),(16,8)); bus <- Seq(64,512)) {
    it should s"load real BI$bi BO$bo vectors on VME$bus and select both halves" in {
      val base = new DefaultTsimConfig
      implicit val p: Parameters = base.alterPartial {
        case CoreKey => base(CoreKey).copy(blockIn=bi, blockOut=bo, inpMemDepth=8192/bi,
          wgtMemDepth=16384/(bi*bo), accMemDepth=32768/(bo*4),outMemDepth=32768/(bo*4))
        case ShellKey => base(ShellKey).copy(memParams=base(ShellKey).memParams.copy(dataBits=bus))
      }
      test(new TensorLoad("inp")) { c =>
        c.io.start.poke(false.B); c.io.inst.poke(0.U); c.io.baddr.poke(0.U)
        c.io.vme_rd.cmd.ready.poke(false.B); c.io.vme_rd.data.valid.poke(false.B)
        c.io.vme_rd.data.bits.data.poke(0.U);c.io.vme_rd.data.bits.tag.poke(0.U)
        c.io.vme_rd.data.bits.last.poke(false.B)
        c.io.tensor.rd(0).idx.valid.poke(false.B);c.io.tensor.rd(0).idx.bits.poke(0.U)
        c.io.tensor.wr(0).valid.poke(false.B)
        c.clock.step()
        // Direct writes stay logical BI vectors, including two nonzero BI16 halves.
        for(n <- 0 until 8) {
          c.io.tensor.wr(0).valid.poke(true.B); c.io.tensor.wr(0).bits.idx.poke(n.U)
          for(j <- 0 until bi) c.io.tensor.wr(0).bits.data(0)(j).poke((n*bi+j+1).U)
          c.clock.step()
        }
        c.io.tensor.wr(0).valid.poke(false.B)
        val lanes=math.min(bi,bo); val parallel=math.max(bi,bo);val ratio=parallel/lanes
        assert(c.io.tensor.rd(0).data.bits(0).length == parallel)
        for(s <- 0 until 8*bi/lanes) {
          c.io.tensor.rd(0).idx.valid.poke(true.B);c.io.tensor.rd(0).idx.bits.poke(s.U)
          c.clock.step();c.io.tensor.rd(0).data.valid.expect(true.B)
          for(j <- 0 until parallel) {
            val selected=(s/ratio)*ratio+(s%ratio+j/lanes)%ratio
            c.io.tensor.rd(0).data.bits(0)(j).expect((selected*lanes+j%lanes+1).U)
          }
        }
      }
    }
  }
}
