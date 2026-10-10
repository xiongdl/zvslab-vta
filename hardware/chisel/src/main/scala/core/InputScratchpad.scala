package vta.core

import chisel3._
import chisel3.util._
import vta.util.config._
import vta.shell._

/** Input SRAM: logical BI DMA writes, subvector compute reads in one cycle.
 * Extra row stripes preserve wide VME throughput without widening any bank.
 */
class InputScratchpad(writePorts: Int)(implicit p: Parameters) extends Module {
  val c = p(CoreKey)
  val tp = new TensorParams("inp")
  val lanes = c.inpBankLanes
  val ratio = c.inpBanksPerBatch
  val slices = c.inpSlices
  val stripes = math.max(1, p(ShellKey).memParams.dataBits / c.inpParallelBits)
  val banksPerBatch = ratio * stripes
  val bankDepth = c.inpSubvectorDepth / banksPerBatch
  require(bankDepth > 0 && c.inpSubvectorDepth % banksPerBatch == 0)
  val io = IO(new Bundle {
    val write = Input(Vec(writePorts, new Bundle {
      val valid = Bool()
      val index = UInt(tp.logicalAddrBits.W)
      val data = Vec(c.batch, Vec(c.blockIn, UInt(c.inpBits.W)))
      val mask = Vec(c.batch, Vec(c.blockIn, Bool()))
    }))
    val read = Flipped(ValidIO(UInt(tp.memAddrBits.W)))
    val result = ValidIO(Vec(c.batch, Vec(math.max(c.blockIn,c.blockOut),UInt(c.inpBits.W))))
  })
  val memories = Seq.fill(c.batch * banksPerBatch) {
    SyncReadMem(bankDepth, Vec(lanes, UInt(c.inpBits.W)))
  }
  for (b <- 0 until c.batch; bank <- 0 until banksPerBatch) {
    val enables = for (port <- 0 until writePorts; q <- 0 until slices) yield {
      val sub = io.write(port).index * slices.U + q.U
      io.write(port).valid && sub % banksPerBatch.U === bank.U &&
        io.write(port).mask(b).slice(q*lanes,(q+1)*lanes).reduce(_ || _)
    }
    val rows = for (port <- 0 until writePorts; q <- 0 until slices) yield {
      (io.write(port).index * slices.U + q.U) / banksPerBatch.U
    }
    val datas = for (port <- 0 until writePorts; q <- 0 until slices) yield {
      VecInit(io.write(port).data(b).slice(q*lanes,(q+1)*lanes))
    }
    val masks = for (port <- 0 until writePorts; q <- 0 until slices) yield {
      VecInit(io.write(port).mask(b).slice(q*lanes,(q+1)*lanes))
    }
    assert(PopCount(VecInit(enables)) <= 1.U, "Input SRAM bank write collision")
    when(enables.reduce(_ || _)) {
      memories(b*banksPerBatch+bank).write(Mux1H(enables,rows), Mux1H(enables,datas), Mux1H(enables,masks))
    }
  }
  val readIndex = ShiftRegister(io.read.bits,tp.readTensorLatency)
  val readValid = ShiftRegister(io.read.valid,tp.readTensorLatency,false.B,true.B)
  val stripe = (readIndex / ratio.U) % stripes.U
  val selector = RegNext(readIndex % ratio.U)
  val stripeDelayed = RegNext(stripe)
  for(b <- 0 until c.batch) {
    val bankData = VecInit((0 until banksPerBatch).map { bank =>
      memories(b*banksPerBatch+bank).read(readIndex / banksPerBatch.U,
        readValid && stripe === (bank/ratio).U)
    })
    for (j <- 0 until math.max(c.blockIn,c.blockOut)) {
      val selected = stripeDelayed * ratio.U + (selector + (j/lanes).U) % ratio.U
      io.result.bits(b)(j) := bankData(selected)(j%lanes)
    }
  }
  io.result.valid := RegNext(readValid,false.B)
}
