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


// Handwritten instructions exercise the real driver through DMA and output stores.
#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <tvm/runtime/registry.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

class Buffer {
 public:
  explicit Buffer(size_t bytes) : data(VTAMemAlloc(bytes, VTA_NOT_CACHED)) {
    memset(data, 0, bytes);
  }
  ~Buffer() { VTAMemFree(data); }
  void* data;
};

static VTAGenericInsn Memory(unsigned opcode, unsigned type, Buffer& buffer,
                             unsigned count, unsigned offset = 0) {
  VTAGenericInsn insn{};
  auto& mem = reinterpret_cast<VTAMemInsn&>(insn);
  mem.opcode = opcode;
  mem.memory_type = type;
  unsigned bytes = type == VTA_MEM_ID_INP ? VTA_INP_ELEM_BYTES
      : type == VTA_MEM_ID_WGT ? VTA_WGT_ELEM_BYTES
      : type == VTA_MEM_ID_ACC ? VTA_ACC_ELEM_BYTES
      : type == VTA_MEM_ID_UOP ? VTA_UOP_ELEM_BYTES : VTA_OUT_ELEM_BYTES;
  mem.dram_base = VTAMemGetPhyAddr(buffer.data) / bytes + offset;
  mem.x_size = count;
  mem.y_size = 1;
  mem.x_stride = count;
  return insn;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  bool gemm = atoi(argv[1]);
  constexpr unsigned taps = 9, outputs = 4;
  constexpr unsigned lanes = std::max(VTA_BLOCK_IN, VTA_BLOCK_OUT);
  constexpr unsigned bank_lanes = std::min(VTA_BLOCK_IN, VTA_BLOCK_OUT);
  constexpr unsigned entries = (taps + VTA_BLOCK_IN - 1) / VTA_BLOCK_IN;
  constexpr unsigned inp_vectors = outputs * taps * lanes / VTA_BLOCK_IN;
  Buffer input(inp_vectors * VTA_INP_ELEM_BYTES);
  Buffer weight(2 * entries * VTA_WGT_ELEM_BYTES);
  Buffer accumulator(outputs * VTA_ACC_ELEM_BYTES);
  Buffer microops((taps + 1) * VTA_UOP_ELEM_BYTES);
  Buffer output(2 * 4 * outputs * VTA_OUT_ELEM_BYTES);
  auto* inp = static_cast<int8_t*>(input.data);
  auto* wgt = static_cast<int8_t*>(weight.data);
  auto* acc = static_cast<int32_t*>(accumulator.data);
  auto* uops = static_cast<VTAUop*>(microops.data);
  for (unsigned pos = 0; pos < outputs; ++pos) {
    for (unsigned k = 0; k < taps; ++k) {
      for (unsigned c = 0; c < lanes; ++c) {
        for (unsigned b = 0; b < VTA_BATCH; ++b) {
          unsigned vector = (pos * taps + k) * lanes / VTA_BLOCK_IN + c / VTA_BLOCK_IN;
          unsigned index = vector * VTA_BATCH * VTA_BLOCK_IN + b * VTA_BLOCK_IN + c % VTA_BLOCK_IN;
          int value = 1 + (pos * 11 + k * 7 + b * 13 + c * 3) % 31;
          inp[index] = (k + c + b) % 2 ? -value : value;
        }
      }
    }
    for (unsigned b = 0; b < VTA_BATCH; ++b) {
      for (unsigned c = 0; c < VTA_BLOCK_OUT; ++c) {
        acc[(pos * VTA_BATCH + b) * VTA_BLOCK_OUT + c] = 101 + pos * 17 + b * 5 + c;
      }
    }
  }
  for (unsigned block = 0; block < 2; ++block) {
    for (unsigned c = 0; c < VTA_BLOCK_OUT; ++c) {
      for (unsigned k = 0; k < taps; ++k) {
        int value = 1 + (block * 5 + c * 3 + k * 2) % 13;
        wgt[((block * entries + k / VTA_BLOCK_IN) * VTA_BLOCK_OUT + c) * VTA_BLOCK_IN
             + k % VTA_BLOCK_IN] = (k + c + block) % 2 ? -value : value;
      }
    }
  }
  // The final entry's unused lanes remain zero; there are exactly nine uops.
  for (unsigned k = 0; k < taps; ++k) {
    uops[k].dst_idx = 0;
    uops[k].src_idx = k * lanes / (gemm ? VTA_BLOCK_IN : bank_lanes);
    uops[k].wgt_idx = k / VTA_BLOCK_IN;
  }
  uops[taps].dst_idx = 0;
  uops[taps].src_idx = 0;

  VTAGenericInsn compute{};
  auto& op = reinterpret_cast<VTAGemInsn&>(compute);
  op.opcode = gemm ? VTA_OPCODE_GEMM : VTA_OPCODE_DWC;
  op.uop_end = taps;
  op.iter_in = op.iter_out = 2;
  op.dst_factor_in = 1;
  op.dst_factor_out = 2;
  unsigned units = gemm ? VTA_BLOCK_IN : bank_lanes;
  op.src_factor_in = taps * lanes / units;
  // Select the second 8-channel half of a real BI16 vector for odd positions.
  if (!gemm && VTA_BLOCK_IN > VTA_BLOCK_OUT) ++op.src_factor_in;
  op.src_factor_out = 2 * taps * lanes / units;
  op.wgt_factor_out = entries;

  std::vector<VTAGenericInsn> instructions{
      Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_INP, input, inp_vectors),
      Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_WGT, weight, 2 * entries),
      Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, accumulator, outputs),
      Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, microops, taps + 1)};
  reinterpret_cast<VTAMemInsn&>(instructions[1]).push_next_dep = 1;
  op.pop_prev_dep = 1;
  instructions.push_back(compute);
  op.pop_prev_dep = 0;
  // Four low-byte stores and arithmetic SHR reconstruct every signed int32.
  auto snapshot = [&](unsigned snapshot_index) {
    for (unsigned byte = 0; byte < 4; ++byte) {
      reinterpret_cast<VTAGemInsn&>(instructions.back()).push_next_dep = 1;
      auto store = Memory(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, output, outputs,
                          (snapshot_index * 4 + byte) * outputs);
      auto& mem = reinterpret_cast<VTAMemInsn&>(store);
      mem.pop_prev_dep = mem.push_prev_dep = 1;
      instructions.push_back(store);
      VTAGenericInsn shift{};
      auto& alu = reinterpret_cast<VTAAluInsn&>(shift);
      alu.opcode = VTA_OPCODE_ALU;
      alu.pop_next_dep = 1;
      alu.uop_bgn = taps;
      alu.uop_end = taps + 1;
      alu.iter_in = outputs;
      alu.iter_out = 1;
      alu.dst_factor_in = 1;
      alu.src_factor_in = 1;
      alu.use_imm = 1;
      alu.imm = 8;
      alu.alu_opcode = VTA_ALU_OPCODE_SHR;
      instructions.push_back(shift);
    }
  };
  snapshot(0);
  auto reset = compute;
  reinterpret_cast<VTAGemInsn&>(reset).reset_reg = 1;
  instructions.push_back(reset);
  instructions.push_back(compute);
  instructions.push_back(compute);  // Accumulation and same-address kernel restart.
  snapshot(1);
  VTAGenericInsn finish{};
  finish.opcode = VTA_OPCODE_FINISH;
  instructions.push_back(finish);
  Buffer stream(instructions.size() * sizeof(VTAGenericInsn));
  memcpy(stream.data, instructions.data(), instructions.size() * sizeof(VTAGenericInsn));
  auto device = VTADeviceAlloc();
  (*tvm::runtime::Registry::Get("vta.simulator.profiler_clear"))();
  assert(VTADeviceRun(device, VTAMemGetPhyAddr(stream.data), instructions.size(), 1000) == 0);
  auto* bytes = static_cast<uint8_t*>(output.data);
  for (unsigned s = 0; s < 2; ++s) {
    for (unsigned pos = 0; pos < outputs; ++pos) {
      for (unsigned b = 0; b < VTA_BATCH; ++b) {
        for (unsigned c = 0; c < VTA_BLOCK_OUT; ++c) {
          uint32_t value = 0;
          for (unsigned byte = 0; byte < 4; ++byte) {
            unsigned index = (((s * 4 + byte) * outputs + pos) * VTA_BATCH + b) * VTA_BLOCK_OUT + c;
            value |= uint32_t(bytes[index]) << (8 * byte);
          }
          printf("RESULT %u %u %u %u %d\n", s, pos, b, c, int32_t(value));
        }
      }
    }
  }
  tvm::runtime::TVMRetValue status = (*tvm::runtime::Registry::Get("vta.simulator.profiler_status"))();
  printf("PROFILE %s\n", status.operator std::string().c_str());
  printf("GEOMETRY %u %u %u\n", unsigned(VTA_BATCH), unsigned(VTA_BLOCK_IN), unsigned(VTA_BLOCK_OUT));
  VTADeviceFree(device);
}
