/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// File-driven ALU execution through the production VTA driver and FSIM library.
#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Stage {
  uint32_t opcode;
  uint32_t rounding;
  uint32_t use_imm;
  int32_t immediate;
};

struct Case {
  int32_t input;
  std::vector<int32_t> operands;
};

class Buffer {
 public:
  explicit Buffer(size_t bytes) : data(VTAMemAlloc(bytes, VTA_NOT_CACHED)), size(bytes) {
    if (data == nullptr) throw std::bad_alloc();
    std::memset(data, 0, size);
  }
  ~Buffer() { VTAMemFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void* data;
  size_t size;
};

static uint32_t ElementBytes(uint32_t memory_type) {
  switch (memory_type) {
    case VTA_MEM_ID_ACC: return VTA_ACC_ELEM_BYTES;
    case VTA_MEM_ID_UOP: return VTA_UOP_ELEM_BYTES;
    case VTA_MEM_ID_OUT: return VTA_OUT_ELEM_BYTES;
    default: throw std::runtime_error("unsupported probe memory type");
  }
}

static VTAGenericInsn Memory(uint32_t opcode, uint32_t memory_type, Buffer& buffer,
                             uint32_t count, uint32_t dram_offset = 0,
                             uint32_t sram_base = 0) {
  VTAGenericInsn insn{};
  auto& mem = reinterpret_cast<VTAMemInsn&>(insn);
  mem.opcode = opcode;
  mem.memory_type = memory_type;
  mem.sram_base = sram_base;
  mem.dram_base = VTAMemGetPhyAddr(buffer.data) / ElementBytes(memory_type) + dram_offset;
  mem.x_size = count;
  mem.y_size = 1;
  mem.x_stride = count;
  return insn;
}

static VTAGenericInsn Alu(uint32_t opcode, uint32_t rounding, uint32_t use_imm,
                          int32_t immediate, uint32_t uop_begin, uint32_t uop_end,
                          bool push_store = false, bool pop_store = false) {
  const int64_t immediate_min = -(int64_t{1} << (VTA_ALUOP_IMM_BIT_WIDTH - 1));
  const int64_t immediate_max = (int64_t{1} << (VTA_ALUOP_IMM_BIT_WIDTH - 1)) - 1;
  if (use_imm && (immediate < immediate_min || immediate > immediate_max)) {
    throw std::out_of_range("VTA ALU immediate must fit signed 16-bit instruction field");
  }
  VTAGenericInsn insn{};
  auto& alu = reinterpret_cast<VTAAluInsn&>(insn);
  alu.opcode = VTA_OPCODE_ALU;
  alu.pop_next_dep = pop_store;
  alu.push_next_dep = push_store;
  alu.uop_bgn = uop_begin;
  alu.uop_end = uop_end;
  alu.iter_in = 1;
  alu.iter_out = 1;
  alu.alu_opcode = opcode;
  alu.use_imm = use_imm;
  alu.imm = immediate;
  alu.rounding = rounding;
  return insn;
}

static std::vector<int32_t> ReadCases(const std::string& input_path,
                                      std::vector<Stage>* stages) {
  std::ifstream input(input_path);
  std::string magic;
  uint64_t case_count = 0;
  uint32_t stage_count = 0;
  if (!(input >> magic) || magic != "VTA_ALU_PROBE_V1" || !(input >> case_count >> stage_count)) {
    throw std::runtime_error("invalid VTA_ALU_PROBE_V1 header");
  }
  if (case_count == 0 || stage_count == 0 || case_count > std::numeric_limits<uint32_t>::max() ||
      stage_count > 32) {
    throw std::runtime_error("invalid probe case or stage count");
  }
  stages->resize(stage_count);
  for (Stage& stage : *stages) {
    if (!(input >> stage.opcode >> stage.rounding >> stage.use_imm >> stage.immediate)) {
      throw std::runtime_error("truncated ALU stage table");
    }
  }
  std::vector<int32_t> inputs(case_count * (stage_count + 1));
  for (uint64_t index = 0; index < case_count; ++index) {
    if (!(input >> inputs[index * (stage_count + 1)])) {
      throw std::runtime_error("truncated ALU case table");
    }
    for (uint32_t stage = 0; stage < stage_count; ++stage) {
      if (!(input >> inputs[index * (stage_count + 1) + stage + 1])) {
        throw std::runtime_error("truncated ALU operand table");
      }
    }
  }
  std::string trailing;
  if (input >> trailing) throw std::runtime_error("unexpected data after ALU cases");
  return inputs;
}

static void AppendSnapshot(std::vector<VTAGenericInsn>* instructions, Buffer& output,
                          uint32_t vector_count, uint32_t total_vectors,
                          uint32_t global_vector, uint32_t scratch_base,
                          uint32_t uop_begin, uint32_t uop_end) {
  for (uint32_t byte = 0; byte < 4; ++byte) {
    const bool has_shift_after = byte < 3;
    VTAGenericInsn store = Memory(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, output, vector_count,
                                  byte * total_vectors + global_vector, scratch_base);
    auto& store_fields = reinterpret_cast<VTAMemInsn&>(store);
    store_fields.pop_prev_dep = 1;
    store_fields.push_prev_dep = 1;
    instructions->push_back(store);
    if (has_shift_after) {
      instructions->push_back(Alu(VTA_ALU_OPCODE_SHR, VTA_ALU_ROUND_NONE, 1, 8,
                                  uop_begin, uop_end, true, true));
    }
  }
}

static void Execute(const std::string& input_path, const std::string& output_path) {
  std::vector<Stage> stages;
  std::vector<int32_t> input_values = ReadCases(input_path, &stages);
  const uint32_t stage_count = static_cast<uint32_t>(stages.size());
  const uint32_t lane_count = VTA_BATCH * VTA_BLOCK_OUT;
  const uint32_t total_cases = static_cast<uint32_t>(input_values.size() / (stage_count + 1));
  const uint32_t total_vectors = (total_cases + lane_count - 1) / lane_count;
  const uint32_t vectors_per_chunk = std::min<uint32_t>(
      VTA_ACC_BUFF_DEPTH / 3, VTA_UOP_BUFF_DEPTH / 2);
  if (vectors_per_chunk == 0) throw std::runtime_error("VTA SRAM geometry cannot fit ALU probe");

  Buffer accumulator((stage_count + 2) * vectors_per_chunk * VTA_ACC_ELEM_BYTES);
  Buffer microops(2 * vectors_per_chunk * VTA_UOP_ELEM_BYTES);
  auto* acc = static_cast<int32_t*>(accumulator.data);
  auto* uops = static_cast<VTAUop*>(microops.data);
  std::vector<int32_t> results(total_cases);

  const uint32_t scratch_base = 2 * vectors_per_chunk;
  const uint32_t copy_uop_begin = vectors_per_chunk;
  const uint32_t copy_uop_end = 2 * vectors_per_chunk;
  const uint32_t zero_tile_offset = (stage_count + 1) * vectors_per_chunk;
  VTADeviceHandle device = VTADeviceAlloc();
  if (device == nullptr) throw std::runtime_error("VTADeviceAlloc failed");
  uint32_t global_vector = 0;
  while (global_vector < total_vectors) {
    const uint32_t vector_count = std::min(vectors_per_chunk, total_vectors - global_vector);
    std::vector<VTAGenericInsn> instructions;
    Buffer chunk_output(4 * vector_count * VTA_OUT_ELEM_BYTES);
    std::memset(chunk_output.data, 0, chunk_output.size);
    const uint64_t chunk_case_base = static_cast<uint64_t>(global_vector) * lane_count;
    for (uint32_t vector = 0; vector < vector_count; ++vector) {
      uops[vector].dst_idx = vector;
      uops[vector].src_idx = vectors_per_chunk + vector;
      uops[vector].wgt_idx = 0;
      uops[copy_uop_begin + vector].dst_idx = scratch_base + vector;
      uops[copy_uop_begin + vector].src_idx = vector;
      uops[copy_uop_begin + vector].wgt_idx = 0;
      for (uint32_t lane = 0; lane < lane_count; ++lane) {
        const uint64_t case_index = chunk_case_base + static_cast<uint64_t>(vector) * lane_count + lane;
        const uint64_t local_index = static_cast<uint64_t>(vector) * lane_count + lane;
        if (case_index < total_cases) {
          acc[local_index] = input_values[case_index * (stage_count + 1)];
        } else {
          acc[local_index] = 0;
        }
      }
    }
    instructions.push_back(Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, accumulator, vector_count));
    instructions.push_back(Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, microops,
                                  2 * vectors_per_chunk));

    for (uint32_t stage_index = 0; stage_index < stage_count; ++stage_index) {
      const uint32_t source_offset = (stage_index + 1) * vectors_per_chunk;
      for (uint32_t vector = 0; vector < vector_count; ++vector) {
        for (uint32_t lane = 0; lane < lane_count; ++lane) {
          const uint64_t case_index = chunk_case_base + static_cast<uint64_t>(vector) * lane_count + lane;
          const uint64_t local_index = static_cast<uint64_t>(source_offset) * lane_count +
                                       static_cast<uint64_t>(vector) * lane_count + lane;
          acc[local_index] = case_index < total_cases
              ? input_values[case_index * (stage_count + 1) + stage_index + 1] : 0;
        }
      }
      instructions.push_back(Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, accumulator,
                                    vector_count, source_offset, vectors_per_chunk));
      const Stage& stage = stages[stage_index];
      instructions.push_back(Alu(stage.opcode, stage.rounding, stage.use_imm,
                                 stage.immediate, 0, vector_count));
    }

    // ADD reads and writes its destination, so initialize scratch for every chunk
    // before using ADD as a register-to-register copy.
    instructions.push_back(Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, accumulator,
                                  vector_count, zero_tile_offset, scratch_base));
    const uint32_t active_copy_uop_end = copy_uop_begin + vector_count;
    instructions.push_back(Alu(VTA_ALU_OPCODE_ADD, VTA_ALU_ROUND_NONE, 0, 0,
                               copy_uop_begin, active_copy_uop_end, true, false));
    AppendSnapshot(&instructions, chunk_output, vector_count, vector_count, 0,
                   scratch_base, copy_uop_begin, active_copy_uop_end);

    // VTADeviceRun is a completion boundary. Keep each SRAM-reusing chunk in
    // its own run so stores from one chunk cannot race the next chunk's loads.
    VTAGenericInsn finish{};
    finish.opcode = VTA_OPCODE_FINISH;
    reinterpret_cast<VTAMemInsn&>(finish).pop_next_dep = 1;
    instructions.push_back(finish);
    Buffer stream(instructions.size() * sizeof(VTAGenericInsn));
    std::memcpy(stream.data, instructions.data(), instructions.size() * sizeof(VTAGenericInsn));
    int status = VTADeviceRun(device, VTAMemGetPhyAddr(stream.data),
                              static_cast<uint32_t>(instructions.size()), 1000);
    if (status != 0) {
      VTADeviceFree(device);
      throw std::runtime_error("VTADeviceRun failed");
    }

    // Decode the completed local output tile before its DRAM pages are freed.
    const auto* bytes = static_cast<const uint8_t*>(chunk_output.data);
    for (uint32_t local = 0; local < vector_count * lane_count; ++local) {
      const uint64_t case_index = static_cast<uint64_t>(global_vector) * lane_count + local;
      if (case_index >= total_cases) break;
      const uint32_t vector = local / lane_count;
      const uint32_t lane = local % lane_count;
      uint32_t bits = 0;
      for (uint32_t byte = 0; byte < 4; ++byte) {
        const uint64_t offset = static_cast<uint64_t>(byte) * vector_count * lane_count +
                                static_cast<uint64_t>(vector) * lane_count + lane;
        bits |= static_cast<uint32_t>(bytes[offset]) << (byte * 8);
      }
      std::memcpy(&results[case_index], &bits, sizeof(bits));
    }
    global_vector += vector_count;
  }
  VTADeviceFree(device);

  std::ofstream result(output_path);
  if (!result) throw std::runtime_error("unable to open ALU output file");
  for (int32_t value : results) result << value << '\n';
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: alu_probe INPUT.txt OUTPUT.txt\n";
    return 2;
  }
  try {
    Execute(argv[1], argv[2]);
  } catch (const std::exception& error) {
    std::cerr << "alu_probe: " << error.what() << '\n';
    return 2;
  }
  return 0;
}
