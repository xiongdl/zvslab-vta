/* Execute tiled int8 convolution through the production VTA driver. */
#include <vta/driver.h>
#include <vta/hw_spec.h>
#include "backend_init.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

constexpr uint32_t kProbeWaitCycles = 100000;

struct Buffer {
  explicit Buffer(size_t bytes) : data(VTAMemAlloc(bytes, VTA_NOT_CACHED)), size(bytes) {
    if (!data) throw std::bad_alloc();
    std::memset(data, 0, size);
  }
  ~Buffer() { VTAMemFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void* data;
  size_t size;
};

static uint32_t ElementBytes(uint32_t type) {
  switch (type) {
    case VTA_MEM_ID_INP: return VTA_INP_ELEM_BYTES;
    case VTA_MEM_ID_WGT: return VTA_WGT_ELEM_BYTES;
    case VTA_MEM_ID_ACC: return VTA_ACC_ELEM_BYTES;
    case VTA_MEM_ID_UOP: return VTA_UOP_ELEM_BYTES;
    case VTA_MEM_ID_OUT: return VTA_OUT_ELEM_BYTES;
    default: throw std::runtime_error("bad memory type");
  }
}

static VTAGenericInsn Memory(uint32_t opcode, uint32_t type, Buffer& buffer,
                             uint32_t count, uint32_t dram_offset = 0,
                             uint32_t sram_base = 0) {
  VTAGenericInsn result{};
  auto& mem = reinterpret_cast<VTAMemInsn&>(result);
  mem.opcode = opcode;
  mem.memory_type = type;
  mem.sram_base = sram_base;
  mem.dram_base = VTAMemGetPhyAddr(buffer.data) / ElementBytes(type) + dram_offset;
  mem.x_size = count;
  mem.y_size = 1;
  mem.x_stride = count;
  return result;
}

static VTAGenericInsn Alu(uint32_t opcode, uint32_t rounding, int32_t immediate,
                          uint32_t begin, uint32_t end, bool use_immediate = false,
                          bool push = false, bool pop = false) {
  VTAGenericInsn result{};
  auto& alu = reinterpret_cast<VTAAluInsn&>(result);
  alu.opcode = VTA_OPCODE_ALU;
  alu.alu_opcode = opcode;
  alu.rounding = rounding;
  alu.imm = immediate;
  alu.use_imm = use_immediate;
  alu.uop_bgn = begin;
  alu.uop_end = end;
  alu.iter_in = alu.iter_out = 1;
  alu.push_next_dep = push;
  alu.pop_prev_dep = pop;
  return result;
}

static void Write(const std::string& path, const void* data, size_t size) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!stream) throw std::runtime_error("cannot write " + path);
}

static int RunDevice(VTADeviceHandle device, Buffer& stream,
                     const std::vector<VTAGenericInsn>& instructions,
                     const char* phase) {
  const bool debug = std::getenv("VTA_QCONV_DEBUG") != nullptr;
  if (debug) {
    std::cerr << "begin " << phase << " instructions=" << instructions.size() << "\n";
    for (size_t i = 0; i < instructions.size(); ++i) {
      const auto& generic = instructions[i];
      const auto& mem = reinterpret_cast<const VTAMemInsn&>(generic);
      std::cerr << i << " op=" << generic.opcode << " type=" << mem.memory_type
                << " popprev=" << mem.pop_prev_dep << " popnext=" << mem.pop_next_dep
                << " pushprev=" << mem.push_prev_dep << " pushnext=" << mem.push_next_dep << "\n";
    }
  }
  const int status = VTADeviceRun(device, VTAMemGetPhyAddr(stream.data),
                                  instructions.size(), kProbeWaitCycles);
  if (debug) std::cerr << "end " << phase << " status=" << status << "\n";
  return status;
}

static void Run(const std::string& fixture_dir, const std::string& mode,
                const std::string& output_dir) {
  CheckBackendABI();
  InitializeSimulatorBackend();
  if (mode != "double" && mode != "single") throw std::runtime_error("mode must be double|single");
  constexpr uint32_t kPixels = 32 * 32;
  constexpr uint32_t kChannels = 16;
  constexpr uint32_t kTaps = 9;
  constexpr uint32_t kTile = 64;
  constexpr uint32_t kAluUopBegin = kTile * kTaps;
  constexpr uint32_t kSnapshotUopBegin = kAluUopBegin + kTile;
  constexpr uint32_t kUopEnd = kSnapshotUopBegin + kTile;
  constexpr uint32_t kRmulUopBegin = 0;
  constexpr uint32_t kRsftUopBegin = kTile;
  constexpr uint32_t kImmediateUopBegin = 2 * kTile;
  constexpr uint32_t kAccParamBase = 2 * kTile;
  constexpr uint32_t kAccCopyBase = 3 * kTile;
  constexpr uint32_t kAccVectors = 4 * kTile;
  constexpr uint32_t kBytes = 8;
  if (VTA_BATCH != 1 || VTA_BLOCK_IN != kBytes || VTA_BLOCK_OUT != kBytes) {
    throw std::runtime_error("conv_probe requires vta_64mac batch=1 and 8x8 GEMM blocks");
  }
  if (VTA_INP_BUFF_DEPTH < kTile * kTaps || VTA_ACC_BUFF_DEPTH < kAccVectors ||
      VTA_UOP_BUFF_DEPTH < kUopEnd) {
    throw std::runtime_error("64mac SRAM cannot fit 64-position qconv tile");
  }
  std::ifstream input_file(fixture_dir + "/fixture.bin", std::ios::binary);
  if (!input_file) throw std::runtime_error("missing fixture.bin");
  std::vector<int8_t> input(kPixels * 3), weights(kChannels * kTaps * 3);
  std::vector<int32_t> bias(kChannels), multiplier(kChannels), shifts(kChannels);
  input_file.read(reinterpret_cast<char*>(input.data()), input.size());
  input_file.read(reinterpret_cast<char*>(weights.data()), weights.size());
  input_file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(int32_t));
  input_file.read(reinterpret_cast<char*>(multiplier.data()), multiplier.size() * sizeof(int32_t));
  input_file.read(reinterpret_cast<char*>(shifts.data()), shifts.size() * sizeof(int32_t));
  if (!input_file || input_file.peek() != EOF) throw std::runtime_error("invalid fixture.bin payload");

  std::vector<int32_t> all_acc(kPixels * kChannels);
  std::vector<int8_t> all_out(kPixels * kChannels);
  const uint32_t inp_vectors = kTile * kTaps;
  Buffer inp(inp_vectors * VTA_INP_ELEM_BYTES);
  Buffer wgt(kTaps * VTA_WGT_ELEM_BYTES);
  Buffer acc(kAccVectors * VTA_ACC_ELEM_BYTES);
  Buffer uop(kUopEnd * VTA_UOP_ELEM_BYTES);
  Buffer out(4 * kTile * VTA_OUT_ELEM_BYTES);
  auto* inp_data = static_cast<int8_t*>(inp.data);
  auto* wgt_data = static_cast<int8_t*>(wgt.data);
  auto* acc_data = static_cast<int32_t*>(acc.data);
  auto* uops = static_cast<VTAUop*>(uop.data);
  auto device = VTADeviceAlloc();
  if (!device) throw std::runtime_error("VTADeviceAlloc failed");

  const char* tile_limit_text = std::getenv("VTA_QCONV_TILE_LIMIT");
  const uint32_t tile_limit = tile_limit_text ? static_cast<uint32_t>(std::strtoul(tile_limit_text, nullptr, 10)) : 0;
  for (uint32_t first = 0; first < kPixels; first += kTile) {
    if (tile_limit && first / kTile >= tile_limit) break;
    const uint32_t count = std::min(kTile, kPixels - first);
    for (uint32_t block = 0; block < 2; ++block) {
      std::memset(inp.data, 0, inp.size);
      std::memset(wgt.data, 0, wgt.size);
      std::memset(acc.data, 0, acc.size);
      std::memset(uop.data, 0, uop.size);
      std::memset(out.data, 0, out.size);
      for (uint32_t p = 0; p < count; ++p) {
        const int32_t oy = (first + p) / 32;
        const int32_t ox = (first + p) % 32;
        for (uint32_t tap = 0; tap < kTaps; ++tap) {
          const int32_t ky = tap / 3, kx = tap % 3;
          const int32_t iy = oy + ky - 1, ix = ox + kx - 1;
          const uint32_t vector = p * kTaps + tap;
          for (uint32_t ic = 0; ic < 3; ++ic) {
            inp_data[vector * kBytes + ic] =
                (iy < 0 || iy >= 32 || ix < 0 || ix >= 32)
                    ? static_cast<int8_t>(-128)
                    : input[(iy * 32 + ix) * 3 + ic];
          }
          uops[p * kTaps + tap].dst_idx = p;
          uops[p * kTaps + tap].src_idx = vector;
          uops[p * kTaps + tap].wgt_idx = tap;
        }
        for (uint32_t lane = 0; lane < kBytes; ++lane) {
          const uint32_t channel = block * kBytes + lane;
          int64_t sum_weight = 0;
          for (uint32_t tap = 0; tap < kTaps; ++tap) {
            for (uint32_t ic = 0; ic < 3; ++ic) {
              const int8_t value = weights[(channel * kTaps + tap) * 3 + ic];
              sum_weight += value;
              const uint32_t wi = tap * VTA_WGT_ELEM_BYTES + lane * kBytes + ic;
              wgt_data[wi] = value;
            }
          }
          const int64_t corrected_bias = static_cast<int64_t>(bias[channel]) + 128 * sum_weight;
          if (corrected_bias < INT32_MIN || corrected_bias > INT32_MAX) {
            throw std::runtime_error("corrected bias exceeds INT32");
          }
          acc_data[p * kBytes + lane] = static_cast<int32_t>(corrected_bias);
          acc_data[(kTile + p) * kBytes + lane] = multiplier[channel];
          acc_data[(2 * kTile + p) * kBytes + lane] = -shifts[channel];
          uops[kAluUopBegin + p].dst_idx = p;
          uops[kAluUopBegin + p].src_idx = kTile + p;
          uops[kAluUopBegin + p].wgt_idx = 0;
        }
      }

      VTAGenericInsn gemm{};
      auto& compute = reinterpret_cast<VTAGemInsn&>(gemm);
      compute.opcode = VTA_OPCODE_GEMM;
      compute.uop_bgn = 0;
      compute.uop_end = count * kTaps;
      compute.iter_in = compute.iter_out = 1;
      compute.pop_prev_dep = 1;
      compute.push_next_dep = 0;
      std::vector<VTAGenericInsn> instructions{
          Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_INP, inp, inp_vectors),
          Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_WGT, wgt, kTaps),
          Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, acc, kAccVectors),
          Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, uop, kSnapshotUopBegin + count),
      };
      reinterpret_cast<VTAMemInsn&>(instructions[1]).push_next_dep = 1;
      instructions.push_back(gemm);

      // Copy raw accumulator to a zeroed scratch bank and export all four bytes.
      for (uint32_t p = 0; p < count; ++p) {
        uops[kAluUopBegin + p].dst_idx = kAccCopyBase + p;
        uops[kAluUopBegin + p].src_idx = p;
        uops[kAluUopBegin + p].wgt_idx = 0;
        uops[kSnapshotUopBegin + p].dst_idx = kAccCopyBase + p;
        uops[kSnapshotUopBegin + p].src_idx = kAccCopyBase + p;
        uops[kSnapshotUopBegin + p].wgt_idx = 0;
      }
      instructions.push_back(Alu(VTA_ALU_OPCODE_ADD, VTA_ALU_ROUND_NONE, 0,
                                 kAluUopBegin, kAluUopBegin + count, false, true, false));
      for (uint32_t byte = 0; byte < 4; ++byte) {
        auto store = Memory(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, out, count,
                            byte * kTile, kAccCopyBase);
        auto& mem = reinterpret_cast<VTAMemInsn&>(store);
        mem.pop_prev_dep = 1;
        mem.push_prev_dep = 1;
        instructions.push_back(store);
        if (byte != 3) {
          auto shift = Alu(VTA_ALU_OPCODE_SHR, VTA_ALU_ROUND_NONE, 8,
                           kSnapshotUopBegin, kSnapshotUopBegin + count, true, true, false);
          auto& alu = reinterpret_cast<VTAAluInsn&>(shift);
          alu.push_next_dep = 1;
          alu.pop_prev_dep = 0;
          alu.pop_next_dep = 1;
          instructions.push_back(shift);
        }
      }
      VTAGenericInsn finish{};
      finish.opcode = VTA_OPCODE_FINISH;
      reinterpret_cast<VTAMemInsn&>(finish).pop_next_dep = 1;
      instructions.push_back(finish);
      Buffer stream(instructions.size() * sizeof(VTAGenericInsn));
      std::memcpy(stream.data, instructions.data(), stream.size);
      int status = RunDevice(device, stream, instructions, "gemm");
      if (status != 0) throw std::runtime_error("VTADeviceRun failed with status " + std::to_string(status));

      const auto* bytes = static_cast<const uint8_t*>(out.data);
      for (uint32_t p = 0; p < count; ++p) {
        const uint32_t pixel = first + p;
        for (uint32_t lane = 0; lane < kBytes; ++lane) {
          uint32_t value = 0;
          for (uint32_t byte = 0; byte < 4; ++byte)
            value |= static_cast<uint32_t>(bytes[(byte * kTile + p) * kBytes + lane]) << (byte * 8);
          all_acc[pixel * kChannels + block * kBytes + lane] = static_cast<int32_t>(value);
        }
      }

      // Requantization runs after the GEMM run boundary and consumes the
      // accumulator read back from the actual GEMM result stores.
      std::memset(acc.data, 0, acc.size);
      std::memset(uop.data, 0, uop.size);
      std::memset(out.data, 0, out.size);
      for (uint32_t p = 0; p < count; ++p) {
        for (uint32_t lane = 0; lane < kBytes; ++lane) {
          const uint32_t channel = block * kBytes + lane;
          acc_data[p * kBytes + lane] = all_acc[(first + p) * kChannels + channel];
          acc_data[(kTile + p) * kBytes + lane] = multiplier[channel];
          acc_data[(2 * kTile + p) * kBytes + lane] = -shifts[channel];
        }
        uops[kRmulUopBegin + p].dst_idx = p;
        uops[kRmulUopBegin + p].src_idx = kTile + p;
        uops[kRsftUopBegin + p].dst_idx = p;
        uops[kRsftUopBegin + p].src_idx = 2 * kTile + p;
        uops[kImmediateUopBegin + p].dst_idx = p;
        uops[kImmediateUopBegin + p].src_idx = p;
      }
      std::vector<VTAGenericInsn> alu_instructions{
          Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, acc, kAccVectors),
          Memory(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, uop, kImmediateUopBegin + count),
      };
      alu_instructions.push_back(Alu(VTA_ALU_OPCODE_RMUL,
          mode == "double" ? VTA_ALU_ROUND_UP : VTA_ALU_ROUND_NONE,
          0, kRmulUopBegin, kRmulUopBegin + count, false, false, false));
      auto rsft = Alu(VTA_ALU_OPCODE_RSFT,
          mode == "double" ? VTA_ALU_ROUND_AWAY : VTA_ALU_ROUND_UP,
          0, kRsftUopBegin, kRsftUopBegin + count, false, false, false);
      alu_instructions.push_back(rsft);
      alu_instructions.push_back(Alu(VTA_ALU_OPCODE_ADD, VTA_ALU_ROUND_NONE, -128,
                                     kImmediateUopBegin, kImmediateUopBegin + count,
                                     true, false, false));
      alu_instructions.push_back(Alu(VTA_ALU_OPCODE_MIN, VTA_ALU_ROUND_NONE, 0,
                                     kImmediateUopBegin, kImmediateUopBegin + count,
                                     true, false, false));
      alu_instructions.push_back(Alu(VTA_ALU_OPCODE_MAX, VTA_ALU_ROUND_NONE, -128,
                                     kImmediateUopBegin, kImmediateUopBegin + count,
                                     true, true, false));
      auto output_store = Memory(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, out, count, 0, 0);
      auto& output_fields = reinterpret_cast<VTAMemInsn&>(output_store);
      output_fields.pop_prev_dep = 1;
      output_fields.push_prev_dep = 1;
      alu_instructions.push_back(output_store);
      VTAGenericInsn alu_finish{};
      alu_finish.opcode = VTA_OPCODE_FINISH;
      reinterpret_cast<VTAMemInsn&>(alu_finish).pop_next_dep = 1;
      alu_instructions.push_back(alu_finish);
      Buffer alu_stream(alu_instructions.size() * sizeof(VTAGenericInsn));
      std::memcpy(alu_stream.data, alu_instructions.data(), alu_stream.size);
      status = RunDevice(device, alu_stream, alu_instructions, "alu");
      if (status != 0) throw std::runtime_error("VTA ALU VTADeviceRun failed with status " + std::to_string(status));
      const auto* final_bytes = static_cast<const uint8_t*>(out.data);
      for (uint32_t p = 0; p < count; ++p)
        for (uint32_t lane = 0; lane < kBytes; ++lane)
          all_out[(first + p) * kChannels + block * kBytes + lane] =
              static_cast<int8_t>(final_bytes[p * kBytes + lane]);
    }
  }
  VTADeviceFree(device);
  Write(output_dir + "/accumulator.bin", all_acc.data(), all_acc.size() * sizeof(int32_t));
  Write(output_dir + "/output.bin", all_out.data(), all_out.size());
}

int main(int argc, char** argv) {
  try {
    std::string fixture, mode, output;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--fixture" && i + 1 < argc) fixture = argv[++i];
      else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
      else if (arg == "--output-dir" && i + 1 < argc) output = argv[++i];
      else throw std::runtime_error("usage: conv_probe --fixture DIR --mode double|single --output-dir DIR");
    }
    if (fixture.empty() || mode.empty() || output.empty())
      throw std::runtime_error("usage: conv_probe --fixture DIR --mode double|single --output-dir DIR");
    Run(fixture, mode, output);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
