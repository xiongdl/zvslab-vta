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


// Exercise the real runtime at its device-driver boundary, without numerical FSIM.
#include "../../../src/runtime/runtime.h"
#include <vta/hw_spec.h>
#include <vta/sim_tlpp.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

static std::vector<VTAGenericInsn> captured;
static std::vector<unsigned> executed;
static std::map<vta_phy_addr_t, void*> physical;
static vta_phy_addr_t next_addr = 4096;

// Replace only the transport: instruction generation and TLPP are production code.
extern "C" {
VTADeviceHandle VTADeviceAlloc() { return reinterpret_cast<void*>(1); }
void VTADeviceFree(VTADeviceHandle) {}
void* VTAMemAlloc(size_t size, int) { return calloc(1, size); }
void VTAMemFree(void* ptr) { free(ptr); }
vta_phy_addr_t VTAMemGetPhyAddr(void* ptr) {
  auto addr = next_addr;
  next_addr += (1 << 26);
  physical[addr] = ptr;
  return addr;
}
void VTAMemCopyFromHost(void* dst, const void* src, size_t size) { memcpy(dst, src, size); }
void VTAMemCopyToHost(void* dst, const void* src, size_t size) { memcpy(dst, src, size); }
void VTAFlushCache(void*, vta_phy_addr_t, int) {}
void VTAInvalidateCache(void*, vta_phy_addr_t, int) {}
int VTADeviceRun(VTADeviceHandle, vta_phy_addr_t addr, uint32_t count, uint32_t) {
  auto* insns = static_cast<VTAGenericInsn*>(physical.at(addr));
  captured.assign(insns, insns + count);
  return 0;
}
}

static int mode;
static int tap_mode;

static int Initialize(void* signature) {
  int reset = *static_cast<int*>(signature);
  VTAUopLoopBegin(3, 11, 13, 17);
  VTAUopLoopBegin(4, 19, 23, 29);
  VTAUopPush(mode, reset, 7, 5, 3, VTA_ALU_OPCODE_ADD, 1, -2);
  VTAUopLoopEnd();
  VTAUopLoopEnd();
  return 0;
}

static int InitializeTaps(void*) {
  for (unsigned k = 0; k < 9; ++k) {
    VTAUopPush(tap_mode, 0, 7, k, k / VTA_BLOCK_IN, 0, 0, 0);
  }
  return 0;
}

static void Observe(const VTAGenericInsn* insn, void*) {
  executed.push_back(reinterpret_cast<const VTAMemInsn*>(insn)->opcode);
}

int main(int argc, char** argv) {
  assert(argc == 3);
  mode = atoi(argv[1]);
  bool serial = atoi(argv[2]);
  void* handle = nullptr;
  auto cmd = VTATLSCommandHandle();
  VTASetDebugMode(cmd, VTA_DEBUG_DUMP_INSN | (serial ? VTA_DEBUG_FORCE_SERIAL : 0));
  void* buffer = VTABufferAlloc(4096);
  VTALoadBuffer2D(cmd, buffer, 0, 1, 1, 1, 0, 0, 0, 0, 0, VTA_MEM_ID_WGT);
  VTADepPush(cmd, 1, 2);
  VTADepPop(cmd, 1, 2);
  int reset = 0;
  auto push = (mode == 1 || mode == 5) ? VTAPushALUOp : VTAPushGEMMOp;
  // Modes 3/4/5 exercise repeated destinations for DwC/GEMM/ALU respectively.
  tap_mode = mode == 3 ? 2 : (mode == 4 ? 0 : 1);
  if (mode >= 3) {
    push(&handle, InitializeTaps, nullptr, 0);
  } else {
    push(&handle, Initialize, &reset, sizeof(reset));
  }
  VTADepPush(cmd, 2, 1);
  VTADepPush(cmd, 2, 3);
  VTADepPop(cmd, 2, 3);
  VTAStoreBuffer2D(cmd, 7, VTA_MEM_ID_OUT, buffer, 0, 1, 1, 1);
  if (mode < 3) {
    reset = 1;
    push(&handle, Initialize, &reset, sizeof(reset));
  }
  VTASynchronize(cmd, 100);

  unsigned count = 0;
  for (const auto& insn : captured) {
    const auto& mem = reinterpret_cast<const VTAMemInsn&>(insn);
    if (mem.opcode == 2 || mem.opcode == 4 || mem.opcode == 5) {
      const auto& gem = reinterpret_cast<const VTAGemInsn&>(insn);
      printf("RESULT %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
             unsigned(gem.opcode), unsigned(gem.reset_reg), unsigned(gem.uop_end - gem.uop_bgn),
             unsigned(gem.iter_out), unsigned(gem.iter_in), unsigned(gem.dst_factor_out),
             unsigned(gem.src_factor_out), mem.opcode == 4 ? 0 : unsigned(gem.wgt_factor_out),
             unsigned(gem.dst_factor_in), unsigned(gem.src_factor_in),
             mem.opcode == 4 ? 0 : unsigned(gem.wgt_factor_in), unsigned(gem.pop_prev_dep),
             unsigned(gem.pop_next_dep), unsigned(gem.push_prev_dep), unsigned(gem.push_next_dep));
      ++count;
    }
  }
  assert(count == (mode >= 3 ? 1 : 2));
  if (mode == 3) {
    bool checked = false;
    for (const auto& insn : captured) {
      const auto& mem = reinterpret_cast<const VTAMemInsn&>(insn);
      if (mem.opcode == VTA_OPCODE_LOAD && mem.memory_type == VTA_MEM_ID_UOP && mem.x_size) {
        auto* uops = static_cast<VTAUop*>(physical.at(mem.dram_base * VTA_UOP_ELEM_BYTES));
        assert(mem.x_size == 9);
        for (unsigned k = 0; k < 9; ++k) {
          assert(uops[k].dst_idx == 7 && uops[k].src_idx == k);
          assert(uops[k].wgt_idx == k / VTA_BLOCK_IN);
        }
        checked = true;
      }
    }
    assert(checked);
  }
  assert(reinterpret_cast<const VTAMemInsn&>(captured.back()).opcode == VTA_OPCODE_FINISH);
  for (const auto& insn : captured) TlppVerify::Global()->TlppPushInsn(&insn);
  TlppVerify::Global()->TlppSynchronization(Observe, nullptr);
  assert(executed.size() == captured.size());
  unsigned compute = 0, store = 0;
  for (unsigned i = 0; i < executed.size(); ++i) {
    if ((executed[i] == 2 || executed[i] == 4 || executed[i] == 5) && !compute) compute = i;
    if (executed[i] == VTA_OPCODE_STORE && !store) store = i;
  }
  assert(compute < store);
  VTABufferFree(buffer);
  VTARuntimeShutdown();
}
