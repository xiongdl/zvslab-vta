// Runtime instruction probe is intentionally compiled against the production runtime.
#include "../../src/runtime/runtime.h"
#include <vta/hw_spec.h>
#include <vta/sim_tlpp.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static std::vector<VTAGenericInsn> captured;
static std::map<vta_phy_addr_t, void*> physical;
static vta_phy_addr_t next_addr = 4096;
extern "C" {
VTADeviceHandle VTADeviceAlloc() { return reinterpret_cast<void*>(1); }
void VTADeviceFree(VTADeviceHandle) {}
void* VTAMemAlloc(size_t size, int) { return calloc(1, size); }
void VTAMemFree(void* ptr) { free(ptr); }
vta_phy_addr_t VTAMemGetPhyAddr(void* ptr) {
  auto addr = next_addr; next_addr += (1 << 26); physical[addr] = ptr; return addr;
}
void VTAMemCopyFromHost(void* dst, const void* src, size_t size) { memcpy(dst, src, size); }
void VTAMemCopyToHost(void* dst, const void* src, size_t size) { memcpy(dst, src, size); }
void VTAFlushCache(void*, vta_phy_addr_t, int) {}
void VTAInvalidateCache(void*, vta_phy_addr_t, int) {}
int VTADeviceRun(VTADeviceHandle, vta_phy_addr_t addr, uint32_t count, uint32_t) {
  auto* insns = static_cast<VTAGenericInsn*>(physical.at(addr));
  captured.assign(insns, insns + count); return 0;
}
}

static uint32_t opcode, use_imm, rounding;
static int32_t immediate;
static int expected_slot;
static void check_signature(void* signature) {
  assert(*static_cast<int*>(signature) == expected_slot);
}
static int init(void* signature) {
  check_signature(signature);
  VTAUopPushEx(VTA_UOP_MODE_ALU, 1, 1, 0, 0, opcode, use_imm, immediate, rounding);
  return 0;
}
static int init_bad_legacy_rounding(void* signature) {
  check_signature(signature);
  VTAUopPushEx(VTA_UOP_MODE_ALU, 1, 1, 0, 0, opcode, use_imm, immediate, rounding);
  return 0;
}
static int init_legacy(void* signature) {
  check_signature(signature);
  VTAUopPush(VTA_UOP_MODE_ALU, 1, 1, 0, 0, opcode, use_imm, immediate);
  return 0;
}
static int init_mixed(void* signature) {
  check_signature(signature);
  VTAUopPush(VTA_UOP_MODE_ALU, 1, 1, 0, 0, opcode, use_imm, immediate);
  VTAUopPushEx(VTA_UOP_MODE_ALU, 1, 2, 0, 0, opcode, use_imm, immediate, rounding);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 5) return 2;
  const char* kind = argv[1]; opcode = std::strtoul(argv[2], nullptr, 0);
  rounding = std::strtoul(argv[3], nullptr, 0); int slot = std::atoi(argv[4]);
  expected_slot = slot;
  use_imm = 1; immediate = 37;
  auto cmd = VTATLSCommandHandle(); void* handle = nullptr;
  if (!std::strcmp(kind, "cache")) {
    rounding = 1;
    VTAPushALUOpEx(&handle, init, &slot, sizeof(slot), rounding);
    rounding = 2;
    VTAPushALUOpEx(&handle, init, &slot, sizeof(slot), rounding);
  } else if (!std::strcmp(kind, "mixed")) {
    VTAPushALUOpEx(&handle, init_mixed, &slot, sizeof(slot), rounding);
  } else if (!std::strcmp(kind, "legacy-mismatch")) {
    VTAPushALUOp(&handle, init_bad_legacy_rounding, &slot, sizeof(slot));
  } else if (!std::strcmp(kind, "expected-mismatch")) {
    VTAPushALUOpEx(&handle, init_legacy, &slot, sizeof(slot), rounding);
  } else if (!std::strcmp(kind, "legacy")) {
    VTAPushALUOp(&handle, init_legacy, &slot, sizeof(slot));
  } else {
    VTAPushALUOpEx(&handle, init, &slot, sizeof(slot), rounding);
  }
  VTASynchronize(cmd, 0);
  for (const auto& item : captured) {
    VTAInsn insn; insn.generic = item;
    if (insn.generic.opcode == VTA_OPCODE_ALU) {
      uint64_t words[2] = {}; std::memcpy(words, &insn, sizeof(insn));
      std::printf("%llu %llu %u %u %lld\n", (unsigned long long)words[0],
                  (unsigned long long)words[1], (unsigned)insn.alu.alu_opcode,
                  (unsigned)insn.alu.rounding, (long long)insn.alu.imm);
    }
  }
  return 0;
}
