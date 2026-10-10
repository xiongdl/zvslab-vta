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
 *
 * \file pynq_driver.c
 * \brief VTA driver for Zynq SoC boards with Pynq support (see pynq.io).
 */

#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <map>
#include <mutex>
#include <thread>
#include <time.h>
#include "pynq_driver.h"
#include "pynq_insn_guard.h"

namespace {
struct CmaAllocation {
  void* virtual_address;
  size_t size;
};

std::map<vta_phy_addr_t, CmaAllocation> cma_allocations;
std::mutex cma_allocations_mutex;

void* FindCmaAllocation(vta_phy_addr_t physical_address, size_t size) {
  std::lock_guard<std::mutex> lock(cma_allocations_mutex);
  auto allocation = cma_allocations.upper_bound(physical_address);
  if (allocation == cma_allocations.begin()) return nullptr;
  --allocation;
  if (!VTAIsPynqCmaRangeValid(allocation->first, allocation->second.size,
                              physical_address, size)) {
    return nullptr;
  }
  const uint64_t offset = static_cast<uint64_t>(physical_address) -
                          static_cast<uint64_t>(allocation->first);
  return static_cast<uint8_t*>(allocation->second.virtual_address) + offset;
}
}  // namespace


void* VTAMemAlloc(size_t size, int cached) {
  assert(size <= VTA_MAX_XFER);
  // Rely on the pynq-specific cma library
  void* buf = cma_alloc(size, cached);
  if (buf != nullptr) {
    const vta_phy_addr_t physical_address = cma_get_phy_addr(buf);
    std::lock_guard<std::mutex> lock(cma_allocations_mutex);
    cma_allocations[physical_address] = {buf, size};
  }
  return buf;
}

void VTAMemFree(void* buf) {
  // Rely on the pynq-specific cma library
  if (buf != nullptr) {
    const vta_phy_addr_t physical_address = cma_get_phy_addr(buf);
    std::lock_guard<std::mutex> lock(cma_allocations_mutex);
    cma_allocations.erase(physical_address);
  }
  cma_free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return cma_get_phy_addr(buf);
}

void VTAMemCopyFromHost(void* dst, const void* src, size_t size) {
  // For SoC-based FPGAs that used shared memory with the CPU, use memcopy()
  memcpy(dst, src, size);
}

void VTAMemCopyToHost(void* dst, const void* src, size_t size) {
  // For SoC-based FPGAs that used shared memory with the CPU, use memcopy()
  memcpy(dst, src, size);
}

void VTAFlushCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  // Call the cma_flush_cache on the CMA buffer
  // so that the FPGA can read the buffer data.
  cma_flush_cache(vir_addr, phy_addr, size);
}

void VTAInvalidateCache(void* vir_addr, vta_phy_addr_t phy_addr, int size) {
  // Call the cma_invalidate_cache on the CMA buffer
  // so that the host needs to read the buffer data.
  cma_invalidate_cache(vir_addr, phy_addr, size);
}

void *VTAMapRegister(uint32_t addr) {
  // Align the base address with the pages
  uint32_t virt_base = addr & ~(getpagesize() - 1);
  // Calculate base address offset w.r.t the base address
  uint32_t virt_offset = addr - virt_base;
  // Open file and mmap
  uint32_t mmap_file = open("/dev/mem", O_RDWR|O_SYNC);
  return mmap(NULL,
              (VTA_IP_REG_MAP_RANGE + virt_offset),
              PROT_READ|PROT_WRITE,
              MAP_SHARED,
              mmap_file,
              virt_base);
}

void VTAUnmapRegister(void *vta) {
  // Unmap memory
  int status = munmap(vta, VTA_IP_REG_MAP_RANGE);
  assert(status == 0);
}

void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val) {
  *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset)) = val;
}

uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset) {
  return *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset));
}

class VTADevice {
 public:
  VTADevice() {
    // VTA stage handles
    vta_fetch_handle_ = VTAMapRegister(VTA_FETCH_ADDR);
    vta_load_handle_ = VTAMapRegister(VTA_LOAD_ADDR);
    vta_compute_handle_ = VTAMapRegister(VTA_COMPUTE_ADDR);
    vta_store_handle_ = VTAMapRegister(VTA_STORE_ADDR);
  }

  ~VTADevice() {
    // Close VTA stage handle
    VTAUnmapRegister(vta_fetch_handle_);
    VTAUnmapRegister(vta_load_handle_);
    VTAUnmapRegister(vta_compute_handle_);
    VTAUnmapRegister(vta_store_handle_);
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    const uint64_t insn_bytes_64 =
        static_cast<uint64_t>(insn_count) * VTA_INS_ELEM_BYTES;
    if (insn_bytes_64 > SIZE_MAX) {
      fprintf(stderr, "PYNQ VTADeviceRun: instruction stream size overflows size_t\n");
      return -1;
    }
    const size_t insn_bytes = static_cast<size_t>(insn_bytes_64);
    void* insns = FindCmaAllocation(insn_phy_addr, insn_bytes);
    if (insns == nullptr) {
      fprintf(stderr,
              "PYNQ VTADeviceRun: instruction stream is outside a tracked CMA allocation\n");
      return -1;
    }
    uint32_t invalid_index = 0;
    uint32_t invalid_opcode = 0;
    uint32_t invalid_rounding = 0;
    if (!VTAValidatePynqInsnStream(insns, insn_count, &invalid_index,
                                  &invalid_opcode, &invalid_rounding)) {
      fprintf(stderr,
              "PYNQ VTADeviceRun: rejecting unsupported ALU instruction %u "
              "(opcode=%u, rounding=%u); legacy Xilinx hardware supports "
              "ALU opcodes MIN/MAX/ADD/SHIFT/MUL with rounding=0\n",
              invalid_index, invalid_opcode, invalid_rounding);
      return -1;
    }
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_COUNT_OFFSET, insn_count);
    VTAWriteMappedReg(vta_fetch_handle_, VTA_FETCH_INSN_ADDR_OFFSET, insn_phy_addr);
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_INP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_load_handle_, VTA_LOAD_WGT_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_UOP_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_compute_handle_, VTA_COMPUTE_BIAS_ADDR_OFFSET, 0);
    VTAWriteMappedReg(vta_store_handle_, VTA_STORE_OUT_ADDR_OFFSET, 0);

    // VTA start
    VTAWriteMappedReg(vta_fetch_handle_, 0x0, VTA_START);
    VTAWriteMappedReg(vta_load_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_compute_handle_, 0x0, VTA_AUTORESTART);
    VTAWriteMappedReg(vta_store_handle_, 0x0, VTA_AUTORESTART);

    // Allow device to respond
    struct timespec ts = { .tv_sec = 0, .tv_nsec = 1000 };
    nanosleep(&ts, &ts);

    // Loop until the VTA is done
    unsigned t, flag = 0;
    for (t = 0; t < wait_cycles; ++t) {
      flag = VTAReadMappedReg(vta_compute_handle_, VTA_COMPUTE_DONE_RD_OFFSET);
      if (flag == VTA_DONE) break;
      std::this_thread::yield();
    }
    // Report error if timeout
    return t < wait_cycles ? 0 : 1;
  }

 private:
  // VTA handles (register maps)
  void* vta_fetch_handle_{nullptr};
  void* vta_load_handle_{nullptr};
  void* vta_compute_handle_{nullptr};
  void* vta_store_handle_{nullptr};
};

VTADeviceHandle VTADeviceAlloc() {
  return new VTADevice();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<VTADevice*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<VTADevice*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}
