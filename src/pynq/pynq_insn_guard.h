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

#ifndef VTA_PYNQ_PYNQ_INSN_GUARD_H_
#define VTA_PYNQ_PYNQ_INSN_GUARD_H_

#include <stdint.h>
#include <string.h>
#include <vta/driver.h>
#include <vta/hw_spec.h>

inline bool VTAIsPynqCmaRangeValid(vta_phy_addr_t allocation_address,
                                   size_t allocation_size,
                                   vta_phy_addr_t requested_address,
                                   size_t requested_size) {
  if (requested_address < allocation_address) return false;
  const uint64_t offset = static_cast<uint64_t>(requested_address) -
                          static_cast<uint64_t>(allocation_address);
  return offset <= allocation_size && requested_size <= allocation_size - offset;
}

/* Validate the ALU subset implemented by the legacy Xilinx accelerator. */
inline bool VTAValidatePynqInsnStream(const void* stream, uint32_t insn_count,
                                     uint32_t* invalid_index,
                                     uint32_t* invalid_alu_opcode,
                                     uint32_t* invalid_rounding) {
  const uint8_t* bytes = static_cast<const uint8_t*>(stream);
  for (uint32_t index = 0; index < insn_count; ++index) {
    VTAInsn insn;
    memcpy(&insn, bytes + index * VTA_INS_ELEM_BYTES, VTA_INS_ELEM_BYTES);
    if (insn.generic.opcode != VTA_OPCODE_ALU) continue;
    if (insn.alu.alu_opcode > VTA_ALU_OPCODE_MUL ||
        insn.alu.rounding != VTA_ALU_ROUND_NONE) {
      *invalid_index = index;
      *invalid_alu_opcode = insn.alu.alu_opcode;
      *invalid_rounding = insn.alu.rounding;
      return false;
    }
  }
  return true;
}

#endif  // VTA_PYNQ_PYNQ_INSN_GUARD_H_
