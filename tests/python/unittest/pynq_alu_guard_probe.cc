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

#include <stdio.h>
#include <string.h>
#include "pynq_insn_guard.h"

int main() {
  VTAInsn stream[5] = {};
  for (uint32_t opcode = VTA_ALU_OPCODE_MIN; opcode <= VTA_ALU_OPCODE_MUL; ++opcode) {
    stream[opcode].alu.opcode = VTA_OPCODE_ALU;
    stream[opcode].alu.alu_opcode = opcode;
    stream[opcode].alu.rounding = VTA_ALU_ROUND_NONE;
  }
  uint32_t index = 99;
  uint32_t opcode = 99;
  uint32_t rounding = 99;
  if (!VTAValidatePynqInsnStream(stream, 5, &index, &opcode, &rounding)) {
    fprintf(stderr, "legacy ALU stream was rejected\n");
    return 1;
  }

  VTAInsn unsupported = {};
  unsupported.alu.opcode = VTA_OPCODE_ALU;
  unsupported.alu.alu_opcode = VTA_ALU_OPCODE_RMUL;
  if (VTAValidatePynqInsnStream(&unsupported, 1, &index, &opcode, &rounding) ||
      index != 0 || opcode != VTA_ALU_OPCODE_RMUL || rounding != VTA_ALU_ROUND_NONE) {
    fprintf(stderr, "unsupported RMUL was not reported\n");
    return 2;
  }
  unsupported.alu.alu_opcode = VTA_ALU_OPCODE_MUL;
  unsupported.alu.rounding = 1;
  if (VTAValidatePynqInsnStream(&unsupported, 1, &index, &opcode, &rounding) ||
      opcode != VTA_ALU_OPCODE_MUL || rounding != 1) {
    fprintf(stderr, "nonzero rounding was not reported\n");
    return 3;
  }

  const vta_phy_addr_t allocation = 0x1000;
  if (!VTAIsPynqCmaRangeValid(allocation, 64, allocation + 16, 48) ||
      !VTAIsPynqCmaRangeValid(allocation, 64, allocation + 64, 0) ||
      VTAIsPynqCmaRangeValid(allocation, 64, allocation - 1, 1) ||
      VTAIsPynqCmaRangeValid(allocation, 64, allocation + 63, 2)) {
    fprintf(stderr, "CMA instruction range boundary check failed\n");
    return 4;
  }
  puts("PYNQ raw ALU guard passed: legacy ALU, RMUL, rounding, CMA bounds");
  return 0;
}
