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

/*!
 * \file vta/runtime.h
 * \brief Stable VTA runtime C interface.
 */

#ifndef VTA_RUNTIME_H_
#define VTA_RUNTIME_H_

#include <stdint.h>

/*! Micro-op modes passed to VTAUopPush; independent of ISA opcodes. */
#define VTA_UOP_MODE_GEMM 0
#define VTA_UOP_MODE_ALU 1
#define VTA_UOP_MODE_DWC 2

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Check that a generated artifact uses the active VTA runtime ABI.
 * \param expected_fingerprint Fingerprint embedded in the generated artifact.
 * \return 0 on a match, or -1 with TVM's last error set on a mismatch.
 */
int VTACheckConfig(uint64_t expected_fingerprint);

/*! Legacy micro-op push, equivalent to VTAUopPushEx with rounding=NONE. */
void VTAUopPush(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
                uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val);

/*! Push one micro-op with an explicit ALU rounding mode. */
void VTAUopPushEx(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
                  uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val,
                  uint32_t rounding);

/*! Push a legacy ALU kernel. Its cache identity remains the caller signature. */
int VTAPushALUOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes);
/*! Push an ALU kernel keyed by caller signature, expected opcode, and rounding.
 *  The initializer must emit the expected opcode and rounding mode.
 */
int VTAPushALUOpEx(void** uop_handle, int (*finit)(void*), void* signature, int nbytes,
                   uint32_t rounding, uint32_t expected_opcode);

#ifdef __cplusplus
}
#endif

#endif  // VTA_RUNTIME_H_
