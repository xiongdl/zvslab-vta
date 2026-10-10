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

#include <stdint.h>
#include "arm_nnsupportfunctions.h"

/* Keep the scalar implementation in the pinned CMSIS-NN header as the oracle. */
int32_t cmsis_requantize(int32_t x, int32_t multiplier, int32_t shift) {
  return arm_nn_requantize(x, multiplier, shift);
}

/* Stable Task 3 entry point. Task 2 intentionally does not run convolution. */
int cmsis_conv2d(const char *fixture_dir, const char *output_dir) {
  (void)fixture_dir;
  (void)output_dir;
  return -1;
}
