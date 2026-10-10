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
#include <stdlib.h>
#include <string.h>
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/* Keep the scalar implementation in the pinned CMSIS-NN header as the oracle. */
int32_t cmsis_requantize(int32_t x, int32_t multiplier, int32_t shift) {
  return arm_nn_requantize(x, multiplier, shift);
}

/* Stable Task 3 entry point. Task 2 intentionally does not run convolution. */
int cmsis_conv2d(const int8_t *input, const int8_t *weights, const int32_t *bias,
                 int32_t *multiplier, int32_t *shift,
                 int32_t *accumulator, int8_t *output) {
  const cmsis_nn_dims input_dims = {1, 32, 32, 3};
  const cmsis_nn_dims filter_dims = {16, 3, 3, 3};
  const cmsis_nn_dims bias_dims = {16, 1, 1, 1};
  const cmsis_nn_dims output_dims = {1, 32, 32, 16};
  const cmsis_nn_dims upscale_dims = {1, 1, 1, 1};
  const cmsis_nn_conv_params conv = {
      .input_offset = 128, .output_offset = -128,
      .stride = {1, 1}, .padding = {1, 1}, .dilation = {1, 1},
      .activation = {-128, 0},
  };
  const cmsis_nn_per_channel_quant_params quant = {multiplier, shift};
  int32_t bytes = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
  void *scratch = bytes > 0 ? malloc((size_t)bytes) : NULL;
  const cmsis_nn_context context = {scratch, bytes};
  if (bytes > 0 && scratch == NULL) return -2;
  const arm_cmsis_nn_status status = arm_convolve_s8(
      &context, &conv, &quant, &input_dims, input, &filter_dims, weights,
      &bias_dims, bias, &upscale_dims, &output_dims, output);
  free(scratch);
  if (status != ARM_CMSIS_NN_SUCCESS) return (int)status;

  /* Independent int64 observer: CMSIS uses x + input_offset, with padded
     input equal to -input_offset. Verify every result fits its int32 kernel. */
  for (int y = 0; y < 32; ++y) {
    for (int x = 0; x < 32; ++x) {
      for (int oc = 0; oc < 16; ++oc) {
        int64_t sum = bias[oc];
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            const int iy = y + ky - 1;
            const int ix = x + kx - 1;
            for (int ic = 0; ic < 3; ++ic) {
              int32_t value = -128;
              if (iy >= 0 && iy < 32 && ix >= 0 && ix < 32) {
                value = input[(iy * 32 + ix) * 3 + ic];
              }
              value += 128;
              const int wi = ((oc * 3 + ky) * 3 + kx) * 3 + ic;
              sum += (int64_t)value * weights[wi];
            }
          }
        }
        if (sum < INT32_MIN || sum > INT32_MAX) return -3;
        accumulator[(y * 32 + x) * 16 + oc] = (int32_t)sum;
      }
    }
  }
  return 0;
}
