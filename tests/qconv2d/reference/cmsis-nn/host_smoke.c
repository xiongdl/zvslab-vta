#include <stdio.h>
#include <stdlib.h>
#include "arm_nnfunctions.h"
int main(void) {
  const cmsis_nn_dims input = {1, 1, 1, 1};
  const cmsis_nn_dims filter = {1, 1, 1, 1};
  const cmsis_nn_dims output = {1, 1, 1, 1};
  const cmsis_nn_dims bias_dims = {1, 1, 1, 1};
  const cmsis_nn_dims upscale = {1, 1, 1, 1};
  cmsis_nn_conv_params conv = {0};
  conv.stride.w = conv.stride.h = 1;
  conv.dilation.w = conv.dilation.h = 1;
  conv.activation.min = -128; conv.activation.max = 127;
  int32_t mult[1] = {1073741824}, shift[1] = {0}, bias[1] = {0};
  cmsis_nn_per_channel_quant_params q = {mult, shift};
  int8_t x[1] = {2}, w[1] = {3}, y[1] = {0};
  int32_t n = arm_convolve_s8_get_buffer_size(&input, &filter);
  size_t size = n > 0 ? (size_t)n : 16;
  void *scratch = calloc(1, size);
  cmsis_nn_context ctx = {scratch, (int32_t)size};
  arm_cmsis_nn_status status = arm_convolve_s8(&ctx, &conv, &q, &input, x, &filter, w, &bias_dims, bias, &upscale, &output, y);
  printf("status=%d output=%d scratch=%d\n", status, y[0], n);
  free(scratch);
  return status != ARM_CMSIS_NN_SUCCESS;
}
