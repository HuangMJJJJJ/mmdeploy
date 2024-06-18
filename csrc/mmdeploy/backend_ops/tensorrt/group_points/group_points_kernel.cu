// Copyright (c) OpenMMLab. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <numeric>
#include <vector>

#include "common_cuda_helper.hpp"
#include "group_points_kernel.hpp"
#include "trt_plugin_helper.hpp"

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void group_points_kernel(const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    int b, int c, int n, int npoints,
                                    int nsample,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

void group_points_impl(const float* features, const int* indices, int b, int c, int n, int npoints, int nsample, float* output, cudaStream_t stream) {
  group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(features, indices, b, c, n, npoints, nsample, output);
}

