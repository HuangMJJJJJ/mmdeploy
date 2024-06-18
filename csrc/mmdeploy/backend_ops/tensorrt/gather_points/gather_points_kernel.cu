// Copyright (c) OpenMMLab. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <numeric>
#include <vector>

#include "common_cuda_helper.hpp"
#include "gather_points_kernel.hpp"
#include "trt_plugin_helper.hpp"

template <typename scalar_t>
__global__ void gather_points_kernel(const scalar_t *__restrict__ features,
                                     const int *__restrict__ indices,
                                     int b, int c, int n, int npoints,
                                     scalar_t *__restrict__ output) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int l = blockIdx.y; l < c; l += gridDim.y) {
      for (int j = threadIdx.x; j < npoints; j += blockDim.x) {
        int a = indices[i * npoints + j];
        output[(i * c + l) * npoints + j] = features[(i * c + l) * n + a];
      }
    }
  }
}

template <typename scalar_t>
void gather_points_impl(const scalar_t* features, const int* indices, int b, int c, int n, int npoints, scalar_t* output, cudaStream_t stream) {
//   int batch = 1;
//   for (int i = 0; i < indice_nbDims - 1; ++i) batch *= dims[i];
//   int num_input = dims[indice_nbDims - 1];
//   int num_indices = indices_dims[indice_nbDims - 1];
//   int channel = 1;
//   for (int i = indice_nbDims; i < nbDims; ++i) channel *= dims[i];
//   const int col_block = DIVUP(batch * num_indices * channel, THREADS_PER_BLOCK);
//   gather_points_kernel<<<col_block, THREADS_PER_BLOCK, 0, stream>>>(input, indices, output, batch,
//                                                                   num_input, num_indices, channel);
  gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0, stream>>>(features, indices, b, c, n, npoints, output);
}

template void gather_points_impl<float>(const float* features, const int* indices, int b, int c, int n, int npoints, float* output, cudaStream_t stream);
