// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_FURTHEST_POINT_SAMPLING_KERNEL_HPP
#define TRT_FURTHEST_POINT_SAMPLING_KERNEL_HPP
#include <cuda_runtime.h>

void furthest_point_sampling_impl(int b, int n, int m,
                                                    const float* dataset,
                                                    float* temp, int* idxs, cudaStream_t stream);


#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
