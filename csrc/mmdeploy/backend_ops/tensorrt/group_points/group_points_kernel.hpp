// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_GROUP_POINTS_KERNEL_HPP
#define TRT_GROUP_POINTS_KERNEL_HPP
#include <cuda_runtime.h>

void group_points_impl(const float *features, const int *idx, int b, int c, int n, int npoints, int nsample, float *out, cudaStream_t stream);

#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
