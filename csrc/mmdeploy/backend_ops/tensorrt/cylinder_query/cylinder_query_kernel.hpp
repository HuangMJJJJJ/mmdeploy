// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_CYLINDER_QUERY_KERNEL_HPP
#define TRT_CYLINDER_QUERY_KERNEL_HPP
#include <cuda_runtime.h>

void query_cylinder_point_impl(int b, int n, int m, float radius, float hmin, float hmax,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx, cudaStream_t stream);
#endif  // TRT_GRID_SAMPLER_KERNEL_HPP