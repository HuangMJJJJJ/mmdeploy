// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_GATHER_POINTS_KERNEL_HPP
#define TRT_GATHER_POINTS_KERNEL_HPP
#include <cuda_runtime.h>

template <typename scalar_t>
void gather_points_impl(const scalar_t *points, const int *idx, int b, int c, int n, int npoints, scalar_t *out, cudaStream_t stream);

#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
