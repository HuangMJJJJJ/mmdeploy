// Copyright (c) OpenMMLab. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <numeric>
#include <vector>
// #include <math.h>

#include "common_cuda_helper.hpp"
#include "cylinder_query_kernel.hpp"
#include "trt_plugin_helper.hpp"


__global__ void query_cylinder_point_kernel(int b, int n, int m, float radius, float hmin, float hmax,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        const float *__restrict__ rot,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  rot += batch_index * m * 9;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    float r0 = rot[j * 9 + 0];
    float r1 = rot[j * 9 + 1];
    float r2 = rot[j * 9 + 2];
    float r3 = rot[j * 9 + 3];
    float r4 = rot[j * 9 + 4];
    float r5 = rot[j * 9 + 5];
    float r6 = rot[j * 9 + 6];
    float r7 = rot[j * 9 + 7];
    float r8 = rot[j * 9 + 8];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0] - new_x;
      float y = xyz[k * 3 + 1] - new_y;
      float z = xyz[k * 3 + 2] - new_z;
      float x_rot = r0 * x + r3 * y + r6 * z;
      float y_rot = r1 * x + r4 * y + r7 * z;
      float z_rot = r2 * x + r5 * y + r8 * z;
      float d2 = y_rot * y_rot + z_rot * z_rot;
      if (d2 < radius2 && x_rot > hmin && x_rot < hmax) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

// template <typename T>
// float data_mean(T *arr, int size) {
//   float sum = 0;
//   for (int i = 0; i < size; ++i) {
//     sum += arr[i];
//   }
//   return sum / size;
// }
// template <typename T>
// float data_std(T *arr, int size) {
//   float mean_val = data_mean(arr, size);
//   float sum = 0;
//   for (int i = 0; i < size; ++i) {
//     sum += (arr[i] - mean_val) * (arr[i] - mean_val);
//   }
//   return sqrt(sum / size);
// }
void query_cylinder_point_impl(int b, int n, int m, float radius, float hmin, float hmax,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx, cudaStream_t stream) {
  // printf("b=%d, n=%d, m=%d, radius=%f, hmin=%f, hmax=%f, nsample=%d\n", b, n, m, radius, hmin, hmax, nsample);
  // float f_new_xyz[b*m*3];
  // float f_xyz[b*n*3];
  // float f_rot[b*m*9];
  // int f_idx[b*m*nsample];
  // cudaMemcpy(f_new_xyz, new_xyz, b*m*3*sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(f_xyz, xyz, b*n*3*sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(f_rot, rot, b*m*9*sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(f_idx, idx, b*m*nsample*sizeof(int), cudaMemcpyDeviceToHost);
  // printf("new_xyz: mean %f, std %f\n", data_mean<float>(f_new_xyz, b*m*3), data_std<float>(f_new_xyz, b*m*3));
  // printf("xyz: mean %f, std %f\n", data_mean<float>(f_xyz, b*n*3), data_std<float>(f_xyz, b*n*3));
  // printf("rot: mean %f, std %f\n", data_mean<float>(f_rot, b*m*9), data_std<float>(f_rot, b*m*9));
  // printf("idx: mean %f, std %f\n", data_mean<int>(f_idx, b*m*nsample), data_std<int>(f_idx, b*m*nsample));
  
  query_cylinder_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, hmin, hmax, nsample, new_xyz, xyz, rot, idx);

}
