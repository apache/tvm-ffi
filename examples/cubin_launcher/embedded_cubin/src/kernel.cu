/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file examples/cubin_launcher/src/kernel.cu
 * \brief Simple CUDA kernel for testing cubin_launcher functionality.
 */

#include <cstdint>

// [kernels.begin]
/*!
 * \brief CUDA kernel that adds 1 to each element of an array.
 *
 * \param x Input array pointer.
 * \param y Output array pointer.
 * \param n Number of elements.
 */
extern "C" __global__ void add_one_cuda(const float* x, float* y, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1.0f;
  }
}

/*!
 * \brief CUDA kernel that multiplies each element by 2.
 *
 * \param x Input array pointer.
 * \param y Output array pointer.
 * \param n Number of elements.
 */
extern "C" __global__ void mul_two_cuda(const float* x, float* y, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] * 2.0f;
  }
}
// [kernels.end]
