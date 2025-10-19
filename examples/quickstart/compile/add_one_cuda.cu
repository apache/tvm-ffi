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

// [example.begin]
// File: compile/add_one_cuda.cu
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

namespace tvm_ffi_example_cuda {

__global__ void AddOneKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] + 1;
  }
}

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int64_t n = x.size(0);
  float* x_data = static_cast<float*>(x.data_ptr());
  float* y_data = static_cast<float*>(y.data_ptr());
  int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  AddOneKernel<<<blocks, threads, 0, stream>>>(x_data, y_data, n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, tvm_ffi_example_cuda::AddOne);
}  // namespace tvm_ffi_example_cuda
// [example.end]
