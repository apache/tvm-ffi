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
#include "tvm_ffi_utils.h"

// [cuda_kernel.begin]
template <typename T>
__global__ void ScaleKernel(T* out, const T* in, T factor, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * factor;
  }
}
// [cuda_kernel.end]

// [function.begin]
void Scale(TensorView output, TensorView input, double factor) {
  // --- 1. Validate inputs ---
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_DIM(1, input);
  CHECK_DEVICE(input, output);
  TVM_FFI_CHECK(input.dtype() == output.dtype(), ValueError) << "input/output dtype mismatch";
  TVM_FFI_CHECK(input.numel() == output.numel(), ValueError) << "input/output size mismatch";

  // --- 2. Device guard and stream ---
  ffi::CUDADeviceGuard guard(input.device().device_id);
  cudaStream_t stream = get_cuda_stream(input.device());

  // --- 3. Dispatch on dtype and launch ---
  int64_t n = input.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  if (input.dtype() == dl_float32) {
    ScaleKernel<<<blocks, threads, 0, stream>>>(static_cast<float*>(output.data_ptr()),
                                                static_cast<float*>(input.data_ptr()),
                                                static_cast<float>(factor), n);
  } else if (input.dtype() == dl_float16) {
    ScaleKernel<<<blocks, threads, 0, stream>>>(static_cast<half*>(output.data_ptr()),
                                                static_cast<half*>(input.data_ptr()),
                                                static_cast<half>(factor), n);
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported dtype: " << input.dtype();
  }
}
// [function.end]

// [export.begin]
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scale, Scale);
// [export.end]
