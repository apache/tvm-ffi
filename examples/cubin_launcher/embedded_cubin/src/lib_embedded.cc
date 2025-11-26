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
 * \file examples/cubin_launcher/src/lib_embedded.cc
 * \brief TVM-FFI library with embedded CUBIN kernels.
 *
 * This library exports TVM-FFI functions to launch CUDA kernels from
 * embedded CUBIN data.
 */

// [example.begin]
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

// Embed CUBIN module with name "env"
// This creates the necessary symbols and singleton struct for accessing the embedded CUBIN
TVM_FFI_EMBED_CUBIN(env);

namespace cubin_embedded {

/*!
 * \brief Launch add_one_cuda kernel on input tensor.
 * \param x Input tensor (float32, 1D)
 * \param y Output tensor (float32, 1D, same shape as x)
 */
void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Get kernel from embedded CUBIN (cached in static variable for efficiency)
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(env, "add_one_cuda");

  TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";
  TVM_FFI_CHECK(y.ndim() == 1, ValueError) << "Output must be 1D tensor";
  TVM_FFI_CHECK(x.size(0) == y.size(0), ValueError) << "Input and output must have same size";

  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  // Prepare kernel arguments
  void* args[] = {reinterpret_cast<void*>(&x_ptr), reinterpret_cast<void*>(&y_ptr),
                  reinterpret_cast<void*>(&n)};

  // Launch configuration
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  // Get CUDA stream
  DLDevice device = x.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  // Launch kernel
  cudaError_t result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_ERROR(result);
}

/*!
 * \brief Launch mul_two_cuda kernel on input tensor.
 * \param x Input tensor (float32, 1D)
 * \param y Output tensor (float32, 1D, same shape as x)
 */
void MulTwo(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Get kernel from embedded CUBIN (cached in static variable for efficiency)
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(env, "mul_two_cuda");

  TVM_FFI_CHECK(x.ndim() == 1, ValueError) << "Input must be 1D tensor";
  TVM_FFI_CHECK(y.ndim() == 1, ValueError) << "Output must be 1D tensor";
  TVM_FFI_CHECK(x.size(0) == y.size(0), ValueError) << "Input and output must have same size";

  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  // Prepare kernel arguments
  void* args[] = {reinterpret_cast<void*>(&x_ptr), reinterpret_cast<void*>(&y_ptr),
                  reinterpret_cast<void*>(&n)};

  // Launch configuration
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  // Get CUDA stream
  DLDevice device = x.device();
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

  // Launch kernel
  cudaError_t result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_ERROR(result);
}

// Export TVM-FFI functions
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, cubin_embedded::AddOne);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_two, cubin_embedded::MulTwo);

}  // namespace cubin_embedded
// [example.end]
