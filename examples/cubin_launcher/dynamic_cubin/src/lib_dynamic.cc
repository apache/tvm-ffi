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
 * \file examples/cubin_launcher/src/lib_dynamic.cc
 * \brief TVM-FFI library with dynamic CUBIN loading.
 *
 * This library exports TVM-FFI functions to load CUBIN from file and
 * launch CUDA kernels.
 */

// [example.begin]
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <memory>

namespace cubin_dynamic {

// Global CUBIN module and kernels (loaded dynamically)
static std::unique_ptr<tvm::ffi::CubinModule> g_cubin_module;
static std::unique_ptr<tvm::ffi::CubinKernel> g_add_one_kernel;
static std::unique_ptr<tvm::ffi::CubinKernel> g_mul_two_kernel;

/*!
 * \brief Set CUBIN module from binary data.
 * \param cubin CUBIN binary data as Bytes object.
 */
void SetCubin(const tvm::ffi::Bytes& cubin) {
  // Load CUBIN module from memory
  g_cubin_module = std::make_unique<tvm::ffi::CubinModule>(cubin);
  g_add_one_kernel = std::make_unique<tvm::ffi::CubinKernel>((*g_cubin_module)["add_one_cuda"]);
  g_mul_two_kernel = std::make_unique<tvm::ffi::CubinKernel>((*g_cubin_module)["mul_two_cuda"]);
}

/*!
 * \brief Launch add_one_cuda kernel on input tensor.
 * \param x Input tensor (float32, 1D)
 * \param y Output tensor (float32, 1D, same shape as x)
 */
void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_CHECK(g_cubin_module != nullptr, RuntimeError)
      << "CUBIN module not loaded. Call set_cubin first.";

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
  cudaError_t result = g_add_one_kernel->Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_ERROR(result);
}

/*!
 * \brief Launch mul_two_cuda kernel on input tensor.
 * \param x Input tensor (float32, 1D)
 * \param y Output tensor (float32, 1D, same shape as x)
 */
void MulTwo(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  TVM_FFI_CHECK(g_cubin_module != nullptr, RuntimeError)
      << "CUBIN module not loaded. Call set_cubin first.";

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
  cudaError_t result = g_mul_two_kernel->Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_ERROR(result);
}

// Export TVM-FFI functions
TVM_FFI_DLL_EXPORT_TYPED_FUNC(set_cubin, cubin_dynamic::SetCubin);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, cubin_dynamic::AddOne);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_two, cubin_dynamic::MulTwo);

}  // namespace cubin_dynamic
// [example.end]
