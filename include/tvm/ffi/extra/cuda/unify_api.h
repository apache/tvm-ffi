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

#ifndef TVM_FFI_EXTRA_CUDA_UNIFY_API_H_
#define TVM_FFI_EXTRA_CUDA_UNIFY_API_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/error.h>

#include <filesystem>
#include <string>

#ifndef TVM_FFI_CUDA_USE_DRIVER_API
#if CUDART_VERSION >= 12080
// Use Runtime API by default if possible
#define TVM_FFI_CUDA_USE_DRIVER_API 0
#else
#define TVM_FFI_CUDA_USE_DRIVER_API 1
#endif
#else if !(TVM_FFI_CUDA_USE_DRIVER_API) && (CUDART_VERSION < 12080)
#error "Runtime API only supported for CUDA >= 12.8"
#endif

#if TVM_FFI_CUDA_USE_DRIVER_API

#include <driver_types.h>

using StreamHandle = CUstream;
using ResultHandle = CUresult;

#define FFI_CUDA_SUCCESS CUDA_SUCCESS

using LibraryHandle = CUmodule;
using KernelHandle = CUfunction;
using LaunchConfigHandle = CUlaunchConfig;
using LaunchAttrHandle = CUlaunchAttribute;

using DeviceAttrHandle = CUdevice_attribute;
using DeviceHandle = CUdevice;

#define load_function cuModuleGetFunction
#define get_device_count cuDeviceGetCount
#define get_device_attr cuDeviceGetAttribute
#define unload_library cuLibraryUnload

#else

using StreamHandle = cudaStream_t;
using ResultHandle = cudaError_t;

#define FFI_CUDA_SUCCESS cudaSuccess

using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfigHandle = cudaLaunchConfig_t;
using LaunchAttrHandle = cudaLaunchAttribute;

using DeviceAttrHandle = cudaDeviceAttr;
using DeviceHandle = int;

#define load_function cudaLibraryGetKernel
#define get_device_count cudaGetDeviceCount
#define get_device_attr cudaDeviceGetAttribute
#define unload_library cudaLibraryUnload

#endif

#define TVM_FFI_CHECK_RUNTIME_CUDA_ERROR(stmt)                                      \
  do {                                                                              \
    cudaError_t __err = (stmt);                                                     \
    if (__err != cudaSuccess) {                                                     \
      const char* __err_name = cudaGetErrorName(__err);                             \
      const char* __err_str = cudaGetErrorString(__err);                            \
      TVM_FFI_THROW(RuntimeError) << "CUDA Runtime Error: " << __err_name << " ("   \
                                  << static_cast<int>(__err) << "): " << __err_str; \
    }                                                                               \
  } while (0)

#define TVM_FFI_CHECK_DRIVER_CUDA_ERROR(stmt)                                  \
  do {                                                                         \
    CUresult __err = (stmt);                                                   \
    if (__err != CUDA_SUCCESS) {                                               \
      const char *name, *info;                                                 \
      cuGetErrorName(__err, &name);                                            \
      cuGetErrorString(__err, &info);                                          \
      TVM_FFI_THROW(RuntimeError) << "CUDA Driver Error: " << name << " ("     \
                                  << static_cast<int>(__err) << "): " << info; \
    }                                                                          \
  } while (0)

static ResultHandle load_image(LibraryHandle* library, const void* image) {
#if TVM_FFI_CUDA_USE_DRIVER_API
  return cuModuleLoad(library, image);
#else
  return cudaLibraryLoadData(library, image, nullptr, nullptr, 0, nullptr, nullptr, 0);
#endif
}

static DeviceHandle idx_to_device(int idx) {
#if TVM_FFI_CUDA_USE_DRIVER_API
  CUdevice o;
  TVM_FFI_CHECK_DRIVER_CUDA_ERROR(cuDeviceGet(&o, idx));
  return o;
#else
  return idx;
#endif
}

static ResultHandle launch_kernel(KernelHandle kernel, void** args, dim3 grid, dim3 block,
                                  StreamHandle stream, uint32_t dyn_smem_bytes = 0) {
#if TVM_FFI_CUDA_USE_DRIVER_API
  return cuLaunchKernel(kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, dyn_smem_bytes,
                        stream, args);
#else
  auto kernel = reinterpret_cast<const void*>(kernel_);
  return cudaLaunchKernel(kernel, {grid.x, grid.y, grid.z}, {block.x, block.y, block.z}, args,
                          dyn_smem_bytes, stream);
#endif
}

#endif
