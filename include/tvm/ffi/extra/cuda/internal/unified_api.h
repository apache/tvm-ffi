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

#ifndef TVM_FFI_EXTRA_CUDA_INTERNAL_UNIFIED_API_H_
#define TVM_FFI_EXTRA_CUDA_INTERNAL_UNIFIED_API_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/cuda/base.h>

#include <string>

// ===========================================================================
// Section 1: Configuration & Version Checks
// ===========================================================================

// We only use unified API for cubin launcher for now
// this name is intentional to avoid confusion of other API usages
#ifndef TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
#if CUDART_VERSION >= 12080
// Use Runtime API by default if possible (CUDA >= 12.8)
#define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 0
#else  // if CUDART_VERSION < 12080
#define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 1
#endif
#else  // if defined(TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API)
// User explicitly defined the macro, check compatibility
#if (!(TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API)) && (CUDART_VERSION < 12080)
#define _STRINGIFY(x) #x
#define STR(x) _STRINGIFY(x)
static_assert(false, "Runtime API only supported for CUDA >= 12.8, got CUDA Runtime version: " STR(
                         CUDART_VERSION));
#undef STR
#undef _STRINGIFY
#endif
#endif

namespace tvm {
namespace ffi {
namespace cuda_api {

// ===========================================================================
// Section 2: Type Definitions & Macros
// ===========================================================================

#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API

// Driver API Types
using StreamHandle = CUstream;
using DeviceHandle = CUdevice;
using LibraryHandle = CUlibrary;
using KernelHandle = CUkernel;
using LaunchConfig = CUlaunchConfig;

using ResultType = CUresult;
using LaunchAttrType = CUlaunchAttribute;
using DeviceAttrType = CUdevice_attribute;

constexpr ResultType kSuccess = CUDA_SUCCESS;

// Driver API Functions
#define _TVM_FFI_CUDA_FUNC(name) cu##name  // NOLINT(bugprone-reserved-identifier)

#else

using StreamHandle = cudaStream_t;
using DeviceHandle = int;
using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfig = cudaLaunchConfig_t;

using ResultType = cudaError_t;
using LaunchAttrType = cudaLaunchAttribute;
using DeviceAttrType = cudaDeviceAttr;

constexpr ResultType kSuccess = cudaSuccess;

// Runtime API Functions
#define _TVM_FFI_CUDA_FUNC(name) cuda##name

#endif

// ===========================================================================
// Section 3: Error Handling
// ===========================================================================

// Helper to get error name/string based on API
inline void GetErrorString(ResultType err, const char** name, const char** str) {
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  cuGetErrorName(err, name);
  cuGetErrorString(err, str);
#else
  *name = cudaGetErrorName(err);
  *str = cudaGetErrorString(err);
#endif
}

// this macro is only used to check cuda errors in cubin launcher where
// we might switch between driver and runtime API.
#define TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(stmt)                               \
  do {                                                                              \
    ::tvm::ffi::cuda_api::ResultType __err = (stmt);                                \
    if (__err != ::tvm::ffi::cuda_api::kSuccess) {                                  \
      const char *__err_name, *__err_str;                                           \
      ::tvm::ffi::cuda_api::GetErrorString(__err, &__err_name, &__err_str);         \
      TVM_FFI_THROW(RuntimeError) << "CUDA Error: " << __err_name << " ("           \
                                  << static_cast<int>(__err) << "): " << __err_str; \
    }                                                                               \
  } while (0)

// ===========================================================================
// Section 4: Unified API Wrappers
// ===========================================================================

inline ResultType LoadLibrary(LibraryHandle* library, const void* image) {
  return _TVM_FFI_CUDA_FUNC(LibraryLoadData)(library, image, nullptr, nullptr, 0, nullptr, nullptr,
                                             0);
}

inline ResultType UnloadLibrary(LibraryHandle library) {
  return _TVM_FFI_CUDA_FUNC(LibraryUnload)(library);
}

inline ResultType GetKernel(KernelHandle* kernel, LibraryHandle library, const char* name) {
  return _TVM_FFI_CUDA_FUNC(LibraryGetKernel)(kernel, library, name);
}

inline DeviceHandle GetDeviceHandle(int device_id) {
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  CUdevice dev;
  // Note: We use CHECK here because this conversion usually shouldn't fail if ID is valid
  // and we need to return a value.
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuDeviceGet(&dev, device_id));
  return dev;
#else
  return device_id;
#endif
}

inline ResultType LaunchKernel(KernelHandle kernel, void** args, tvm::ffi::dim3 grid,
                               tvm::ffi::dim3 block, StreamHandle stream,
                               uint32_t dyn_smem_bytes = 0) {
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  return cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), grid.x, grid.y, grid.z, block.x,
                        block.y, block.z, dyn_smem_bytes, stream, args, nullptr);
#else
  return cudaLaunchKernel(reinterpret_cast<const void*>(kernel), {grid.x, grid.y, grid.z},
                          {block.x, block.y, block.z}, args, dyn_smem_bytes, stream);
#endif
}

inline ResultType GetKernelSharedMem(KernelHandle kernel, int& out, DeviceHandle device) {
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  return cuKernelGetAttribute(&out, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel, device);
#else
  cudaFuncAttributes func_attr;
  cudaError_t err = cudaFuncGetAttributes(&func_attr, kernel);
  if (err == cudaSuccess) {
    out = func_attr.sharedSizeBytes;
  }
  return err;
#endif
}

inline ResultType SetKernelMaxDynamicSharedMem(KernelHandle kernel, int shmem,
                                               DeviceHandle device) {
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  return cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem, kernel,
                              device);
#else
  return cudaKernelSetAttributeForDevice(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem,
                                         device);
#endif
}

// Additional wrappers for device operations used in CubinLauncher
inline ResultType GetDeviceCount(int* count) {
#if TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
  return cuDeviceGetCount(count);
#else
  return cudaGetDeviceCount(count);
#endif
}

inline ResultType GetDeviceAttribute(int* value, DeviceAttrType attr, DeviceHandle device) {
  return _TVM_FFI_CUDA_FUNC(DeviceGetAttribute)(value, attr, device);
}

}  // namespace cuda_api
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_CUDA_INTERNAL_UNIFIED_API_H_
