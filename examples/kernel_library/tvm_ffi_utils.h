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
#ifndef KERNEL_LIBRARY_TVM_FFI_UTILS_H_
#define KERNEL_LIBRARY_TVM_FFI_UTILS_H_

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;
using ffi::Optional;
using ffi::Tensor;
using ffi::TensorView;

// [check_macros.begin]
// --- Reusable validation macros ---
#define CHECK_CUDA(x) \
  TVM_FFI_CHECK((x).device().device_type == kDLCUDA, ValueError) << #x " must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) \
  TVM_FFI_CHECK((x).IsContiguous(), ValueError) << #x " must be contiguous"
#define CHECK_INPUT(x)   \
  do {                   \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
  } while (0)
#define CHECK_DIM(d, x) \
  TVM_FFI_CHECK((x).ndim() == (d), ValueError) << #x " must be a " #d "D tensor"
#define CHECK_DEVICE(a, b)                                                          \
  do {                                                                              \
    TVM_FFI_CHECK((a).device().device_type == (b).device().device_type, ValueError) \
        << #a " and " #b " must be on the same device type";                        \
    TVM_FFI_CHECK((a).device().device_id == (b).device().device_id, ValueError)     \
        << #a " and " #b " must be on the same device";                             \
  } while (0)
// [check_macros.end]

// [get_stream.begin]
// --- Stream helper ---
inline cudaStream_t get_cuda_stream(DLDevice device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}
// [get_stream.end]

// [alloc_tensor.begin]
// --- Tensor allocation helper ---
inline ffi::Tensor alloc_tensor(const ffi::Shape& shape, DLDataType dtype, DLDevice device) {
  return ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape, dtype, device);
}
// [alloc_tensor.end]

// [dtype_constants.begin]
// --- DLPack dtype constants ---
constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
// [dtype_constants.end]

#endif  // KERNEL_LIBRARY_TVM_FFI_UTILS_H_
