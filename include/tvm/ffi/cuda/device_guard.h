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
 * \file tvm/ffi/cuda/device_guard.h
 * \brief Device guard structs.
 */
#ifndef TVM_FFI_CUDA_DEVICE_GUARD_H_
#define TVM_FFI_CUDA_DEVICE_GUARD_H_

#include <cuda_runtime_api.h>
#include <tvm/ffi/error.h>

namespace tvm {
namespace ffi {

/*!
 * \brief CUDA Device Guard.
 *
 * Example usage:
 * \code
 * void kernel(ffi::TensorView x) {
 *   ffi::CUDADeviceGuard guard(x.device().device_id);
 *   ...
 * }
 * \endcode
 */
struct CUDADeviceGuard {
  CUDADeviceGuard() = delete;
  /*!
   * \brief Constructor from a device index, and backup the current device index.
   * \param device_index The device index to guard.
   */
  explicit CUDADeviceGuard(int device_index) {
    cudaError_t err;
    err = cudaGetDevice(&previous_device_index);
    TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaGetDevice failed: " << cudaGetErrorString(err);
    if (previous_device_index != device_index) {
      err = cudaSetDevice(device_index);
      TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaSetDevice failed: " << cudaGetErrorString(err);
    }
  }

  /*!
   * \brief Destructor to set the current device index back to backup one.
   */
  ~CUDADeviceGuard() {
    cudaError_t err = cudaSetDevice(previous_device_index);
    TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaSetDevice failed: " << cudaGetErrorString(err);
  }

 private:
  int previous_device_index;
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CUDA_DEVICE_GUARD_H_
