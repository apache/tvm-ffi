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
 * \file tvm/ffi/extra/cuda/device_guard.h
 * \brief Device guard structs.
 */
#ifndef TVM_FFI_EXTRA_CUDA_DEVICE_GUARD_H_
#define TVM_FFI_EXTRA_CUDA_DEVICE_GUARD_H_

#include <tvm/ffi/extra/cuda/base.h>

namespace tvm {
namespace ffi {

/*!
 * \brief CUDA Device Guard. On construction, it calls `cudaGetDevice` to set the CUDA device to
 * target index, and stores the original current device index. And on destruction, it sets the
 * current CUDA device back to original device index.
 *
 * Example usage:
 * \code{.cpp}
 * void kernel(ffi::TensorView x) {
 *   ffi::CUDADeviceGuard guard(x.device().device_id);
 *   ...
 * }
 * \endcode
 */
struct CUDADeviceGuard {
  CUDADeviceGuard() = delete;
  /*!
   * \brief Constructor from a device index, and store the original device index.
   * \param device_index The device index to guard.
   */
  explicit CUDADeviceGuard(int device_index) {
    target_device_index_ = device_index;
    TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&original_device_index_));
    if (target_device_index_ != original_device_index_) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(device_index));
    }
  }

  /*!
   * \brief Destructor to set the current device index back to original one if different.
   */
  ~CUDADeviceGuard() noexcept(false) {
    if (original_device_index_ != target_device_index_) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(original_device_index_));
    }
  }

 private:
  int original_device_index_;
  int target_device_index_;
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_CUDA_DEVICE_GUARD_H_
