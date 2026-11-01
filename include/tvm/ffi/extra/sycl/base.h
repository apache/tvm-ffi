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
 * \file tvm/ffi/extra/sycl/base.h
 * \brief SYCL base utilities for kernels on kDLOneAPI devices (e.g. Intel XPU).
 *
 * This header requires a SYCL 2020 compiler in C++20 mode (e.g. ``icpx -fsycl -std=c++20``).
 * It is never included by the TVM FFI core library.
 */
#ifndef TVM_FFI_EXTRA_SYCL_BASE_H_
#define TVM_FFI_EXTRA_SYCL_BASE_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <sycl/sycl.hpp>
#include <vector>

namespace tvm {
namespace ffi {

/*!
 * \brief Macro for translating a sycl::exception thrown by a statement into a RuntimeError.
 *
 * \param stmt The statement that may throw a sycl::exception.
 */
#define TVM_FFI_CHECK_SYCL_ERROR(stmt)                                      \
  do {                                                                      \
    try {                                                                   \
      stmt;                                                                 \
    } catch (const ::sycl::exception& __err) {                              \
      TVM_FFI_THROW(RuntimeError) << "SYCL Error (" << __err.code().value() \
                                  << "): " << __err.what();                 \
    }                                                                       \
  } while (0)

/*!
 * \brief Get the SYCL queue set as the current FFI stream of a kDLOneAPI device.
 *
 * The stream handle of a kDLOneAPI device is a non-owning ``sycl::queue*`` that is
 * valid for the duration of the call, e.g. the current ``torch.xpu`` stream.
 *
 * \param device The kDLOneAPI device.
 * \return The current queue of the device.
 * \throws ValueError if the device is not kDLOneAPI or no queue is set.
 */
inline ::sycl::queue& GetCurrentSYCLQueue(DLDevice device) {
  if (device.device_type != kDLOneAPI) {
    TVM_FFI_THROW(ValueError) << "Expected a kDLOneAPI device, got device type "
                              << static_cast<int>(device.device_type);
  }
  void* stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
  if (stream == nullptr) {
    TVM_FFI_THROW(ValueError) << "No SYCL queue is set for oneapi:" << device.device_id
                              << ", use tvm_ffi.use_torch_stream or TVMFFIEnvSetStream";
  }
  return *static_cast<::sycl::queue*>(stream);
}

/*!
 * \brief Get the SYCL device of a kDLOneAPI device.
 *
 * Following DLPack, the device_id of kDLOneAPI is the index of the device in
 * ``sycl::device::get_devices()``, which may differ from the framework device index.
 *
 * \note The list is intentionally unfiltered, like PyTorch's DLPack exporter; a GPU-only list
 *       such as ``get_devices(sycl::info::device_type::gpu)`` yields different indices.
 *
 * \param device The kDLOneAPI device.
 * \return The SYCL device.
 * \throws ValueError if the device is not kDLOneAPI or the index is out of range.
 */
inline const ::sycl::device& SYCLDeviceFromDLDevice(DLDevice device) {
  static const std::vector<::sycl::device> devices = ::sycl::device::get_devices();
  if (device.device_type != kDLOneAPI) {
    TVM_FFI_THROW(ValueError) << "Expected a kDLOneAPI device, got device type "
                              << static_cast<int>(device.device_type);
  }
  if (device.device_id < 0 || static_cast<size_t>(device.device_id) >= devices.size()) {
    TVM_FFI_THROW(ValueError) << "oneapi:" << device.device_id << " is out of range, "
                              << devices.size() << " SYCL devices are available";
  }
  return devices[device.device_id];
}

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_SYCL_BASE_H_
