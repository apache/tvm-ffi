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
 * \file tvm/ffi/device.h
 * \brief Device handling.
 */
#ifndef TVM_FFI_DEVICE_H_
#define TVM_FFI_DEVICE_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/type_traits.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace tvm {
namespace ffi {
namespace details {

TVM_FFI_INLINE static std::optional<DLDeviceType> TryParseDLDeviceType(std::string_view name) {
  if (name == "cpu") return kDLCPU;
  if (name == "cuda") return kDLCUDA;
  if (name == "opencl") return kDLOpenCL;
  if (name == "vulkan") return kDLVulkan;
  if (name == "metal" || name == "mps") return kDLMetal;
  if (name == "vpi") return kDLVPI;
  if (name == "rocm") return kDLROCM;
  if (name == "ext_dev") return kDLExtDev;
  if (name == "hexagon") return kDLHexagon;
  if (name == "wgpu" || name == "webgpu") return kDLWebGPU;
  if (name == "maia") return kDLMAIA;
  if (name == "trn") return kDLTrn;
  return std::nullopt;
}

TVM_FFI_INLINE static std::optional<int32_t> TryParseDLDeviceIndex(std::string_view index) {
  if (index.empty()) return std::nullopt;
  int64_t value = 0;
  for (char ch : index) {
    if (ch < '0' || ch > '9') return std::nullopt;
    value = value * 10 + (ch - '0');
    if (value > std::numeric_limits<int32_t>::max()) return std::nullopt;
  }
  return static_cast<int32_t>(value);
}

TVM_FFI_INLINE static std::optional<DLDevice> TryStringViewToDLDevice(std::string_view str) {
  size_t space_pos = str.find(' ');
  if (space_pos != std::string_view::npos) {
    str = str.substr(0, space_pos);
  }
  size_t colon_pos = str.find(':');
  std::string_view name = colon_pos == std::string_view::npos ? str : str.substr(0, colon_pos);
  if (name.empty()) return std::nullopt;
  if (str.find(':', colon_pos == std::string_view::npos ? str.size() : colon_pos + 1) !=
      std::string_view::npos) {
    return std::nullopt;
  }
  std::optional<DLDeviceType> device_type = TryParseDLDeviceType(name);
  if (!device_type.has_value()) return std::nullopt;
  int32_t device_id = 0;
  if (colon_pos != std::string_view::npos) {
    std::optional<int32_t> parsed_device_id = TryParseDLDeviceIndex(str.substr(colon_pos + 1));
    if (!parsed_device_id.has_value()) return std::nullopt;
    device_id = parsed_device_id.value();
  }
  return DLDevice{device_type.value(), device_id};
}

}  // namespace details

// Device
template <>
struct TypeTraits<DLDevice> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDevice;

  TVM_FFI_INLINE static void CopyToAnyView(const DLDevice& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDevice;
    result->zero_padding = 0;
    result->v_device = src;
  }

  TVM_FFI_INLINE static void MoveToAny(DLDevice src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDevice;
    result->zero_padding = 0;
    result->v_device = src;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDevice;
  }

  TVM_FFI_INLINE static DLDevice CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIDevice);
    return src->v_device;
  }

  TVM_FFI_INLINE static DLDevice MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<DLDevice> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDevice) {
      return src->v_device;
    }
    if (auto opt_str = TypeTraits<String>::TryCastFromAnyView(src)) {
      return details::TryStringViewToDLDevice(std::string_view(opt_str->data(), opt_str->size()));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIDevice; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIDevice) + R"("})";
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_DEVICE_H_
