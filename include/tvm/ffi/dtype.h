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
 * \file tvm/ffi/dtype.h
 * \brief Data type handling.
 */
#ifndef TVM_FFI_DTYPE_H_
#define TVM_FFI_DTYPE_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/type_traits.h>

#include <cstdint>
#include <string>

namespace tvm {
namespace ffi {
/*!
 * \brief Extension code beyond the DLDataType.
 *
 * This class is always consistent with the DLPack.
 */
enum DLExtDataTypeCode { kDLExtCustomBegin = 129 };

namespace details {

/*
 * \brief Convert a DLDataTypeCode to a string.
 * \param os The output stream.
 * \param type_code The DLDataTypeCode to convert.
 */
inline const char* DLDataTypeCodeAsCStr(DLDataTypeCode type_code) {  // NOLINT(*)
  switch (static_cast<int>(type_code)) {
    case kDLInt: {
      return "int";
    }
    case kDLUInt: {
      return "uint";
    }
    case kDLFloat: {
      return "float";
    }
    case kDLOpaqueHandle: {
      return "handle";
    }
    case kDLBfloat: {
      return "bfloat";
    }
    case kDLBool: {
      return "bool";
    }
    case kDLFloat8_e3m4: {
      return "float8_e3m4";
    }
    case kDLFloat8_e4m3: {
      return "float8_e4m3";
    }
    case kDLFloat8_e4m3b11fnuz: {
      return "float8_e4m3b11fnuz";
    }
    case kDLFloat8_e4m3fn: {
      return "float8_e4m3fn";
    }
    case kDLFloat8_e4m3fnuz: {
      return "float8_e4m3fnuz";
    }
    case kDLFloat8_e5m2: {
      return "float8_e5m2";
    }
    case kDLFloat8_e5m2fnuz: {
      return "float8_e5m2fnuz";
    }
    case kDLFloat8_e8m0fnu: {
      return "float8_e8m0fnu";
    }
    case kDLFloat6_e2m3fn: {
      return "float6_e2m3fn";
    }
    case kDLFloat6_e3m2fn: {
      return "float6_e3m2fn";
    }
    case kDLFloat4_e2m1fn: {
      return "float4_e2m1fn";
    }
    default: {
      if (static_cast<int>(type_code) >= static_cast<int>(DLExtDataTypeCode::kDLExtCustomBegin)) {
        return "custom";
      } else {
        TVM_FFI_THROW(ValueError) << "DLDataType contains unknown type_code="
                                  << static_cast<int>(type_code);
      }
      TVM_FFI_UNREACHABLE();
    }
  }
}
}  // namespace details

/*!
 * \brief Convert a string to a DLDataType.
 * \param str The string to convert.
 * \return The DLDataType.
 */
inline DLDataType StringToDLDataType(const String& str) {
  DLDataType out;
  TVMFFIByteArray data{str.data(), str.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIDataTypeFromString(&data, &out));
  return out;
}

/*!
 * \brief Convert a DLDataType to a string.
 * \param dtype The DLDataType to convert.
 * \return The string.
 */
inline String DLDataTypeToString(DLDataType dtype) {
  TVMFFIAny out;
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIDataTypeToString(&dtype, &out));
  return TypeTraits<String>::MoveFromAnyAfterCheck(&out);
}

/*!
 * \brief Check whether a DLDataType is a floating-point type.
 * \param dtype The DLDataType to check.
 * \return True if dtype is a floating-point type, false otherwise.
 */
inline bool DTypeIsFloat(DLDataType dtype) {
  switch (static_cast<int>(dtype.code)) {
    case kDLFloat:
    case kDLBfloat:
    case kDLFloat8_e3m4:
    case kDLFloat8_e4m3:
    case kDLFloat8_e4m3b11fnuz:
    case kDLFloat8_e4m3fn:
    case kDLFloat8_e4m3fnuz:
    case kDLFloat8_e5m2:
    case kDLFloat8_e5m2fnuz:
    case kDLFloat8_e8m0fnu:
    case kDLFloat6_e2m3fn:
    case kDLFloat6_e3m2fn:
    case kDLFloat4_e2m1fn:
      return true;
    default:
      return false;
  }
}

/*!
 * \brief Check whether a DLDataType is an integer type.
 * \param dtype The DLDataType to check.
 * \return True if dtype is a signed or unsigned integer type, false otherwise.
 */
TVM_FFI_INLINE bool DTypeIsInt(DLDataType dtype) {
  return dtype.code == kDLInt || dtype.code == kDLUInt;
}

/*!
 * \brief Check whether a DLDataType is a boolean type.
 * \param dtype The DLDataType to check.
 * \return True if dtype is a boolean type, false otherwise.
 */
TVM_FFI_INLINE bool DTypeIsBool(DLDataType dtype) { return dtype.code == kDLBool; }

/*!
 * \brief Convert a DLDataType to a compact abbreviation for text formats.
 * \param dtype The DLDataType to convert.
 * \return The compact string, such as ``i32``, ``f16``, or ``bf16``.
 */
inline std::string DTypeAbbrev(DLDataType dtype) {
  if (dtype.bits == 8 && dtype.lanes == 1 && dtype.code == kDLBool) {
    return "bool";
  }
  if (dtype.code == kDLOpaqueHandle && dtype.lanes == 0 && dtype.bits == 0) {
    return "void";
  }

  std::string result;
  switch (static_cast<int>(dtype.code)) {
    case kDLInt: {
      result = "i";
      break;
    }
    case kDLUInt: {
      result = "u";
      break;
    }
    case kDLFloat: {
      result = "f";
      break;
    }
    case kDLOpaqueHandle: {
      return "handle";
    }
    case kDLBfloat: {
      result = "bf";
      break;
    }
    case kDLBool: {
      result = "bool";
      break;
    }
    case kDLFloat8_e3m4: {
      result = "f8_e3m4";
      break;
    }
    case kDLFloat8_e4m3: {
      result = "f8_e4m3";
      break;
    }
    case kDLFloat8_e4m3b11fnuz: {
      result = "f8_e4m3b11fnuz";
      break;
    }
    case kDLFloat8_e4m3fn: {
      result = "f8_e4m3fn";
      break;
    }
    case kDLFloat8_e4m3fnuz: {
      result = "f8_e4m3fnuz";
      break;
    }
    case kDLFloat8_e5m2: {
      result = "f8_e5m2";
      break;
    }
    case kDLFloat8_e5m2fnuz: {
      result = "f8_e5m2fnuz";
      break;
    }
    case kDLFloat8_e8m0fnu: {
      result = "f8_e8m0fnu";
      break;
    }
    case kDLFloat6_e2m3fn: {
      result = "f6_e2m3fn";
      break;
    }
    case kDLFloat6_e3m2fn: {
      result = "f6_e3m2fn";
      break;
    }
    case kDLFloat4_e2m1fn: {
      result = "f4_e2m1fn";
      break;
    }
    default: {
      if (static_cast<int>(dtype.code) >= static_cast<int>(DLExtDataTypeCode::kDLExtCustomBegin)) {
        String full_name = DLDataTypeToString(dtype);
        return std::string(full_name.data(), full_name.size());
      }
      TVM_FFI_THROW(ValueError) << "DLDataType contains unknown type_code="
                                << static_cast<int>(dtype.code);
    }
  }

  int16_t lanes = static_cast<int16_t>(dtype.lanes);
  if (dtype.code < kDLFloat8_e3m4 && (dtype.code != kDLBool || dtype.bits != 8)) {
    result += std::to_string(static_cast<int>(dtype.bits));
  }
  if (lanes > 1) {
    result += "x" + std::to_string(lanes);
  } else if (lanes < -1) {
    result += "xvscalex" + std::to_string(-lanes);
  }
  return result;
}

// DLDataType
template <>
struct TypeTraits<DLDataType> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDataType;

  TVM_FFI_INLINE static void CopyToAnyView(const DLDataType& src, TVMFFIAny* result) {
    // clear padding part to ensure the equality check can always check the v_uint64 part
    result->v_uint64 = 0;
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->zero_padding = 0;
    result->v_dtype = src;
  }

  TVM_FFI_INLINE static void MoveToAny(DLDataType src, TVMFFIAny* result) {
    // clear padding part to ensure the equality check can always check the v_uint64 part
    result->v_uint64 = 0;
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->zero_padding = 0;
    result->v_dtype = src;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDataType;
  }

  TVM_FFI_INLINE static DLDataType CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIDataType);
    return src->v_dtype;
  }

  TVM_FFI_INLINE static DLDataType MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type — move is just copy.
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<DLDataType> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDataType) {
      return src->v_dtype;
    }
    // enable string to dtype auto conversion
    if (auto opt_str = TypeTraits<std::string>::TryCastFromAnyView(src)) {
      return StringToDLDataType(*opt_str);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return ffi::StaticTypeKey::kTVMFFIDataType; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(ffi::StaticTypeKey::kTVMFFIDataType) + R"("})";
  }
};
}  // namespace ffi
}  // namespace tvm

// define DLDataType comparison and printing in root namespace
inline std::ostream& operator<<(std::ostream& os, DLDataType dtype) {  // NOLINT(*)
  return os << tvm::ffi::DLDataTypeToString(dtype);
}

inline bool operator==(const DLDataType& lhs, const DLDataType& rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

inline bool operator!=(const DLDataType& lhs, const DLDataType& rhs) { return !(lhs == rhs); }
#endif  // TVM_FFI_DTYPE_H_
