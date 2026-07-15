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
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/string.h>

#include <string>
#include <string_view>

namespace tvm {
namespace ffi {
namespace details {
/*!
 * \brief Get the custom type name for a given type code.
 */
inline String DLDataTypeCodeGetCustomTypeName(DLDataTypeCode type_code) {
  static Function fget_custom_type_name = Function::GetGlobalRequired("dtype.get_custom_type_name");
  return fget_custom_type_name(static_cast<int>(type_code)).cast<String>();
}

/*!
 * \brief Get the custom type name for a given type code.
 * \param str The string to parse.
 * \param scan The scan pointer.
 * \return The custom type name.
 */
inline int ParseCustomDataTypeCode(const std::string_view& str, const char** scan) {
  TVM_FFI_ICHECK(str.substr(0, 6) == "custom") << "Not a valid custom datatype string";
  auto tmp = str.data();
  TVM_FFI_ICHECK(str.data() == tmp);
  *scan = str.data() + 6;
  TVM_FFI_ICHECK(str.data() == tmp);
  if (**scan != '[') {
    TVM_FFI_THROW(ValueError) << "expected opening brace after 'custom' type in" << str;
  }
  TVM_FFI_ICHECK(str.data() == tmp);
  *scan += 1;
  TVM_FFI_ICHECK(str.data() == tmp);
  size_t custom_name_len = 0;
  TVM_FFI_ICHECK(str.data() == tmp);
  while (*scan + custom_name_len <= str.data() + str.length() &&
         *(*scan + custom_name_len) != ']') {
    ++custom_name_len;
  }
  TVM_FFI_ICHECK(str.data() == tmp);
  if (*(*scan + custom_name_len) != ']') {
    TVM_FFI_THROW(ValueError) << "expected closing brace after 'custom' type in" << str;
  }
  TVM_FFI_ICHECK(str.data() == tmp);
  *scan += custom_name_len + 1;
  TVM_FFI_ICHECK(str.data() == tmp);
  auto type_name = str.substr(7, custom_name_len);
  TVM_FFI_ICHECK(str.data() == tmp);
  static Function fget_custom_type_code = Function::GetGlobalRequired("dtype.get_custom_type_code");
  return fget_custom_type_code(std::string(type_name)).cast<int>();
}

/*
 * \brief Convert a DLDataTypeCode to a string.
 * \param os The output stream.
 * \param type_code The DLDataTypeCode to convert.
 */
inline void PrintDLDataTypeCodeAsStr(std::ostream& os, DLDataTypeCode type_code) {  // NOLINT(*)
  switch (static_cast<int>(type_code)) {
    case kDLInt: {
      os << "int";
      break;
    }
    case kDLUInt: {
      os << "uint";
      break;
    }
    case kDLFloat: {
      os << "float";
      break;
    }
    case kDLOpaqueHandle: {
      os << "handle";
      break;
    }
    case kDLBfloat: {
      os << "bfloat";
      break;
    }
    case kDLFloat8_e3m4: {
      os << "float8_e3m4";
      break;
    }
    case kDLFloat8_e4m3: {
      os << "float8_e4m3";
      break;
    }
    case kDLFloat8_e4m3b11fnuz: {
      os << "float8_e4m3b11fnuz";
      break;
    }
    case kDLFloat8_e4m3fn: {
      os << "float8_e4m3fn";
      break;
    }
    case kDLFloat8_e4m3fnuz: {
      os << "float8_e4m3fnuz";
      break;
    }
    case kDLFloat8_e5m2: {
      os << "float8_e5m2";
      break;
    }
    case kDLFloat8_e5m2fnuz: {
      os << "float8_e5m2fnuz";
      break;
    }
    case kDLFloat8_e8m0fnu: {
      os << "float8_e8m0fnu";
      break;
    }
    case kDLFloat6_e2m3fn: {
      os << "float6_e2m3fn";
      break;
    }
    case kDLFloat6_e3m2fn: {
      os << "float6_e3m2fn";
      break;
    }
    case kDLFloat4_e2m1fn: {
      os << "float4_e2m1fn";
      break;
    }
    default: {
      if (static_cast<int>(type_code) >= static_cast<int>(DLExtDataTypeCode::kDLExtCustomBegin)) {
        os << "custom[" << details::DLDataTypeCodeGetCustomTypeName(type_code) << "]";
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
 *  \brief Printer function for DLDataType.
 *  \param os The output stream.
 *  \param dtype The DLDataType to print.
 *  \return The output stream.
 */
inline std::string DLDataTypeToString_(DLDataType dtype) {  // NOLINT(*)
  if (dtype.bits == 8 && dtype.lanes == 1 && dtype.code == kDLBool) {
    return "bool";
  }
  // specially handle void
  if (dtype.code == kDLOpaqueHandle && dtype.lanes == 0 && dtype.bits == 0) {
    return "";
  }

  std::ostringstream os;
  if (dtype.code >= kDLExtCustomBegin) {
    os << "custom["
       << details::DLDataTypeCodeGetCustomTypeName(static_cast<DLDataTypeCode>(dtype.code)) << "]";
  } else {
    os << details::DLDataTypeCodeAsCStr(static_cast<DLDataTypeCode>(dtype.code));
  }
  if (dtype.code == kDLOpaqueHandle) return os.str();
  int16_t lanes = static_cast<int16_t>(dtype.lanes);
  if (dtype.code < kDLFloat8_e3m4 && (dtype.code != kDLBool || dtype.bits != 8)) {
    os << static_cast<int>(dtype.bits);
  }
  if (lanes > 1) {
    os << 'x' << lanes;
  } else if (lanes < -1) {
    os << "xvscalex" << -lanes;
  }
  return os.str();
}

/*!
 * \brief Parse a string to a DLDataType.
 * \param str The string to convert.
 * \return The corresponding DLDataType.
 */
inline DLDataType StringViewToDLDataType_(std::string_view str) {
  DLDataType dtype;
  // handle void type
  if (str.length() == 0 || str == "void") {
    dtype.code = kDLOpaqueHandle;
    dtype.bits = 0;
    dtype.lanes = 0;
    return dtype;
  }

  auto is_digit = [](char ch) { return ch >= '0' && ch <= '9'; };

  std::string normalized_dtype;
  auto normalize_prefix = [&](std::string_view canonical_prefix, size_t alias_prefix_size) {
    normalized_dtype.assign(canonical_prefix);
    normalized_dtype.append(str.substr(alias_prefix_size));
    str = normalized_dtype;
  };
  if (str.size() >= 2 && str[0] == 'i' && is_digit(str[1])) {
    normalize_prefix("int", 1);
  } else if (str.size() >= 2 && str[0] == 'u' && is_digit(str[1])) {
    normalize_prefix("uint", 1);
  } else if (str.size() >= 3 && str[0] == 'b' && str[1] == 'f' && is_digit(str[2])) {
    normalize_prefix("bfloat", 2);
  } else if (str.compare(0, 4, "fp8_") == 0) {
    normalize_prefix("float8_", 4);
  } else if (str.compare(0, 4, "fp6_") == 0) {
    normalize_prefix("float6_", 4);
  } else if (str.compare(0, 4, "fp4_") == 0) {
    normalize_prefix("float4_", 4);
  } else if (str.size() >= 2 && str[0] == 'f' && is_digit(str[1])) {
    normalize_prefix("float", 1);
  }

  // set the default values;
  dtype.bits = 32;
  dtype.lanes = 1;
  const char* scan;
  const char* str_end = str.data() + str.length();

  // Helper lambda to parse decimal digits from a bounded string_view
  // Returns the parsed value and updates *ptr to point past the last digit
  auto parse_digits = [](const char** ptr, const char* end) -> uint32_t {
    uint64_t value = 0;
    const char* start_ptr = *ptr;
    while (*ptr < end && **ptr >= '0' && **ptr <= '9') {
      value = value * 10 + (**ptr - '0');
      (*ptr)++;
    }
    if (value > UINT32_MAX) {
      TVM_FFI_THROW(ValueError) << "Integer value in dtype string '"
                                << std::string_view(start_ptr, *ptr - start_ptr)
                                << "' is out of range for uint32_t";
    }
    return static_cast<uint32_t>(value);
  };

  // Helper lambda to parse lanes specification (e.g., "x16" or "xvscalex4")
  // Returns the parsed lanes value and updates *ptr to point past the lanes specification
  // Supports scalable vectors with the "xvscale" prefix (represented as negative lanes)
  auto parse_lanes = [&](const char** ptr, const char* end, const std::string_view& dtype_str,
                         bool allow_scalable = false) -> uint16_t {
    int multiplier = 1;
    // Check for "xvscale" prefix for scalable vectors
    if (allow_scalable && (end - *ptr >= 7) && strncmp(*ptr, "xvscale", 7) == 0) {
      multiplier = -1;
      *ptr += 7;
    }
    if (*ptr >= end || **ptr != 'x') {
      return 1;  // No lanes specification, default to 1
    }
    (*ptr)++;  // Skip 'x'
    const char* digits_start = *ptr;
    uint32_t lanes_val = parse_digits(ptr, end);
    if (*ptr == digits_start || lanes_val == 0) {
      TVM_FFI_THROW(ValueError) << "Invalid lanes specification in dtype '" << dtype_str
                                << "'. Lanes must be a positive integer.";
    }
    if (lanes_val > UINT16_MAX) {
      TVM_FFI_THROW(ValueError) << "Lanes value " << lanes_val
                                << " is out of range for uint16_t in dtype '" << dtype_str << "'";
    }
    return static_cast<uint16_t>(multiplier * lanes_val);
  };

  auto parse_float = [&](const std::string_view& str, int offset, int code, int bits,
                         bool allow_underscore_lanes = false) {
    dtype.code = static_cast<uint8_t>(code);
    dtype.bits = static_cast<uint8_t>(bits);
    scan = str.data() + offset;
    if (allow_underscore_lanes && scan < str_end && *scan == '_') {
      ++scan;
    }
    const char* endpt = scan;
    dtype.lanes = parse_lanes(&endpt, str_end, str);
    scan = endpt;
    if (scan != str_end) {
      TVM_FFI_THROW(ValueError) << "unknown dtype `" << str << '`';
    }
    return dtype;
  };

  auto match_float_variant = [&](int offset, std::string_view pattern,
                                 bool allow_underscore_lanes = false) -> int {
    size_t end = static_cast<size_t>(offset) + pattern.size();
    if (str.size() < end || str.compare(offset, pattern.size(), pattern) != 0) {
      return -1;
    }
    if (end == str.size() || str[end] == 'x' || (allow_underscore_lanes && str[end] == '_')) {
      return static_cast<int>(end);
    }
    return -1;
  };

  auto parse_float8 = [&](int offset) {
    if (int end = match_float_variant(offset, "e4m3b11fnuz"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e4m3b11fnuz, 8);
    } else if (int end = match_float_variant(offset, "e4m3fnuz"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e4m3fnuz, 8);
    } else if (int end = match_float_variant(offset, "e4m3fn"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e4m3fn, 8);
    } else if (int end = match_float_variant(offset, "e5m2fnuz"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e5m2fnuz, 8);
    } else if (int end = match_float_variant(offset, "e8m0fnu"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e8m0fnu, 8);
    } else if (int end = match_float_variant(offset, "e3m4"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e3m4, 8);
    } else if (int end = match_float_variant(offset, "e4m3"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e4m3, 8);
    } else if (int end = match_float_variant(offset, "e5m2"); end >= 0) {
      return parse_float(str, end, kDLFloat8_e5m2, 8);
    } else {
      TVM_FFI_THROW(ValueError) << "unknown float8 type `" << str << '`';
      TVM_FFI_UNREACHABLE();
    }
  };

  auto parse_float6 = [&](int offset) {
    if (int end = match_float_variant(offset, "e2m3fn"); end >= 0) {
      return parse_float(str, end, kDLFloat6_e2m3fn, 6);
    } else if (int end = match_float_variant(offset, "e3m2fn"); end >= 0) {
      return parse_float(str, end, kDLFloat6_e3m2fn, 6);
    } else {
      TVM_FFI_THROW(ValueError) << "unknown float6 type `" << str << '`';
      TVM_FFI_UNREACHABLE();
    }
  };

  auto parse_float4 = [&](int offset) {
    if (int end = match_float_variant(offset, "e2m1fn", /*allow_underscore_lanes=*/true);
        end >= 0) {
      return parse_float(str, end, kDLFloat4_e2m1fn, 4, /*allow_underscore_lanes=*/true);
    } else {
      TVM_FFI_THROW(ValueError) << "unknown float4 type `" << str << '`';
      TVM_FFI_UNREACHABLE();
    }
  };

  if (str.compare(0, 3, "int") == 0) {
    dtype.code = kDLInt;
    scan = str.data() + 3;
  } else if (str.size() >= 2 && str[0] == 'i' && is_digit(str[1])) {
    dtype.code = kDLInt;
    scan = str.data() + 1;
  } else if (str.compare(0, 4, "uint") == 0) {
    dtype.code = kDLUInt;
    scan = str.data() + 4;
  } else if (str.size() >= 2 && str[0] == 'u' && is_digit(str[1])) {
    dtype.code = kDLUInt;
    scan = str.data() + 1;
  } else if (str.compare(0, 4, "bool") == 0) {
    dtype.code = kDLBool;
    dtype.bits = 8;
    scan = str.data() + 4;
  } else if (str.compare(0, 4, "fp8_") == 0) {
    return parse_float8(4);
  } else if (str.compare(0, 3, "f8_") == 0) {
    return parse_float8(3);
  } else if (str.compare(0, 4, "fp6_") == 0) {
    return parse_float6(4);
  } else if (str.compare(0, 3, "f6_") == 0) {
    return parse_float6(3);
  } else if (str.compare(0, 4, "fp4_") == 0) {
    return parse_float4(4);
  } else if (str.compare(0, 3, "f4_") == 0) {
    return parse_float4(3);
  } else if (str.compare(0, 5, "float") == 0) {
    if (str.compare(5, 2, "8_") == 0) {
      return parse_float8(7);
    } else if (str.compare(5, 2, "6_") == 0) {
      return parse_float6(7);
    } else if (str.compare(5, 2, "4_") == 0) {
      return parse_float4(7);
    } else {
      dtype.code = kDLFloat;
      scan = str.data() + 5;
    }
  } else if (str.size() >= 2 && str[0] == 'f' && is_digit(str[1])) {
    dtype.code = kDLFloat;
    scan = str.data() + 1;
  } else if (str.compare(0, 6, "handle") == 0) {
    dtype.code = kDLOpaqueHandle;
    dtype.bits = 64;  // handle uses 64 bit by default.
    scan = str.data() + 6;
  } else if (str.compare(0, 6, "bfloat") == 0) {
    dtype.code = kDLBfloat;
    dtype.bits = 16;
    scan = str.data() + 6;
  } else if (str.size() >= 3 && str[0] == 'b' && str[1] == 'f' && is_digit(str[2])) {
    dtype.code = kDLBfloat;
    dtype.bits = 16;
    scan = str.data() + 2;
  } else if (str.compare(0, 6, "custom") == 0) {
    dtype.code = static_cast<uint8_t>(details::ParseCustomDataTypeCode(str, &scan));
  } else {
    scan = str.data();
    TVM_FFI_THROW(ValueError) << "unknown dtype `" << str << '`';
  }
  // Parse bits manually to handle non-null-terminated string_view
  const char* xdelim = scan;
  uint32_t bits_val = parse_digits(&xdelim, str_end);
  if (bits_val > UINT8_MAX) {
    TVM_FFI_THROW(ValueError) << "Bits value " << bits_val
                              << " is out of range for uint8_t in dtype '" << str << "'";
  }
  uint8_t bits = static_cast<uint8_t>(bits_val);
  if (bits != 0) dtype.bits = bits;
  const char* endpt = xdelim;
  dtype.lanes = parse_lanes(&endpt, str_end, str, /*allow_scalable=*/true);
  if (endpt != str_end) {
    TVM_FFI_THROW(ValueError) << "unknown dtype `" << str << '`';
  }
  return dtype;
}

}  // namespace ffi
}  // namespace tvm

int TVMFFIDataTypeFromString(const TVMFFIByteArray* str, DLDataType* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = tvm::ffi::StringViewToDLDataType_(std::string_view(str->data, str->size));
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIDataTypeToString(const DLDataType* dtype, TVMFFIAny* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::String out_str(tvm::ffi::DLDataTypeToString_(*dtype));
  tvm::ffi::TypeTraits<tvm::ffi::String>::MoveToAny(std::move(out_str), out);
  TVM_FFI_SAFE_CALL_END();
}
