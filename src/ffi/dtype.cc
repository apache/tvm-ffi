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
  // set the default values;
  dtype.bits = 32;
  dtype.lanes = 1;
  const char* scan;
  const char* str_end = str.data() + str.length();

  // Helper lambda to parse decimal digits from a bounded string_view
  // Returns the parsed value and updates ptr to point past the last digit
  auto parse_digits = [](const char*& ptr, const char* end) -> uint32_t {
    uint64_t value = 0;
    const char* start_ptr = ptr;
    while (ptr < end && *ptr >= '0' && *ptr <= '9') {
      value = value * 10 + (*ptr - '0');
      ptr++;
    }
    if (value > UINT32_MAX) {
      TVM_FFI_THROW(ValueError) << "Integer value in dtype string '"
                                << std::string_view(start_ptr, ptr - start_ptr)
                                << "' is out of range for uint32_t";
    }
    return static_cast<uint32_t>(value);
  };

  auto parse_float = [&](const std::string_view& str, int offset, int code, int bits) {
    dtype.code = static_cast<uint8_t>(code);
    dtype.bits = static_cast<uint8_t>(bits);
    scan = str.data() + offset;
    const char* endpt = scan;
    if (scan < str_end && *scan == 'x') {
      endpt = scan + 1;
      const char* digits_start = endpt;
      uint32_t lanes_val = parse_digits(endpt, str_end);
      if (endpt == digits_start || lanes_val == 0) {
        TVM_FFI_THROW(ValueError) << "Invalid lanes specification in dtype '" << str
                                  << "'. Lanes must be a positive integer.";
      }
      if (lanes_val > UINT16_MAX) {
        TVM_FFI_THROW(ValueError) << "Lanes value " << lanes_val
                                  << " is out of range for uint16_t in dtype '" << str << "'";
      }
      dtype.lanes = static_cast<uint16_t>(lanes_val);
      scan = endpt;
    }
    if (scan != str_end) {
      TVM_FFI_THROW(ValueError) << "unknown dtype `" << str << '`';
    }
    return dtype;
  };

  if (str.compare(0, 3, "int") == 0) {
    dtype.code = kDLInt;
    scan = str.data() + 3;
  } else if (str.compare(0, 4, "uint") == 0) {
    dtype.code = kDLUInt;
    scan = str.data() + 4;
  } else if (str.compare(0, 4, "bool") == 0) {
    dtype.code = kDLBool;
    dtype.bits = 8;
    scan = str.data() + 4;
  } else if (str.compare(0, 5, "float") == 0) {
    if (str.compare(5, 2, "8_") == 0) {
      if (str.compare(7, 4, "e3m4") == 0) {
        return parse_float(str, 11, kDLFloat8_e3m4, 8);
      } else if (str.compare(7, 4, "e4m3") == 0) {
        if (str.compare(11, 7, "b11fnuz") == 0) {
          return parse_float(str, 18, kDLFloat8_e4m3b11fnuz, 8);
        } else if (str.compare(11, 2, "fn") == 0) {
          if (str.compare(13, 2, "uz") == 0) {
            return parse_float(str, 15, kDLFloat8_e4m3fnuz, 8);
          } else {
            return parse_float(str, 13, kDLFloat8_e4m3fn, 8);
          }
        } else {
          return parse_float(str, 11, kDLFloat8_e4m3, 8);
        }
      } else if (str.compare(7, 8, "e5m2fnuz") == 0) {
        return parse_float(str, 15, kDLFloat8_e5m2fnuz, 8);
      } else if (str.compare(7, 4, "e5m2") == 0) {
        return parse_float(str, 11, kDLFloat8_e5m2, 8);
      } else if (str.compare(7, 7, "e8m0fnu") == 0) {
        return parse_float(str, 14, kDLFloat8_e8m0fnu, 8);
      } else {
        TVM_FFI_THROW(ValueError) << "unknown float8 type `" << str << '`';
        TVM_FFI_UNREACHABLE();
      }
    } else if (str.compare(5, 2, "6_") == 0) {
      if (str.compare(7, 6, "e2m3fn") == 0) {
        return parse_float(str, 13, kDLFloat6_e2m3fn, 6);
      } else if (str.compare(7, 6, "e3m2fn") == 0) {
        return parse_float(str, 13, kDLFloat6_e3m2fn, 6);
      } else {
        TVM_FFI_THROW(ValueError) << "unknown float6 type `" << str << '`';
        TVM_FFI_UNREACHABLE();
      }
    } else if (str.compare(5, 2, "4_") == 0) {
      // kFloat4_e2m1fn
      if (str.compare(7, 6, "e2m1fn") == 0) {
        return parse_float(str, 13, kDLFloat4_e2m1fn, 4);
      } else {
        TVM_FFI_THROW(ValueError) << "unknown float4 type `" << str << '`';
        TVM_FFI_UNREACHABLE();
      }
    } else {
      dtype.code = kDLFloat;
      scan = str.data() + 5;
    }
  } else if (str.compare(0, 6, "handle") == 0) {
    dtype.code = kDLOpaqueHandle;
    dtype.bits = 64;  // handle uses 64 bit by default.
    scan = str.data() + 6;
  } else if (str.compare(0, 6, "bfloat") == 0) {
    dtype.code = kDLBfloat;
    dtype.bits = 16;
    scan = str.data() + 6;
  } else if (str.compare(0, 6, "custom") == 0) {
    dtype.code = static_cast<uint8_t>(details::ParseCustomDataTypeCode(str, &scan));
  } else {
    scan = str.data();
    TVM_FFI_THROW(ValueError) << "unknown dtype `" << str << '`';
  }
  // Parse bits manually to handle non-null-terminated string_view
  const char* xdelim = scan;
  uint32_t bits_val = parse_digits(xdelim, str_end);
  if (bits_val > UINT8_MAX) {
    TVM_FFI_THROW(ValueError) << "Bits value " << bits_val << " is out of range for uint8_t in dtype '"
                              << str << "'";
  }
  uint8_t bits = static_cast<uint8_t>(bits_val);
  if (bits != 0) dtype.bits = bits;
  int scalable_multiplier = 1;
  // Check bounds before dereferencing xdelim
  if (xdelim < str_end && strncmp(xdelim, "xvscale", 7) == 0) {
    scalable_multiplier = -1;
    xdelim += 7;
  }
  const char* endpt = xdelim;
  // Check bounds before dereferencing xdelim
  if (xdelim < str_end && *xdelim == 'x') {
    endpt = xdelim + 1;
    const char* digits_start = endpt;
    uint32_t lanes_val = parse_digits(endpt, str_end);
    if (endpt == digits_start || lanes_val == 0) {
      TVM_FFI_THROW(ValueError) << "Invalid lanes specification in dtype '" << str
                                << "'. Lanes must be a positive integer.";
    }
    if (lanes_val > UINT16_MAX) {
      TVM_FFI_THROW(ValueError) << "Lanes value " << lanes_val
                                << " is out of range for uint16_t in dtype '" << str << "'";
    }
    dtype.lanes = static_cast<uint16_t>(scalable_multiplier * lanes_val);
  }
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
