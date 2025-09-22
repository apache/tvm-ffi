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

#include <tvm/ffi/type_parser.h>

#include <string>
#include <string_view>
#include <vector>

namespace tvm {
namespace ffi {
namespace type_parser {
namespace detail {

TVM_FFI_DLL std::string EscapeJSONString(std::string_view input) {
  std::string result;
  result.reserve(input.size() + 8);
  constexpr char kHexDigits[] = "0123456789abcdef";
  for (unsigned char ch : input) {
    switch (ch) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (ch < 0x20) {
          result += "\\u00";
          result.push_back(kHexDigits[(ch >> 4) & 0xF]);
          result.push_back(kHexDigits[ch & 0xF]);
        } else {
          result.push_back(static_cast<char>(ch));
        }
        break;
    }
  }
  return result;
}

TVM_FFI_DLL std::string QuoteJSONString(std::string_view input) {
  std::string result;
  result.reserve(input.size() + 2);
  result.push_back('"');
  result += EscapeJSONString(input);
  result.push_back('"');
  return result;
}

TVM_FFI_DLL std::string ComposeComposite(std::string_view kind, std::vector<std::string> args) {
  std::string result;
  result.reserve(kind.size() + 16);
  result.append("{\"kind\":");
  result.append(QuoteJSONString(kind));
  result.append(",\"args\":[");
  bool first = true;
  for (std::string& arg : args) {
    if (!first) {
      result.push_back(',');
    }
    first = false;
    result += std::move(arg);
  }
  result.push_back(']');
  result.push_back('}');
  return result;
}

}  // namespace detail
}  // namespace type_parser
}  // namespace ffi
}  // namespace tvm
