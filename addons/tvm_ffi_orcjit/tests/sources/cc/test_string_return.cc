// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// String return type tests (C++): JIT functions that return String objects.
// Tests the conversion of kTVMFFIStr type_index from JIT to Python.

#include <tvm/ffi/c_api.h>

#include <cstring>
#include <string>

// test_get_hello_world: returns a simple ASCII string
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_get_hello_world(void* self, const TVMFFIAny* args,
                                                      int32_t num_args, TVMFFIAny* result) {
  std::string message = "Hello, World!";
  TVMFFIByteArray input;
  input.data = message.c_str();
  input.size = message.size();

  if (TVMFFIStringFromByteArray(&input, result) != 0) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to create string");
    return -1;
  }

  return 0;
}

// test_get_empty_string: returns an empty string
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_get_empty_string(void* self, const TVMFFIAny* args,
                                                       int32_t num_args, TVMFFIAny* result) {
  TVMFFIByteArray input;
  input.data = "";
  input.size = 0;

  if (TVMFFIStringFromByteArray(&input, result) != 0) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to create empty string");
    return -1;
  }

  return 0;
}

// test_concatenate_strings: takes two string args and returns concatenated result
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_concatenate_strings(void* self, const TVMFFIAny* args,
                                                          int32_t num_args, TVMFFIAny* result) {
  try {
    if (num_args != 2) {
      TVMFFIErrorSetRaisedFromCStr("ValueError", "Expected 2 arguments");
      return -1;
    }

    // Accept both SmallStr and Str for input arguments
    bool is_str0 = args[0].type_index == kTVMFFIStr || args[0].type_index == kTVMFFISmallStr;
    bool is_str1 = args[1].type_index == kTVMFFIStr || args[1].type_index == kTVMFFISmallStr;
    if (!is_str0 || !is_str1) {
      TVMFFIErrorSetRaisedFromCStr("TypeError", "Arguments must be strings");
      return -1;
    }

    // Get byte arrays (handling both SmallStr and heap Str)
    TVMFFIByteArray* bytes0 = (args[0].type_index == kTVMFFISmallStr)
                                  ? TVMFFISmallBytesGetContentByteArray(&args[0])
                                  : TVMFFIBytesGetByteArrayPtr(args[0].v_ptr);
    TVMFFIByteArray* bytes1 = (args[1].type_index == kTVMFFISmallStr)
                                  ? TVMFFISmallBytesGetContentByteArray(&args[1])
                                  : TVMFFIBytesGetByteArrayPtr(args[1].v_ptr);

    if (bytes0 == nullptr || bytes1 == nullptr) {
      TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to extract string data");
      return -1;
    }

    std::string str0(bytes0->data, bytes0->size);
    std::string str1(bytes1->data, bytes1->size);
    std::string concatenated = str0 + str1;

    TVMFFIByteArray output;
    output.data = concatenated.c_str();
    output.size = concatenated.size();
    if (TVMFFIStringFromByteArray(&output, result) != 0) {
      TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to create concatenated string");
      return -1;
    }

    return 0;
  } catch (const std::exception& e) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", e.what());
    return -1;
  }
}

// test_string_length: returns length of input string
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_string_length(void* self, const TVMFFIAny* args,
                                                    int32_t num_args, TVMFFIAny* result) {
  try {
    if (num_args != 1) {
      TVMFFIErrorSetRaisedFromCStr("ValueError", "Expected 1 argument");
      return -1;
    }

    // Accept both SmallStr and Str for input argument
    bool is_str = args[0].type_index == kTVMFFIStr || args[0].type_index == kTVMFFISmallStr;
    if (!is_str) {
      TVMFFIErrorSetRaisedFromCStr("TypeError", "Argument must be a string");
      return -1;
    }

    // Get byte array (handling both SmallStr and heap Str)
    TVMFFIByteArray* bytes = (args[0].type_index == kTVMFFISmallStr)
                                 ? TVMFFISmallBytesGetContentByteArray(&args[0])
                                 : TVMFFIBytesGetByteArrayPtr(args[0].v_ptr);
    if (bytes == nullptr) {
      TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to extract string data");
      return -1;
    }

    result->type_index = kTVMFFIInt;
    result->zero_padding = 0;
    result->v_int64 = static_cast<int64_t>(bytes->size);
    return 0;
  } catch (const std::exception& e) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", e.what());
    return -1;
  }
}
