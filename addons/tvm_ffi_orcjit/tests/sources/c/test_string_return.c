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

/*
 * String return type tests: JIT functions that return String objects
 * Tests the conversion of kTVMFFIStr type_index from JIT to Python.
 */

#include <string.h>
#include <tvm/ffi/c_api.h>

/* test_get_hello_world: returns a simple ASCII string */
TVM_FFI_DLL_EXPORT int __tvmffi_test_get_hello_world(void* self, const TVMFFIAny* args,
                                                     int32_t num_args, TVMFFIAny* result) {
  /* Create a string object with "Hello, World!" */
  const char* message = "Hello, World!";
  TVMFFIByteArray input = {.data = message, .size = strlen(message)};

  /* Convert byte array to String (automatically selects SmallStr or Str based on size) */
  if (TVMFFIStringFromByteArray(&input, result) != 0) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to create string");
    return -1;
  }

  return 0;
}

/* test_get_empty_string: returns an empty String */
TVM_FFI_DLL_EXPORT int __tvmffi_test_get_empty_string(void* self, const TVMFFIAny* args,
                                                      int32_t num_args, TVMFFIAny* result) {
  TVMFFIByteArray input = {.data = "", .size = 0};

  if (TVMFFIStringFromByteArray(&input, result) != 0) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to create empty string");
    return -1;
  }

  return 0;
}

/* test_concatenate_strings: takes two string args and returns concatenated result */
TVM_FFI_DLL_EXPORT int __tvmffi_test_concatenate_strings(void* self, const TVMFFIAny* args,
                                                         int32_t num_args, TVMFFIAny* result) {
  if (num_args != 2) {
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expected 2 arguments");
    return -1;
  }

  /* Accept both SmallStr and Str for input arguments */
  int is_str0 = args[0].type_index == kTVMFFIStr || args[0].type_index == kTVMFFISmallStr;
  int is_str1 = args[1].type_index == kTVMFFIStr || args[1].type_index == kTVMFFISmallStr;
  if (!is_str0 || !is_str1) {
    TVMFFIErrorSetRaisedFromCStr("TypeError", "Both arguments must be strings");
    return -1;
  }

  /* Manually extract data and size from  both SmallStr and heap Str */
  const char* data0;
  size_t size0;
  if (args[0].type_index == kTVMFFISmallStr) {
    data0 = args[0].v_bytes;
    size0 = args[0].small_str_len;
  } else {
    TVMFFIByteArray* bytes = (TVMFFIByteArray*)((char*)args[0].v_ptr + sizeof(TVMFFIObject));
    data0 = bytes->data;
    size0 = bytes->size;
  }

  const char* data1;
  size_t size1;
  if (args[1].type_index == kTVMFFISmallStr) {
    data1 = args[1].v_bytes;
    size1 = args[1].small_str_len;
  } else {
    TVMFFIByteArray* bytes = (TVMFFIByteArray*)((char*)args[1].v_ptr + sizeof(TVMFFIObject));
    data1 = bytes->data;
    size1 = bytes->size;
  }

  /* Concatenate the two strings */
  size_t total_len = size0 + size1;
  char buffer[512]; /* Fixed-size buffer for small test strings */

  if (total_len >= sizeof(buffer)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Concatenated string too large");
    return -1;
  }

  memcpy(buffer, data0, size0);
  memcpy(buffer + size0, data1, size1);

  /* Create result string (TVMFFIStringFromByteArray will select SmallStr or Str) */
  TVMFFIByteArray output = {.data = buffer, .size = total_len};
  if (TVMFFIStringFromByteArray(&output, result) != 0) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Failed to create concatenated string");
    return -1;
  }

  return 0;
}

/* test_string_length: returns length of input string */
TVM_FFI_DLL_EXPORT int __tvmffi_test_string_length(void* self, const TVMFFIAny* args,
                                                   int32_t num_args, TVMFFIAny* result) {
  if (num_args != 1) {
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expected 1 argument");
    return -1;
  }

  /* Accept both SmallStr and Str */
  int is_str = args[0].type_index == kTVMFFIStr || args[0].type_index == kTVMFFISmallStr;
  if (!is_str) {
    TVMFFIErrorSetRaisedFromCStr("TypeError", "Argument must be a string");
    return -1;
  }

  /* Manually extract size from SmallStr or heap Str */
  size_t size;
  if (args[0].type_index == kTVMFFISmallStr) {
    size = args[0].small_str_len;
  } else {
    TVMFFIByteArray* bytes = (TVMFFIByteArray*)((char*)args[0].v_ptr + sizeof(TVMFFIObject));
    size = bytes->size;
  }

  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = (int64_t)size;
  return 0;
}
