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

// One JIT-allocated FFI container per test. Every returned Object is freshly
// constructed inside this translation unit, so its header.deleter points into
// JIT-mapped memory — the escape hazard that keep_module_alive=True pins the
// JITDylib against.

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <string>

using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::String;
using tvm::ffi::Tuple;

// Character replace in a String.
String test_string_replace_impl(String s, String from, String to) {
  std::string out(s);
  char c_from = std::string(from)[0];
  char c_to = std::string(to)[0];
  for (char& c : out) {
    if (c == c_from) c = c_to;
  }
  return String(out);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_string_replace, test_string_replace_impl);

// Concatenate two Array<int64_t>s.
Array<int64_t> test_array_concat_impl(Array<int64_t> a, Array<int64_t> b) {
  Array<int64_t> out;
  for (int64_t x : a) out.push_back(x);
  for (int64_t x : b) out.push_back(x);
  return out;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_array_concat, test_array_concat_impl);

// Hand-built Map<String, int64_t>.
Map<String, int64_t> test_map_build_impl() {
  Map<String, int64_t> out;
  out.Set(String("a"), 1);
  out.Set(String("b"), 2);
  return out;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_map_build, test_map_build_impl);

// Reverse a four-element Tuple.
Tuple<String, int64_t, String, int64_t> test_tuple_reverse_impl(
    Tuple<int64_t, String, int64_t, String> t) {
  return Tuple<String, int64_t, String, int64_t>(t.get<3>(), t.get<2>(), t.get<1>(), t.get<0>());
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_tuple_reverse, test_tuple_reverse_impl);
