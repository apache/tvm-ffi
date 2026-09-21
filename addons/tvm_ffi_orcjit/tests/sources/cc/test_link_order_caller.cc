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

// Caller object for the intra-module linking test (C++ version).
// References __tvm_ffi_helper_add from test_link_order_base.cc and
// exports cross_lib_add which forwards to it.

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>

extern "C" int __tvm_ffi_helper_add(void* self, const TVMFFIAny* args, int32_t num_args,
                                    TVMFFIAny* result);

int cross_lib_add_impl(int a, int b) {
  TVMFFIAny call_args[2];
  TVMFFIAny call_result;
  call_args[0].type_index = kTVMFFIInt;
  call_args[0].zero_padding = 0;
  call_args[0].v_int64 = a;
  call_args[1].type_index = kTVMFFIInt;
  call_args[1].zero_padding = 0;
  call_args[1].v_int64 = b;
  call_result.type_index = kTVMFFINone;
  call_result.zero_padding = 0;
  call_result.v_int64 = 0;
  __tvm_ffi_helper_add(nullptr, call_args, 2, &call_result);
  return static_cast<int>(call_result.v_int64);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cross_lib_add, cross_lib_add_impl);
