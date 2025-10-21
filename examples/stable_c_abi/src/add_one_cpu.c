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
// NOLINTBEGIN(bugprone-reserved-identifier,google-readability-braces-around-statements)

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

// clang-format off
// [example.begin]
// File: src/add_one_cpu.cc
TVM_FFI_DLL int __tvm_ffi_add_one_cpu(void* handle, const TVMFFIAny* args, int32_t num_args,
                                      TVMFFIAny* result) {
  // Step 1. Extract inputs from `Any`
  // Step 1.1. Extract `x := args[0]`
  DLTensor* x;
  if (args[0].type_index == kTVMFFIDLTensorPtr) x = (DLTensor*)(args[0].v_ptr);
  else if (args[0].type_index == kTVMFFITensor) x = (DLTensor*)(args[0].v_c_str + sizeof(TVMFFIObject));
  else { TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input"); return -1; }
  // Step 1.2. Extract `y := args[1]`
  DLTensor* y;
  if (args[1].type_index == kTVMFFIDLTensorPtr) y = (DLTensor*)(args[1].v_ptr);
  else if (args[1].type_index == kTVMFFITensor) y = (DLTensor*)(args[1].v_c_str + sizeof(TVMFFIObject));
  else { TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor output"); return -1; }

  // Step 2. Perform add one: y = x + 1
  for (int64_t i = 0; i < x->shape[0]; ++i) {
    ((float*)y->data)[i] = ((float*)x->data)[i] + 1.0f;
  }

  // Step 3. Return error code 0 (success)
  //
  // Note that `result` is not set, as the output is passed in via `y` argument,
  // which is functionally similar to a Python function with signature:
  //
  //   def add_one(x: Tensor, y: Tensor) -> None: ...
  return 0;
}
// [example.end]
// clang-format on
// NOLINTEND(bugprone-reserved-identifier,google-readability-braces-around-statements)
