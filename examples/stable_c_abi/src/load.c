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
// NOLINTBEGIN(modernize-deprecated-headers,modernize-use-nullptr,bugprone-assignment-in-if-condition,modernize-loop-convert)
// [main.begin]
// File: src/load.c
#include <stdio.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

// Global functions are looked up during `Initialize` and deallocated during `Finalize`
// - global function: "ffi.Module.load_from_file.so"
static TVMFFIObjectHandle fn_load_module = NULL;
// - global function: "ffi.ModuleGetFunction"
static TVMFFIObjectHandle fn_get_function = NULL;

int Run(DLTensor* x, DLTensor* y) {
  int ret_code = 0;
  TVMFFIAny call_args[3] = {};
  TVMFFIAny mod = {.type_index = kTVMFFINone, .v_obj = NULL};
  TVMFFIAny func = {.type_index = kTVMFFINone, .v_obj = NULL};
  TVMFFIAny none = {.type_index = kTVMFFINone};  // ignore the return value

  // Step 1. Load module
  // Equivalent to:
  //    mod = tvm::ffi::Module::LoadFromFile("build/add_one_cpu.so")
  call_args[0] = (TVMFFIAny){.type_index = kTVMFFIRawStr, .v_c_str = "build/add_one_cpu.so"};
  call_args[1] = (TVMFFIAny){.type_index = kTVMFFISmallStr, .v_int64 = 0};
  if ((ret_code = TVMFFIFunctionCall(fn_load_module, call_args, 2, &mod))) goto _RAII;

  // Step 2. Get function `add_one_cpu` from module
  // Equivalent to:
  //    func = mod->GetFunction("add_one_cpu", /*query_imports=*/false).value()
  call_args[0] = (TVMFFIAny){.type_index = mod.type_index, .v_obj = mod.v_obj};
  call_args[1] = (TVMFFIAny){.type_index = kTVMFFIRawStr, .v_c_str = "add_one_cpu"};
  call_args[2] = (TVMFFIAny){.type_index = kTVMFFIBool, .v_int64 = 0};
  if ((ret_code = TVMFFIFunctionCall(fn_get_function, call_args, 3, &func))) goto _RAII;

  // Step 3. Call function `add_one_cpu(x, y)`
  // Equivalent to:
  //    func(x, y)
  call_args[0] = (TVMFFIAny){.type_index = kTVMFFIDLTensorPtr, .v_ptr = x};
  call_args[1] = (TVMFFIAny){.type_index = kTVMFFIDLTensorPtr, .v_ptr = y};
  if ((ret_code = TVMFFIFunctionCall(func.v_ptr, call_args, 2, &none))) goto _RAII;

_RAII:
  if (mod.type_index >= kTVMFFIObject) TVMFFIObjectDecRef(mod.v_obj);
  if (func.type_index >= kTVMFFIObject) TVMFFIObjectDecRef(func.v_obj);
  if (none.type_index >= kTVMFFIObject) TVMFFIObjectDecRef(none.v_obj);
  return ret_code;
}
// [main.end]

/************* Auxiliary Logics *************/

// [aux.begin]
static inline int Initialize() {
  int ret_code = 0;
  TVMFFIByteArray name_load_module = {.data = "ffi.Module.load_from_file.so", .size = 28};
  TVMFFIByteArray name_get_function = {.data = "ffi.ModuleGetFunction", .size = 21};
  if ((ret_code = TVMFFIFunctionGetGlobal(&name_load_module, &fn_load_module))) return ret_code;
  if ((ret_code = TVMFFIFunctionGetGlobal(&name_get_function, &fn_get_function))) return ret_code;
  return 0;
}

static inline void Finalize(int ret_code) {
  TVMFFIObjectHandle err = NULL;
  TVMFFIErrorCell* cell = NULL;
  if (fn_load_module) TVMFFIObjectDecRef(fn_load_module);
  if (fn_get_function) TVMFFIObjectDecRef(fn_get_function);
  if (ret_code) {
    TVMFFIErrorMoveFromRaised(&err);
    cell = (TVMFFIErrorCell*)((char*)(err) + sizeof(TVMFFIObject));
    printf("%.*s: %.*s\n", (int)(cell->kind.size), cell->kind.data, (int)(cell->message.size),
           cell->message.data);
  }
}

int main() {
  int ret_code = 0;
  float x_data[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  float y_data[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t shape[1] = {5};
  int64_t strides[1] = {1};
  DLDataType f32 = {.code = kTVMFFIFloat, .bits = 32, .lanes = 1};
  DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
  DLTensor x = {//
                .data = x_data, .device = cpu,      .ndim = 1,       .dtype = f32,
                .shape = shape, .strides = strides, .byte_offset = 0};
  DLTensor y = {//
                .data = y_data, .device = cpu,      .ndim = 1,       .dtype = f32,
                .shape = shape, .strides = strides, .byte_offset = 0};
  if ((ret_code = Initialize())) goto _RAII;
  if ((ret_code = Run(&x, &y))) goto _RAII;

  printf("[ ");
  for (int i = 0; i < 5; ++i) printf("%f ", y_data[i]);
  printf("]\n");

_RAII:
  Finalize(ret_code);
  return ret_code;
}
// [aux.end]
// NOLINTEND(modernize-deprecated-headers,modernize-use-nullptr,bugprone-assignment-in-if-condition,modernize-loop-convert)
