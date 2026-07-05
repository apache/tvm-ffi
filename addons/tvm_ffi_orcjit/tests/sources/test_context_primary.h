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

#include <stdint.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

#ifndef TVM_FFI_TEST_EXTERN_C
#define TVM_FFI_TEST_EXTERN_C
#endif

#ifdef _MSC_VER
#define TVM_FFI_TEST_WEAK __declspec(selectany)
#else
#define TVM_FFI_TEST_WEAK __attribute__((weak))
#endif

typedef int (*TVMFFITestFunctionCall)(TVMFFIObjectHandle, TVMFFIAny*, int32_t, TVMFFIAny*);
typedef int (*TVMFFITestLookupFromImports)(TVMFFIObjectHandle, const char*, TVMFFIObjectHandle*);
typedef void (*TVMFFITestContextRecorder)(int32_t);

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT TVM_FFI_TEST_WEAK void* __tvm_ffi__library_ctx = NULL;
TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT TVM_FFI_TEST_WEAK TVMFFITestFunctionCall
    __TVMFFIFunctionCall = NULL;
TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT TVM_FFI_TEST_WEAK TVMFFITestLookupFromImports
    __TVMFFIEnvModLookupFromImports = NULL;
TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT TVM_FFI_TEST_WEAK TVMFFITestContextRecorder
    __TVMFFITestContextPrimary = NULL;

enum {
  kPrimaryLibraryContext = 1,
  kPrimaryFunctionCall = 2,
  kPrimaryLookupFromImports = 4,
  kPrimaryCustomContext = 8,
  kPrimaryAllContexts = 15,
};

static int primary_current_contexts(void) {
  int contexts = 0;
  if (__tvm_ffi__library_ctx != NULL) contexts |= kPrimaryLibraryContext;
  if (__TVMFFIFunctionCall != NULL) contexts |= kPrimaryFunctionCall;
  if (__TVMFFIEnvModLookupFromImports != NULL) contexts |= kPrimaryLookupFromImports;
  if (__TVMFFITestContextPrimary != NULL) contexts |= kPrimaryCustomContext;
  return contexts;
}

static int primary_return_int(TVMFFIAny* result, int64_t value) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = value;
  return 0;
}

static void primary_poison_recorder(int32_t event) { (void)event; }

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int tvm_ffi_test_context_primary_ready(void) {
  return primary_current_contexts() == kPrimaryAllContexts;
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_primary_status(void* self,
                                                                              const TVMFFIAny* args,
                                                                              int32_t num_args,
                                                                              TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  return primary_return_int(result, primary_current_contexts());
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_primary_owner(void* self,
                                                                             const TVMFFIAny* args,
                                                                             int32_t num_args,
                                                                             TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  return primary_return_int(result, (int64_t)(uintptr_t)__tvm_ffi__library_ctx);
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_primary_token(void* self,
                                                                             const TVMFFIAny* args,
                                                                             int32_t num_args,
                                                                             TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  return primary_return_int(result, (int64_t)(uintptr_t)__TVMFFITestContextPrimary);
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_poison_primary(void* self,
                                                                              const TVMFFIAny* args,
                                                                              int32_t num_args,
                                                                              TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  __TVMFFITestContextPrimary = primary_poison_recorder;
  return primary_return_int(result, (int64_t)(uintptr_t)__TVMFFITestContextPrimary);
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_call_global(void* self,
                                                                           const TVMFFIAny* args,
                                                                           int32_t num_args,
                                                                           TVMFFIAny* result) {
  static const char name_data[] = "test_orcjit_context_add_five";
  TVMFFIByteArray name = {name_data, sizeof(name_data) - 1};
  TVMFFIObjectHandle function = NULL;
  TVMFFIAny call_arg;
  int status;
  (void)self;

  if (__TVMFFIFunctionCall == NULL) return primary_return_int(result, -1001);
  if (num_args != 1 || args[0].type_index != kTVMFFIInt) {
    return primary_return_int(result, -1002);
  }
  if (TVMFFIFunctionGetGlobal(&name, &function) != 0 || function == NULL) {
    return primary_return_int(result, -1003);
  }
  call_arg = args[0];
  status = __TVMFFIFunctionCall(function, &call_arg, 1, result);
  TVMFFIObjectDecRef(function);
  return status;
}
