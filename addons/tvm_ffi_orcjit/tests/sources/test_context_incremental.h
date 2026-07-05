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
TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT TVM_FFI_TEST_WEAK TVMFFITestContextRecorder
    __TVMFFITestContextIncremental = NULL;

TVM_FFI_TEST_EXTERN_C int tvm_ffi_test_context_primary_ready(void);

enum {
  kIncrementalLibraryContext = 1,
  kIncrementalFunctionCall = 2,
  kIncrementalLookupFromImports = 4,
  kIncrementalPrimaryContext = 8,
  kIncrementalNewContext = 16,
  kIncrementalAllContexts = 31,
  kIncrementalPrimaryReady = 32,
};

static int incremental_return_int(TVMFFIAny* result, int64_t value) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = value;
  return 0;
}

static int incremental_current_contexts(void) {
  int contexts = 0;
  if (__tvm_ffi__library_ctx != NULL) contexts |= kIncrementalLibraryContext;
  if (__TVMFFIFunctionCall != NULL) contexts |= kIncrementalFunctionCall;
  if (__TVMFFIEnvModLookupFromImports != NULL) contexts |= kIncrementalLookupFromImports;
  if (__TVMFFITestContextPrimary != NULL) contexts |= kIncrementalPrimaryContext;
  if (__TVMFFITestContextIncremental != NULL) contexts |= kIncrementalNewContext;
  if (tvm_ffi_test_context_primary_ready()) contexts |= kIncrementalPrimaryReady;
  return contexts;
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_incremental_status(
    void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  return incremental_return_int(result, incremental_current_contexts());
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_incremental_primary_token(
    void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  return incremental_return_int(result, (int64_t)(uintptr_t)__TVMFFITestContextPrimary);
}

TVM_FFI_TEST_EXTERN_C TVM_FFI_DLL_EXPORT int __tvm_ffi_context_incremental_new_token(
    void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  return incremental_return_int(result, (int64_t)(uintptr_t)__TVMFFITestContextIncremental);
}
