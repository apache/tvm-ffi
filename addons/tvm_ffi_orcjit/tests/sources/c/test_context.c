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

#include <tvm/ffi/c_api.h>

// The host writes this slot after compilation and before constructors run.
// Volatile keeps the constructor's read from being folded to NULL.
TVM_FFI_DLL_EXPORT void* volatile __tvm_ffi__library_ctx = NULL;
static int context_was_set_during_init = 0;

static void record_context_during_init(void) {
  context_was_set_during_init = __tvm_ffi__library_ctx != NULL;
}

#ifdef _MSC_VER
typedef void(__cdecl* ctor_t)(void);
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) ctor_t __tvm_test_context_init = record_context_during_init;
#else
__attribute__((constructor)) static void context_init(void) { record_context_during_init(); }
#endif

TVM_FFI_DLL_EXPORT int __tvm_ffi_context_is_set(void* self, const TVMFFIAny* args, int32_t num_args,
                                                TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = __tvm_ffi__library_ctx != NULL;
  return 0;
}

TVM_FFI_DLL_EXPORT int __tvm_ffi_context_was_set_during_init(void* self, const TVMFFIAny* args,
                                                             int32_t num_args, TVMFFIAny* result) {
  (void)self;
  (void)args;
  (void)num_args;
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = context_was_set_during_init;
  return 0;
}
