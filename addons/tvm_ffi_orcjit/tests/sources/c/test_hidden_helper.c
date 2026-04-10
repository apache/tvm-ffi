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
 * Helper library for ADRP overflow test.
 * Defines a hidden-visibility function whose ADDRESS is taken
 * by test_hidden_caller.c.  On AArch64, the caller uses
 * ADRP+ADD (no GOT) to compute the address — this overflows
 * when the two objects are in different allocations >4GB apart.
 *
 * Reference: LLVM issue #173269
 */
#include <stdint.h>
#include <tvm/ffi/c_api.h>

/* Hidden visibility: caller uses ADRP+ADD instead of GOT */
__attribute__((visibility("hidden"))) int64_t hidden_helper_add(int64_t a, int64_t b) {
  return a + b;
}

/* Export a TVM FFI function that calls hidden_helper_add directly */
TVM_FFI_DLL_EXPORT int __tvm_ffi_hidden_add(void* self, const TVMFFIAny* args, int32_t num_args,
                                            TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = hidden_helper_add(args[0].v_int64, args[1].v_int64);
  return 0;
}

/* Return the address of this function's code — for co-location tests */
TVM_FFI_DLL_EXPORT int __tvm_ffi_helper_code_address(void* self, const TVMFFIAny* args,
                                                     int32_t num_args, TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = (int64_t)(uintptr_t)&__tvm_ffi_helper_code_address;
  return 0;
}
