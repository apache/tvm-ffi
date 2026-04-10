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

// Caller library for ADRP overflow test (LLVM issue #173269).
//
// Takes the ADDRESS of hidden_helper_add via ADRP+ADD (no GOT,
// because of hidden visibility).  When this object and
// test_hidden_helper.o are in different mmap allocations >4GB
// apart, the ADRP immediate overflows — silent truncation on
// AArch64 causes a segfault.
//
// The arena memory manager fixes this by placing all objects
// in contiguous VA space (<< 4GB).

#include <tvm/ffi/c_api.h>

#include <cstdint>

// Same hidden declaration — compiler uses ADRP+ADD to take address
__attribute__((visibility("hidden"))) extern int64_t hidden_helper_add(int64_t a, int64_t b);

using binop_t = int64_t (*)(int64_t, int64_t);

// call_hidden_add: take address of hidden_helper_add, then call via pointer.
// On AArch64, generates:
//   ADRP x0, hidden_helper_add@PAGE       (R_AARCH64_ADR_PREL_PG_HI21, ±4GB)
//   ADD  x0, x0, hidden_helper_add@PAGEOFF (R_AARCH64_ADD_ABS_LO12_NC)
// When hidden_helper_add is in a different allocation >4GB away, ADRP overflows.
extern "C" {
TVM_FFI_DLL_EXPORT int __tvm_ffi_call_hidden_add(void* self, const TVMFFIAny* args,
                                                 int32_t num_args, TVMFFIAny* result) {
  volatile binop_t fn = &hidden_helper_add;
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = fn(args[0].v_int64, args[1].v_int64);
  return 0;
}
}
