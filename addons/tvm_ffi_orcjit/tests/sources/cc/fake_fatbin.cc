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

// Simulates an NVCC-compiled object with a large .nv_fatbin device blob.
// The fatbin data is referenced only by absolute relocations (R_*_64 /
// R_AARCH64_ABS64), never by PC-relative relocations.  This lets us test
// overflow-region classification without needing a real CUDA toolchain.

#include <tvm/ffi/c_api.h>

#include <cstdint>

#ifdef __APPLE__
__attribute__((section("__DATA,.nv_fatbin"), used))
#else
__attribute__((section(".nv_fatbin"), used))
#endif
static const uint8_t fake_fatbin_data[4 * 1024 * 1024] = {0};

// get_fatbin_size: return the size of the fake fatbin blob.
// The reference to fake_fatbin_data generates an absolute relocation
// (R_*_64 / R_AARCH64_ABS64), NOT PC-relative.
extern "C" {
TVM_FFI_DLL_EXPORT int __tvm_ffi_get_fatbin_size(void* self, const TVMFFIAny* args,
                                                 int32_t num_args, TVMFFIAny* result) {
  volatile const void* p = fake_fatbin_data;
  (void)p;
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = static_cast<int64_t>(sizeof(fake_fatbin_data));
  return 0;
}
}
