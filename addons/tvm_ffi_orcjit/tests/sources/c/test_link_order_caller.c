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
 * Caller object for the intra-module linking test.
 * References __tvm_ffi_helper_add from test_link_order_base.c via extern
 * declaration, and exports cross_lib_add which forwards to it.
 */
#include <tvm/ffi/c_api.h>

/* Declare the external symbol from the base object. */
extern int __tvm_ffi_helper_add(void* self, const TVMFFIAny* args, int32_t num_args,
                                TVMFFIAny* result);

/* cross_lib_add: forwards to helper_add in the base object. */
TVM_FFI_DLL_EXPORT int __tvm_ffi_cross_lib_add(void* self, const TVMFFIAny* args, int32_t num_args,
                                               TVMFFIAny* result) {
  return __tvm_ffi_helper_add(self, args, num_args, result);
}
