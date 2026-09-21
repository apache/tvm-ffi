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
 * Base object for the intra-module linking test.
 * Exports helper_add which is called by test_link_order_caller.c.
 */
#include <tvm/ffi/c_api.h>

/* helper_add: add two integers — used as a dependency by the caller library */
TVM_FFI_DLL_EXPORT int __tvm_ffi_helper_add(void* self, const TVMFFIAny* args, int32_t num_args,
                                            TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = args[0].v_int64 + args[1].v_int64;
  return 0;
}
