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
#include <tvm/ffi/extra/c_env_api.h>

// This is a raw C variant of the add_one_cpu function
// it is used to demonstrate how low-level mechanism works
// to construct a tvm ffi compatible function
//
// This function can also serve as a reference for how to implement
// a compiler codegen to target tvm ffi
//
// if you are looking for a more high-level way to construct a tvm ffi compatible function,
// please refer to the add_one_cpu.cc instead
/*!
 * \brief Helper code to read DLTensor from TVMFFIAny, can be inlined into generated code
 * \param value The TVMFFIAny to read from
 * \param out The DLTensor to read into
 * \return 0 on success, -1 on error
 */
int ReadDLTensorPtr(const TVMFFIAny* value, DLTensor** out) {
  if (value->type_index == kTVMFFIDLTensorPtr) {
    *out = (DLTensor*)(value->v_ptr);
    return 0;
  }
  if (value->type_index != kTVMFFITensor) {
    // Use TVMFFIErrorSetRaisedFromCStr to set an error which will
    // be propagated to the caller
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input");
    return -1;
  }
  *out = (DLTensor*)((char*)(value->v_obj) + sizeof(TVMFFIObject));
  return 0;
}

// FFI function implementing add_one operation
int __tvm_ffi_add_one_c(                                                      //
    void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result  //
) {
  DLTensor *x, *y;
  // Extract tensor arguments
  // return -1 for error, error is set through TVMFFIErrorSetRaisedFromCStr
  if (ReadDLTensorPtr(&args[0], &x) == -1) return -1;
  if (ReadDLTensorPtr(&args[1], &y) == -1) return -1;

  // Get current stream for device synchronization (e.g., CUDA)
  // not needed for CPU, just keep here for demonstration purpose
  void* stream = TVMFFIEnvGetStream(x->device.device_type, x->device.device_id);

  // perform the actual operation
  for (int i = 0; i < x->shape[0]; ++i) {
    ((float*)(y->data))[i] = ((float*)(x->data))[i] + 1;
  }
  // return 0 for success run
  return 0;
}
