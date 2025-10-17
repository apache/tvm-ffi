<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
# Stable C ABI

**C ABI** is provided for easy low-level integration.

For those who need to understand the low-level C ABI or are implementing
compiler codegen, we also provided an example that is C only as follows:

```c
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

// Helper to extract DLTensor from TVMFFIAny
int ReadDLTensorPtr(const TVMFFIAny *value, DLTensor** out) {
  if (value->type_index == kTVMFFIDLTensorPtr) {
    *out = (DLTensor*)(value->v_ptr);
    return 0;
  }
  if (value->type_index != kTVMFFITensor) {
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input");
    return -1;
  }
  *out = (DLTensor*)((char*)(value->v_obj) + sizeof(TVMFFIObject));
  return 0;
}

// Raw C FFI function
int __tvm_ffi_add_one_c(
  void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
) {
  DLTensor *x, *y;

  // Extract tensor arguments
  if (ReadDLTensorPtr(&args[0], &x) == -1) return -1;
  if (ReadDLTensorPtr(&args[1], &y) == -1) return -1;

  // Get current stream for device synchronization (e.g., CUDA)
  // not needed for CPU, just keep here for demonstration purpose
  void* stream = TVMFFIEnvGetStream(x->device.device_type, x->device.device_id);

  // Perform computation
  for (int i = 0; i < x->shape[0]; ++i) {
    ((float*)(y->data))[i] = ((float*)(x->data))[i] + 1;
  }
  return 0;  // Success
}
```

To compile this code, you need to add {py:func}`tvm_ffi.libinfo.find_include_paths` to your include
path and link the shared library that can be found through {py:func}`tvm_ffi.libinfo.find_libtvm_ffi`.
We also provide command line tools to link, so you can compile with the following command:

```bash
gcc -shared -fPIC `tvm-ffi-config --cflags`  \
    src/add_one_c.c -o build/add_one_c.so    \
    `tvm-ffi-config --ldflags` `tvm-ffi-config --libs`
```

The main takeaway points are:

- Function symbols follow name `int __tvm_ffi_<name>`
- The function follows signature of `TVMFFISafeCallType`
- Use `TVMFFIAny` to handle dynamic argument types
- Return `0` for success, `-1` for error (set via `TVMFFIErrorSetRaisedFromCStr`)
- This function can be compiled using a c compiler and loaded in the same one as
  other libraries in this example.
