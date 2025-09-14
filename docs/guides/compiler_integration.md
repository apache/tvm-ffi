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

# Compiler Integration

TVM FFI is a standard ABI designed as a standalone module
that is independent from compiler or intermediate representation implementations.
It specifies a runtime ABI that DSL compilers and languages can integrate with.

## Kernel Language Compilers

Kernel languages such as OpenAI Triton, TileLang, Mojo, cuteDSL, Helion,
and Hidet usually leverage their own internal compilation mechanisms to
build code. To connect these functions to the FFI convention, one can use the
following options:

- For compilers that generate host functions via codegen (e.g., LLVM), one can
  generate the symbol `__tvm_ffi_<func_name>`, where `<funcname>` is the exported
  function.
- For kernel generators that generate C++ host code, one can directly
  use {c:macro}`TVM_FFI_DLL_EXPORT_TYPED_FUNC` to expose the symbol.

The following code snippet shows C code that corresponds to a
function performing `add_one` under the ABI. It is reasonably straightforward for
low-level code generators to replicate this C logic.

```c
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

// Helper function to extract DLTensor from TVMFFIAny (can be inlined into generated code)
int ReadDLTensorPtr(const TVMFFIAny *value, DLTensor* out) {
  if (value->type_index == kTVMFFIDLTensorPtr) {
    *out = static_cast<DLTensor*>(value->v_ptr);
    return 0;
  }
  if (value->type_index == kTVMFFITensor) {
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input");
    return -1;
  }
  *out = reinterpret_cast<DLTensor*>(
    reinterpret_cast<char*>(value->v_obj) + sizeof(TVMFFIObject));
  return 0;
}

// FFI function implementing add_one operation
int __tvm_ffi_add_one(
  void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
) {
  DLTensor *a, *b, *c;
  // Extract tensor arguments
  if (ReadDLTensorPtr(&args[0], &a) == -1) return -1;
  if (ReadDLTensorPtr(&args[1], &b) == -1) return -1;
  if (ReadDLTensorPtr(&args[2], &c) == -1) return -1;

  // Get current stream for device synchronization (e.g., CUDA)
  void* stream = TVMFFIEnvGetStream(a->device.device_type, a->device.device_id);

  // Generated computation code would follow here to perform the actual operation
  // on tensors a, b, c and store result in c
  return 0;
}
```

Some of the key takeaways include:
- Prefix the symbol with `__tvm_ffi_`
- Call {cpp:func}`TVMFFIEnvGetStream` to get the current environment stream
- Use return value for error handling, set error via {cpp:func}`TVMFFIErrorSetRaisedFromCStr`.

You can also check out the [ABI overview](../concepts/abi_overview.md) for a more complete guide.


## Graph Compilers

Machine learning graph compilers take computational graphs and can integrate with TVM FFI through:

- Supporting the `call_tvm_ffi` primitive that calls into `my_func` that follows the ABI:
```python
Op.call_tvm_ffi("my_func", *args)
```
- Using the module API to load the modules into context and run. Alternatively, look up
  global functions that are registered and invoke them.
- For ahead-of-time compilation (AOT) with minimum runtime, the AOT compiler can generate
  direct calls into FFI functions:
  - Use the TVMFFIFunctionCall API to call into custom {cpp:class}`tvm::ffi::Function`s
  - If the function exposes a C symbol following the FFI ABI, call it directly.

This approach provides a unified mechanism to call into any libraries and other DSLs
that expose kernels following the FFI convention, enabling seamless interoperability
with various kernel DSLs and libraries.

## Advanced: Custom Modules

While the standard dynamic library module is sufficient for many use cases,
sometimes it may be helpful to package a custom runtime module that wraps over a driver API.
For example, using `cuModuleLoad` explicitly to load generated PTX code and expose it as an {cpp:class}`tvm::ffi::Function`.
The {cpp:class}`tvm::ffi::ModuleObj` interface provides a way to support this need.
Generally, the steps include subclassing the {cpp:class}`tvm::ffi::ModuleObj`:

- Provide a specific {cpp:func}`tvm::ffi::ModuleObj::kind` string to identify the module type
- Override {cpp:func}`tvm::ffi::ModuleObj::GetPropertyMask` to indicate:
  - The module is `ffi::Module::kRunnable` (executable)
  - If binary serialization is supported, also add `ffi::Module::kBinarySerializable`
- Override {cpp:func}`tvm::ffi::ModuleObj::GetFunction` to specify how functions loaded
- Register binary serialization/deserialization {cpp:func}`tvm::ffi::ModuleObj::SaveToBytes` and register a global
  function `ffi.Module.load_from_bytes.<kind>`

### Enable Export and Loading

We also support export and loading of modules that import custom modules.
We allow libraries to embed a binary symbol `__tvm_ffi__library_bin` in the following binary layout:

- `<nbytes : u64> <import_tree> <key0: str> [val0: bytes] <key1: str> [val1: bytes] ...`
- `nbytes` indicates the total number of bytes following the nbytes header
- `<import_tree>` uses CSR sparse array format: `<indptr: vec<u64>> <child_indices: vec<u64>>`
  to store child indices of each node (each node is a Module instance)
- `<key>` stores the module kind, or can be `_lib`:
  - `_lib` indicates the module corresponds to the dynamic library itself
  - For other cases, `val: bytes` contains the serialized bytes from the custom module
- Both `bytes` and `str` are serialized as `<size: u64> <content>`

This information allows us to deserialize the custom modules by calling `ffi.Module.load_from_bytes.<kind>` and then reconstruct
the overall import relations from `<import_tree>` and return the final composed modules back to the user.
As long as the compiler generates the `__tvm_ffi__library_bin` in the above format, {py:func}`tvm_ffi.load_module` will correctly
handle the loading and recover the original module. Note that we will need the custom module class definition to be available
during loading, either by importing another runtime DLL, or embedding it in the generated library.

