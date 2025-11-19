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
  function. Optionally, also generate `__tvm_ffi__metadata_<func_name>` for reflection.
- For kernel generators that generate C++ host code, use {c:macro}`TVM_FFI_DLL_EXPORT_TYPED_FUNC`.
  This macro automatically exports function metadata when {c:macro}`TVM_FFI_DLL_EXPORT_INCLUDE_METADATA`
  is set to 1.
- To export documentation strings, use {c:macro}`TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC` separately
  after exporting the function. This enhances tooling support (stub generation, IDE tooltips).
  Documentation export is also controlled by {c:macro}`TVM_FFI_DLL_EXPORT_INCLUDE_METADATA`.

The following code snippet shows C code that corresponds to a
function performing `add_one_c` under the ABI. It is reasonably straightforward for
low-level code generators to replicate this C logic.
You can run this code as part of the [quick start example](https://github.com/apache/tvm-ffi/tree/dev/examples/quick_start).

```c
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

// Helper function to extract DLTensor from TVMFFIAny (can be inlined into generated code)
int ReadDLTensorPtr(const TVMFFIAny *value, DLTensor** out) {
  if (value->type_index == kTVMFFIDLTensorPtr) {
    *out = (DLTensor*)(value->v_ptr);
    return 0;
  }
  if (value->type_index != kTVMFFITensor) {
    // Use TVMFFIErrorSetRaisedFromCStr / TVMFFIErrorSetRaisedFromCStrParts to set an error which will
    // be propagated to the caller
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input");
    return -1;
  }
  *out = (DLTensor*)((char*)(value->v_obj) + sizeof(TVMFFIObject));
  return 0;
}

// FFI function implementing add_one operation
int __tvm_ffi_add_one_c(
  void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result
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
```

Some of the key takeaways include:

- Prefix the symbol with `__tvm_ffi_`
- Call {cpp:func}`TVMFFIEnvGetStream` to get the current environment stream
- Use return value for error handling, set error via {cpp:func}`TVMFFIErrorSetRaisedFromCStr`
  or {cpp:func}`TVMFFIErrorSetRaisedFromCStrParts`.

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

## Runtime and State Management for Compilers

While TVM FFI provides a standard ABI for compiler-generated kernels, many compilers and domain-specific languages
(DSLs) require their own **runtime** to manage states like dynamic shapes, workspace memory, or other
application-specific data. This runtime can be a separate shared library accessible to all kernels from a specific
compiler.

### Recommended Approach for State Management

The recommended approach for managing compiler-specific state is to define the state within a **separate shared library**.
This library exposes its functionality by registering functions as global `tvm::ffi::Function`s.

Here's a breakdown of the process:

1. **Define a Global State**: Create a class or structure to hold your compiler's runtime state. A simple singleton pattern is often used for this.
2. **Register Global Functions**: Use the `TVM_FFI_STATIC_INIT_BLOCK()` macro to register a global function that returns a pointer to your state. For example:

   ```c++
   class GlobalState {
     ... // your state variables here
    public:
      GlobalState* Global() {
         static auto *inst = new GlobalState();
         return inst;
      }
   };
   TVM_FFI_STATIC_INIT_BLOCK() {
     using refl = tvm::ffi::reflection;
     refl.GlobalDef().def("mylang.get_global_state", []()-> void*{ return GlobalState::Global()});
     // other runtime APIs can be registered here
   }
   ```

   This method allows both C++ and Python to access the runtime state through a consistent API.
3. **Access State from Kernels**: Within your compiler-generated kernels, you can use
    `GetGlobalRequired("mylang.get_global_state")` in C++ or the C equivalent
    `TVMFFIGetGlobalFunction("mylang.get_global_state", ...)` to get the function and then call it to retrieve the state
    pointer.

### Distributing the Runtime

For a user to use a kernel from your compiler, they must have access to your runtime library. The preferred method is to
package the runtime shared library (e.g., `libmylang_runtime.so`) as part of a Python or C++ package. Users must install
and import this package before loading any kernels compiled by your system.
This approach ensures the state is shared among different kernels.

### Common vs. Custom State

It's important to distinguish between compiler-specific state and **common state** managed by TVM FFI. TVM FFI handles
common states like **streams** and **memory allocators** through environment functions (e.g., `TVMFFIEnvGetStream`),
allowing kernels to access these without managing their own. However, for any unique state required by your compiler,
the global function registration approach is the most suitable method.

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
