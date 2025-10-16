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
# Quick Start

This is a quick start guide explaining the basic features and usage of tvm-ffi.
The source code can be found at `examples/quick_start` in the project source.

## Build and Run the Example

Let us first get started by build and run the example. The example will show us:

- How to expose c++ functions as tvm ffi ABI function
- How to load and run tvm-ffi based library from python
- How to load and run tvm-ffi based library from c++

Before starting, ensure you have:

- TVM FFI installed
- C++ compiler with C++17 support
- CMake 3.18 or later
- (Optional) Ninja build system (the quick-start uses Ninja for fast incremental builds)
- (Optional) CUDA toolkit for GPU examples
- (Optional) PyTorch for checking torch integrations

Then obtain a copy of the tvm-ffi source code.

```bash
git clone https://github.com/apache/tvm-ffi --recursive
cd tvm-ffi
```

The examples are now in the example folder, you can quickly build
the example using the following command.

```bash
cd examples/quick_start

# with ninja or omit -G Ninja to use default generator
cmake --fresh -G Ninja -B build -S .
cmake --build build --parallel
```

After the build finishes, you can run the python examples by

```bash
python run_example.py
```

You can also run the c++ example

```bash
./build/run_example
```

If the CUDA toolkit is available, the GPU demo binary is built alongside the CPU sample:

```bash
./build/run_example_cuda
```

## Walk through the Example

Now we have quickly try things out. Let us now walk through the details of the example.
Specifically, in this example, we create a simple "add one" operation that adds 1 to each element of an input
tensor and expose that function as TVM FFI compatible function. The key file structures are as follows:

```text
examples/quick_start/
├── src/
│   ├── add_one_cpu.cc      # CPU implementation
│   ├── add_one_c.c         # A low-level C based implementation
│   ├── add_one_cuda.cu     # CUDA implementation
│   ├── run_example.cc      # C++ usage example
│   └── run_example_cuda.cc # C++ with CUDA kernel usage example
├── run_example.py          # Python usage example
├── run_example.sh          # Build and run script
└── CMakeLists.txt          # Build configuration
```

### CPU Implementation

```cpp
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>

namespace tvm_ffi_example {

namespace ffi = tvm::ffi;

void AddOne(ffi::TensorView x, ffi::TensorView y) {
  // Validate inputs
  TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
  DLDataType f32_dtype{kDLFloat, 32, 1};
  TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
  TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
  TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
  TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";

  // Perform the computation
  for (int i = 0; i < x.size(0); ++i) {
    static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
  }
}

// Expose the function through TVM FFI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, tvm_ffi_example::AddOne);
}
```

**Key Points:**

- Functions take `tvm::ffi::Tensor` parameters for cross-language compatibility
- The `TVM_FFI_DLL_EXPORT_TYPED_FUNC` macro exposes the function with a given name

### CUDA Implementation

```cpp
void AddOneCUDA(ffi::TensorView x, ffi::TensorView y) {
  // Validation (same as CPU version)
  // ...

  int64_t n = x.size(0);
  int64_t nthread_per_block = 256;
  int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;

  // Get current CUDA stream from environment
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

  // Launch kernel
  AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(
      static_cast<float*>(x.data_ptr()), static_cast<float*>(y.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, tvm_ffi_example::AddOneCUDA);
```

**Key Points:**

- We use `TVMFFIEnvGetStream` to obtain the current stream from the environement
- When invoking ffi Function from python end with PyTorch tensor as argument,
  the stream will be populated with torch's current stream.

### Working with PyTorch

Atfer build, we will create library such as `build/add_one_cuda.so`, that can be loaded by
with api {py:func}`tvm_ffi.load_module` that returns a {py:class}`tvm_ffi.Module`
Then the function will become available as property of the loaded module.
The tensor arguments in the ffi functions automatically consumes `torch.Tensor`. The following code shows how
to use the function in torch.

```python
import torch
import tvm_ffi

if torch.cuda.is_available():
    mod = tvm_ffi.load_module("build/add_one_cuda.so")

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    # TVM FFI automatically handles CUDA streams
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        mod.add_one_cuda(x, y)
    stream.synchronize()
```

### Working with Python Data Arrays

TVM FFI functions works automaticaly with python data arrays that are compatible with dlpack.
The following examples how to use the function with numpy.

```python
import tvm_ffi
import numpy as np

# Load the compiled module
mod = tvm_ffi.load_module("build/add_one_cpu.so")

# Create input and output arrays
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.empty_like(x)

# Call the function
mod.add_one_cpu(x, y)
print("Result:", y)  # [2, 3, 4, 5, 6]
```

### Working with C++

One important design goal of tvm-ffi is to be universally portable.
As a result, the result libraries do not have explicit dependencies in python
and can be loaded in other language environments, such as c++. The following code
shows how to run the example exported function in C++.

```cpp
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>

namespace ffi = tvm::ffi;

void CallAddOne(ffi::TensorView x, ffi::TensorView y) {
  ffi::Module mod = ffi::Module::LoadFromFile("build/add_one_cpu.so");
  ffi::Function add_one_cpu = mod->GetFunction("add_one_cpu").value();
  add_one_cpu(x, y);
}
```

## Advanced: Minimal C ABI demonstration

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
- The function follows signaure of `TVMFFISafeCallType`
- Use `TVMFFIAny` to handle dynamic argument types
- Return `0` for success, `-1` for error (set via `TVMFFIErrorSetRaisedFromCStr`)
- This function can be compiled using a c compiler and loaded in the same one as
  other libraries in this example.

## Summary Key Concepts

- **TVM_FFI_DLL_EXPORT_TYPED_FUNC** exposes a c++ function into tvm-ffi C ABI
- **ffi::Tensor** is a universal tensor structure that enables zero-copy exchange of array data
- **Module loading** is provided by tvm ffi APIs in multiple languages.
- **C ABI** is provided for easy low-level integration
