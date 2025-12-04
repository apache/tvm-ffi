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

# CUBIN Launcher

## Overview

Demonstrates loading and executing CUDA kernels from CUBIN files using TVM-FFI. The `cubin_launcher.h` header wraps CUDA Runtime API to provide lightweight CUBIN module and kernel management.

## Techniques

The implementation uses CUDA Runtime API Library Management:

- **`cudaLibraryLoadData()`** - Load CUBIN from memory buffer
- **`cudaLibraryGetKernel()`** - Get kernel handle by name
- **`cudaKernelGetFunction()`** - Get function handle for current CUDA context
- **`cudaLaunchKernel()`** - Launch kernel with grid/block dimensions

Key features:

- Multi-GPU support via CUDA primary contexts
- RAII-based resource management (CubinModule, CubinKernel)
- CUBIN embedding at compile time (via `ld` + `objcopy`)
- TVM-FFI integration for tensor argument passing
- **New:** `TVM_FFI_EMBED_CUBIN` and `TVM_FFI_EMBED_CUBIN_GET_KERNEL` macros for easy CUBIN embedding
- **New:** `embed_cubin` parameter in `tvm_ffi.cpp.load_inline` for seamless CUBIN integration
- **New:** `tvm_ffi.cpp.nvrtc` module for runtime CUDA compilation

## Examples

### 1. Embedded CUBIN

Demonstrates embedding CUBIN data directly into the shared library at build time using the `tvm_ffi_embed_cubin` CMake utility.

**Location:** `embedded_cubin/`

**Build and run:**

```bash
cd examples/cubin_launcher/embedded_cubin
mkdir build && cd build
cmake ..
make
cd ..
python main.py
```

**Key features:**

- CUBIN is embedded at compile time using `ld` and `objcopy`
- No separate CUBIN file needed at runtime
- Symbols are localized to prevent conflicts
- `.note.GNU-stack` section automatically added for security

### 2. Dynamic CUBIN Loading

Demonstrates loading CUBIN data from a file at runtime using the CUDA Driver API.

**Location:** `dynamic_cubin/`

**Build and run:**

```bash
cd examples/cubin_launcher/dynamic_cubin
mkdir build && cd build
cmake ..
make
cd ..
python main.py
```

**Key features:**

- CUBIN loaded from file at runtime
- More flexible - can swap CUBIN files
- Useful for JIT-compiled kernels

### 3. Triton Kernel with Embedded CUBIN (Experimental)

`example_triton_cubin.py` - Triton kernel compiled to CUBIN and embedded inline using the `embed_cubin` parameter.

```bash
# Requires: triton, torch
python examples/cubin_launcher/example_triton_cubin.py
```

### 4. NVRTC with Embedded CUBIN

`example_nvrtc_cubin.py` - CUDA source compiled to CUBIN using NVRTC and embedded inline.

```bash
# Requires: cuda-python, torch
python examples/cubin_launcher/example_nvrtc_cubin.py
```

## Using Embedded CUBIN with `tvm_ffi.cpp.load_inline`

The new `embed_cubin` parameter makes it easy to embed CUBIN binaries into your module:

```python
from tvm_ffi import cpp
from tvm_ffi.cpp import nvrtc

# Compile CUDA source to CUBIN
cuda_source = """
extern "C" __global__ void my_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}
"""
cubin_bytes = nvrtc.nvrtc_compile(cuda_source)

# C++ code using the embedded CUBIN
cpp_code = """
#include <tvm/ffi/extra/cuda/cubin_launcher.h>

TVM_FFI_EMBED_CUBIN(my_module);

void launch_kernel(TensorView data) {
    static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_module, "my_kernel");
    // ... launch kernel
}
"""

# Load with embedded CUBIN
mod = cpp.load_inline(
    "my_module",
    cpp_sources=cpp_code,
    embed_cubin={"my_module": cubin_bytes},
    extra_ldflags=["-lcudart"],
)
```

## Project Structure

### Core Files

- `include/tvm/ffi/extra/cuda/cubin_launcher.h` - Header-only C++ library with CUBIN utilities
- `python/tvm_ffi/utils/embed_cubin.py` - Python utility for embedding CUBIN into object files
- `python/tvm_ffi/cpp/nvrtc.py` - NVRTC compilation utilities
- `cmake/Utils/EmbedCubin.cmake` - CMake utilities (`tvm_ffi_generate_cubin`, `tvm_ffi_embed_cubin`)

### Example Directories

**`embedded_cubin/`** - CUBIN embedded at build time

- `CMakeLists.txt` - Build configuration using `tvm_ffi_embed_cubin`
- `main.py` - Python test script
- `src/lib_embedded.cc` - C++ code using `TVM_FFI_EMBED_CUBIN` macro
- `src/kernel.cu` - CUDA kernels (add_one, mul_two)

**`dynamic_cubin/`** - CUBIN loaded at runtime

- `CMakeLists.txt` - Build configuration using `tvm_ffi_generate_cubin`
- `main.py` - Python test script
- `src/lib_dynamic.cc` - C++ code using `CubinModule::GetKernel()`
- `src/kernel.cu` - CUDA kernels (add_one, mul_two)

**Additional Examples** (at root level)

- `example_triton_cubin.py` - Triton kernel with embedded CUBIN
- `example_nvrtc_cubin.py` - NVRTC compilation with embedded CUBIN
