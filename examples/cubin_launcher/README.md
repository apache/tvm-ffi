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

The implementation supports both CUDA Runtime API (CUDA >= 12.8) and Driver API for Library Management.

**Runtime API (CUDA >= 12.8):**

- **`cudaLibraryLoadData()`** - Load CUBIN from memory buffer
- **`cudaLibraryGetKernel()`** - Get kernel handle by name
- **`cudaLaunchKernel()`** - Launch kernel with grid/block dimensions

**Driver API:**

- **`cuLibraryLoadData()`** - Load CUBIN from memory buffer
- **`cuLibraryGetKernel()`** - Get kernel handle by name
- **`cuLaunchKernel()`** - Launch kernel with grid/block dimensions

**Customization:**

By default, the implementation uses the Runtime API if compiled with CUDA >= 12.8, falling back to the Driver API for older versions. You can force the usage of the Driver API (or Runtime API) by defining the macro `TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API` (set to `1` for Driver API, `0` for Runtime API) before including the header.

Key features:

- Multi-GPU support via CUDA primary contexts
- RAII-based resource management (CubinModule, CubinKernel)
- CUBIN embedding at compile time
  - Object linking (via `ld` + `objcopy`)
  - Header inclusion (via `bin2c`)
  - Modern C++ embedding (via `#embed`)
- TVM-FFI integration for tensor argument passing
- **Macros:**
  - `TVM_FFI_EMBED_CUBIN`: Declare symbols for object-linked CUBIN
  - `TVM_FFI_EMBED_CUBIN_FROM_BYTES`: Load CUBIN from byte array (for `#embed`/`bin2c`)
  - `TVM_FFI_EMBED_CUBIN_GET_KERNEL`: Helper to retrieve kernels
- **Python Integration:** `embed_cubin` parameter in `tvm_ffi.cpp.load_inline` for seamless CUBIN integration
- **Runtime Compilation:** `tvm_ffi.cpp.nvrtc` module for runtime CUDA compilation

## Examples

### 1. Embedded CUBIN

The `embedded_cubin` directory contains three examples demonstrating different embedding techniques.

#### 1.1 Object Linking (Standard)

Demonstrates embedding CUBIN data directly into the shared library at build time using the `tvm_ffi_embed_bin_into` CMake utility. This is the most robust method for CMake projects.

**Location:** `embedded_cubin/embed_with_tvm_ffi/`

**Build and run:**

```bash
cd examples/cubin_launcher/embedded_cubin/embed_with_tvm_ffi
mkdir build && cd build
cmake ..
make
cd ..
python main.py
```

#### 1.2 Header Inclusion (Portable)

Demonstrates converting the CUBIN to a C header file using `bin2c` and including it in the C++ source. This is highly portable and works with any compiler.

**Location:** `embedded_cubin/include_bin2c/`

**Build and run:**

```bash
cd examples/cubin_launcher/embedded_cubin/include_bin2c
mkdir build && cd build
cmake ..
make
cd ..
python main.py
```

#### 1.3 C++ Embedding (Modern)

Demonstrates using C++23 `#embed` (or compiler extensions in GCC/Clang) to directly include binary data. This is the cleanest approach for modern toolchains.

**Location:** `embedded_cubin/cpp_embed/`

**Build and run:**

```bash
cd examples/cubin_launcher/embedded_cubin/cpp_embed
mkdir build && cd build
cmake ..
make
cd ..
python main.py
```

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
- `cmake/Utils/EmbedCubin.cmake` - CMake utilities

### Example Directories

**`embedded_cubin/`** - Different CUBIN embedding techniques:

- `embed_with_tvm_ffi/` - Standard object linking
- `include_bin2c/` - Header inclusion
- `cpp_embed/` - Modern C++ `#embed`

**`dynamic_cubin/`** - CUBIN loaded at runtime

**Additional Examples** (at root level)

- `example_triton_cubin.py` - Triton kernel with embedded CUBIN
- `example_nvrtc_cubin.py` - NVRTC compilation with embedded CUBIN
