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

Demonstrates loading and executing CUDA kernels from CUBIN files using TVM-FFI. The `cubin_launcher.h` header wraps CUDA Driver API to provide lightweight CUBIN module and kernel management.

## Techniques

The implementation uses CUDA Driver API Library Management:

- **`cuLibraryLoadData()`** - Load CUBIN from memory buffer
- **`cuLibraryGetKernel()`** - Get kernel handle by name
- **`cuKernelGetFunction()`** - Get function handle for current CUDA context
- **`cuLaunchKernel()`** - Launch kernel with grid/block dimensions

Key features:

- Multi-GPU support via CUDA primary contexts
- RAII-based resource management (CubinModule, CubinKernel)
- CUBIN embedding at compile time (via `ld` + `objcopy`)
- TVM-FFI integration for tensor argument passing
- **New:** `TVM_FFI_EMBED_CUBIN` and `TVM_FFI_EMBED_CUBIN_GET_KERNEL` macros for easy CUBIN embedding
- **New:** `embed_cubin` parameter in `tvm_ffi.cpp.load_inline` for seamless CUBIN integration
- **New:** `tvm_ffi.cpp.nvrtc` module for runtime CUDA compilation

## Examples

### 1. Embedded CUBIN (TVM-FFI Library)

`example_embeded_cubin.py` - CUBIN linked into shared library at build time.

```bash
cd build
cmake ..
make
cd ..
python examples/cubin_launcher/example_embedded_cubin.py
```

### 2. Dynamic CUBIN Loading (TVM-FFI Library)

`example_dynamic_cubin.py` - CUBIN data passed dynamically at runtime.

```bash
python examples/cubin_launcher/example_dynamic_cubin.py
```

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
    extra_ldflags=["-lcuda"],
)
```

## Files

- `include/tvm/ffi/extra/cubin_launcher.h` - Header-only C++ library with embedding macros
- `python/tvm_ffi/cpp/nvrtc.py` - NVRTC compilation utilities
- `src/lib_embedded.cc` - Embedded CUBIN example (lib_embedded.so)
- `src/lib_dynamic.cc` - Dynamic CUBIN loading example (lib_dynamic.so)
- `src/kernel.cu` - CUDA kernels (add_one, mul_two)
- `example_embedded_cubin.py` - Pre-compiled CUBIN embedded at build time
- `example_dynamic_cubin.py` - CUBIN data passed dynamically at runtime
- `example_triton_cubin.py` - Triton kernel with embedded CUBIN
- `example_nvrtc_cubin.py` - NVRTC compilation with embedded CUBIN
- `CMakeLists.txt` - Build configuration
