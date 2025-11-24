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

# CUBIN Launcher Guide

This guide demonstrates how to load and launch CUDA kernels from CUBIN (CUDA Binary) modules using TVM-FFI. The CUBIN launcher enables you to execute pre-compiled or runtime-compiled CUDA kernels efficiently through the CUDA Driver API.

## Overview

TVM-FFI provides utilities for loading and launching CUDA kernels from CUBIN modules. The implementation is in `tvm/ffi/extra/cuda/cubin_launcher.h` and provides:

- {cpp:class}`tvm::ffi::CubinModule`: RAII wrapper for loading CUBIN modules from memory
- {cpp:class}`tvm::ffi::CubinKernel`: Handle for launching CUDA kernels with specified parameters
- {c:macro}`TVM_FFI_EMBED_CUBIN`: Macro for embedding CUBIN data at compile time
- {c:macro}`TVM_FFI_EMBED_CUBIN_GET_KERNEL`: Macro for retrieving kernels from embedded CUBIN

The CUBIN launcher supports:

- Loading CUBIN from memory (embedded data or runtime-generated)
- Multi-GPU execution using CUDA primary contexts
- Kernel parameter management and launch configuration
- Integration with NVRTC, Triton, and other CUDA compilation tools

## Python Usage

### Basic Workflow

The typical workflow for launching CUBIN kernels from Python involves:

1. **Generate CUBIN**: Compile your CUDA kernel to CUBIN format
2. **Define C++ Wrapper**: Write C++ code to load and launch the kernel
3. **Load Module**: Use {py:func}`tvm_ffi.cpp.load_inline` with `embed_cubin` parameter
4. **Call Kernel**: Invoke the kernel function from Python

### Example: NVRTC Compilation

Here's a complete example using NVRTC to compile CUDA source at runtime:

```python
import torch
from tvm_ffi import cpp
from tvm_ffi.cpp import nvrtc

# Step 1: Define CUDA kernel source
cuda_source = """
extern "C" __global__ void add_one(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + 1.0f;
    }
}
"""

# Step 2: Compile to CUBIN using NVRTC
cubin_bytes = nvrtc.nvrtc_compile(cuda_source, name="kernel.cu")

# Step 3: Define C++ wrapper with embedded CUBIN
cpp_wrapper = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

// Declare embedded CUBIN module
TVM_FFI_EMBED_CUBIN(my_cubin);

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Get kernel from embedded CUBIN (cached for efficiency)
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_cubin, "add_one");

  // Prepare kernel arguments
  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();
  void* args[] = {&x_ptr, &y_ptr, &n};

  // Configure launch parameters
  tvm::ffi::dim3 grid((n + 255) / 256);
  tvm::ffi::dim3 block(256);

  // Get CUDA stream and launch
  DLDevice device = x.device();
  CUstream stream = static_cast<CUstream>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  CUresult result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, AddOne);
"""

# Step 4: Load module with embedded CUBIN
mod = cpp.load_inline(
    "my_module",
    cuda_sources=cpp_wrapper,
    embed_cubin={"my_cubin": cubin_bytes}
)

# Step 5: Use the kernel
x = torch.arange(1024, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)
mod.add_one(x, y)

# Verify results
assert torch.allclose(y, x + 1)
```

**Key Points:**

- The `embed_cubin` parameter is a dictionary mapping CUBIN names to their binary data
- CUBIN names in `embed_cubin` must match names in {c:macro}`TVM_FFI_EMBED_CUBIN`
- Use `cuda_sources` parameter (instead of `cpp_sources`) to automatically link with CUDA libraries
- The C++ wrapper handles device management, stream handling, and kernel launching

### Example: Using Triton Kernels

You can compile Triton kernels to CUBIN and launch them through TVM-FFI:

```python
import torch
import triton
import triton.language as tl
from tvm_ffi import cpp

# Define Triton kernel
@triton.jit
def square_kernel(X_ptr, Y_ptr, n, BLOCK: tl.constexpr = 1024):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = x * x
    tl.store(Y_ptr + offsets, y, mask=mask)

# Compile kernel by triggering a dummy launch
x_dummy = torch.ones(1024, dtype=torch.float32, device="cuda")
y_dummy = torch.empty_like(x_dummy)
square_kernel[1, 1](x_dummy, y_dummy, 1024)

# Extract compiled CUBIN from cache
device_caches = square_kernel.device_caches
device_id = next(iter(device_caches.keys()))
cache_tuple = device_caches[device_id]
compiled_kernel = next(iter(cache_tuple[0].values()))
cubin_bytes = compiled_kernel.kernel

# Define C++ wrapper (similar to NVRTC example)
cpp_wrapper = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

TVM_FFI_EMBED_CUBIN(triton_cubin);

void LaunchSquare(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(triton_cubin, "square_kernel");

  uint32_t n = static_cast<uint32_t>(x.size(0));
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();
  uint64_t dummy_ptr = 0;

  // Triton may require extra dummy parameters
  void* args[] = {&x_ptr, &y_ptr, &n, &dummy_ptr, &dummy_ptr};

  tvm::ffi::dim3 grid((n + 127) / 128);
  tvm::ffi::dim3 block(128);

  DLDevice device = x.device();
  CUstream stream = static_cast<CUstream>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  CUresult result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_square, LaunchSquare);
"""

# Load with TVM-FFI
mod = cpp.load_inline(
    "triton_module",
    cuda_sources=cpp_wrapper,
    embed_cubin={"triton_cubin": cubin_bytes}
)

# Use the kernel
x = torch.arange(4096, dtype=torch.float32, device="cuda") * 0.5
y = torch.empty_like(x)
mod.launch_square(x, y)

# Verify
assert torch.allclose(y, x * x)
```

### Loading Pre-Compiled CUBIN

If you have a pre-compiled CUBIN file, you can load it directly:

```python
from pathlib import Path
from tvm_ffi import cpp

# Read CUBIN from file
cubin_path = Path("kernel.cubin")
cubin_bytes = cubin_path.read_bytes()

# Load with wrapper code using tvm_ffi.cpp.load_inline
mod = cpp.load_inline(
    "precompiled_module",
    cuda_sources=cpp_wrapper,
    embed_cubin={"my_kernels": cubin_bytes}
)
```

## C++ Usage

### Embedding CUBIN at Compile Time

The recommended approach in C++ is to embed CUBIN data directly into your shared library:

```cpp
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

// Declare embedded CUBIN module (symbols created via objcopy or cpp.load_inline)
TVM_FFI_EMBED_CUBIN(my_kernels);

void LaunchMyKernel(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  // Step 1: Get kernel from embedded CUBIN (cached in static variable)
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "add_one");

  // Step 2: Prepare kernel arguments (pointers to GPU memory)
  int64_t n = x.size(0);
  void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();
  void* args[] = {&x_ptr, &y_ptr, &n};

  // Step 3: Configure launch parameters (grid and block dimensions)
  tvm::ffi::dim3 grid((n + 255) / 256);   // Number of blocks
  tvm::ffi::dim3 block(256);               // Threads per block

  // Step 4: Get CUDA stream from device
  DLDevice device = x.device();
  CUstream stream = static_cast<CUstream>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  // Step 5: Launch kernel and check for errors
  CUresult result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
}

// Export the function for FFI access
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_my_kernel, LaunchMyKernel);
```

**Key Points:**

- Use `static auto kernel` to cache the kernel lookup for efficiency
- Kernel arguments must be pointers to the actual values (use `&` for addresses)
- {cpp:type}`tvm::ffi::dim3` supports 1D, 2D, or 3D configurations: `dim3(x)`, `dim3(x, y)`, `dim3(x, y, z)`
- `TVMFFIEnvGetStream` retrieves the correct CUDA stream for the device
- Always check kernel launch results with {c:macro}`TVM_FFI_CHECK_CUDA_DRIVER_ERROR`

### Loading CUBIN at Runtime

You can also load CUBIN modules dynamically from memory:

```cpp
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/string.h>

void LoadAndLaunchCubin(const tvm::ffi::Bytes& cubin_data) {
  // Load CUBIN module from memory
  tvm::ffi::CubinModule module(cubin_data);

  // Get kernel by name
  tvm::ffi::CubinKernel kernel = module["my_kernel"];

  // Launch kernel (same as embedded example)
  void* args[] = {...};
  tvm::ffi::dim3 grid(...);
  tvm::ffi::dim3 block(...);
  CUstream stream = ...;

  CUresult result = kernel.Launch(args, grid, block, stream);
  TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
}
```

### Creating CUBIN Symbols with objcopy

To embed CUBIN at build time, you need to create symbols using `objcopy`:

#### Step 1: Compile CUDA kernel to CUBIN

```bash
nvcc --cubin -arch=sm_75 kernel.cu -o kernel.cubin
```

#### Step 2: Convert CUBIN to object file

```bash
ld -r -b binary -o kernel_data.o kernel.cubin
```

#### Step 3: Rename symbols with objcopy

```bash
objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents \
        --redefine-sym _binary_kernel_cubin_start=__tvm_ffi__cubin_my_kernels \
        --redefine-sym _binary_kernel_cubin_end=__tvm_ffi__cubin_my_kernels_end \
        kernel_data.o
```

#### Step 4: Link with your library

```bash
g++ -o mylib.so -shared mycode.cc kernel_data.o -Wl,-z,noexecstack -lcuda
```

The symbol names must match the name used in {c:macro}`TVM_FFI_EMBED_CUBIN`.

**CMake Example:**

```cmake
# Compile CUDA kernel to CUBIN
add_custom_command(
    OUTPUT kernel.cubin
    COMMAND ${CMAKE_CUDA_COMPILER} --cubin -arch=sm_75
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu -o kernel.cubin
    DEPENDS kernel.cu
)

# Embed CUBIN into object file
add_custom_command(
    OUTPUT kernel_data.o
    COMMAND ${CMAKE_LINKER} -r -b binary -o kernel_data.o kernel.cubin
    COMMAND ${CMAKE_OBJCOPY}
        --rename-section .data=.rodata,alloc,load,readonly,data,contents
        --redefine-sym _binary_kernel_cubin_start=__tvm_ffi__cubin_my_kernels
        --redefine-sym _binary_kernel_cubin_end=__tvm_ffi__cubin_my_kernels_end
        kernel_data.o
    DEPENDS kernel.cubin
)

# Link into library
add_library(mylib SHARED mycode.cc kernel_data.o)
target_link_libraries(mylib PRIVATE cuda)
target_link_options(mylib PRIVATE "LINKER:-z,noexecstack")
```

## Advanced Topics

### Multi-GPU Support

The CUBIN launcher automatically handles multi-GPU execution through CUDA primary contexts. Kernels will execute on the device associated with the input tensors:

```cpp
void MultiGPUExample(tvm::ffi::TensorView x_gpu0, tvm::ffi::TensorView x_gpu1) {
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "process");

  // Launch on GPU 0 (device determined by x_gpu0.device())
  LaunchOnDevice(kernel, x_gpu0);

  // Launch on GPU 1 (device determined by x_gpu1.device())
  LaunchOnDevice(kernel, x_gpu1);
}
```

The {cpp:class}`tvm::ffi::CubinKernel` automatically uses the device context from the input tensors.

### Kernel Launch Configuration

When writing the C++ wrapper, important considerations include:

- **Grid/Block Dimensions**: Use {cpp:type}`tvm::ffi::dim3` for 1D, 2D, or 3D configurations
  - 1D: `dim3(x)` → `(x, 1, 1)`
  - 2D: `dim3(x, y)` → `(x, y, 1)`
  - 3D: `dim3(x, y, z)` → `(x, y, z)`

- **Kernel Arguments**: Must be pointers to actual values
  - For device pointers: `void* ptr = tensor.data_ptr(); args[] = {&ptr}`
  - For scalars: `int n = 42; args[] = {&n}`

- **Stream Management**: Use `TVMFFIEnvGetStream` to get the correct CUDA stream for synchronization with DLPack tensors

- **Error Checking**: Always use {c:macro}`TVM_FFI_CHECK_CUDA_DRIVER_ERROR` to validate CUDA Driver API results

### Dynamic Shared Memory

To use dynamic shared memory, specify the size in the {cpp:func}`tvm::ffi::CubinKernel::Launch` call:

```cpp
// Allocate 1KB of dynamic shared memory
uint32_t shared_mem_bytes = 1024;
CUresult result = kernel.Launch(args, grid, block, stream, shared_mem_bytes);
```

### Integration with Different Compilers

The CUBIN launcher works with various CUDA compilation tools:

- **NVCC**: Standard NVIDIA compiler, produces highly optimized CUBIN
- **NVRTC**: Runtime compilation for JIT scenarios (via {py:mod}`tvm_ffi.cpp.nvrtc`)
- **Triton**: High-level DSL that compiles to CUBIN
- **Custom compilers**: Any tool that generates valid CUDA CUBIN

## Complete Examples

For complete working examples, see the `examples/cubin_launcher/` directory:

- `example_embedded_cubin.py` - Pre-compiled CUBIN embedded at build time
- `example_dynamic_cubin.py` - CUBIN data passed dynamically at runtime
- `example_nvrtc_cubin.py` - NVRTC runtime compilation
- `example_triton_cubin.py` - Triton kernel compilation

These examples demonstrate:

- Compiling CUDA kernels to CUBIN
- Embedding CUBIN in C++ modules
- Launching kernels with proper error handling
- Testing and verification

## API Reference

### C++ Classes

- {cpp:class}`tvm::ffi::CubinModule`: RAII wrapper for CUBIN module lifecycle
  - {cpp:func}`tvm::ffi::CubinModule::CubinModule`: Load CUBIN from memory
  - {cpp:func}`tvm::ffi::CubinModule::GetKernel`: Get kernel by name
  - {cpp:func}`tvm::ffi::CubinModule::operator[]`: Convenient kernel access

- {cpp:class}`tvm::ffi::CubinKernel`: Handle for launching kernels
  - {cpp:func}`tvm::ffi::CubinKernel::Launch`: Launch kernel with specified parameters

- {cpp:type}`tvm::ffi::dim3`: 3D dimension structure
  - `dim3()`: Default (1, 1, 1)
  - `dim3(unsigned int x)`: 1D
  - `dim3(unsigned int x, unsigned int y)`: 2D
  - `dim3(unsigned int x, unsigned int y, unsigned int z)`: 3D

### C++ Macros

- {c:macro}`TVM_FFI_EMBED_CUBIN`: Declare embedded CUBIN module
- {c:macro}`TVM_FFI_EMBED_CUBIN_GET_KERNEL`: Get kernel from embedded module
- {c:macro}`TVM_FFI_CHECK_CUDA_DRIVER_ERROR`: Check CUDA Driver API result

### Python Functions

- {py:func}`tvm_ffi.cpp.nvrtc.nvrtc_compile`: Compile CUDA source to CUBIN
- {py:func}`tvm_ffi.cpp.load_inline`: Load inline module with embedded CUBIN
