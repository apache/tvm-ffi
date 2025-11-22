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

`example_file_cubin.py` - CUBIN loaded from file at runtime.

```bash
python examples/cubin_launcher/example_file_cubin.py
```

### 3. Triton Kernel (Experimental)

`example_triton_cubin.py` - Triton kernel compiled to CUBIN, with C++ wrapper via `tvm_ffi.cpp.load_inline`.

```bash
# Requires: triton, torch
python examples/cubin_launcher/example_triton_cubin.py
```

## Files

- `include/tvm/ffi/extra/cubin_launcher.h` - Header-only C++ library
- `src/lib_embedded.cc` - Embedded CUBIN example (lib_embedded.so)
- `src/lib_dynamic.cc` - Dynamic loading example (lib_dynamic.so)
- `src/kernel.cu` - CUDA kernels (add_one, mul_two)
- `CMakeLists.txt` - Build configuration
