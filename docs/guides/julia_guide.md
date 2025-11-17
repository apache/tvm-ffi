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

# Julia Guide

This guide demonstrates how to use TVM FFI from Julia applications.

## Installation

### Prerequisites

The Julia support depends on `libtvm_ffi`. First, build the TVM FFI library:

```bash
cd tvm-ffi
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### Adding to Your Project

Add the TVMFFI package to your Julia project:

```julia
using Pkg
Pkg.add(path="/path/to/tvm-ffi/julia/TVMFFI")
```

### Environment Setup

Set the library path so `libtvm_ffi` can be found at runtime:

```bash
export LD_LIBRARY_PATH=/path/to/tvm-ffi/build/lib:$LD_LIBRARY_PATH
```

## Basic Usage

### Loading a Module

Load a compiled TVM FFI module and call its functions:

```julia
using TVMFFI

# Load compiled module
mod = load_module("build/add_one_cpu.so")

# Get function by name
add_one = get_function(mod, "add_one_cpu")

# Or use bracket notation (Python-style)
add_one = mod["add_one_cpu"]
```

### Calling Functions

Call functions with automatic array conversion:

```julia
# Create input and output arrays
x = Float32[1, 2, 3, 4, 5]
y = zeros(Float32, 5)

# Call function - arrays auto-converted!
add_one(x, y)

println(y)  # [2.0, 3.0, 4.0, 5.0, 6.0]
```

### Working with Slices

Julia's `@view` creates zero-copy slices:

```julia
matrix = Float32[1 2 3; 4 5 6; 7 8 9]
col = @view matrix[:, 2]  # Column slice (zero-copy)

add_one(col, output)  # Pass slice directly
```

## Advanced Topics

### Global Functions

Access globally registered functions:

```julia
# Get global function
func = get_global_func("my_function")

if func !== nothing
    result = func(arg1, arg2)
end
```

### GPU Support

Work with GPU arrays using CUDA.jl, AMDGPU.jl, Metal.jl, or oneAPI.jl:

```julia
using CUDA

# Create CUDA arrays
x_gpu = CUDA.CuArray(Float32[1, 2, 3, 4, 5])
y_gpu = CUDA.zeros(Float32, 5)

# Same API - device auto-detected!
add_one_cuda(x_gpu, y_gpu)

CUDA.synchronize()
println(Array(y_gpu))
```

The Julia bindings support multiple GPU backends:

```julia
# Check available backends
backends = list_available_gpu_backends()

# NVIDIA CUDA
using CUDA
x = CUDA.CuArray(data)

# AMD ROCm
using AMDGPU
x = AMDGPU.ROCArray(data)

# Apple Metal
using Metal
x = Metal.MtlArray(data)

# All work with the same API
func(x)  # Auto-detects backend
```

### Manual Holder Creation

For performance optimization in hot loops:

```julia
# Create holder once
holder = from_julia_array(x)

# Reuse in loop (no allocation)
for i in 1:1000000
    func(holder)
end
```

### Device Context

Specify device explicitly if needed:

```julia
# Create devices
cpu_dev = cpu(0)
cuda_dev = cuda(0)
rocm_dev = rocm(0)

# Create holder with device
holder = from_julia_array(x, cuda_dev)
```

### Data Types

Work with DLPack data types:

```julia
# From Julia types
dt = DLDataType(Float32)

# From strings
dt = DLDataType("float32")

# Convert to string
println(string(dt))  # "float32"
```

### Error Handling

TVM FFI errors are translated to Julia exceptions:

```julia
try
    result = func(args...)
catch e
    if e isa TVMError
        println("Error: ", e.kind, " - ", e.message)
    end
end
```

## Examples

The repository includes complete examples in `julia/TVMFFI/examples/`.

Run the CPU example:

```bash
cd julia/TVMFFI
julia --project=.. examples/load_add_one.jl
```

Run the CUDA example:

```bash
julia --project=.. examples/load_add_one_cuda.jl
```

## Testing

Run the test suite:

```bash
cd julia/TVMFFI
julia --project=.. -e 'using Pkg; Pkg.test("TVMFFI")'
```

Or run tests directly:

```julia
include("julia/TVMFFI/test/runtests.jl")
```

## Design Philosophy

The Julia bindings follow these principles:

1. **Simplicity** - Direct C API calls via `ccall`, no intermediate layers
2. **Trust the GC** - Julia's garbage collector manages lifetimes via finalizers
3. **Zero-Copy** - Efficient interop with Julia arrays and slices
4. **Idiomatic Julia** - Multiple dispatch, standard interfaces (`size`, `length`, etc.)

## Related Resources

- [Quick Start Guide](../get_started/quickstart.rst) - General TVM FFI introduction
- [C++ Guide](cpp_guide.md) - C++ API usage
- [Python Guide](python_guide.md) - Python API usage
- [Rust Guide](rust_guide.md) - Rust API usage
