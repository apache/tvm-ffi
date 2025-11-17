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

# Julia Interface for TVM FFI

This directory contains the Julia language bindings for TVM FFI.

## Features

- ✅ **Module Loading** - Load compiled TVM modules (.so files)
- ✅ **Function Calling** - Call TVM functions with type safety
- ✅ **Zero-Copy Tensors** - Efficient array passing
- ✅ **CPU Execution** - Verified working with real examples
- ✅ **GPU Support** - CUDA integration via CUDA.jl
- ✅ **Automatic Memory** - GC-based, no manual management
- ✅ **Error Handling** - Julia exceptions with detailed messages

## Quick Start

### 1. Build TVM FFI Library

```bash
cd tvm-ffi
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### 2. Run Working Demo

```bash
cd tvm-ffi/julia/TVMFFI

# CPU example (verified working!)
julia examples/load_add_one.jl
# Output: ✅ SUCCESS! Output matches expected values!

# Complete demo
julia examples/complete_demo.jl
```

### 3. Use in Your Code

```julia
using Pkg
Pkg.add(path="/path/to/tvm-ffi/julia/TVMFFI")

using TVMFFI

# Load TVM module
mod = load_module("my_module.so")

# Get function
my_func = get_function(mod, "my_function")

# Or use bracket notation
my_func = mod["my_function"]

# Create arrays
x = Float32[1, 2, 3, 4, 5]
y = zeros(Float32, 5)

# NEW: Direct array passing! (Auto-conversion)
# Arrays are automatically converted to DLTensorHolder
my_func(x, y)  # ← Simple! Just pass arrays!

# Check results
println(y)  # Results from TVM!

# Works with slices too! (Zero-copy)
matrix = Float32[1 2 3; 4 5 6; 7 8 9]
col = @view matrix[:, 2]           # Column slice (contiguous)
my_func(col)                       # Pass slice directly!

# For optimization: pre-create holders
holder = from_julia_array(x)
for i in 1:1000000
    my_func(holder)  # Reuse holder, no allocation
end
```

## Directory Structure

```text
julia/
├── TVMFFI/              Main package directory
│   ├── Project.toml     Package metadata
│   ├── README.md        Package documentation
│   ├── DESIGN.md        Design philosophy and decisions
│   ├── src/             Source code
│   │   ├── TVMFFI.jl    Main module
│   │   ├── LibTVMFFI.jl C API bindings
│   │   ├── error.jl     Error handling
│   │   ├── dtype.jl     Data types
│   │   ├── device.jl    Devices
│   │   ├── string.jl    Strings and bytes
│   │   ├── object.jl    Object system
│   │   ├── function.jl  Function calling
│   │   └── tensor.jl    Tensor support
│   ├── test/            Unit tests
│   │   └── runtests.jl  Test suite
│   └── examples/        Usage examples
│       ├── README.md    Examples documentation
│       └── basic_usage.jl  Basic usage example
└── README.md            This file
```

## Documentation

- **[Package README](TVMFFI/README.md)**: User-facing documentation
- **[Design Document](TVMFFI/DESIGN.md)**: Architecture and implementation details
- **[Examples](TVMFFI/examples/)**: Code examples

## Design Philosophy

TVMFFI.jl follows these principles:

1. **Simplicity** - Direct, clear code over complex abstractions
2. **Trust the Runtime** - Use Julia's GC instead of manual memory management
3. **Zero-Copy** - Efficient interop with Julia arrays
4. **Idiomatic Julia** - Follow Julia conventions and idioms

For details, see [DESIGN.md](TVMFFI/DESIGN.md).

## Testing

```julia
using Pkg
Pkg.activate("/path/to/tvm-ffi/julia/TVMFFI")
Pkg.test("TVMFFI")
```

Or run tests directly:

```julia
include("TVMFFI/test/runtests.jl")
```

## Requirements

- Julia 1.10 or later
- TVM FFI library (build from source in parent directory)
- Library must be in system path or build directory

## Contributing

Contributions welcome! Please:

- Follow Julia style guide
- Add tests for new features
- Update documentation
- Keep it simple (no unnecessary complexity)

## License

Licensed under Apache License 2.0. See LICENSE file in repository root.

## Acknowledgments

Design inspired by:

- TVM FFI C API
- Rust implementation (simplified for Julia)
- Julia's C interop best practices
