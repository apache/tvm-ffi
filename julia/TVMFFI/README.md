# TVMFFI.jl

Julia bindings for TVM FFI (Foreign Function Interface).

## Overview

TVMFFI.jl provides a clean, idiomatic Julia interface to TVM's C API, enabling machine learning model compilation and execution from Julia with minimal overhead.

## Design Philosophy

**Simple. Direct. No Bullshit.**

This package follows a pragmatic design philosophy:

1. **Direct C API calls** - Uses Julia's `ccall` directly, no intermediate layers
2. **GC-based memory management** - Julia's garbage collector handles lifetimes via finalizers
3. **Zero unnecessary abstractions** - If it's not solving a real problem, it's not here
4. **Interop over isolation** - Works naturally with Julia arrays and types

Unlike the Rust version with its complex trait hierarchies and Arc wrappers, Julia doesn't need that complexity. We trust the GC. We keep it simple.

## Installation

```julia
# From the Julia REPL
using Pkg
Pkg.add(path="/path/to/tvm-ffi/julia/TVMFFI")
```

Or add to your `Project.toml`:

```toml
[deps]
TVMFFI = {path = "/path/to/tvm-ffi/julia/TVMFFI"}
```

## Quick Start

```julia
using TVMFFI

# Create a device context
device = cpu()  # or cuda(0), opencl(0), etc.

# Create a data type
dtype = DLDataType("float32")
# Or from Julia types
dtype = DLDataType(Float32)

# Get a global function
func = get_global_func("my_tvm_function")
if func !== nothing
    result = func(arg1, arg2, arg3)
end

# Error handling
try
    # TVM operations that might fail
    func = get_global_func("nonexistent")
catch e
    if e isa TVMError
        println("TVM Error: ", e.kind)
        println("Message: ", e.message)
    end
end
```

## Core Types

### Device and DataType

```julia
# Devices
cpu_dev = cpu()           # CPU:0
gpu0 = cuda(0)            # CUDA:0
gpu1 = cuda(1)            # CUDA:1
opencl_dev = opencl(0)    # OpenCL:0

# Data types
int32 = DLDataType("int32")
float64 = DLDataType("float64")
bool_type = DLDataType("bool")

# From Julia types
int32 = DLDataType(Int32)
float64 = DLDataType(Float64)
```

### Strings and Bytes

```julia
# TVM strings (ABI-stable)
s = TVMString("hello")
julia_str = String(s)

# TVM bytes (for binary data)
b = TVMBytes([0x01, 0x02, 0x03])
julia_bytes = Vector{UInt8}(b)
```

### Functions

```julia
# Get global function
func = get_global_func("my_function")

# Call with Julia values (automatic conversion)
result = func(42, 3.14, "hello", cpu())

# Functions return native Julia types when possible
# Int64, Float64, String, etc.
```

### Tensors

```julia
# Assuming you have a TVM tensor object
tensor = # ... get tensor from TVM ...

# Query tensor properties
shape_tuple = size(tensor)    # e.g., (3, 224, 224)
ndim = ndims(tensor)          # e.g., 3
n_elements = length(tensor)   # e.g., 150528
dt = dtype(tensor)            # DLDataType
dev = device(tensor)          # DLDevice

# Convert to Julia array (zero-copy when possible)
arr = to_julia_array(tensor, Float32)

# arr now shares memory with tensor
arr[1, 1, 1] = 1.0  # modifies tensor data

# Copy if you need independence
arr_copy = copy_to_julia(tensor, Float32)
```

## Error Handling

TVM FFI uses exception-based error handling with detailed error information:

```julia
try
    func = get_global_func("my_func")
    result = func(invalid_arg)
catch e
    if e isa TVMError
        @show e.kind        # "ValueError", "TypeError", etc.
        @show e.message     # Detailed error message
        @show e.backtrace   # Stack trace from C++/TVM side
    end
end
```

Standard error kinds:
- `ValueError` - Invalid value
- `TypeError` - Type mismatch
- `RuntimeError` - Runtime execution error
- `AttributeError` - Attribute not found
- `KeyError` - Key not found
- `IndexError` - Index out of bounds

## Architecture

```
TVMFFI.jl
├── LibTVMFFI.jl    - Raw C API bindings (ccall wrappers)
├── error.jl        - Error types and handling
├── dtype.jl        - Data type support
├── device.jl       - Device abstraction
├── string.jl       - String and bytes types
├── object.jl       - Object system (simplified vs Rust)
├── function.jl     - Function calling interface
└── tensor.jl       - Tensor/NDArray support
```

### Key Design Decisions

1. **No manual reference counting** - Julia's GC + finalizers handle this
2. **Direct ccall, no sys crate** - Julia can call C directly, keep it simple
3. **Minimal type wrappers** - Only wrap when it adds value
4. **Zero-copy where possible** - `unsafe_wrap` for array interop
5. **Exception-based errors** - Natural for Julia, no Result<T, E> needed

## Comparison with Rust Implementation

| Aspect | Rust | Julia |
|--------|------|-------|
| Memory Management | Manual Arc/RefCount | GC + Finalizers |
| C Interop | `tvm_ffi_sys` crate | Direct `ccall` |
| Error Handling | `Result<T, Error>` | Exceptions |
| Type Conversion | Trait-based | Multiple dispatch |
| Lines of Code | More complex | Simpler, clearer |

**Linus would approve**: Julia version is simpler, more direct, and trusts the runtime. No needless abstraction.

## Requirements

- Julia 1.6 or later
- TVM FFI library (`libtvm_ffi.so` or equivalent)
- The library must be in your system library path or build directory

## Building TVM FFI

```bash
cd /path/to/tvm-ffi
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Library will be in build/lib/libtvm_ffi.so
```

## Examples

See the `examples/` directory for complete examples:
- `basic_usage.jl` - Basic API usage
- `array_interop.jl` - Working with tensors and Julia arrays
- `error_handling.jl` - Error handling patterns

## Contributing

Contributions welcome! Guidelines:
- Keep it simple - no unnecessary abstractions
- Match Julia idioms - don't cargo-cult from other languages
- Write tests for new functionality
- Update docs for API changes

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.

## Acknowledgments

Based on the TVM FFI C API design and inspired by the Rust implementation,
but simplified for Julia's runtime model and idioms.

