# Julia Interface for TVM FFI

This directory contains the Julia language bindings for TVM FFI.

## Quick Start

```bash
# 1. Build TVM FFI library
cd /path/to/tvm-ffi
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 2. Use the Julia package
julia
```

```julia
using Pkg
Pkg.add(path="/path/to/tvm-ffi/julia/TVMFFI")

using TVMFFI

# Create a device
device = cpu(0)

# Create a data type
dtype = DLDataType(Float32)

# Get and call a function
func = get_global_func("my_function")
if func !== nothing
    result = func(arg1, arg2)
end
```

## Directory Structure

```
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

- Julia 1.6 or later
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

