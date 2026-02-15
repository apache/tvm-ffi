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

# TVM-FFI OrcJIT

A Python package that enables dynamic loading of compiled object files (`.o`) using LLVM ORC JIT v2, providing a flexible JIT execution environment for TVM-FFI exported functions.

## Features

- **JIT Execution**: Load and execute compiled object files at runtime using LLVM's ORC JIT v2
- **Multiple Libraries**: Create separate dynamic libraries with independent symbol namespaces
- **Incremental Loading**: Add multiple object files to the same library incrementally
- **Symbol Isolation**: Different libraries can define the same symbol without conflicts
- **TVM-FFI Integration**: Seamlessly works with TVM-FFI's stable C ABI
- **Cross-Platform**: Supports Linux, macOS, and Windows (on the plan)
- **Python API**: Simple Pythonic interface for JIT compilation and execution

## Installation

### Prerequisites

- Python 3.9+
- CMake 3.18+
- LLVM 18+ (with ORC JIT support)
- C/C++ compiler with C++17 support
- Ninja build system (optional, recommended)

### Installing LLVM

**Ubuntu/Debian:**

```bash
sudo apt-get install -y llvm-18-dev
```

**macOS:**

```bash
brew install llvm@18
export LLVM_DIR=$(brew --prefix llvm@18)
export CC=$(brew --prefix llvm@18)/bin/clang
export CXX=$(brew --prefix llvm@18)/bin/clang++
```

### Build from Source

1. Install the core TVM-FFI package:

   ```bash
   pip install apache-tvm-ffi
   ```

1. Install tvm-ffi-orcjit:

   ```bash
   pip install tvm-ffi-orcjit
   ```

### Development Installation

For development, install from source:

```bash
git clone --recursive https://github.com/yaoyaoding/tvm-ffi.git
cd tvm-ffi/addons/tvm-ffi-orcjit
pip install -e .
```

## Usage

### Basic Example

```python
from tvm_ffi_orcjit import create_session

# Create an execution session
session = create_session()

# Create a dynamic library
lib = session.create_library()

# Load an object file
lib.add("example.o")

# Get and call a function
add_func = lib.get_function("test_add")
result = add_func(1, 2)
print(f"Result: {result}")  # Output: Result: 3
```

### Multiple Libraries with Symbol Isolation

Create separate libraries to avoid symbol conflicts:

```python
from tvm_ffi_orcjit import create_session

session = create_session()

# Create two separate libraries
lib1 = session.create_library("lib1")
lib2 = session.create_library("lib2")

# Each library can have its own version of the same symbol
lib1.add("implementation_v1.o")  # Contains test_add
lib2.add("implementation_v2.o")  # Contains test_add with different behavior

# Get functions from different libraries
add_v1 = lib1.get_function("test_add")
add_v2 = lib2.get_function("test_add")

print(add_v1(5, 3))  # Uses implementation from lib1
print(add_v2(5, 3))  # Uses implementation from lib2
```

### Incremental Loading

Add multiple object files to the same library:

```python
from tvm_ffi_orcjit import create_session

session = create_session()
lib = session.create_library()

# Load multiple object files incrementally
lib.add("math_ops.o")
lib.add("string_ops.o")
lib.add("utils.o")

# Access functions from any loaded object file
add = lib.get_function("test_add")
subtract = lib.get_function("test_subtract")
concat = lib.get_function("string_concat")

print(add(10, 5))              # From math_ops.o
print(subtract(10, 5))         # From math_ops.o
print(concat("Hello", " World"))  # From string_ops.o
```

## How It Works

1. **ExecutionSession**: Manages the LLVM ORC JIT execution session and multiple dynamic libraries
2. **DynamicLibrary**: Represents a JITDylib that can load object files and resolve symbols
3. **Symbol Resolution**: Uses LLVM's ORC JIT v2 symbol lookup with proper linkage semantics
4. **Memory Management**: Allocates `__dso_handle` in JIT memory to ensure proper relocations
5. **TVM-FFI Integration**: Functions are exposed through TVM-FFI's PackedFunc interface

### Technical Details

- **ORC JIT v2**: Uses LLVM's modern JIT infrastructure (LLJIT)
- **Weak Linkage**: Each library gets its own `__dso_handle` with weak linkage
- **IR-based Allocation**: Creates LLVM IR modules for runtime symbols to ensure JIT memory allocation
- **Cross-Platform**: Correctly handles `.so` (Linux), `.dylib` (macOS), and `.dll` (Windows)

## Development

### Building Tests

The project includes comprehensive tests with CMake-built test objects:

```bash
cd tests
cmake -B build
cmake --build build
pytest -v
```

### Project Structure

```text
tvm-ffi-orcjit/
├── CMakeLists.txt              # CMake build configuration
├── pyproject.toml              # Python package metadata
├── README.md                   # This file
├── include/
│   └── tvm/ffi/orcjit/
│       ├── orcjit_session.h    # ExecutionSession C++ header
│       └── orcjit_dylib.h      # DynamicLibrary C++ header
├── src/
│   └── ffi/
│       ├── orcjit_session.cc   # ExecutionSession implementation
│       └── orcjit_dylib.cc     # DynamicLibrary implementation
├── python/
│   └── tvm_ffi_orcjit/
│       ├── __init__.py         # Module exports and library loading
│       ├── session.py          # Python ExecutionSession wrapper
│       └── dylib.py            # Python DynamicLibrary wrapper
├── tests/
│   ├── CMakeLists.txt          # Test object file builds
│   ├── test_basic.py           # Python tests
│   └── sources/
│       ├── test_funcs.cc       # Test functions
│       ├── test_funcs2.cc      # Additional test functions
│       └── test_funcs_conflict.cc  # Conflicting symbols for testing
└── examples/
    └── quick-start/            # Complete example with CMake
```

## Examples

Complete examples are available in the `examples/` directory:

- **quick-start/**: Demonstrates basic usage with a simple add function
  - Shows how to compile C++ code with TVM-FFI exports
  - Loads and executes the compiled object file
  - Uses CMake for building the example

## Writing Functions for OrcJIT

Functions must use TVM-FFI's export macros:

```cpp
#include <tvm/ffi/c_ffi_api.h>

// Simple function
TVM_FFI_DLL_EXPORT_TYPED_FUNC(simple_add, [](int a, int b) {
    return a + b;
});

// Function with implementation
static int multiply_impl(int a, int b) {
    return a * b;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(simple_multiply, multiply_impl);
```

Compile with C++17:

```bash
clang++ -std=c++17 -fPIC -c -o example.o example.cc
```

## Requirements

The package depends on:

- `apache-tvm-ffi>=0.1.0` - TVM-FFI core library
- LLVM 18+ (linked at build time) - For ORC JIT v2 functionality
- Python 3.9+ - For runtime

## Known Limitations

### Optimized Code and Relocations

When compiling object files with optimization enabled (`-O2`, `-O3`), ensure your code doesn't generate PC-relative relocations that exceed ±2GB range. The package allocates `__dso_handle` in JIT memory to mitigate this, but extremely large programs may still encounter issues.

**Workaround**: Compile test objects with `-O0` if you encounter "relocation out of range" errors during sequential test runs.

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please ensure that:

1. Code follows the existing C++17 and Python style
2. New features include tests in `tests/test_basic.py`
3. Documentation is updated (README and docstrings)
4. CI tests pass on all platforms (Linux, macOS)

## Troubleshooting

### "Cannot find global function" error

The shared library wasn't loaded. This usually means:

- The library file extension doesn't match your platform
- The library wasn't installed correctly
- Python can't find the library file

**Solution**: Reinstall the package:

```bash
pip install --force-reinstall tvm-ffi-orcjit
```

### "Duplicate definition of symbol" error

You're adding multiple object files with the same symbol to the same library.

**Solution**: Use separate libraries for different implementations:

```python
lib1 = session.create_library("lib1")
lib2 = session.create_library("lib2")
```

### "Symbol not found" error

The symbol wasn't exported with TVM-FFI macros.

**Solution**: Use `TVM_FFI_DLL_EXPORT_TYPED_FUNC`:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_function, impl);
```

### Relocation errors with optimized code

Object files compiled with `-O2` or higher may fail with "relocation out of range" in some scenarios.

**Solution**:

- Use `-O0` for test/development builds
- Run tests in separate processes (using `pytest-xdist`)
- This limitation primarily affects test scenarios, not production use

### LLVM version mismatch

The package requires LLVM 18+. Using older versions will cause build failures.

**Solution**: Install LLVM 18:

```bash
# Ubuntu
sudo ./llvm.sh 18

# macOS
brew install llvm@18
```

### CMake can't find LLVM

Set the `LLVM_DIR` environment variable:

```bash
# macOS
export LLVM_DIR=$(brew --prefix llvm@18)/lib/cmake/llvm

# Linux
export LLVM_DIR=/usr/lib/llvm-18/cmake
```
