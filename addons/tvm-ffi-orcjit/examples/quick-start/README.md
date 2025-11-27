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

# Quick Start Example

This example demonstrates the basic usage of tvm-ffi-orcjit to compile C++ functions and load them dynamically at runtime.

## What's Included

- `add.cc` - Simple C++ source file with math functions exported via TVM-FFI
- `run.py` - Python script that loads and calls the compiled functions
- `CMakeLists.txt` - CMake configuration to compile the C++ code into an object file
- `compile.sh` - Legacy shell script (CMake is recommended for cross-platform support)

## Prerequisites

- Python 3.8+
- CMake 3.18+
- C++ compiler (g++, clang++, or MSVC)
- TVM-FFI and tvm-ffi-orcjit packages

## Installation

First, install the required packages:

```bash
# Navigate to the repository root
cd ../../..

# Install TVM-FFI in editable mode
pip install -e .

# Install tvm-ffi-orcjit in editable mode
pip install -e addons/tvm-ffi-orcjit

# Return to the example directory
cd addons/tvm-ffi-orcjit/examples/quick-start
```

After installation, `tvm-ffi-config` will be available in your PATH and used by the compile script to get the correct include directories and compiler flags.

## Steps

### 1. Compile the C++ code

Using CMake (recommended for cross-platform):

```bash
cmake -B build
cmake --build build
```

Or using the legacy shell script (Unix-like systems only):

```bash
./compile.sh
```

Both methods will create `add.o` - a compiled object file with exported functions.

### 2. Run the Python loader

```bash
python run.py
```

This will:

- Load the object file using tvm-ffi-orcjit
- Call the exported functions
- Print the results

## Expected Output

```text
Loading object file: add.o
✓ Object file loaded successfully

=== Testing add function ===
add(10, 20) = 30

=== Testing multiply function ===
multiply(7, 6) = 42

=== Testing fibonacci function ===
fibonacci(10) = 55

=== Testing concat function ===
concat('Hello, ', 'World!') = 'Hello, World!'

==================================================
✓ All tests passed successfully!
==================================================
```

## How It Works

1. **C++ Side** (`add.cc`):
   - Functions are exported using `TVM_FFI_DLL_EXPORT_TYPED_FUNC` macro
   - The macro registers functions with TVM-FFI's global function registry

2. **Python Side** (`run.py`):
   - `create_session()` creates an ORC JIT execution session
   - `session.create_library()` creates a dynamic library (JITDylib)
   - `lib.add()` loads the `.o` file into the JIT
   - `lib.get_function()` looks up symbols in the JIT-compiled code
   - Functions are called like normal Python functions

## Next Steps

- Modify `add.cc` to add your own functions
- Recompile with CMake: `cmake --build build`
- Load and test in Python

For more details on the API, see the main package documentation.
