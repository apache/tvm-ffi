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

Demonstrates basic usage of tvm_ffi_orcjit: compile functions to object files,
load them into the ORC JIT at runtime, and call them from Python.

## Files

| File | Description |
| ------ | ------------- |
| `add.cc` | C++ source using `TVM_FFI_DLL_EXPORT_TYPED_FUNC` (automatic type marshaling, supports `std::string`) |
| `add_c.c` | Pure C source using `TVMFFISafeCallType` ABI (`__tvm_ffi_` prefix). No C++ runtime needed. |
| `run.py` | Python script that compiles, loads, and calls the functions |
| `CMakeLists.txt` | Alternative CMake build configuration |

## Prerequisites

- Python 3.11+, C/C++ compiler
- `apache-tvm-ffi` and `apache-tvm_ffi_orcjit` packages installed

## Run

`run.py` compiles the source files to object files using `tvm_ffi.cpp.build`,
then loads them into the ORC JIT:

```bash
# C++ variant (Linux/macOS)
python run.py

# Pure C variant (all platforms including Windows)
python run.py --lang c
```

## How It Works

**C++ variant** (`add.cc`): Functions are exported with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC`, which wraps a typed C++ lambda/function into
TVM-FFI's packed calling convention. Supports C++ types like `std::string`.

**C variant** (`add_c.c`): Functions follow the `TVMFFISafeCallType` ABI
directly — each function is named `__tvm_ffi_<name>` and manually packs
arguments/results via `TVMFFIAny`. Zero C++ dependencies, works on all
platforms including Windows with MSVC or clang-cl.

**Python side** (`run.py`):

```python
from tvm_ffi_orcjit import ExecutionSession

session = ExecutionSession()       # Create ORC JIT session
lib = session.create_library()     # Create a JITDylib
lib.add("add_c.o")                 # Load object file into JIT
add = lib.get_function("add")      # Look up symbol
print(add(10, 20))                 # Call like a normal function → 30
```

## Platform Notes

| Platform | C++ (`add.o`) | C (`add_c.o`) |
| ---------- | :-: | :-: |
| Linux (Clang/GCC) | yes | yes |
| macOS (Clang/Apple Clang) | yes | yes |
| Windows (all compilers) | no | yes |

C++ is not supported for ORC JIT on Windows. The `TVM_FFI_DLL_EXPORT_TYPED_FUNC`
macro uses `try`/`catch` which requires Itanium exception ABI symbols that the
MSVC-built host process cannot provide. Use the pure C variant on Windows.
