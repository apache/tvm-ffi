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

A Python package that enables dynamic loading of compiled object files (`.o`)
using LLVM ORC JIT v2, providing a flexible JIT execution environment for
TVM-FFI exported functions.

## Features

- **JIT Execution**: Load and execute compiled object files at runtime using LLVM's ORC JIT v2
- **High-Level Loading**: `default_session().load_module(...)` mirrors `tvm_ffi.load_module`, returning a plain `tvm_ffi.Module`
- **Unified Input**: Load from a file path, in-memory object bytes, or a list mixing both
- **Shared Session**: A process-wide session so callers share process-symbol resolution and the Linux slab pool while loaded modules remain isolated
- **Symbol Isolation**: Separate `load_module` calls define independent symbol namespaces, so they can define the same symbol without conflicts
- **Bounded-Range Memory**: A growable Linux slab pool keeps JIT code/data close enough for 32-bit PC-relative relocations and supports explicit drained-slab reclamation
- **Init/Fini Support**: Handles static constructors/destructors across ELF (`.init_array`/`.ctors`), Mach-O (`__mod_init_func`), and COFF (`.CRT$XC*`/`.CRT$XT*`)
- **Cross-Platform**: Linux (x86_64, aarch64), macOS (arm64), Windows (AMD64)
- **Multi-Compiler**: Tested with LLVM Clang, GCC, Apple Clang, MSVC, and clang-cl
- **TVM-FFI Integration**: Seamlessly works with TVM-FFI's stable C ABI
- **Python API**: Simple Pythonic interface for JIT compilation and execution

## Supported Platforms and Compilers

Object files compiled with any of the following compiler/platform combinations
can be loaded and executed by the ORC JIT:

| Platform | Compilers | C | C++ |
| -------- | --------- | :-: | :-: |
| Linux (x86_64, aarch64) | LLVM Clang, GCC | yes | yes |
| macOS (arm64) | LLVM Clang, Apple Clang | yes | yes |
| Windows (AMD64) | LLVM Clang, MSVC, clang-cl | yes | no |

Windows is C-only across all compilers. C++ objects compiled with
`TVM_FFI_DLL_EXPORT_TYPED_FUNC` use `try`/`catch` (via `TVM_FFI_SAFE_CALL_BEGIN/END`),
which requires Itanium exception ABI symbols (`__cxa_begin_catch`,
`__gxx_personality_v0`, etc.) that the MSVC-built host process cannot provide.
Pure C objects using the `TVMFFISafeCallType` ABI work on all platforms.

## Installation

### Install from PyPI

```bash
pip install apache-tvm-ffi apache-tvm_ffi_orcjit
```

### Build from Source

#### Prerequisites

- Python 3.10+, CMake 3.20+, C++17 compiler
- LLVM 23.1.1+ development libraries (`llvmdev`, `llvm-config`)
- Static `zlib` and `zstd` libraries (in the same prefix as LLVM)

#### Install LLVM via conda-forge

The easiest way to get all dependencies is via conda-forge:

```bash
conda create -p /opt/llvm -c conda-forge \
  llvmdev=23.1.1 clangdev=23.1.1 compiler-rt=23.1.1 zlib zstd-static -y
export LLVM_PREFIX=/opt/llvm
```

On Windows:

```cmd
conda create -p C:\opt\llvm -c conda-forge llvmdev=23.1.1 zlib zstd-static -y
set LLVM_PREFIX=C:\opt\llvm
```

#### Build and install

```bash
git clone --recursive https://github.com/apache/tvm-ffi.git
cd tvm-ffi

# Install tvm-ffi first
pip install -e .

# Build and install the orcjit addon
cd addons/tvm_ffi_orcjit
pip install -e .
```

The `LLVM_PREFIX` environment variable tells CMake where to find LLVM. If
LLVM is installed in a conda env or a standard system path, CMake can
auto-discover it and `LLVM_PREFIX` is not needed.

On Linux, each `load_module` call follows the same default compiler selection
as `tvm_ffi.cpp.build`: `$CXX`, or `c++` when unset. ORC asks that compiler for
its `libstdc++.so.6` and optional `libstdc++_nonshared.a`, then keeps both local
to the new JITDylib. Discovery is cached, and `load_module(..., cxx="g++-14")`
can select a different toolchain without supplying runtime-library paths.

## Usage

### Basic Example

The high-level API mirrors `tvm_ffi.load_module`: a process-wide shared session
plus a `load_module` that accepts a path, in-memory object bytes, or a list of
either, and returns a plain `tvm_ffi.Module`.

```python
import tvm_ffi_orcjit as oj

# Shared process-wide session (created once, cached).
session = oj.default_session()

# Load a single object file by path.
mod = session.load_module("example.o")

# Call an exported function.
result = mod.add(1, 2)
print(f"Result: {result}")  # Output: Result: 3
```

### Loading Multiple Objects and In-Memory Bytes

Objects passed together are linked into one module (the same way a multi-object
shared library links). Each element may be a path or an object-file image in
memory.

```python
from pathlib import Path

session = oj.default_session()

mod = session.load_module(
    [
        "math_ops.o",                    # path
        Path("kernel.o").read_bytes(),   # in-memory object bytes
    ]
)
result = mod.call_math(10, 20)
```

### Isolated Sessions

`default_session()` is shared across the process. For an isolated symbol
namespace or a tuned memory arena, construct an `ExecutionSession` directly:

```python
from tvm_ffi_orcjit import ExecutionSession

session = ExecutionSession()          # independent LLVM ExecutionSession
mod = session.load_module("impl.o")
```

## Writing Functions for OrcJIT

### C++ (Linux/macOS)

```cpp
#include <tvm/ffi/function.h>

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, [](int a, int b) {
    return a + b;
});
```

Compile: `clang++ -std=c++17 -fPIC -O2 -c -o example.o example.cc`

### Pure C (all platforms including Windows)

```c
#include <tvm/ffi/c_api.h>

TVM_FFI_DLL_EXPORT int __tvm_ffi_add(
    void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->v_int64 = args[0].v_int64 + args[1].v_int64;
  return 0;
}
```

Compile: `clang -O2 -c -o example.o example.c`

## How It Works

- **LLJIT**: Built on LLVM's ORC JIT v2 with `ObjectLinkingLayer` (JITLink) for
  all platforms.
- **ELF Lifecycle** (Linux): LLVM 23.1.1+ `ELFNixPlatform` runs ordered
  constructors and destructors through `LLJIT::initialize` / `deinitialize`.
- **InitFiniPlugin** (macOS/Windows): A local `ObjectLinkingLayer::Plugin`
  handles Mach-O `__mod_init_func`/`__mod_term_func` and COFF
  `.CRT$XC*`/`.CRT$XT*` while those ORC platforms remain unsuitable.
- **DLL Import Stubs** (Windows): Custom `DefinitionGenerator` that resolves host
  process symbols from all loaded DLLs and creates `__imp_*` pointer stubs in
  JIT memory, keeping all fixups within PCRel32 range.
- **SEH Stripping** (Windows): `ObjectTransformLayer` strips `.pdata`/`.xdata`
  relocations from COFF objects before JITLink graph building, working around a
  JITLink limitation with COMDAT section symbols.

Refer to [ORCJIT_PRIMER.md](./ORCJIT_PRIMER.md) for background on object files,
linking, LLVM ORC JIT v2, and the addon's architecture.

## Project Structure

```text
tvm_ffi_orcjit/
├── CMakeLists.txt              # Build configuration
├── pyproject.toml              # Python package metadata
├── src/ffi/
│   ├── orcjit_session.cc       # ExecutionSession (LLJIT setup, plugins)
│   ├── orcjit_session.h
│   ├── orcjit_dylib.cc         # JIT dylib module (object loading, symbol lookup)
│   ├── orcjit_dylib.h
│   ├── orcjit_memory_manager.*  # Growable Linux slab pool
│   ├── orcjit_slab.*            # Contiguous-VA allocator and page protection
│   ├── llvm_patches/            # Isolated upstream LLVM workarounds
│   └── orcjit_utils.h          # LLVM error handling utilities
├── python/tvm_ffi_orcjit/
│   ├── __init__.py             # Module exports and library loading
│   └── session.py              # Python ExecutionSession + default_session
├── tests/                      # See tests/README.md
└── examples/quick-start/       # Complete example with CMake
```

## CI

Runs on Linux (x86_64, aarch64), macOS (arm64), Windows (AMD64) via
`cibuildwheel`. Each platform builds test objects with multiple compilers
and runs the full test suite. See the `orcjit` job in `.github/workflows/ci_test.yml`.

## Troubleshooting

### "Cannot find global function" error

The shared library wasn't loaded. Reinstall: `pip install --force-reinstall apache-tvm_ffi_orcjit`

### "Duplicate definition of symbol" error

Use separate libraries for different implementations of the same symbol.

### "Symbol not found" error

Ensure functions are exported with TVM-FFI macros (`TVM_FFI_DLL_EXPORT_TYPED_FUNC`
for C++, or `__tvm_ffi_` prefix for C).

### Relocation errors on Windows

MSVC/clang-cl objects must be compiled with `/GS-` to disable buffer security
checks (`__security_cookie`) which are CRT symbols the JIT cannot resolve.

### LLVM version mismatch

The package requires LLVM 23.1.1+. Set `LLVM_PREFIX` to the LLVM install prefix:

```bash
export LLVM_PREFIX=/path/to/llvm
```

### Reclaiming unused JIT memory on Linux

Dropping a module returns its regions to the session's slab pool for reuse. To
return fully drained slabs to the operating system, call
`session.clear_free_slabs()`. The call is synchronized with concurrent JIT work.

## License

Apache License 2.0
