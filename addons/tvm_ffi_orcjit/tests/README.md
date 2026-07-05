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

# TVM-FFI-OrcJIT Tests

## Quick Start

Run pytest directly — test objects are auto-built by `utils.build_test_objects()`
into a temporary directory using `tvm_ffi.cpp.build`:

```bash
pytest tests/ -v
```

Run the quick-start example:

```bash
python examples/quick-start/run.py --lang c
```

### Alternative: cmake-based build

A `CMakeLists.txt` is also provided for cmake-based workflows.
See `TEST_OBJ_INSTALL_SUFFIX` for multi-compiler builds.

## Tested Configurations

Tests are parametrized over compiler variants. Each platform builds objects
with multiple compilers; `test_basic.py` auto-discovers which are available.

| Platform | Variant | Subdir | Compiler | Languages |
| -------- | ------- | ------ | -------- | --------- |
| Linux, macOS | default | `c/`, `cc/` | LLVM Clang | C, C++ |
| Linux | gcc | `c-gcc/`, `cc-gcc/` | GCC | C, C++ |
| macOS | appleclang | `c-appleclang/`, `cc-appleclang/` | Apple Clang (`/usr/bin/clang`) | C, C++ |
| Windows | default | `c/` | LLVM Clang | C only |
| Windows | msvc | `c-msvc/` | MSVC (`cl`) | C only |
| Windows | clang-cl | `c-clang-cl/` | clang-cl | C only |

All Windows variants are C-only. C++ objects use `try`/`catch`
(via `TVM_FFI_SAFE_CALL_BEGIN/END`) which requires Itanium exception ABI
symbols that the MSVC-built host process cannot resolve.

### Platform-specific compiler flags

| Flag | Applies to | Reason |
| ---- | ---------- | ------ |
| `/O2 /GS-` | MSVC, clang-cl | `/GS-` disables buffer security checks (`__security_cookie`) which are CRT symbols the ORC JIT cannot resolve |
| `-O2` | All others | Standard optimization |
| `-mno-outline-atomics` | aarch64 | Avoids libgcc outline atomics helper calls that the JIT cannot resolve |

## Test Structure

```text
tests/
  sources/
    c/          C source files (.c)
    cc/         C++ source files (.cc)
    cuda/       CUDA source files (optional)
  utils.py              Build helpers (compile sources to objects in /tmp)
  test_basic.py         Python test cases
  CMakeLists.txt        Alternative cmake build
```

Built objects are placed in `{tempdir}/tvm_ffi_orcjit_tests/` with per-compiler
variant subdirectories (c/, cc/, c-gcc/, etc.).

### Source files

| Source | Description |
| ------ | ----------- |
| `test_funcs` | Basic arithmetic (add, multiply) |
| `test_funcs2` | More arithmetic (subtract, divide) |
| `test_funcs_conflict` | Symbol conflict testing (duplicate `add`) |
| `test_call_global` | Callbacks into Python-registered global functions |
| `test_context_primary` / `test_context_incremental` | Generated-style context materialization, incremental coalescing, imports, and link-order initialization |
| `test_types` | Zero-arg, multi-arg, float, void return types |
| `test_link_order_base` / `test_link_order_caller` | Cross-library symbol resolution |
| `test_error` | Error propagation from JIT'd code |
| `test_ctor_dtor` | Constructor/destructor and init/fini sections |

### Constructor/destructor test coverage

`test_ctor_dtor` verifies that the ORC JIT correctly handles platform-specific
static initialization and finalization:

| Platform | Init mechanism | Fini mechanism |
| -------- | -------------- | -------------- |
| Linux (ELF) | `.init_array` (with priorities), `.ctors` (reversed) | `.dtors` |
| macOS (Mach-O) | `.init_array` (with priorities), `__mod_init_func` | `.fini_array` (with priorities) |
| Windows (COFF) | `.CRT$XCA`..`XCU` (alphabetical order) | `.CRT$XTA`..`XTZ` (alphabetical order) |

## CI

The CI workflow (`.github/workflows/tvm_ffi_orcjit.yml`) runs `pytest` and the
quick-start example via `cibuildwheel`.
