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

# CLAUDE.md — Apache TVM FFI

## What is this project?

TVM FFI is an open ABI and FFI (Foreign Function Interface) for machine learning
systems. It provides a stable C ABI, C++17 API, Python bindings (via Cython),
and Rust bindings. The core abstractions are type-erased values (`Any`/`AnyView`),
reference-counted objects (`Object`/`ObjectRef`), and packed functions (`Function`).

## Repository layout

```text
include/tvm/ffi/    C++ public headers (core API)
src/ffi/            C++ implementation
python/tvm_ffi/     Python package (Cython bindings in cython/)
rust/               Rust crate workspace (tvm-ffi, tvm-ffi-sys, tvm-ffi-macros)
tests/cpp/          GoogleTest C++ tests
tests/python/       pytest Python tests
tests/lint/         Lint scripts (ASF header, file type, version check)
docs/               Sphinx documentation (RST + Markdown)
examples/           Runnable examples
cmake/Utils/        CMake utility modules
3rdparty/           Vendored deps (dlpack, libbacktrace)
addons/             Optional addons (torch_c_dlpack_ext)
```

## Building

All Python commands use `uv`. The default virtualenv is `.venv` in the repo root.

### Python editable install (primary workflow)

```bash
uv pip install --force-reinstall --verbose -e .
```

C++/Cython changes always require re-running this command. Pure Python changes
are reflected immediately.

### C++-only build

```bash
cmake . -B build_cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_cpp --parallel --config RelWithDebInfo --target tvm_ffi_shared
```

### Prerequisites

- Python 3.9+, C++17 compiler, CMake 3.18+, Ninja
- Submodules: `git submodule update --init --recursive`

## Testing

### C++ tests

```bash
cmake . -B build_test -DTVM_FFI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build_test --clean-first --config Debug --target tvm_ffi_tests
ctest -V -C Debug --test-dir build_test --output-on-failure
```

### Python tests

```bash
uv pip install --force-reinstall --verbose --group test -e .
uv run pytest -vvs tests/python
```

### Rust tests

```bash
cd rust && cargo test   # requires Python package installed first
```

## Linting

### Pre-commit (primary lint workflow)

```bash
pre-commit run --all-files     # all hooks
pre-commit run <hook-id> --all-files  # single hook (e.g. ruff-check, clang-format)
```

Key linters: `ruff` (Python), `clang-format` (C++, Google style, 100-col),
`cython-lint`, `cmake-format`, `shfmt`/`shellcheck`, `markdownlint-cli2`.

### clang-tidy (separate from pre-commit)

```bash
uv run --no-project --with "clang-tidy==21.1.1" \
  python tests/lint/clang_tidy_precommit.py \
    --build-dir=build-pre-commit \
    --jobs=$(sysctl -n hw.ncpu) \
    ./src/ ./include ./tests
```

## Code conventions

### C++

- Source files: `.cc` (not `.cpp`). Headers: `.h`.
- Style: Google (via clang-format), 100-col limit, pointer-left (`int* ptr`).
- Namespaces: `namespace tvm { namespace ffi { ... } }` (no C++17 nested form).
- Header guards: `#ifndef TVM_FFI_<PATH>_H_`.
- Object pattern: `FooObj` (data) + `Foo` (ref wrapper extending `ObjectRef`).
  - `TVM_FFI_DECLARE_OBJECT_INFO("key", FooObj, ParentObj)` in the Obj class.
  - `TVM_FFI_DEFINE_OBJECT_REF_METHODS(Foo, ParentRef, FooObj)` in the Ref class.
- Errors: `TVM_FFI_THROW(ErrorType) << "message"`.
- Doc comments: Doxygen (`/*! \brief ... */`).
- Every file needs an Apache 2.0 license header.

### Python

- `from __future__ import annotations` at the top of every file.
- Style: `ruff` (100-col, double quotes, Google docstrings).
- Register objects: `@register_object("type.Key")`.
- Register functions: `register_global_func("name", fn)`.

### Commit messages

Tag-based style: `[FEAT]`, `[FIX]`, `[ERROR]`, `[TEST]`, `[CORE]`, `[EXTRA]`, etc.
Conventional style (`feat:`, `fix:`, `doc:`) is also used. Include PR number.

## Key architecture concepts

- **Any/AnyView**: Type-erased value containers. `AnyView` is non-owning, `Any` owns.
- **Object system**: Ref-counted heap objects. `ObjectObj` holds data, `Object` is the
  ref wrapper. Created via `make_object<T>(args...)`.
- **Function**: Type-erased callable with packed calling convention
  `(const AnyView* args, int32_t num_args, Any* rv)`.
- **Global registry**: Functions registered by string name, accessible cross-language
  via `register_global_func`/`get_global_func`.
- **Containers**: `Array<T>` (immutable), `List<T>` (mutable), `Map<K,V>` (immutable),
  `Dict` (mutable), `String`, `Tensor`, `Shape`, `Tuple`, `Variant<T...>`.
- **Reflection**: `ObjectDef<T>` builder with `def_field`/`def_method`. Exposed to
  Python as `tvm_ffi.dataclasses.c_class`.
- **Module system**: `load_module("path.so")` wraps shared libraries, exposing
  functions via `__tvm_ffi_<name>` symbol prefix.

## CI

Runs on: Linux x86_64 + aarch64, macOS arm64, Windows AMD64.
Jobs: lint -> clang-tidy (if C++ changed) -> doc build -> test (C++, Python, Rust).

## Further reading

The `docs/` directory contains the full Sphinx documentation site, including:

- `docs/concepts/` — design docs (ABI overview, Any, Object/Class, Tensor, Function, etc.)
- `docs/guides/` — usage guides (exporting functions/classes, kernel libraries, C++/Python/Rust)
- `docs/get_started/` — quickstart and stable C ABI
- `docs/dev/` — developer instructions (source build, CI/CD, release process, doc build)
- `docs/packaging/` — Python packaging guide
