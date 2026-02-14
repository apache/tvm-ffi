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

---
name: devtools
description: Developer reference for Apache TVM-FFI.
argument-hint: "[lint | cpp | python | docs]"
---

# TVM-FFI Developer Guide

Condensed reference from `docs/dev/`. Use this when working on the TVM-FFI codebase.

## Prerequisites

- **Python**: 3.9+ (managed via `uv`; default virtualenv at `.venv`)
- **Compiler**: C++17-capable toolchain (GCC/Clang on Linux, Apple Clang on macOS, MSVC on Windows)
- **Build tools**: CMake 3.18+, Ninja
- **Source**: Always clone with `--recursive`, or run `git submodule update --init --recursive`

All Python-related commands below use [`uv`](https://docs.astral.sh/uv/). The default virtual environment is `.venv` in the repo root.

---

## 1. Run Linters and clang-tidy

### Pre-commit hooks (primary linting workflow)

Install and register git hooks:

```bash
uv tool install pre-commit
pre-commit install          # register git hooks (runs automatically before each commit)
```

Run hooks manually:

```bash
pre-commit run --all-files          # all hooks on every file
pre-commit run                      # all hooks on staged files only
pre-commit run <hook-id> --all-files  # single hook, e.g. ruff-check, clang-format
```

**Linters by language:**

| Language | Formatter                       | Linter / Type Checker            |
|----------|---------------------------------|----------------------------------|
| Python   | `ruff` (format)                 | `ruff` (lint), `ty` (type check) |
| C/C++    | `clang-format`                  | `clang-tidy` (see below)         |
| Cython   | `double-quote-cython-strings`   | `cython-lint`                    |
| CMake    | `cmake-format`                  | `cmake-lint`                     |
| Shell    | `shfmt`                         | `shellcheck`                     |
| YAML     | `yamllint`                      |                                  |
| TOML     | `taplo-format`                  |                                  |
| Markdown | `markdownlint-cli2`             |                                  |
| RST      | `rstcheck`                      |                                  |

**Troubleshooting pre-commit:**

- **Version problems**: Ensure pre-commit 2.18.0+ (`pre-commit --version`).
- **Stale cache**: Run `pre-commit clean` to clear the hook cache.
- **Auto-fixed files**: Most formatting hooks fix issues in place. Review changes, stage with `git add -u`, and commit again.

### clang-tidy (separate from pre-commit)

`clang-tidy` is **not** a pre-commit hook. It runs as a separate CI job and only checks changed C++ files. To reproduce locally:

```bash
# Run on specific files
uv run --no-project --with "clang-tidy==21.1.1" \
  python tests/lint/clang_tidy_precommit.py \
    --build-dir=build-pre-commit \
    --jobs=$(sysctl -n hw.ncpu) \
    include/tvm/ffi/c_api.h src/some_file.cc

# Run on all C++ sources
uv run --no-project --with "clang-tidy==21.1.1" \
  python tests/lint/clang_tidy_precommit.py \
    --build-dir=build-pre-commit \
    --jobs=$(sysctl -n hw.ncpu) \
    ./src/ ./include ./tests
```

> On macOS, `clang-tidy` is resolved via `xcrun`; the wrapper script handles this automatically.
> On Linux, replace `$(sysctl -n hw.ncpu)` with `$(nproc)`.

---

## 2. Build and Test the C++ Package

### Build the C++ library (no Python)

```bash
cmake . -B build_cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_cpp --parallel --config RelWithDebInfo --target tvm_ffi_shared
cmake --install build_cpp --config RelWithDebInfo --prefix ./dist
```

After installation:
- Headers: `dist/include/`
- Libraries: `dist/lib/`

### Build and run C++ tests

Set parallel build level first:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.ncpu)  # macOS
# export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)             # Linux
```

Configure, build, and run:

```bash
cmake . -B build_test -DTVM_FFI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build_test --clean-first --config Debug --target tvm_ffi_tests
ctest -V -C Debug --test-dir build_test --output-on-failure
```

### Key CMake options

| Option                           | Default | Description                                  |
|----------------------------------|---------|----------------------------------------------|
| `TVM_FFI_BUILD_TESTS`           | `OFF`   | Enable C++ test targets                      |
| `TVM_FFI_ATTACH_DEBUG_SYMBOLS`  | `OFF`   | Attach debug symbols in release mode         |
| `TVM_FFI_USE_LIBBACKTRACE`      | `ON`    | Enable libbacktrace                          |
| `TVM_FFI_USE_EXTRA_CXX_API`     | `ON`    | Enable extra C++ API in shared lib           |
| `TVM_FFI_BACKTRACE_ON_SEGFAULT` | `ON`    | Print backtrace on segfault                  |
| `CMAKE_EXPORT_COMPILE_COMMANDS`  | `OFF`   | Generate `compile_commands.json` for clangd  |

> On Windows, run from a **Developer Command Prompt for VS** or ensure the MSVC toolchain is on your `PATH`.

---

## 3. Build and Test the Python Package

### Build (editable install)

The Python build uses `scikit-build-core` which drives CMake to compile the C++ core and Cython extension:

```bash
uv pip install --force-reinstall --verbose -e .
```

- `--force-reinstall` forces a full rebuild.
- `-e` (editable) means pure Python changes are reflected immediately without rebuilding.
- **C++/Cython changes always require re-running this command.**

Pass CMake flags via `--config-settings`:

```bash
uv pip install --force-reinstall --verbose -e . \
  --config-settings cmake.define.TVM_FFI_ATTACH_DEBUG_SYMBOLS=ON
```

Verify the install:

```bash
uv run python -c "import tvm_ffi; print(tvm_ffi.__version__)"
```

### Run Python tests

Install with test dependencies, then run pytest:

```bash
uv pip install --force-reinstall --verbose --group test -e .
uv run pytest -vvs tests/python
```

### Run Rust tests

Rust tests require the Python package to be installed first (Rust FFI bindings link against the built shared library):

```bash
cd rust && cargo test
```

### Troubleshooting

- **Rebuilding after C++/Cython changes**: Re-run `uv pip install --force-reinstall -e .`. Editable installs only auto-reflect pure Python changes.
- **Submodules missing**: Run `git submodule update --init --recursive` from the repo root.
- **Library not found at import time**: Ensure the dynamic loader can find the shared library. Add the `lib` directory to `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` (macOS), or `PATH` (Windows).

---

## 4. Build Documentation Website

Building documentation requires the Python package to be installed first (see section 3).

### Interactive build (auto-reload)

Serves locally at `http://127.0.0.1:8000` with live reload:

```bash
uv run --group docs sphinx-autobuild docs docs/_build/html \
  --ignore docs/reference/cpp/generated
```

### One-off build

```bash
uv run --group docs sphinx-build -M html docs docs/_build
```

### With C++ API reference (requires Doxygen)

```bash
# Install Doxygen
brew install doxygen        # macOS
# sudo apt install doxygen  # Linux

# Interactive
BUILD_CPP_DOCS=1 uv run --group docs sphinx-autobuild docs docs/_build/html \
  --ignore docs/reference/cpp/generated --watch include

# One-off
BUILD_CPP_DOCS=1 uv run --group docs sphinx-build -M html docs docs/_build
```

### With Rust API reference (requires cargo)

```bash
# Interactive
BUILD_RUST_DOCS=1 uv run --group docs sphinx-autobuild docs docs/_build/html \
  --ignore docs/reference/rust/generated --watch rust

# One-off
BUILD_RUST_DOCS=1 uv run --group docs sphinx-build -M html docs docs/_build
```

### Build all documentation

```bash
BUILD_CPP_DOCS=1 BUILD_RUST_DOCS=1 uv run --group docs sphinx-build \
  -M html docs docs/_build
```

### Cleanup

```bash
rm -rf docs/_build/ docs/reference/python/generated docs/reference/cpp/generated docs/reference/rust/generated
```
