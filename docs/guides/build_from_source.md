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
# Build from Source

This guide covers two common workflows:

- Python package building, which automatically includes building the core C++ library;
- C++-only package Building without Python.

```{admonition} Prerequisite
:class: tip

- Python: 3.9 or newer
- Compiler: C++17-capable toolchain
  - Linux: GCC or Clang with C++17 support
  - macOS: Apple Clang (via Xcode Command Line Tools)
  - Windows: MSVC (Visual Studio 2019 or 2022, x64)
- Build tools: CMake 3.18+; Ninja
```

## Build the Python Package

Download the source via:

```bash
git clone --recursive https://github.com/apache/tvm-ffi
cd tvm-ffi
```

```{note}
Always clone with ``--recursive`` to pull submodules. If you already cloned without it, run:

    git submodule update --init --recursive

```

Follow the instruction below to build the Python package with scikit-build-core, which drives CMake to compile the C++ core and Cython extension.

```bash
pip install --reinstall --verbose -e . \
  --config-settings cmake.define.TVM_FFI_ATTACH_DEBUG_SYMBOLS=ON
```

The ``--reinstall`` flag forces a rebuild, and ``-e/--editable`` install means future Python-only changes are reflected immediately without having to rebuild.

```{warning}
However, changes to C++/Cython always require re-running the install command.
```

Verify the install:

```bash
python -c "import tvm_ffi; print(tvm_ffi.__version__)"
tvm-ffi-config -h
```

```{tip}
Use ``tvm-ffi-config`` to query include and link flags when consuming TVM FFI from external C/C++ projects:
    tvm-ffi-config --includedir
    tvm-ffi-config --dlpack-includedir
    tvm-ffi-config --libfiles   # or --libs/--ldflags on Unix
```

## Build the C/C++ Library Only

TVM FFI can be used as a standalone C/C++ library without Python. The instruction below should work for Linux, macOS and Windows.

```bash
cmake . -B build_cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_cpp --parallel --config RelWithDebInfo --target tvm_ffi_shared
cmake --install build_cpp --config RelWithDebInfo --prefix ./dist
```

After installation, you should see:

- Headers are installed under ``dist/include/``;
- Libraries are installed under ``dist/lib/``.

## Troubleshooting

- Rebuilding after C++/Cython changes: re-run ``pip install --reinstall -e .``. Editable installs only auto-reflect pure Python changes.
- Submodules missing: run ``git submodule update --init --recursive`` from the repo root.
- Library not found at import time: ensure your dynamic loader can find the shared library. If built from source, add the ``lib`` directory to ``LD_LIBRARY_PATH`` (Linux), ``DYLD_LIBRARY_PATH`` (macOS), or ``PATH`` (Windows).
- Wrong generator/build type: Ninja/Unix Makefiles use ``-DCMAKE_BUILD_TYPE=...``; Visual Studio requires ``--config ...`` at build/ctest time.
