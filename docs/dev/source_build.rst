..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Build from Source
=================

This guide covers two common workflows:

- Python package building, which automatically includes building the core C++
  library;
- C++-only package building without Python.

.. admonition:: Prerequisite
   :class: tip

   - Python: 3.9 or newer
   - Compiler: C++17-capable toolchain

     - Linux: GCC or Clang with C++17 support
     - macOS: Apple Clang (via Xcode Command Line Tools)
     - Windows: MSVC (Visual Studio 2019 or 2022, x64)

   - Build tools: CMake 3.18+; Ninja

Build the Python Package
------------------------

Download the source via:

.. code-block:: bash

   git clone --recursive https://github.com/apache/tvm-ffi
   cd tvm-ffi

.. note::

   Always clone with ``--recursive`` to pull submodules. If you already cloned
   without it, run:

   .. code-block:: bash

      git submodule update --init --recursive

Build the Python package with scikit-build-core, which drives CMake to compile
the C++ core and Cython extension:

.. code-block:: bash

   uv pip install --force-reinstall --verbose -e .

The ``--force-reinstall`` flag forces a rebuild, and ``-e`` (editable) install
means future Python-only changes are reflected immediately without having to
rebuild.

**CMake flags** can be passed via ``--config-settings cmake.define.<FLAG>=<VALUE>``.
For example, to attach debug symbols:

.. code-block:: bash

   uv pip install --force-reinstall --verbose -e . \
     --config-settings cmake.define.TVM_FFI_ATTACH_DEBUG_SYMBOLS=ON

Available CMake options (see `CMakeLists.txt <https://github.com/apache/tvm-ffi/blob/main/CMakeLists.txt>`__) include:

- ``TVM_FFI_ATTACH_DEBUG_SYMBOLS`` -- Attach debug symbols even in release
  mode (default: ``OFF``).
- ``TVM_FFI_USE_LIBBACKTRACE`` -- Enable libbacktrace (default: ``ON``).
- ``TVM_FFI_USE_EXTRA_CXX_API`` -- Enable extra C++ API in shared lib
  (default: ``ON``).
- ``TVM_FFI_BACKTRACE_ON_SEGFAULT`` -- Set signal handler to print backtrace
  on segfault (default: ``ON``).
- ``CMAKE_EXPORT_COMPILE_COMMANDS`` -- Generate ``compile_commands.json`` for
  clangd and other tools (default: ``OFF``).

.. warning::

   However, changes to C++/Cython always require re-running the install
   command.

Verify the install:

.. code-block:: bash

   uv run python -c "import tvm_ffi; print(tvm_ffi.__version__)"
   uv run tvm-ffi-config -h

.. tip::

   Use ``tvm-ffi-config`` to query include and link flags when consuming TVM
   FFI from external C/C++ projects:

   .. code-block:: bash

      tvm-ffi-config --includedir
      tvm-ffi-config --dlpack-includedir
      tvm-ffi-config --libfiles   # or --libs/--ldflags on Unix

Build the C/C++ Library Only
----------------------------

TVM FFI can be used as a standalone C/C++ library without Python. The
instruction below should work for Linux, macOS and Windows.

.. code-block:: bash

   cmake . -B build_cpp -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build_cpp --parallel --config RelWithDebInfo --target tvm_ffi_shared
   cmake --install build_cpp --config RelWithDebInfo --prefix ./dist

After installation, you should see:

- Headers are installed under ``dist/include/``;
- Libraries are installed under ``dist/lib/``.

Troubleshooting
---------------

- **Rebuilding after C++/Cython changes**: re-run
  ``uv pip install --force-reinstall -e .``. Editable installs only
  auto-reflect pure Python changes.
- **Submodules missing**: run ``git submodule update --init --recursive`` from
  the repo root.
- **Library not found at import time**: ensure your dynamic loader can find the
  shared library. If built from source, add the ``lib`` directory to
  ``LD_LIBRARY_PATH`` (Linux), ``DYLD_LIBRARY_PATH`` (macOS), or ``PATH``
  (Windows).
- **Wrong generator/build type**: Ninja/Unix Makefiles use
  ``-DCMAKE_BUILD_TYPE=...``; Visual Studio requires ``--config ...`` at
  build/ctest time.

.. seealso::

   - :doc:`ci_cd`: Reproduce linters, unit tests, and wheel builds locally.
   - :doc:`../get_started/quickstart`: End-to-end walkthrough of building and
     running a C++/CUDA kernel.
   - :doc:`../packaging/cpp_tooling`: CMake integration, compiler flags, and
     library distribution for downstream projects.
