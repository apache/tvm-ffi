.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
..
..   http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.

C++ Tooling
===========

This guide covers the TVM-FFI C++ toolchain: header layout, build integration,
library distribution, and editor setup.
For an end-to-end walkthrough that builds, loads, and calls a C++/CUDA kernel,
see :doc:`../get_started/quickstart`.

.. admonition:: Prerequisite
   :class: hint

   - Compiler: C++17-capable toolchain (GCC/Clang/MSVC)
   - CMake: 3.18 or newer
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi


C++ Headers
-----------

**Core APIs.** A single umbrella header exposes most of the API surface,
including :doc:`functions <../concepts/func_module>`,
:doc:`objects <../concepts/object_and_class>`,
:doc:`Any/AnyView <../concepts/any>`,
:doc:`tensors <../concepts/tensor>`, and
:doc:`exception handling <../concepts/exception_handling>`.

.. code-block:: cpp

  #include <tvm/ffi/tvm_ffi.h>


Extra features live in dedicated headers under
`tvm/ffi/extra/ <https://github.com/apache/tvm-ffi/tree/main/include/tvm/ffi/extra>`_
and should be included only when needed:

- **Environment API** -- allocator, stream, and device access
  (see :doc:`../concepts/tensor` for usage):
  ``#include <tvm/ffi/extra/c_env_api.h>``
- **Dynamic module loading** (see :ref:`sec:module`):
  ``#include <tvm/ffi/extra/module.h>``
- **CUBIN launcher** (see :doc:`../guides/cubin_launcher`):
  ``#include <tvm/ffi/extra/cuda/cubin_launcher.h>``


Build with TVM-FFI
------------------

CMake
~~~~~

TVM-FFI ships CMake utilities and imported targets through its package configuration.
The two primary functions are ``tvm_ffi_configure_target`` and ``tvm_ffi_install``,
both defined in ``cmake/Utils/Library.cmake``.

.. hint::

   See `examples/python_packaging/CMakeLists.txt <https://github.com/apache/tvm-ffi/blob/main/examples/python_packaging/CMakeLists.txt>`_
   for a complete working example.

Configure Target
""""""""""""""""

``tvm_ffi_configure_target`` wires a CMake target to TVM-FFI with sensible
defaults: it links headers and the shared library, configures debug symbols,
and optionally runs :ref:`stub generation <sec-stubgen>` as a post-build step.

.. code-block:: cmake

  tvm_ffi_configure_target(
    <target>
    [LINK_SHARED  ON|OFF  ]
    [LINK_HEADER  ON|OFF  ]
    [DEBUG_SYMBOL ON|OFF  ]
    [MSVC_FLAGS   ON|OFF  ]
    [STUB_INIT    ON|OFF  ]
    [STUB_DIR     <dir>   ]
    [STUB_PKG     <pkg>   ]
    [STUB_PREFIX  <prefix>]
  )

:LINK_SHARED: (default: ON) Link against the TVM-FFI shared library
  ``tvm_ffi::shared``. Set OFF for header-only usage or deferred runtime loading.
:LINK_HEADER: (default: ON) Link against the TVM-FFI headers via
  ``tvm_ffi::header``. Set OFF when you manage include paths and compile
  definitions manually.
:DEBUG_SYMBOL: (default: ON) Enable debug symbol post-processing hooks. On
  Apple platforms this runs ``dsymutil``.
:MSVC_FLAGS: (default: ON) Apply MSVC compatibility flags.
:STUB_DIR: (default: "") Output directory for generated Python stubs. When set,
  stub generation runs as a post-build step.
:STUB_INIT: (default: OFF) Allow the stub generator to initialize a new package layout.
  Requires ``STUB_DIR``.
:STUB_PKG: (default: "") Package name passed to the stub generator. Requires ``STUB_DIR``
  and ``STUB_INIT=ON``.
:STUB_PREFIX: (default: "") Module prefix passed to the stub generator. Requires
  ``STUB_DIR`` and ``STUB_INIT=ON``.

See :ref:`sec-stubgen-cmake` for a detailed explanation of each ``STUB_*`` option
and the generation modes they control.


Install Target
""""""""""""""

``tvm_ffi_install`` installs platform-specific artifacts for a target.
On Apple platforms it installs the ``.dSYM`` bundle when present;
on other platforms this is currently a no-op.

.. code-block:: cmake

  tvm_ffi_install(
    <target>
    [DESTINATION <dir>]
  )

:DESTINATION: Install destination directory relative to ``CMAKE_INSTALL_PREFIX``.


Set ``tvm_ffi_ROOT``
""""""""""""""""""""

If ``find_package(tvm_ffi CONFIG REQUIRED)`` fails because CMake cannot locate
the package, pass ``tvm_ffi_ROOT`` explicitly:

.. code-block:: bash

   cmake -S . -B build \
     -Dtvm_ffi_ROOT="$(tvm-ffi-config --cmakedir)"

.. note::

   When packaging Python wheels with scikit-build-core, ``tvm_ffi_ROOT`` is
   discovered automatically from the active Python environment.


GCC/NVCC
~~~~~~~~

For quick prototyping or CI scripts without CMake, invoke ``g++`` or ``nvcc``
directly with flags from ``tvm-ffi-config``.
The examples below are from the :doc:`Quick Start <../get_started/quickstart>` tutorial:

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../examples/quickstart/raw_compile.sh
      :language: bash
      :start-after: [cpp_compile.begin]
      :end-before: [cpp_compile.end]

  .. group-tab:: CUDA

    .. literalinclude:: ../../examples/quickstart/raw_compile.sh
      :language: bash
      :start-after: [cuda_compile.begin]
      :end-before: [cuda_compile.end]

The three ``tvm-ffi-config`` flags provide:

:``--cxxflags``: Include paths and compile definitions (``-I...``, ``-D...``)
:``--ldflags``: Library search paths (``-L...``, ``-Wl,-rpath,...``)
:``--libs``: Libraries to link (``-ltvm_ffi``)

**RPATH handling.** The resulting shared library links against ``libtvm_ffi.so``,
so the dynamic linker must be able to find it at load time:

- **Python distribution.** ``import tvm_ffi`` preloads ``libtvm_ffi.so`` into the
  process before any user library is loaded, so the RPATH requirement is already
  satisfied without additional linker flags.
- **Pure C++ distribution.** You must ensure ``libtvm_ffi.so`` is on the library
  search path. Either set ``-Wl,-rpath,$(tvm-ffi-config --libdir)`` at link time,
  or place ``libtvm_ffi.so`` alongside your binary.


Library Distribution
--------------------

When distributing pre-built shared libraries on Linux, glibc symbol versioning
can cause load-time failures on systems with a different glibc version.
The standard solution is the `manylinux <https://github.com/pypa/manylinux>`_
approach: **build on old glibc, run on new**.

**Build environment.** Use a manylinux Docker image:

.. code-block:: bash

   docker pull quay.io/pypa/manylinux2014_x86_64

Build host and device code inside the container. For CUDA:

.. code-block:: bash

   nvcc -shared -Xcompiler -fPIC your_kernel.cu -o kernel.so \
       $(tvm-ffi-config --cxxflags) \
       $(tvm-ffi-config --ldflags) \
       $(tvm-ffi-config --libs)

**Verify glibc requirements.** Inspect the minimum glibc version your binary requires:

.. code-block:: bash

   objdump -T your_kernel.so | grep GLIBC_

The ``apache-tvm-ffi`` wheel is already manylinux-compatible, so linking against
it inside a manylinux build environment produces portable binaries.


Editor Setup
------------

The following configuration enables code completion and diagnostics in
VSCode, Cursor, or any editor backed by clangd.

**CMake Tools (VSCode/Cursor).** Add these workspace settings so CMake Tools
can locate TVM-FFI and generate ``compile_commands.json``:

.. code-block:: json

   {
       "cmake.buildDirectory": "${workspaceFolder}/build-vscode",
       "cmake.configureArgs": [
           "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
       ],
       "cmake.configureSettings": {
           "Python_EXECUTABLE": "${workspaceFolder}/.venv/bin/python3",
           "tvm_ffi_ROOT": "${workspaceFolder}/.venv/lib/pythonX.Y/site-packages/tvm_ffi/share/cmake/tvm_ffi"
       }
   }

.. important::

  Make sure ``Python_EXECUTABLE`` and ``tvm_ffi_ROOT`` match the virtual
  environment you intend to use.

**clangd.** Create a ``.clangd`` file at the project root pointing to the CMake
compilation database. The snippet below also strips NVCC flags that clangd
does not recognize:

.. code-block:: yaml

   CompileFlags:
     CompilationDatabase: build-vscode/
     Remove: # for NVCC compatibility
       - -forward-unknown-to-host-compiler
       - --generate-code*
       - -Xcompiler*


Further Reading
---------------

- :doc:`../get_started/quickstart`: End-to-end walkthrough building and shipping a C++/CUDA kernel
- :doc:`../dev/source_build`: Building TVM-FFI from source with CMake flags
- :doc:`stubgen`: Generating Python type stubs from C++ reflection metadata
- :doc:`python_packaging`: Packaging shared libraries as Python wheels
- :doc:`../concepts/func_module`: Defining and exporting TVM-FFI functions
- :doc:`../concepts/object_and_class`: Defining C++ classes with cross-language reflection
- :doc:`../concepts/exception_handling`: Error handling across language boundaries
- :doc:`../concepts/abi_overview`: Low-level C ABI details
