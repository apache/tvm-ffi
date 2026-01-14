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

This guide covers the TVM-FFI C++ toolchain, focusing on header layout, CMake
integration, and editor setup for a smooth local workflow.

.. admonition:: Prerequisite
   :class: hint

   - Compiler: C++17-capable toolchain (GCC/Clang/MSVC)
   - CMake: 3.18 or newer
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi


C++ Headers
-----------

**Core APIs.** Most of the APIs are exposed via a single umbrella header.

.. code-block:: cpp

  #include <tvm/ffi/tvm_ffi.h>


Extra features live in dedicated headers under the
`tvm/ffi/extra/ <https://github.com/apache/tvm-ffi/tree/main/include/tvm/ffi/extra>`_
subdirectory and should be included only when needed.

**Environment API**. Use the environment API to access the caller's allocator,
stream, and device:

.. code-block:: cpp

  #include <tvm/ffi/extra/c_env_api.h>


**Dynamic module loading**. Dynamic module loading lives in the extra API and
requires its own header:

.. code-block:: cpp

  #include <tvm/ffi/extra/module.h>


**CUBIN launcher**. See :doc:`CUBIN launching utilities <../guides/cubin_launcher>`; the header is:

.. code-block:: cpp

  #include <tvm/ffi/extra/cuda/cubin_launcher.h>


CMake Usage
-----------

TVM-FFI ships CMake utilities and imported targets through its package configuration.
The two primary functions are ``tvm_ffi_configure_target`` and ``tvm_ffi_install``,
both defined in ``cmake/Utils/Library.cmake``.

Configure Target
~~~~~~~~~~~~~~~~

The configure helper wires a target to TVM-FFI and provides sensible defaults.
It links the TVM-FFI headers and shared library, and it configures debug symbol handling.
Optionally, it runs the Python :ref:`stub generation <sec-stubgen>` tool after
the build completes.

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


Install Target
~~~~~~~~~~~~~~

The install helper handles extra artifacts associated with a target.
``DESTINATION`` defaults to ``.`` (relative to ``CMAKE_INSTALL_PREFIX``).
On Apple platforms the target ``.dSYM`` bundle is installed when present.
On non-Apple platforms, this is currently a no-op.

.. code-block:: cmake

  tvm_ffi_install(
    <target>
    [DESTINATION <dir>]
  )

:DESTINATION: Install destination directory relative to ``CMAKE_INSTALL_PREFIX``.


CMake Example
~~~~~~~~~~~~~

.. code-block:: cmake

  find_package(tvm_ffi CONFIG REQUIRED)  # requires tvm_ffi_ROOT
  tvm_ffi_configure_target(my-shared-lib)  # configure TVM-FFI linkage
  install(TARGETS my-shared-lib DESTINATION .)
  tvm_ffi_install(my-shared-lib DESTINATION .)  # install extra artifacts


Set ``tvm_ffi_ROOT``
~~~~~~~~~~~~~~~~~~~~

For a pure C++ build, CMake may fail when it reaches

.. code-block:: cmake

  find_package(tvm_ffi CONFIG REQUIRED)

if it cannot locate the TVM-FFI package. In that case, set
``tvm_ffi_ROOT`` to the TVM-FFI CMake package directory.

.. code-block:: bash

   cmake -S . -B build \
     -Dtvm_ffi_ROOT="$(tvm-ffi-config --cmakedir)"


.. note::

   When packaging Python wheels with scikit-build-core, ``tvm_ffi_ROOT`` is
   discovered automatically from the active Python environment, so you usually
   do not need to set it explicitly.


VSCode/Cursor
-------------

The following settings help CMake Tools integrate with TVM-FFI and generate the
``compile_commands.json`` used by clangd:

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


clangd
------

Create a ``.clangd`` file at your project root and point it to the CMake
compilation database. The snippet below also removes NVCC flags that clangd
does not understand:

.. code-block:: yaml

   CompileFlags:
     CompilationDatabase: build-vscode/
     Remove: # for NVCC compatibility
       - -forward-unknown-to-host-compiler
       - --generate-code*
       - -Xcompiler*

Make sure your CMake configure step enables ``compile_commands.json`` via
``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON``.
