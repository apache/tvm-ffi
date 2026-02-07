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

Python Packaging
================

This guide walks through a small, complete workflow for packaging a TVM-FFI extension
as a Python wheel. The goal is to help you wire up a simple extension, produce a wheel,
and ship user-friendly typing annotations without needing to know every detail of TVM
internals. We cover three checkpoints:

- Build a Python wheel;
- Export C++ to Python;
- Generate Python package stubs.

.. note::

  All code used in this guide is under
  `examples/python_packaging <https://github.com/apache/tvm-ffi/tree/main/examples/python_packaging>`_.

.. admonition:: Prerequisite
   :class: hint

   - Python: 3.9 or newer (for the ``tvm_ffi.config``/``tvm-ffi-config`` helpers)
   - Compiler: C11-capable toolchain (GCC/Clang/MSVC)
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi


Build Python Wheel
------------------

Start by defining the Python packaging and build wiring. TVM-FFI provides helpers to build and ship
ABI-agnostic Python extensions using standard packaging tools. The steps below set up the build so
you can plug in the C++ exports from the next section.

The flow below uses :external+scikit_build_core:doc:`scikit-build-core <index>`
to drive a CMake build, but the same ideas apply to setuptools or other :pep:`517` backends.

CMake Target
~~~~~~~~~~~~

Assume the source tree contains ``src/extension.cc``. Create a ``CMakeLists.txt`` that
creates a shared target ``my_ffi_extension`` and configures it against TVM-FFI.

.. literalinclude:: ../../examples/python_packaging/CMakeLists.txt
  :language: cmake
  :start-after: [example.cmake.begin]
  :end-before: [example.cmake.end]

Function ``tvm_ffi_configure_target`` sets up TVM-FFI include paths and links against the TVM-FFI library.
Additional options for stub generation are covered in :ref:`sec-stubgen`.

Function ``tvm_ffi_install`` places necessary information (e.g., debug symbols on macOS) next to
the shared library for packaging.

Python Build Backend
~~~~~~~~~~~~~~~~~~~~

Define a :pep:`517` build backend in ``pyproject.toml`` with the following steps:

- Specify ``apache-tvm-ffi`` as a build requirement, so that CMake can find TVM-FFI;
- Configure ``wheel.py-api`` that indicates a Python ABI-agnostic wheel;
- Specify the source directory of the package via ``wheel.packages``, and the installation
  destination via ``wheel.install-dir``.

.. literalinclude:: ../../examples/python_packaging/pyproject.toml
  :language: toml
  :start-after: [pyproject.build.begin]
  :end-before: [pyproject.build.end]

Once specified, scikit-build-core will invoke CMake and drive the extension build.


Wheel Auditing
~~~~~~~~~~~~~~

**Build wheels**. You can build wheels using standard workflows, for example:

- `pip workflow <https://pip.pypa.io/en/stable/cli/pip_wheel/>`_ or `editable install <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_

.. code-block:: bash

  # editable install
  pip install -e .
  # standard wheel build
  pip wheel -w dist .

- `uv workflow <https://docs.astral.sh/uv/guides/package/>`_

.. code-block:: bash

  uv build --wheel --out-dir dist .

- `cibuildwheel <https://cibuildwheel.pypa.io/>`_ for multi-platform build

.. code-block:: bash

  cibuildwheel --output-dir dist

**Audit wheels**. In practice, an extra step is usually needed to remove redundant
and error-prone shared library dependencies. In our case, because ``libtvm_ffi.so``
(or its platform variants) is guaranteed to be loaded by importing ``tvm_ffi``,
we can safely exclude this dependency from the final wheel.

.. code-block:: bash

   # Linux
   auditwheel repair --exclude libtvm_ffi.so dist/*.whl
   # macOS
   delocate-wheel -w dist -v --exclude libtvm_ffi.dylib dist/*.whl
   # Windows
   delvewheel repair --exclude tvm_ffi.dll -w dist dist\\*.whl

.. _sec-load-the-library:

Load the Library
~~~~~~~~~~~~~~~~

Once the wheel is installed, use :py:func:`tvm_ffi.libinfo.load_lib_module` to load
the shared library:

.. code-block:: python

   from tvm_ffi.libinfo import load_lib_module

   LIB = load_lib_module(
       package="my-ffi-extension",
       target_name="my_ffi_extension",
   )

The parameters are:

- ``package``: The Python package name as registered with pip (e.g., ``"my-ffi-extension"``
  or ``"apache-tvm-ffi"``). This is the name in ``pyproject.toml``, **not** the import name
  (e.g., ``tvm_ffi``). The function uses ``importlib.metadata.distribution(package)`` internally
  to locate installed package files.

- ``target_name``: The CMake target name (e.g., ``"my_ffi_extension"``). It is used to derive
  the platform-specific shared library filename:

  * Linux: ``lib{target_name}.so``
  * macOS: ``lib{target_name}.dylib``
  * Windows: ``{target_name}.dll``

Once the library is loaded, functions and classes can be exported from C++ and
called from Python. See :doc:`../guides/export_func_cls` for the three export
mechanisms (C symbols, global functions, and classes) with complete examples.


Stub Generation
---------------

TVM-FFI provides a stub generation tool ``tvm-ffi-stubgen`` that creates Python type hints
from C++ reflection metadata. The tool integrates with CMake and can generate complete
stub files automatically, or update existing files using special directive comments.

For most projects, enable automatic stub generation in CMake:

.. code-block:: cmake

   tvm_ffi_configure_target(my_ffi_extension
       STUB_DIR "python"
       STUB_INIT ON
   )

This generates ``_ffi_api.py`` and ``__init__.py`` files with proper type hints for all
registered global functions and classes.

.. seealso::

   :doc:`stubgen` for the complete stub generation guide, including directive-based
   customization and command-line usage.
