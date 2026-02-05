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

Export C++ to Python
--------------------

Include the umbrella header to access the core TVM-FFI C++ API.

.. code-block:: cpp

   #include <tvm/ffi/tvm_ffi.h>

TVM-FFI offers three ways to expose code:

- C symbols in the TVM-FFI ABI: export code as plain C symbols. This is the recommended way for
  most use cases because it keeps the boundary thin and works well with compiler codegen;
- Functions: expose functions via the global registry;
- Classes: register C++ classes derived from :cpp:class:`tvm::ffi::Object` as Python dataclasses.

Metadata is captured automatically and later turned into Python type hints for LSP support.
The examples below show C++ code and its Python usage. The "Python (Generated)" tab
shows code produced by the stub generation tool (see :ref:`sec-stubgen`).

TVM-FFI ABI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to export plain C symbols, TVM-FFI provides helpers to make them accessible
from Python. This option keeps the boundary thin and works well with LLVM-based compilers where
C symbols are easier to call.

.. tabs::

  .. group-tab:: C++

    Macro :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` exports the function ``AddTwo`` as
    a C symbol ``__tvm_ffi_add_two`` inside the shared library.

    .. literalinclude:: ../../examples/python_packaging/src/extension.cc
      :language: cpp
      :start-after: [tvm_ffi_abi.begin]
      :end-before: [tvm_ffi_abi.end]

  .. group-tab:: Python (User)

    Symbol ``__tvm_ffi_add_two`` is made available via ``LIB.add_two`` to users.

    .. code-block:: python

      import my_ffi_extension
      my_ffi_extension.LIB.add_two(1)  # -> 3

  .. group-tab:: Python (Generated)

    The shared library is loaded by :py:func:`tvm_ffi.libinfo.load_lib_module`.

    .. code-block:: python

      # File: my_ffi_extension/_ffi_api.py

      LIB = tvm_ffi.libinfo.load_lib_module(
        package="my-ffi-extension",
        target_name="my_ffi_extension",
      )


Global Function
~~~~~~~~~~~~~~~

This example registers a function in the global registry and then calls it from Python.
The registry handles type translation, error handling, and metadata.

.. tabs::

  .. group-tab:: C++

    C++ function ``AddOne`` is registered with name ``my_ffi_extension.add_one``
    in the global registry using :cpp:class:`tvm::ffi::reflection::GlobalDef`.

    .. literalinclude:: ../../examples/python_packaging/src/extension.cc
      :language: cpp
      :start-after: [global_function.begin]
      :end-before: [global_function.end]

  .. group-tab:: Python (User)

    The global function is accessible after importing the extension,
    and the import path matches the registered name, i.e. ``my_ffi_extension.add_one``.

    .. code-block:: python

      import my_ffi_extension

      my_ffi_extension.add_one(3)  # -> 4

  .. group-tab:: Python (Generated)

    Under the hood, the shared library is loaded by :py:func:`tvm_ffi.init_ffi_api`
    during package initialization.

    .. code-block:: python

      # File: my_ffi_extension/_ffi_api.py

      tvm_ffi.init_ffi_api(
        namespace="my_ffi_extension",
        target_module_name=__name__,
      )

      def add_one(x: int) -> int: ...


.. note::

  Global functions can be retrieved via :py:func:`tvm_ffi.get_global_func` in Python, :cpp:func:`TVMFFIFunctionGetGlobal` in C,
  or :cpp:func:`tvm::ffi::Function::GetGlobal` in C++.

  .. code-block:: python

    func = tvm_ffi.get_global_func("my_ffi_extension.add_one")
    func(3)  # -> 4


Class
~~~~~

Any class derived from :cpp:class:`tvm::ffi::Object` can be registered, exported, and
instantiated from Python. The reflection helper :cpp:class:`tvm::ffi::reflection::ObjectDef`
makes it easy to expose:

- Fields

  * Immutable field via :cpp:func:`ObjectDef::def_ro <tvm::ffi::reflection::ObjectDef::def_ro>`;
  * Mutable field via :cpp:func:`ObjectDef::def_rw <tvm::ffi::reflection::ObjectDef::def_rw>`;

- Methods

  * Member method via :cpp:func:`ObjectDef::def <tvm::ffi::reflection::ObjectDef::def>`.
  * Static method via :cpp:func:`ObjectDef::def_static <tvm::ffi::reflection::ObjectDef::def_static>`;
  * Constructors via :cpp:class:`tvm::ffi::reflection::init`.


.. tabs::

  .. group-tab:: C++

    The example below defines a class ``my_ffi_extension.IntPair`` with

    - two integer fields ``a``, ``b``,
    - a constructor, and
    - a method ``Sum`` that returns the sum of the two fields.

    .. literalinclude:: ../../examples/python_packaging/src/extension.cc
      :language: cpp
      :start-after: [object.begin]
      :end-before: [object.end]

  .. group-tab:: Python (User)

    The class is available immediately after importing the extension,
    with the import path matching the registered name, i.e. ``my_ffi_extension.IntPair``.

    .. code-block:: python

      import my_ffi_extension

      pair = my_ffi_extension.IntPair(1, 2)
      pair.sum()  # -> 3

  .. group-tab:: Python (Generated)

    Type hints are generated for both fields and methods.

    .. code-block:: python

      # File: my_ffi_extension/_ffi_api.py (auto generated)

      @tvm_ffi.register_object("my_ffi_extension.IntPair")
      class IntPair(tvm_ffi.Object):
          a: int
          b: int

          def __init__(self, a: int, b: int) -> None: ...
          def sum(self) -> int: ...


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
