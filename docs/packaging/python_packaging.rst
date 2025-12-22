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

This guide walks through a small but complete workflow for packaging a TVM-FFI extension
as a Python wheel. The goal is to help you wire up a simple extension, produce a wheel,
and ship user-friendly typing annotations without needing to know every detail of TVM
internals. We will cover three checkpoints:

- Export C++ to Python;
- Build Python wheel;
- Automatic Python package generation tools.

.. note::

  All code used in this guide lives under
  `examples/python_packaging <https://github.com/apache/tvm-ffi/tree/main/examples/python_packaging>`_.

.. admonition:: Prerequisite
   :class: hint

   - Python: 3.9 or newer (for the ``tvm_ffi.config``/``tvm-ffi-config`` helpers)
   - Compiler: C11-capable toolchain (GCC/Clang/MSVC)
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi


Export C++ to Python
--------------------

TVM-FFI offers three ways to expose code:

- C symbols in TVM FFI ABI: Export code as plain C symbols. This is the recommended way for
  most usecases as it keeps the boundary thin and works well with compiler codegen;
- Functions: Reflect functions via the global registry;
- Classes: Register C++ classes derived from :cpp:class:`tvm::ffi::Object` to Python dataclasses.

Metadata is automatically captured and is later be turned into type hints for proper LSP help.

TVM-FFI ABI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to export plain C symbols, TVM-FFI provides helpers to make them accessible
to Python. This option keeps the boundary thin and works well with LLVM compilers where
C symbols are easier to call into.

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

This example registers a function into the global registry and then calls it from Python.
It registry handles type translation, error handling, and metadata.

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

Any class derived from :cpp:class:`tvm::ffi::Object` can be registered, exported and
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
      pair.sum() # -> 3

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


Build Python Wheel
------------------

Once the C++ side is ready, TVM-FFI provides convenient helpers to build and ship
ABI-agnostic Python extensions using any standard packaging tool.

The flow below uses :external+scikit_build_core:doc:`scikit-build-core <index>`
that drives CMake build, but the same ideas translate to setuptools or other :pep:`517` backends.

CMake Target
~~~~~~~~~~~~

Assume the source tree contains ``src/extension.cc``. Create a ``CMakeLists.txt`` that
creates a shared target ``my_ffi_extension`` and configures it against TVM-FFI.

.. literalinclude:: ../../examples/python_packaging/CMakeLists.txt
  :language: cmake
  :start-after: [example.cmake.begin]
  :end-before: [example.cmake.end]

Function ``tvm_ffi_configure_target`` sets up TVM-FFI include paths, link against TVM-FFI library,
generates stubs under ``STUB_DIR``, and can scaffold stub files when ``STUB_INIT`` is
enabled.

Function ``tvm_ffi_install`` places necessary information, e.g. debug symbols in macOS, next to
the shared library for proper packaging.

Python Build Backend
~~~~~~~~~~~~~~~~~~~~

Define a :pep:`517` build backend in ``pyproject.toml``, with the following steps:

- Sepcfiy ``apache-tvm-ffi`` as a build requirement, so that CMake can find TVM-FFI;
- Configure ``wheel.py-api`` that indicates a Python ABI-agnostic wheel;
- Specify the source directory of the package via ``wheel.packages``, and the installation
  destination via ``wheel.install-dir``.

.. literalinclude:: ../../examples/python_packaging/pyproject.toml
  :language: toml
  :start-after: [pyproject.build.begin]
  :end-before: [pyproject.build.end]

Once fully specified, scikit-build-core will invoke CMake and drive the extension building process.


Wheel Auditing
~~~~~~~~~~~~~~

**Build wheels**. The wheel can be built using the standard workflows, e.g.:

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

**Audit wheels**. In practice, an extra step is usually necessary to remove redundant
and error-prone shared library dependencies. In our case, given ``libtvm_ffi.so``
(or its respective platform variants) is guaranteed to be loaded by importing ``tvm_ffi``,
we can safely exclude this dependency from the final wheel.

.. code-block:: bash

   # Linux
   auditwheel repair --exclude libtvm_ffi.so dist/*.whl
   # macOS
   delocate-wheel -w dist -v --exclude libtvm_ffi.dylib dist/*.whl
   # Windows
   delvewheel repair --exclude tvm_ffi.dll -w dist dist\\*.whl

Stub Generation Tool
--------------------

TVM-FFI comes with a command-line tool ``tvm-ffi-stubgen`` that automates
the generation of type stubs for both global functions and classes.
It turns reflection metadata into proper Python type hints, and generates
corresponding Python code **inline** and **statically**.

Inline Directives
~~~~~~~~~~~~~~~~~

Similar to linter tools, ``tvm-ffi-stubgen`` uses special comments
to identify what to generate and where to write generated code.

**Directive 1 (Global functions)**. Example below shows an directive
``global/${prefix}`` marking a type stub section of global functions.

.. code-block:: python

   # tvm-ffi-stubgen(begin): global/my_ext.arith
   tvm_ffi.init_ffi_api("my_ext.arith", __name__)
   if TYPE_CHECKING:
     def add_one(_0: int, /) -> int: ...
     def add_two(_0: int, /) -> int: ...
     def add_three(_0: int, /) -> int: ...
   # tvm-ffi-stubgen(end)

Running ``tvm-ffi-stubgen`` fills in the function stubs between the
``begin`` and ``end`` markers based on the loaded registry, and in this case
introduces all the global functions named ``my_ext.arith.*``.

**Directive 2 (Classes)**. Example below shows an directive
``object/${type_key}`` marking the fields and methods of a registered class.

.. code-block:: python

   @tvm_ffi.register_object("my_ffi_extension.IntPair")
   class IntPair(_ffi_Object):
     # tvm-ffi-stubgen(begin): object/my_ffi_extension.IntPair
     a: int
     b: int
     if TYPE_CHECKING:
       def __init__(self, a: int, b: int) -> None: ...
       def sum(self) -> int: ...
     # tvm-ffi-stubgen(end)

Directive-based Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

After TVM-FFI extension is built as a shared library, say at
``build/libmy_ffi_extension.so``

**Command line tool**. The command below generates stubs for
the package located at ``python/my_ffi_extension``, updating
all sections marked by the directives.

.. code-block:: bash

   tvm-ffi-stubgen                          \
     python/my_ffi_extension                \
     --dlls build/libmy_ffi_extension.so    \


**CMake Integration**. CMake function ``tvm_ffi_configure_target``
is integrated with this command and can be used to keep stubs up to date
every time the target is built.

.. code-block:: cmake

   tvm_ffi_configure_target(my_ffi_extension
       STUB_DIR "python"
   )

Inside the function, CMake manages to find proper ``--dlls`` arguments
via ``$<TARGET_FILE:${target}>``.

Scaffold Missing Directives
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Command line tool**. Beyond updating existing directives, ``tvm-ffi-stubgen``
can be used to scaffold missing directives if they are not yet present in the
package with a few extra flags.

.. code-block:: bash

   tvm-ffi-stubgen                          \
     python/my_ffi_extension                \
     --dlls build/libmy_ffi_extension.so    \
     --init-pypkg my-ffi-extension          \
     --init-lib my_ffi_extension            \
     --init-prefix "my_ffi_extension."      \

- ``--init-pypkg <pypkg>``: Specifies the name of the Python package to initialize, e.g. ``apache-tvm-ffi``, ``my-ffi-extension``;
- ``--init-lib <libtarget>``: Specifies the name of the CMake target (shared library) to load for reflection metadata;
- ``--init-prefix <prefix>``: Specifies the registry prefix to include for stub generation, e.g. ``my_ffi_extension.``. If names of global functions or classes start with this prefix, they will be included in the generated stubs.

**CMake Integration**. CMake function ``tvm_ffi_configure_target``
also supports scaffolding missing directives via the extra options
``STUB_INIT``, ``STUB_PKG``, and ``STUB_PREFIX``.

.. code-block:: cmake

   tvm_ffi_configure_target(my_ffi_extension
       STUB_DIR "python"
       STUB_INIT ON
   )

The ``STUB_INIT`` option instructs CMake to scaffold missing directives
based on the target and package information already specified.

Other Directives
~~~~~~~~~~~~~~~~

All the supported directives are documented via:

.. code-block:: bash

   tvm-ffi-stubgen --help


It includes:

**Directive 3 (Import section)**. It populates all the imported names used by generated stubs. Example:

.. code-block:: python

   # tvm-ffi-stubgen(begin): import-section
   from __future__ import annotations
   from ..registry import init_ffi_api as _FFI_INIT_FUNC
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from collections.abc import Mapping, Sequence
       from tvm_ffi import Device, Object, Tensor, dtype
       from tvm_ffi.testing import TestIntPair
       from typing import Any, Callable
   # tvm-ffi-stubgen(end)

**Directive 4 (Export)**. It re-exports names defined in `_ffi_api.__all__` into the current file. Usually
used in ``__init__.py`` to aggregate all exported names. Example:

.. code-block:: python

   # tvm-ffi-stubgen(begin): export/_ffi_api
   from ._ffi_api import *  # noqa: F403
   from ._ffi_api import __all__ as _ffi_api__all__
   if "__all__" not in globals():
       __all__ = []
   __all__.extend(_ffi_api__all__)
   # tvm-ffi-stubgen(end)

**Directive 5 (__all__)**. It populates the ``__all__`` variable with all generated
classes and functions, as well as ``LIB`` if present. It's usually placed at the end of
``_ffi_api.py``. Example:

.. code-block:: python

   __all__ = [
       # tvm-ffi-stubgen(begin): __all__
       "LIB",
       "IntPair",
       "raise_error",
       # tvm-ffi-stubgen(end)
   ]

**Directive 6 (ty-map)**. It maps the type key of a class to Python types used in generation. Example:

.. code-block:: python

   # tvm-ffi-stubgen(ty-map): ffi.reflection.AccessStep -> ffi.access_path.AccessStep

means the class with type key ``ffi.reflection.AccessStep``, is instead class ``ffi.access_path.AccessStep``
in Python.

**Directive 7 (Import object)**. It injects a custom import into generated code, optionally
TYPE_CHECKING-only. Example:


.. code-block:: python

   # tvm-ffi-stubgen(import-object): ffi.Object;False;_ffi_Object

imports ``ffi.Object`` as ``_ffi_Object`` for use in generated code,
where the second field ``False`` indicates the import is not TYPE_CHECKING-only.

**Directive 8 (Skip file)**. It prevents the stub generation tool from modifying the file.
This is useful when the file contains custom code that should not be altered.
