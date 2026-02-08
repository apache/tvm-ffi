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

Export Functions and Classes
============================

TVM-FFI provides three mechanisms to make functions and classes available across
C, C++, and Python. Each targets a different use case:

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Mechanism
     - When to use
     - How it works
   * - :ref:`C Symbols <export-c-symbols>`
     - Kernel libraries, compiler codegen
     - Export ``__tvm_ffi_<name>`` in a shared library; load via
       :py:func:`tvm_ffi.load_module`
   * - :ref:`Global Functions <export-global-functions>`
     - Application-level APIs, cross-language callbacks
     - Register by string name; retrieve from any language
   * - :ref:`Classes <export-classes>`
     - Structured data with fields and methods
     - Define a C++ :cpp:class:`~tvm::ffi::Object` subclass; use from Python
       as a dataclass

Include the umbrella header to access all C++ APIs used in this guide:

.. code-block:: cpp

   #include <tvm/ffi/tvm_ffi.h>

Metadata (type signatures, field names, docstrings) is captured automatically
and can be turned into Python type hints via stub generation.

.. seealso::

   - All C++ examples in this guide are under
     `examples/python_packaging/ <https://github.com/apache/tvm-ffi/tree/main/examples/python_packaging>`_;
     ANSI C examples are under
     `examples/stable_c_abi/ <https://github.com/apache/tvm-ffi/tree/main/examples/stable_c_abi>`_.
   - :doc:`../concepts/func_module`: Calling convention, module system, and
     global registry concepts.
   - :doc:`../get_started/stable_c_abi`: Low-level C ABI walkthrough.
   - :doc:`../packaging/python_packaging`: Packaging extensions as Python wheels.
   - :doc:`../packaging/stubgen`: Generating Python type stubs from C++ metadata.


.. _export-c-symbols:

C Symbols
---------

C symbols are the most direct export mechanism. A function is compiled into a
shared library with a ``__tvm_ffi_<name>`` symbol, then loaded dynamically at
runtime. This is the recommended approach for kernel libraries and compiler
codegen because it keeps the language boundary thin.

.. tip::

   For exporting and calling C symbols from pure ANSI C code, see
   :doc:`../get_started/stable_c_abi`.


Export and Look up in C++
~~~~~~~~~~~~~~~~~~~~~~~~~

**Export.** Use :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` to export a C++ function
as a C symbol that follows the
:ref:`TVM-FFI calling convention <sec:function-calling-convention>`:

.. literalinclude:: ../../examples/python_packaging/src/extension.cc
   :language: cpp
   :start-after: [tvm_ffi_abi.begin]
   :end-before: [tvm_ffi_abi.end]

This creates a symbol ``__tvm_ffi_add_two`` in the shared library. The macro
handles argument unmarshalling and error propagation automatically.

**Look up.** Use :cpp:func:`tvm::ffi::Module::LoadFromFile` to load a shared
library and retrieve functions by name:

.. code-block:: cpp

   namespace ffi = tvm::ffi;

   ffi::Module mod = ffi::Module::LoadFromFile("path/to/library.so");
   ffi::Function func = mod->GetFunction("add_two").value();
   int result = func(40).cast<int>();  // -> 42


Load from Python
~~~~~~~~~~~~~~~~

Use :py:func:`tvm_ffi.load_module` to load the shared library and call its
functions by name:

.. code-block:: python

   import tvm_ffi

   mod = tvm_ffi.load_module("path/to/library.so")
   result = mod.add_two(40)  # -> 42

.. seealso::

   For DSO loading at Python package level, see :ref:`sec-load-the-library` in
   the Python packaging guide.

.. _export-embedded-binary-data:

Embedded Binary Data
~~~~~~~~~~~~~~~~~~~~

Shared libraries can embed a binary symbol ``__tvm_ffi__library_bin`` alongside
function symbols to support **composite modules** — modules that import custom
sub-modules (e.g., a PTX module loaded via ``cuModuleLoad``). The binary layout
is:

.. code-block:: text

   <nbytes: u64> <import_tree> <key0: str> [val0: bytes] <key1: str> [val1: bytes] ...

- ``nbytes``: total byte count following this header.
- ``import_tree``: a CSR sparse array
  (``<indptr: vec<u64>> <child_indices: vec<u64>>``) encoding the parent–child
  relationships among module nodes.
- Each ``key`` is a module kind string, or the special value ``_lib`` for the
  host dynamic library itself. For non-\ ``_lib`` entries, ``val`` contains the
  serialized bytes of the custom sub-module.
- Both ``str`` and ``bytes`` values are length-prefixed: ``<size: u64> <content>``.

When :py:func:`tvm_ffi.load_module` opens a library containing this symbol, it
deserializes each sub-module by calling ``ffi.Module.load_from_bytes.<kind>``,
reconstructs the import tree, and returns the composed module. The custom module
class must be available at load time — either by importing its runtime library
beforehand or by embedding the class definition in the generated library.
See :ref:`Custom Modules <sec:custom-modules>` in :doc:`../concepts/func_module`
for details on custom module subclassing.


.. _export-global-functions:

Global Functions
----------------

Global functions are registered by string name in a shared registry, making them
accessible from any language without loading a specific module. This is useful
for application-level APIs and cross-language callbacks.


Register and Retrieve in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Register.** Use :cpp:class:`tvm::ffi::reflection::GlobalDef` inside a
:c:macro:`TVM_FFI_STATIC_INIT_BLOCK` to register a function during library
initialization:

.. literalinclude:: ../../examples/python_packaging/src/extension.cc
   :language: cpp
   :start-after: [global_function.begin]
   :end-before: [global_function.end]

The function becomes available by its string name
(``my_ffi_extension.add_one``) from any language once the library is loaded.

**Retrieve.** Use :cpp:func:`tvm::ffi::Function::GetGlobal` (returns
``Optional``) or :cpp:func:`tvm::ffi::Function::GetGlobalRequired` (throws if
missing):

.. code-block:: cpp

   namespace ffi = tvm::ffi;

   // Optional retrieval
   ffi::Optional<ffi::Function> maybe_func =
       ffi::Function::GetGlobal("my_ffi_extension.add_one");
   if (maybe_func.has_value()) {
     int result = maybe_func.value()(3).cast<int>();
   }

   // Required retrieval (throws if not found)
   ffi::Function func =
       ffi::Function::GetGlobalRequired("my_ffi_extension.add_one");
   int result = func(3).cast<int>();  // -> 4


Register and Retrieve in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Register.** Use the :py:func:`tvm_ffi.register_global_func` decorator:

.. code-block:: python

   import tvm_ffi

   @tvm_ffi.register_global_func("my_ffi_extension.add_one")
   def add_one(x: int) -> int:
       return x + 1

**Retrieve.** Use :py:func:`tvm_ffi.get_global_func` to look up a function by
name:

.. code-block:: python

   import tvm_ffi

   add_one = tvm_ffi.get_global_func("my_ffi_extension.add_one")
   result = add_one(3)  # -> 4

.. seealso::

   For packaged extensions, :ref:`stub generation <sec-stubgen>` produces
   type-annotated bindings so that users can call global functions directly
   (e.g. ``my_ffi_extension.add_one(3)``) with full IDE support.

.. note::

   Global functions can also be retrieved via :cpp:func:`TVMFFIFunctionGetGlobal`
   in C.


.. _export-classes:

Classes
-------

Any class derived from :cpp:class:`tvm::ffi::Object` can be registered, exported,
and instantiated from Python. The reflection helper
:cpp:class:`tvm::ffi::reflection::ObjectDef` makes it easy to expose:

- **Fields**: immutable via
  :cpp:func:`ObjectDef::def_ro <tvm::ffi::reflection::ObjectDef::def_ro>`,
  mutable via
  :cpp:func:`ObjectDef::def_rw <tvm::ffi::reflection::ObjectDef::def_rw>`
- **Methods**: instance via
  :cpp:func:`ObjectDef::def <tvm::ffi::reflection::ObjectDef::def>`,
  static via
  :cpp:func:`ObjectDef::def_static <tvm::ffi::reflection::ObjectDef::def_static>`
- **Constructors** via :cpp:class:`tvm::ffi::reflection::init`


Register in C++
~~~~~~~~~~~~~~~

Define a class that inherits from :cpp:class:`tvm::ffi::Object`, then register
it with :cpp:class:`tvm::ffi::reflection::ObjectDef` inside a
:c:macro:`TVM_FFI_STATIC_INIT_BLOCK`:

.. literalinclude:: ../../examples/python_packaging/src/extension.cc
   :language: cpp
   :start-after: [object.begin]
   :end-before: [object.end]

Key elements:

- :c:macro:`TVM_FFI_DECLARE_OBJECT_INFO_FINAL` declares the type with a unique
  string key (``my_ffi_extension.IntPair``), the class name, and parent class.
- ``static constexpr bool _type_mutable = true`` allows field modification from
  Python. Omit this (or set to ``false``) for immutable objects.


Use in Python
~~~~~~~~~~~~~

After importing the extension, the class is available with property access and
method calls:

.. code-block:: python

   import my_ffi_extension

   pair = my_ffi_extension.IntPair(1, 2)
   print(pair.a)      # -> 1
   print(pair.b)      # -> 2
   print(pair.sum())  # -> 3

.. seealso::

   For packaged extensions, :ref:`stub generation <sec-stubgen>` produces
   type-annotated Python classes with full IDE support.


Further Reading
---------------

- :doc:`../get_started/stable_c_abi`: End-to-end C ABI walkthrough with callee and caller examples
- :doc:`../concepts/func_module`: Calling convention, module system, and global registry concepts
- :doc:`../concepts/object_and_class`: Object system, type hierarchy, and reference counting
- :doc:`../packaging/python_packaging`: Packaging extensions as Python wheels with stub generation
- :doc:`../packaging/cpp_tooling`: Build toolchain, CMake integration, and library distribution
- :doc:`cpp_lang_guide`: Full C++ API guide covering Any, Function, containers, and more
