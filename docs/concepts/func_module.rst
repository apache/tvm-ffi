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

Function and Module
===================

TVM-FFI provides a unified and ABI-stable calling convention that enables
cross-language function calls between C++, Python, Rust, and other languages.
Functions are first-class :doc:`TVM-FFI objects <object_and_class>`.

This tutorial covers defining, registering, and calling TVM-FFI functions,
exception handling, and working with modules.

Glossary
--------

TVM-FFI ABI, or "Packed Function". :cpp:type:`TVMFFISafeCallType`
  A stable C calling convention where every function is represented by a single signature,
  which enables type-erased, cross-language function calls.
  This calling convention is used across all TVM-FFI function calls at the ABI boundary.
  See :ref:`Stable C ABI <tvm_ffi_c_abi>` for a quick introduction.

TVM-FFI Function. :py:class:`tvm_ffi.Function`, :cpp:class:`tvm::ffi::FunctionObj`, :cpp:class:`tvm::ffi::Function`
  A reference-counted :doc:`function object <object_and_class>` and its managed reference, which wraps any callable,
  including language-agnostic functions and lambdas (C++, Python, Rust, etc.),
  member functions, external C symbols, and other callable objects,
  all sharing the same calling convention.

TVM-FFI Module. :py:class:`tvm_ffi.Module`, :cpp:class:`tvm::ffi::ModuleObj`, :cpp:class:`tvm::ffi::Module`
  A namespace for a collection of functions, loaded from a shared library via ``dlopen`` (Linux, macOS) or ``LoadLibraryW`` (Windows),
  or statically linked to the current executable.

Global Functions and Registry. :py:func:`tvm_ffi.get_global_func` and :py:func:`tvm_ffi.register_global_func`
  A registry is a table that maps string names to :cpp:class:`~tvm::ffi::Function` objects
  and their metadata (name, docs, signatures, etc.) for cross-language access.
  Functions in the registry are called **global functions**.

Common Usage
------------

TVM-FFI C Symbols
~~~~~~~~~~~~~~~~~

**Shared library**. Use :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` to export
a function as a C symbol that follows the TVM-FFI ABI:

.. code-block:: cpp

   static int AddTwo(int x) { return x + 2; }

   TVM_FFI_DLL_EXPORT_TYPED_FUNC(/*ExportName=*/add_two, /*Function=*/AddTwo)

This creates a C symbol ``__tvm_ffi_<ExportName>`` in the shared library,
which can then be loaded and called via :py:func:`tvm_ffi.load_module`:

.. code-block:: python

   import tvm_ffi

   mod = tvm_ffi.load_module("path/to/library.so")
   result = mod.add_two(40)  # -> 42

**System library**. For symbols bundled in the same executable, use :cpp:func:`TVMFFIEnvModRegisterSystemLibSymbol`
to register each symbol during static initialization within a :c:macro:`TVM_FFI_STATIC_INIT_BLOCK`.
See :py:func:`tvm_ffi.system_lib` for a complete workflow.


Global Functions
~~~~~~~~~~~~~~~~

**Register a global function**. In C++, use :cpp:class:`tvm::ffi::reflection::GlobalDef` to
register a function:

.. code-block:: cpp

   #include <tvm/ffi/tvm_ffi.h>

   static int AddOne(int x) { return x + 1; }

   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = tvm::ffi::reflection;
     refl::GlobalDef()
         .def("my_ext.add_one", AddOne, "Add one to the input");
   }

The :c:macro:`TVM_FFI_STATIC_INIT_BLOCK` macro ensures that registration occurs
during library initialization. The registered function is then accessible from
Python by the name ``my_ext.add_one``.

In Python, use the decorator :py:func:`tvm_ffi.register_global_func` to register a global function:

.. code-block:: python

   import tvm_ffi

   @tvm_ffi.register_global_func("my_ext.add_one")
   def add_one(x: int) -> int:
       return x + 1

**Retrieve a global function**. After registration, functions are accessible by name.

In Python, use :py:func:`tvm_ffi.get_global_func` to retrieve a global function:

.. code-block:: python

   import tvm_ffi

   # Get a function from the global registry
   add_one = tvm_ffi.get_global_func("my_ext.add_one")
   result = add_one(41)  # -> 42

In C++, use :cpp:func:`tvm::ffi::Function::GetGlobal` or :cpp:func:`tvm::ffi::Function::GetGlobalRequired`
to retrieve a global function:

.. code-block:: cpp

   ffi::Function func = ffi::Function::GetGlobalRequired("my_ext.add_one");
   int result = func(41);  // -> 42


Create Functions
~~~~~~~~~~~~~~~~

**From C++**. An :cpp:class:`tvm::ffi::Function` can be created via :cpp:func:`tvm::ffi::Function::FromTyped`
or :cpp:class:`tvm::ffi::TypedFunction`'s constructor.

.. code-block:: cpp

   // Create type-erased function: add_type_erased
   ffi::Function add_type_erased = ffi::Function::FromTyped([](int x, int y) {
     return x + y;
   });

   // Create a typed function: add_typed
   ffi::TypedFunction<int(int, int)> add_typed = [](int x, int y) {
     return x + y;
   };

   // Convert a typed function to a type-erased function
   ffi::Function generic = add_typed;

**From Python**. Any Python :py:class:`Callable <collections.abc.Callable>` is automatically converted
to a :py:class:`tvm_ffi.Function` at the ABI boundary. The example below demonstrates that in ``my_ext.bind``:

- The input ``func`` is automatically converted to a :py:class:`tvm_ffi.Function`.
- The returned lambda is also automatically converted to a :py:class:`tvm_ffi.Function`.

.. code-block:: python

   import tvm_ffi

   @tvm_ffi.register_global_func("my_ext.bind")
   def bind(func, x):
     assert isinstance(func, tvm_ffi.Function)
     return lambda *args: func(x, *args)  # converted to `tvm_ffi.Function`

   def add_x_y(x, y):
     return x + y

   func_bind = tvm_ffi.get_global_func("my_ext.bind")
   add_y = func_bind(add_x_y, 1)  # bind x = 1
   assert isinstance(add_y, tvm_ffi.Function)
   print(add_y(2))  # -> 3


:py:func:`tvm_ffi.convert` explicitly converts a Python callable to :py:class:`tvm_ffi.Function`:

.. code-block:: python

   import tvm_ffi

   def add(x, y):
     return x + y

   func_add = tvm_ffi.convert(add)
   print(func_add(1, 2))


.. _sec:function:

Function
--------

.. _sec:function-calling-convention:

Calling Convention
~~~~~~~~~~~~~~~~~~

All TVM-FFI functions ultimately conform to the :cpp:type:`TVMFFISafeCallType` signature,
which provides a stable C ABI for cross-language calls. The C calling convention is defined as:

.. code-block:: cpp

   int tvm_ffi_c_abi(
     void* handle,           // Resource handle
     const TVMFFIAny* args,  // Input arguments (non-owning)
     int32_t num_args,       // Number of input arguments
     TVMFFIAny* result       // Output argument (owning, zero-initialized)
   );

**Input arguments**. The input arguments are passed as an array of :cpp:class:`tvm::ffi::AnyView` values
(see :ref:`any-ownership` for ownership semantics), specified by ``args`` and ``num_args``.

**Output argument**. The output argument ``result`` is an owning :cpp:type:`tvm::ffi::Any`
that the caller must zero-initialize before the call.

.. important::
   The caller must zero-initialize the output argument ``result`` before the call.

**Return value**. The ABI returns an **error code** that indicates:

- **Error code 0**: Success
- **Error code -1**: Error occurred, retrievable with :cpp:func:`TVMFFIErrorMoveFromRaised`

.. hint::
  See :doc:`Any <any>` for more details on the semantics of :cpp:type:`tvm::ffi::AnyView` and :cpp:type:`tvm::ffi::Any`.

This design is called a **packed function**, because it "packs" all arguments into a single array of type-erased :cpp:type:`tvm::ffi::AnyView`,
and further unifies calling convention across all languages without resorting to JIT compilation.

More specifically, this mechanism enables the following scenarios:

- **Dynamic languages**. Well-optimized bindings are provided for, e.g. Python, to translate arguments into packed function format, and translate return value back to the host language.
- **Static languages**. Metaprogramming techniques, such as C++ templates, are usually available to directly instantiate packed format on stack, saving the need for dynamic examination.
- **Cross-language callbacks**. Language-agnostic :cpp:class:`tvm::ffi::Function` makes it easy to call between languages without depending on language-specific features such as GIL.

**Performance Implications**. This approach is in practice highly efficient in machine learning workloads.

- In Python/C++ calls, we can get to microsecond level overhead, which is generally similar to overhead for eager mode;
- When both sides of calls are static languages, the overhead will go down to tens of nanoseconds.

.. note::
  Although we found it less necessary in practice, further link time optimization (LTO) is still theoretically possible
  in scenarios where both sides are static languages with a known symbol and linked into a single binary.
  In this case, the callee can be inlined into caller side and the stack argument memory can be passed into register passing.

.. _sec:function-layout:

Layout and ABI
~~~~~~~~~~~~~~

:cpp:class:`tvm::ffi::FunctionObj` stores two call pointers in :cpp:class:`TVMFFIFunctionCell`:

- ``safe_call``: Used for cross-ABI function calls; intercepts exceptions and stores them in TLS.
- ``cpp_call``: Used within the same DSO; exceptions are thrown directly for better performance.

See :ref:`abi-function` for the C struct definition.

.. important::

  :cpp:func:`TVMFFIFunctionCall` is the idiomatic way to call a :cpp:class:`tvm::ffi::FunctionObj` in C,
  while ``safe_call`` or ``cpp_call`` remain low-level ABIs for fast access.

**Conversion with Any**. Since :py:class:`tvm_ffi.Function` is a TVM-FFI object, it follows the same
conversion rules as any other TVM-FFI object. See :ref:`Object Conversion with Any <object-conversion-with-any>` for details.


Exception Handling
~~~~~~~~~~~~~~~~~~

See :doc:`exception_handling` for details on throwing, catching, and propagating exceptions
across language boundaries.

Compiler developers commonly need to look up global functions in generated code.
Use :cpp:func:`TVMFFIFunctionGetGlobal` to retrieve a function by name, then call it with :cpp:func:`TVMFFIFunctionCall`.
See :ref:`abi-function` for C code examples.


.. _sec:module:

Modules
-------

A :py:class:`tvm_ffi.Module` is a namespace for a collection of functions that can be loaded
from a shared library or bundled with the current executable. Modules provide namespace isolation
and dynamic loading capabilities for TVM-FFI functions.

Shared Library
~~~~~~~~~~~~~~

Shared library modules are loaded dynamically at runtime via ``dlopen`` (Linux, macOS) or
``LoadLibraryW`` (Windows). This is the most common way to distribute and load compiled functions.

**Export functions from C++**. Use :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` to export
a function as a C symbol that follows the TVM-FFI ABI:

.. code-block:: cpp

   #include <tvm/ffi/tvm_ffi.h>

   static int AddTwo(int x) { return x + 2; }

   // Exports as symbol `__tvm_ffi_add_two`
   TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_two, AddTwo);

**Load and call from Python**. Use :py:func:`tvm_ffi.load_module` to load the shared library:

.. code-block:: python

   import tvm_ffi

   # Load the shared library
   mod = tvm_ffi.load_module("path/to/library.so")

   # Access functions by name
   result = mod.add_two(40)  # -> 42

   # Alternative: explicit function retrieval
   func = mod.get_function("add_two")
   result = func(40)  # -> 42

**Build and load from source**. For rapid prototyping, :py:func:`tvm_ffi.cpp.load` compiles
C++/CUDA source files and loads them as a module in one step:

.. code-block:: python

   import tvm_ffi.cpp

   # Compile and load in one step
   mod = tvm_ffi.cpp.load(
       name="my_ops",
       cpp_files="my_ops.cpp",
   )
   result = mod.add_two(40)

Essentially, :py:func:`tvm_ffi.cpp.load` is a convenience function that JIT-compiles the source
files and loads the resulting library as a :py:class:`tvm_ffi.Module`.

System Library
~~~~~~~~~~~~~~

System library modules contain symbols that are statically linked to the current executable.

This technique is useful when you want to simulate dynamic module loading behavior but cannot
or prefer not to use ``dlopen`` or ``LoadLibraryW`` (e.g., on iOS). Functions are statically
linked to the executable as a system library module. Symbols can be registered via
:cpp:func:`TVMFFIEnvModRegisterSystemLibSymbol` and looked up via :py:func:`tvm_ffi.system_lib`.

**Register symbols in C/C++**. Use :cpp:func:`TVMFFIEnvModRegisterSystemLibSymbol` to register
a symbol during static initialization:

.. code-block:: cpp

   #include <tvm/ffi/c_api.h>
   #include <tvm/ffi/extra/c_env_api.h>

   // A function following the TVM-FFI ABI
   static int add_one_impl(void*, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
     TVM_FFI_SAFE_CALL_BEGIN();
     int64_t x = reinterpret_cast<const tvm::ffi::AnyView*>(args)[0].cast<int64_t>();
     reinterpret_cast<tvm::ffi::Any*>(result)[0] = x + 1;
     TVM_FFI_SAFE_CALL_END();
   }

   // Register during static initialization
   // The symbol name follows the convention `__tvm_ffi_<prefix>.<name>`
   TVM_FFI_STATIC_INIT_BLOCK() {
     TVMFFIEnvModRegisterSystemLibSymbol(
         "__tvm_ffi_my_prefix.add_one",
         reinterpret_cast<void*>(add_one_impl)
     );
   }

**Access from Python**. Use :py:func:`tvm_ffi.system_lib` to get the system library module:

.. code-block:: python

   import tvm_ffi

   # Get system library with symbol prefix "my_prefix."
   # This looks up symbols prefixed with `__tvm_ffi_my_prefix.`
   mod = tvm_ffi.system_lib("my_prefix.")

   # Call the registered function
   func = mod.add_one  # looks up `__tvm_ffi_my_prefix.add_one`
   result = func(10)  # -> 11

.. note::

   The system library is intended for statically linked symbols that exist for the entire
   program lifetime. For dynamic loading with the ability to unload, use shared library modules instead.


.. _sec:custom-modules:

Custom Modules
~~~~~~~~~~~~~~

While the standard shared library and system library modules cover most use
cases, some scenarios require a **custom module** that wraps a platform-specific
driver API — for example, using ``cuModuleLoad`` to load generated PTX code and
expose each kernel as a :cpp:class:`tvm::ffi::Function`.

To create a custom module, subclass :cpp:class:`tvm::ffi::ModuleObj` and
implement the following:

- :cpp:func:`~tvm::ffi::ModuleObj::kind` — return a unique string identifying
  the module type (e.g., ``"cuda"``).
- :cpp:func:`~tvm::ffi::ModuleObj::GetPropertyMask` — return a bitmask
  indicating the module's capabilities:

  - ``ffi::Module::kRunnable`` if the module can execute functions.
  - ``ffi::Module::kBinarySerializable`` if the module supports
    serialization to and from bytes.

- :cpp:func:`~tvm::ffi::ModuleObj::GetFunction` — look up a function by name
  within the module.
- If the module is serializable, override
  :cpp:func:`~tvm::ffi::ModuleObj::SaveToBytes` and register a global function
  ``ffi.Module.load_from_bytes.<kind>`` so the module can be reconstructed from
  its serialized form.

.. seealso::

   :ref:`Embedded Binary Data <export-embedded-binary-data>` in the export guide describes
   the ``__tvm_ffi__library_bin`` binary layout used to serialize composite
   modules that contain custom sub-modules.


Further Reading
---------------

- :doc:`exception_handling`: Throwing, catching, and propagating exceptions across language boundaries
- :doc:`any`: How functions are stored in :cpp:class:`~tvm::ffi::Any` containers
- :doc:`object_and_class`: The object system that backs :cpp:class:`~tvm::ffi::FunctionObj`
- :doc:`abi_overview`: Low-level C ABI details for functions and exceptions
- :doc:`../packaging/python_packaging`: Packaging functions for Python wheels
