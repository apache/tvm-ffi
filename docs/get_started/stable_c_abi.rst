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

Stable C ABI
============

.. note::

  All code used in this guide lives under
  `examples/stable_c_abi <https://github.com/apache/tvm-ffi/tree/main/examples/stable_c_abi>`_.

.. admonition:: Prerequisite
   :class: hint

   - Python: 3.9 or newer (for the ``tvm_ffi.config``/``tvm-ffi-config`` helpers)
   - Compiler: C11-capable toolchain (GCC/Clang/MSVC)
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi

This guide introduces TVM-FFI's stable C ABI: a single, minimal and stable
ABI that represents any cross-language calls, with DSL and ML compiler codegen
in mind.

TVM-FFI builds on the following key idea:

.. _tvm_ffi_c_abi:

.. admonition:: Key Idea: A Single C ABI for all Functions
  :class: important

  Every function call can be represented by a single stable C ABI:

  .. code-block:: c

      int tvm_ffi_c_abi(          // returns 0 on success; non-zero on failure
        void*            handle,  // library handle
        const TVMFFIAny* args,    // inputs: args[0 ... N - 1]
        int              N,       // number of inputs
        TVMFFIAny*       result,  // output: *result
      );

  where :cpp:class:`TVMFFIAny`, is a tagged union of all supported types, e.g. integers, floats, Tensors, strings, etc., and can be further extended to arbitrary user-defined types.

Built on top of this stable C ABI, TVM-FFI defines a common C ABI protocol for all functions, and further provides an extensible, performant, and ecosystem-friendly open solution for all.

The rest of this guide covers:

- The stable C layout and calling convention of ``tvm_ffi_c_abi``;
- C examples from both callee and caller side of this ABI.

Stable C Layout
---------------

TVM-FFI's :ref:`C ABI <tvm_ffi_c_abi>` uses a stable layout for all the input and output arguments.

Layout of :cpp:class:`TVMFFIAny`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cpp:class:`TVMFFIAny` is a fixed-size (128-bit) tagged union that represents all supported types.

- First 32 bits: type index indicating which value is stored (supports up to 2^32 types).
- Next 32 bits: reserved (used for flags in rare cases, e.g. small-string optimization).
- Last 64 bits: payload that is either a 64-bit integer, a 64-bit floating-point number, or a pointer to a heap-allocated object.

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/tvm-ffi/stable-c-abi-layout-any.svg
   :alt: Layout of the 128-bit Any tagged union
   :name: fig:layout-any

   Figure 1. Layout spec for the :cpp:class:`TVMFFIAny` tagged union.

The following conventions apply when representing values in :cpp:class:`TVMFFIAny`:

- Primitive types: the last 64 bits directly store the value, for example:

  * Integers
  * Floating-point numbers

- Heap-allocated objects: the last 64 bits store a pointer to the actual object, for example:

  * Managed tensor objects that follow `DLPack <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html#dlpack-an-in-memory-tensor-structure>`_ (i.e. `DLTensor <https://dmlc.github.io/dlpack/latest/c_api.html#c.DLTensor>`_) layout.

- Arbitrary objects: the type index identifies the concrete type, and the last 64 bits store a pointer to a reference-counted object in TVM-FFI's object format, for example:

  * :py:class:`tvm_ffi.Function`, representing all functions, such as Python/C++ functions/lambdas, etc.;
  * :py:class:`tvm_ffi.Array` and :py:class:`tvm_ffi.Map` (list/dict containers of :cpp:class:`TVMFFIAny` values);
  * Extending to up to 2^32 types is supported.

Function Calling Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function calls in TVM-FFI share the same calling convention, :ref:`tvm_ffi_c_abi <tvm_ffi_c_abi>`, as described above.

- ``handle: void*``: optional library/closure handle passed to the callee. For exported symbols this is typically ``NULL``; closures may use it to capture context.
- ``args: TVMFFIAny*``: pointer to a contiguous array of input arguments.
- ``num_args: int``: number of input arguments.
- ``result: TVMFFIAny*``: out-parameter that receives the function result (use ``kTVMFFINone`` for "no return value").

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/tvm-ffi/stable-c-abi-layout-func.svg
   :alt: Layout and calling convention for tvm_ffi_c_abi
   :name: fig:layout-func

   Figure 2. Layout and calling convention of :ref:`tvm_ffi_c_abi <tvm_ffi_c_abi>`, where ``Any`` in this figure refers to :cpp:class:`TVMFFIAny`.


Stability and Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Stability.** The pure C layout and the calling convention are stable across compiler versions and independent of host languages or frameworks.

**Cross-language.** TVM-FFI implements this calling convention in multiple languages (C, C++, Python, Rust, ...), enabling code written in one language—or generated by a DSL targeting the ABI—to be called from another language.

**Cross-framework.** TVM-FFI uses standard data structures such as `DLPack tensors <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html#dlpack-an-in-memory-tensor-structure>`_ to represent arrays, so compiled functions can be used from any array framework that implements the DLPack protocol (NumPy, PyTorch, TensorFlow, CuPy, JAX, and others).


Stable ABI in C Code
--------------------

.. hint::

  You can build and run the examples either with raw compiler commands or with CMake.
  Both approaches are demonstrated below.

TVM FFI's :ref:`C ABI <tvm_ffi_c_abi>` is designed with DSL and ML compilers in mind. DSL codegen usually relies on MLIR, LLVM or low-level C as the compilation target, where no access to C++ features is available, and where stable C ABIs are preferred for simplicity and stability.

This section shows how to write C code that follows the stable C ABI. Specifically, we provide two examples:

- Callee side: A CPU ``add_one_cpu`` kernel in C that is equivalent to the :ref:`C++ example <cpp_add_one_kernel>`.
- Caller side: A loader and runner in C that invokes the kernel, a direct C translation of the :ref:`C++ example <cpp_load>`.

The C code is minimal and dependency-free, so it can serve as a direct reference for DSL compilers that want to expose or invoke kernels through the ABI.

Callee: ``add_one_cpu`` Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a minimal ``add_one_cpu`` kernel in C that follows the stable C ABI. It has three steps:

- **Step 1**. Extract input ``x`` and output ``y`` as DLPack tensors;
- **Step 2**. Implement the kernel ``y = x + 1`` on CPU with a simple for-loop;
- **Step 3**. Set the output result to ``result``.

.. literalinclude:: ../../examples/stable_c_abi/src/add_one_cpu.c
   :language: c
   :start-after: [example.begin]
   :end-before: [example.end]

Build it with either approach:

.. tabs::

  .. group-tab:: Raw command

    .. literalinclude:: ../../examples/stable_c_abi/raw_compile.sh
      :language: bash
      :start-after: [kernel.begin]
      :end-before: [kernel.end]

  .. group-tab:: CMake

    .. code-block:: bash

       cmake . -B build -DEXAMPLE_NAME="kernel" -DCMAKE_BUILD_TYPE=RelWithDebInfo
       cmake --build build --config RelWithDebInfo


**Compiler codegen.** This C code serves as a direct reference for DSL compilers. To emit a function that follows the stable C ABI, ensure the following:

- Symbol naming: define the exported symbol name as ``__tvm_ffi_{func_name}``;
- Type checking: check input types via :cpp:member:`TVMFFIAny::type_index`, then marshal inputs from :cpp:class:`TVMFFIAny` to the desired types;
- Error handling: return 0 on success, or a non-zero code on failure. When an error occurs, set an error message via :cpp:func:`TVMFFIErrorSetRaisedFromCStr` or :cpp:func:`TVMFFIErrorSetRaisedFromCStrParts`.

**C vs. C++.** Compared to the :ref:`C++ example <cpp_add_one_kernel>`, there are a few key differences:

- The explicit marshalling in **Step 1** is only needed in C. In C++, templates hide these details.
- The C++ macro :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` (used to export ``add_one_cpu``) is not needed in C, because this example directly defines the exported C symbol ``__tvm_ffi_add_one_cpu``.

.. hint::

  In TVM-FFI's C++ APIs, many invocables (functions, lambdas, functors) are automatically converted into the universal C ABI form by :cpp:class:`tvm::ffi::Function` and :cpp:class:`tvm::ffi::TypedFunction`.

  Rule of thumb: if an invocable's arguments and result can be converted to/from :cpp:class:`tvm::ffi::Any` (the C++ equivalent of :cpp:class:`TVMFFIAny`), it can be wrapped as a universal C ABI function.


Caller: Kernel Loader
~~~~~~~~~~~~~~~~~~~~~

Next, a minimal C loader invokes the ``add_one_cpu`` kernel. It is functionally identical to the :ref:`C++ example <cpp_load>` and performs:

- **Step 1**. Load the shared library ``build/add_one_cpu.so`` that contains the kernel;
- **Step 2**. Get function ``add_one_cpu`` from the library;
- **Step 3**. Invoke the function with two `DLTensor <https://dmlc.github.io/dlpack/latest/c_api.html#c.DLTensor>`_ inputs ``x`` and ``y``;

.. literalinclude:: ../../examples/stable_c_abi/src/load.c
   :language: c
   :start-after: [main.begin]
   :end-before: [main.end]


.. dropdown:: Auxiliary Logics

  .. literalinclude:: ../../examples/stable_c_abi/src/load.c
    :language: c
    :start-after: [aux.begin]
    :end-before: [aux.end]

Build and run the loader with either approach:

.. tabs::

  .. group-tab:: Raw command

    .. literalinclude:: ../../examples/stable_c_abi/raw_compile.sh
      :language: bash
      :start-after: [load.begin]
      :end-before: [load.end]

  .. group-tab:: CMake

    .. code-block:: bash

       cmake . -B build -DEXAMPLE_NAME="load" -DCMAKE_BUILD_TYPE=RelWithDebInfo
       cmake --build build --config RelWithDebInfo
       build/load

To call a function via the stable C ABI in C, idiomatically:

- Convert input arguments to the :cpp:class:`TVMFFIAny` type;
- Call the target function (e.g., ``add_one_cpu``) via :cpp:func:`TVMFFIFunctionCall`;
- Optionally convert the output :cpp:class:`TVMFFIAny` back to the desired type, if the function returns a value.

What's Next
-----------

**ABI specification.** See the complete ABI specification in :doc:`../concepts/abi_overview`.

**Convenient compiler target.** The stable C ABI is a simple, portable codegen target for DSL compilers. Emit C that follows this ABI to integrate with TVM-FFI and call the result from multiple languages and frameworks. See :doc:`../guides/compiler_integration`.

**Rich and extensible type system.** TVM-FFI supports a rich set of types in the stable C ABI: primitive types (integers, floats), DLPack tensors, strings, built-in reference-counted objects (functions, arrays, maps), and user-defined reference-counted objects. See :doc:`../guides/cpp_guide`.
