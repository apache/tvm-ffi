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

Quick Start
===========

.. note::

  All the code in this tutorial can be found under `examples/quickstart <https://github.com/apache/tvm-ffi/tree/main/examples/quickstart>`_ in the repository.

This guide walks through shipping a minimal ``add_one`` function that computes
``y = x + 1`` in C++ and CUDA.
TVM-FFI's Open ABI and FFI make it possible to **ship one library** for multiple frameworks and languages.
We can build a single shared library that works across:

- **ML frameworks**, e.g. PyTorch, JAX, NumPy, CuPy, etc., and
- **Languages**, e.g. C++, Python, Rust, etc.,
- **Python ABI versions**, e.g. ship one wheel to support all Python versions, including free-threaded ones.

.. admonition:: Prerequisite
   :class: hint
   :name: prerequisite

   - Python: 3.9 or newer
   - Compiler: C++17-capable toolchain (GCC/Clang/MSVC)
   - Optional ML frameworks for testing: NumPy, PyTorch, JAX, CuPy
   - CUDA: Any modern version (if you want to try the CUDA part)
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi


Write a Simple ``add_one``
--------------------------

Source Code
~~~~~~~~~~~

Suppose we implement a C++ function ``AddOne`` that performs elementwise ``y = x + 1`` for a 1-D ``float32`` vector. The source code (C++, CUDA) is:

.. tabs::

  .. group-tab:: C++

    .. _cpp_add_one_kernel:

    .. literalinclude:: ../../examples/quickstart/compile/add_one_cpu.cc
      :language: cpp
      :emphasize-lines: 8, 17
      :start-after: [example.begin]
      :end-before: [example.end]

  .. group-tab:: CUDA

    .. literalinclude:: ../../examples/quickstart/compile/add_one_cuda.cu
      :language: cpp
      :emphasize-lines: 15, 22, 26
      :start-after: [example.begin]
      :end-before: [example.end]


The macro :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` exports the C++ function ``AddOne``
as a TVM FFI compatible symbol with the name ``__tvm_ffi_add_one_cpu/cuda`` in the resulting library.

The class :cpp:class:`tvm::ffi::TensorView` allows zero-copy interop with tensors from different ML frameworks:

- NumPy, CuPy,
- PyTorch, JAX, or
- any array type that supports the standard `DLPack protocol <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html>`_.

Finally, :cpp:func:`TVMFFIEnvGetStream` can be used in the CUDA code to launch a kernel on the caller's stream.

.. _sec-cpp-compile-with-tvm-ffi:

Compile with TVM-FFI
~~~~~~~~~~~~~~~~~~~~

**Raw command.** We can use the following minimal commands to compile the source code:

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

This step produces a shared library ``add_one_cpu.so`` and ``add_one_cuda.so`` that can be used across languages and frameworks.

.. hint::

   For a single-file C++/CUDA project, a convenient method :py:func:`tvm_ffi.cpp.load_inline`
   is provided to minimize boilerplate code in compilation, linking, and loading.


**CMake.** CMake is the preferred approach for building across platforms.
TVM-FFI natively integrates with CMake via ``find_package`` as demonstrated below:

.. tabs::

  .. group-tab:: C++

    .. code-block:: cmake

      # Run `tvm-ffi-config --cmakedir` to set `tvm_ffi_DIR`
      find_package(Python COMPONENTS Interpreter REQUIRED)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
      find_package(tvm_ffi CONFIG REQUIRED)

      # Link C++ target to `tvm_ffi_header` and `tvm_ffi_shared`
      add_library(add_one_cpu SHARED compile/add_one_cpu.cc)
      target_link_libraries(add_one_cpu PRIVATE tvm_ffi_header)
      target_link_libraries(add_one_cpu PRIVATE tvm_ffi_shared)

  .. group-tab:: CUDA

    .. code-block:: cmake

      enable_language(CUDA)
      # Run `tvm-ffi-config --cmakedir` to set `tvm_ffi_DIR`
      find_package(Python COMPONENTS Interpreter REQUIRED)
      execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
      find_package(tvm_ffi CONFIG REQUIRED)

      # Link CUDA target to `tvm_ffi_header` and `tvm_ffi_shared`
      add_library(add_one_cuda SHARED compile/add_one_cuda.cu)
      target_link_libraries(add_one_cuda PRIVATE tvm_ffi_header)
      target_link_libraries(add_one_cuda PRIVATE tvm_ffi_shared)

**Artifact.** The resulting ``add_one_cpu.so`` and ``add_one_cuda.so`` are minimal libraries that are agnostic to:

- Python version/ABI. It is not compiled/linked with Python and depends only on TVM-FFI's stable C ABI;
- Languages, including C++, Python, Rust or any other language that can interop with C ABI;
- ML frameworks, such as PyTorch, JAX, NumPy, CuPy, or anything with standard `DLPack protocol <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html>`_.

.. _sec-use-across-framework:

Ship Across ML Frameworks
-------------------------

TVM-FFI's Python package provides :py:func:`tvm_ffi.load_module`, which can load either
the ``add_one_cpu.so`` or ``add_one_cuda.so`` into :py:class:`tvm_ffi.Module`.

.. code-block:: python

   import tvm_ffi
   mod  : tvm_ffi.Module   = tvm_ffi.load_module("add_one_cpu.so")
   func : tvm_ffi.Function = mod.add_one_cpu

``mod.add_one_cpu`` retrieves a callable :py:class:`tvm_ffi.Function` that accepts tensors from host frameworks
directly. This process is done zero-copy, without any boilerplate code, under extremely low latency.

We can then use these functions in the following ways:

.. tab-set::

    .. tab-item:: PyTorch

        .. literalinclude:: ../../examples/quickstart/load/load_pytorch.py
          :language: python
          :start-after: [example.begin]
          :end-before: [example.end]

    .. tab-item:: JAX

        Support via `nvidia/jax-tvm-ffi <https://github.com/nvidia/jax-tvm-ffi>`_. This can be installed via

        .. code-block:: bash

          pip install jax-tvm-ffi

        After installation, ``add_one_cuda`` can be registered as a target to JAX's ``ffi_call``.

        .. code-block:: python

          # Step 1. Load `build/add_one_cuda.so`
          import tvm_ffi
          mod = tvm_ffi.load_module("build/add_one_cuda.so")

          # Step 2. Register `mod.add_one_cuda` into JAX
          import jax_tvm_ffi
          jax_tvm_ffi.register_ffi_target("add_one", mod.add_one_cuda, platform="gpu")

          # Step 3. Run `mod.add_one_cuda` with JAX
          import jax
          import jax.numpy as jnp
          jax_device, *_ = jax.devices("gpu")
          x = jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32, device=jax_device)
          y = jax.ffi.ffi_call(
              "add_one",  # name of the registered function
              jax.ShapeDtypeStruct(x.shape, x.dtype),  # shape and dtype of the output
              vmap_method="broadcast_all",
          )(x)
          print(y)

    .. tab-item:: NumPy

        .. literalinclude:: ../../examples/quickstart/load/load_numpy.py
          :language: python
          :start-after: [example.begin]
          :end-before: [example.end]

    .. tab-item:: CuPy

        .. literalinclude:: ../../examples/quickstart/load/load_cupy.py
          :language: python
          :start-after: [example.begin]
          :end-before: [example.end]


Ship Across Languages
---------------------

TVM-FFI's core loading mechanism is ABI stable and works across language boundaries.
A single library can be loaded in every language TVM-FFI supports,
without having to recompile different libraries targeting different ABIs or languages.

Python
~~~~~~

As shown in the :ref:`previous section<sec-use-across-framework>`, :py:func:`tvm_ffi.load_module` loads a language-
and framework-independent ``add_one_cpu.so`` or ``add_one_cuda.so`` and can be used to incorporate it into all Python
array frameworks that implement the standard `DLPack protocol <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html>`_.

.. _cpp_load:

C++
~~~

TVM-FFI's C++ API :cpp:func:`tvm::ffi::Module::LoadFromFile` loads ``add_one_cpu.so`` or ``add_one_cuda.so`` and
can be used directly in C/C++ with no Python dependency.

.. literalinclude:: ../../examples/quickstart/load/load_cpp.cc
   :language: cpp
   :start-after: [main.begin]
   :end-before: [main.end]

.. dropdown:: Auxiliary Logics

  .. literalinclude:: ../../examples/quickstart/load/load_cpp.cc
    :language: cpp
    :start-after: [aux.begin]
    :end-before: [aux.end]

Compile and run it with:

.. literalinclude:: ../../examples/quickstart/raw_compile.sh
   :language: bash
   :start-after: [load_cpp.begin]
   :end-before: [load_cpp.end]

.. note::

  Don't like loading shared libraries? Static linking is also supported.

  In such cases, we can use :cpp:func:`tvm::ffi::Function::FromExternC` to create a
  :cpp:class:`tvm::ffi::Function` from the exported symbol, or directly use
  :cpp:func:`tvm::ffi::Function::InvokeExternC` to invoke the function.

  This feature can be useful on iOS, or when the exported module is generated by another DSL compiler matching the ABI.

  .. code-block:: cpp

      // Linked with `add_one_cpu.o` or `add_one_cuda.o`
      #include <tvm/ffi/function.h>
      #include <tvm/ffi/container/tensor.h>

      // declare reference to the exported symbol
      extern "C" int __tvm_ffi_add_one_cpu(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

      namespace ffi = tvm::ffi;

      int bundle_add_one(ffi::TensorView x, ffi::TensorView y) {
        void* closure_handle = nullptr;
        ffi::Function::InvokeExternC(closure_handle, __tvm_ffi_add_one_cpu, x, y);
        return 0;
      }

Rust
~~~~

TVM-FFI's Rust API ``tvm_ffi::Module::load_from_file`` loads ``add_one_cpu.so`` or ``add_one_cuda.so`` and
then retrieves a function ``add_one_cpu`` or ``add_one_cuda`` from it.
This procedure is identical to those in C++ and Python:

.. code-block:: rust

    fn run_add_one(x: &Tensor, y: &Tensor) -> Result<()> {
        let module = tvm_ffi::Module::load_from_file("add_one_cpu.so")?;
        let func = module.get_function("add_one_cpu")?;
        let typed_fn = into_typed_fn!(func, Fn(&Tensor, &Tensor) -> Result<()>);
        typed_fn(x, y)?;
        Ok(())
    }


.. hint::

    We can also use the Rust API to target the TVM FFI ABI. This means we can use Rust to write the function
    implementation and export to Python/C++ in the same fashion.


Troubleshooting
---------------

- ``OSError: cannot open shared object file``: Add an rpath (Linux/macOS) or ensure the DLL is on ``PATH`` (Windows). Example run-path: ``-Wl,-rpath,`tvm-ffi-config --libdir```.
- ``undefined symbol: __tvm_ffi_add_one_cpu``: Ensure you used :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` and compiled with default symbol visibility (``-fvisibility=hidden`` is fine; the macro ensures export).
- ``CUDA error: invalid device function``: Rebuild with the correct ``-arch=sm_XX`` for your GPU, or include multiple ``-gencode`` entries.
