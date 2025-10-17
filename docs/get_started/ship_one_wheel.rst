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

Ship Across Boundaries
======================

This quickstart guide walks through shipping a minimal ``add_one`` function that computes
``y = x + 1`` for 1-D float32 tensors in two variants:

- A simplistic C++ implementation;
- A CUDA implementation that launches a kernel on the caller's stream.

Built with TVM-FFI's open ABI and FFI, a single shared library works across boundaries of:

- languages, e.g. C++, Python, Rust, etc., and
- ML frameworks, e.g. NumPy, PyTorch, JAX, CuPy, etc.

without having to write any boilerplate code or worry about ABIs.

.. admonition:: Prerequisite
   :class: hint
   :name: prerequisite

   - Python: 3.9 or newer
   - Compiler: C++17-capable toolchain (GCC/Clang/MSVC)
   - Optional ML frameworks for testing: NumPy, PyTorch, JAX, CuPy
   - CUDA: Any modern version if you want to try the CUDA part
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi


Simple ``add_one``
------------------

.. _sec-cpp-source-code:

Source Code
~~~~~~~~~~~

Suppose we implement a C++ function ``AddOne`` that performs elementwise ``y = x + 1`` for a 1-D ``float32`` vector. The source file ``main.cc`` is:

.. tabs::

  .. group-tab:: C++

    .. code-block:: cpp
      :emphasize-lines: 8, 14

      // File: main.cc
      #include <tvm/ffi/container/tensor.h>
      #include <tvm/ffi/function.h>

      namespace tvm_ffi_example_cpp {

      /*! \brief Perform vector add one: y = x + 1 (1-D float32) */
      void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
        for (int i = 0; i < x->shape[0]; ++i) {
          static_cast<float *>(y->data)[i] = static_cast<float *>(x->data)[i] + 1;
        }
      }

      TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, tvm_ffi_example_cpp::AddOne);
      }

  .. group-tab:: CUDA

    .. code-block:: cpp
      :emphasize-lines: 15, 20, 25

      // File: main.cu
      #include <tvm/ffi/container/tensor.h>
      #include <tvm/ffi/extra/c_env_api.h>
      #include <tvm/ffi/function.h>

      namespace tvm_ffi_example_cuda {

      __global__ void AddOneKernel(float* x, float* y, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
          y[idx] = x[idx] + 1;
        }
      }

      void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
        int64_t n = x->shape[0];
        int64_t threads = 256;
        int64_t blocks = (n + threads - 1) / threads;
        cudaStream_t stream = static_cast<cudaStream_t>(
          TVMFFIEnvGetStream(x->device.device_type, x->device.device_id));
        AddOneKernel<<<blocks, threads, 0, stream>>>(static_cast<float*>(x->data),
                                                     static_cast<float*>(y->data), n);
      }

      TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, tvm_ffi_example_cuda::AddOne);
      }



Macro :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` exports the C++ function ``AddOne`` with public name ``add_one`` in the resulting library.
TVM-FFI looks it up at runtime to make the function available across languages.

Class :cpp:class:`tvm::ffi::TensorView` allows zero-copy interop with tensors from different ML frameworks:

- NumPy/CuPy ``ndarray``,
- a PyTorch ``Tensor``,
- a JAX ``Array``, or
- any array type that supports the standard `DLPack protocol <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html>`_.

Finally, :cpp:func:`TVMFFIEnvGetStream` used in CUDA code makes it possible to launch a kernel on caller's stream.

.. _sec-cpp-compile-with-tvm-ffi:

Compile with TVM-FFI
~~~~~~~~~~~~~~~~~~~~

**Raw command.** Basic command to compile the source code can be as concise as below:

.. tabs::

  .. group-tab:: C++

    .. code-block:: bash

      g++ -shared -fPIC -fvisibility=hidden -O3  \
          main.cc                                \
          `tvm-ffi-config --cxxflags`            \
          `tvm-ffi-config --ldflags`             \
          `tvm-ffi-config --libs`                \
          -o libmain.so

  .. group-tab:: CUDA

    .. code-block:: bash

      nvcc -shared -fPIC -fvisibility=hidden -O3 \
          main.cu                                \
          `tvm-ffi-config --cxxflags`            \
          `tvm-ffi-config --ldflags`             \
          `tvm-ffi-config --libs`                \
          -o libmain.so

This produces a shared library ``libmain.so``. TVM-FFI automatically embeds the metadata needed to call the function across language and framework boundaries.

**CMake.** As the preferred approach to build across platforms, CMake relies on CMake package ``tvm_ffi``, which can be found via ``tvm-ffi-config --cmakedir``.

.. tabs::

  .. group-tab:: C++

    .. code-block:: cmake

      # Run `tvm-ffi-config --cmakedir` to find tvm-ffi targets
      find_package(Python COMPONENTS Interpreter REQUIRED)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -m tvm-ffi-config --cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE tvm_ffi_ROOT
      )
      find_package(tvm_ffi CONFIG REQUIRED)
      # Create C++ target `add_one_cpp`
      add_library(add_one_cpp SHARED main.cc)
      target_link_libraries(add_one_cpp PRIVATE tvm_ffi_header)
      target_link_libraries(add_one_cpp PRIVATE tvm_ffi_shared)

  .. group-tab:: CUDA

    .. code-block:: cmake

      # Run `tvm-ffi-config --cmakedir` to find tvm-ffi targets
      find_package(Python COMPONENTS Interpreter REQUIRED)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -m tvm-ffi-config --cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE tvm_ffi_ROOT
      )
      find_package(tvm_ffi CONFIG REQUIRED)
      # Create C++ target `add_one_cuda`
      enable_language(CUDA)
      add_library(add_one_cuda SHARED main.cu)
      target_link_libraries(add_one_cuda PRIVATE tvm_ffi_header)
      target_link_libraries(add_one_cuda PRIVATE tvm_ffi_shared)

.. hint::

   For a single-file C++/CUDA, a convenient method :py:func:`tvm_ffi.cpp.load_inline`
   is provided to minimize boilerplate code in compilation, linking and loading.

Note that ``libmain.so`` is neutral and agnostic to:

- Python version/ABI, because it is pure C++ and not compiled or linked against Python
- C++ ABI, because TVM-FFI interacts with the artifact only via stable C APIs
- Frontend languages, which can be C++, Rust, Python, TypeScript, etc.

.. _sec-use-across-framework:

Use across ML Frameworks
------------------------

TVM FFI's Python package provides :py:func:`tvm_ffi.load_module`, which loads either C++ or CUDA's ``libmain.so`` into :py:class:`tvm_ffi.Module`.

.. code-block:: python

   import tvm_ffi
   mod  : tvm_ffi.Module   = tvm_ffi.load_module("libmain.so")
   func : tvm_ffi.Function = mod["add_one"]

``mod["add_one"]`` retrieves a callable :py:class:`tvm_ffi.Function` that accepts tensors from host frameworks directly, which can be zero-copy incorporated in all popular ML frameworks. This process is done seamlessly without any boilerplate code, and with ultra low latency.

.. tab-set::

    .. tab-item:: NumPy (C++)

        .. code-block:: python

          import numpy as np
          x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
          y = np.empty_like(x)
          func(x, y)
          print(y)

    .. tab-item:: PyTorch (C++/CUDA)

        .. code-block:: python

          import torch
          device = "cpu" # or "cuda"
          x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device)
          y = torch.empty_like(x)
          func(x, y)
          print(y)

    .. tab-item:: JAX (C++/CUDA)

        .. code-block:: python

          # TBD: add JAX example

    .. tab-item:: CuPy (CUDA)

        .. code-block:: python

          # TBD: add CuPy example


Use across Languages
--------------------

TVM-FFI's core loading mechanism is ABI stable and works across language boundaries.
That said, a single artifact can be loaded in every language TVM-FFI supports,
without having to recompile different artifact targeting different ABIs or languages.


Python
~~~~~~

As introduced in the :ref:`previous section<sec-use-across-framework>`, :py:func:`tvm_ffi.load_module` loads a language- and framework-neutral ``libmain.so`` and supports incorporating it into all Python frameworks that implements the standard `DLPack protocol <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html>`_.

C++
~~~

TVM-FFI's C++ API :cpp:func:`tvm::ffi::Module::LoadFromFile` loads ``libmain.so`` and can be used directly in C/C++ with no Python dependency. Note that it is also ABI stable and can be used without having to worry about C++ compilers and ABIs.

.. code-block:: cpp

   // File: test_load.cc
   #include <tvm/ffi/extra/module.h>

   int main() {
     namespace ffi = tvm::ffi;
     ffi::Module   mod  = ffi::Module::LoadFromFile("libmain.so");
     ffi::Function func = mod->GetFunction("add_one").value();
     return 0;
   }

Compile it with:

.. code-block:: bash

    g++ -fvisibility=hidden -O3               \
        test_load.cc                          \
        `tvm-ffi-config --cxxflags`           \
        `tvm-ffi-config --ldflags`            \
        `tvm-ffi-config --libs`               \
        -Wl,-rpath,`tvm-ffi-config --libdir`  \
        -o test_load

    ./test_load


Rust
~~~~

TVM-FFI's Rust API ``tvm_ffi::Module::load_from_file`` loads ``libmain.so``, and then retrieves a function ``add_one`` from it. This procedure is strictly identical to C++ and Python:

.. code-block:: rust

    let module : tvm_ffi::Module   = tvm_ffi::Module::load_from_file("libmain.so").unwrap();
    let func   : tvm_ffi::Function = module.get_function("add_one").unwrap();


Troubleshooting
---------------

- ``OSError: cannot open shared object file``: Add an rpath (Linux/macOS) or ensure the DLL is on ``PATH`` (Windows). Example run-path: ``-Wl,-rpath,`tvm-ffi-config --libdir```.
- ``undefined symbol: __tvm_ffi_add_one``: Ensure you used ``TVM_FFI_DLL_EXPORT_TYPED_FUNC`` and compiled with default symbol visibility (``-fvisibility=hidden`` is fine; the macro ensures export).
- ``CUDA error: invalid device function``: Rebuild with the right ``-arch=sm_XX`` for your GPU, or include multiple ``-gencode`` entries.

What's next?
------------

TBD
