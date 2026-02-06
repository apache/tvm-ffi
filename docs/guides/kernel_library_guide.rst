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

Kernel Library Guide
====================

This guide covers shipping C++/CUDA kernel libraries with TVM-FFI. The resulting
libraries are agnostic to Python version and ML framework — a single ``.so`` works
with PyTorch, JAX, PaddlePaddle, NumPy, and more.

.. seealso::

   - :doc:`../get_started/quickstart`: End-to-end walkthrough of a simpler ``add_one`` kernel
   - :doc:`../packaging/cpp_tooling`: Build toolchain, CMake integration, and library distribution
   - All example code in this guide is under
     `examples/kernel_library/ <https://github.com/apache/tvm-ffi/tree/main/examples/kernel_library>`_.
   - Production examples:
     `FlashInfer <https://github.com/flashinfer-ai/flashinfer>`_ ships CUDA kernels via TVM-FFI.


Anatomy of a Kernel Function
-----------------------------

Every TVM-FFI CUDA kernel follows the same sequence:

1. **Validate** inputs (device, dtype, shape, contiguity)
2. **Set device guard** to match the tensor's device
3. **Acquire stream** from the host framework
4. **Dispatch** on dtype and **launch** the kernel

Here is a complete ``Scale`` kernel that computes ``y = x * factor``:

.. literalinclude:: ../../examples/kernel_library/scale_kernel.cu
   :language: cpp
   :start-after: [function.begin]
   :end-before: [function.end]

The CUDA kernel itself is a standard ``__global__`` function:

.. literalinclude:: ../../examples/kernel_library/scale_kernel.cu
   :language: cpp
   :start-after: [cuda_kernel.begin]
   :end-before: [cuda_kernel.end]

The following subsections break down each step.


Input Validation
~~~~~~~~~~~~~~~~

Kernel functions should validate inputs early and fail with clear error messages.
A common pattern is to define reusable ``CHECK_*`` macros on top of
:c:macro:`TVM_FFI_CHECK` (see :doc:`../concepts/exception_handling`):

.. literalinclude:: ../../examples/kernel_library/tvm_ffi_utils.h
   :language: cpp
   :start-after: [check_macros.begin]
   :end-before: [check_macros.end]

For **user-facing errors** (bad arguments, unsupported dtypes, shape mismatches),
use :c:macro:`TVM_FFI_THROW` or :c:macro:`TVM_FFI_CHECK` with a specific error kind
so that callers receive an actionable message:

.. code-block:: cpp

   TVM_FFI_THROW(TypeError) << "Unsupported dtype: " << input.dtype();
   TVM_FFI_CHECK(input.numel() > 0, ValueError) << "input must be non-empty";
   TVM_FFI_CHECK(input.numel() == output.numel(), ValueError) << "size mismatch";

For **internal invariants** that indicate bugs in the kernel itself, use
:c:macro:`TVM_FFI_ICHECK`:

.. code-block:: cpp

   TVM_FFI_ICHECK_GE(n, 0) << "element count must be non-negative";


Device Guard and Stream
~~~~~~~~~~~~~~~~~~~~~~~

Before launching a CUDA kernel, two things must happen:

1. **Set the CUDA device** to match the tensor's device. :cpp:class:`tvm::ffi::CUDADeviceGuard`
   is an RAII guard that calls ``cudaSetDevice`` on construction and restores the
   original device on destruction.

2. **Acquire the stream** from the host framework via :cpp:func:`TVMFFIEnvGetStream`.
   When Python code calls a kernel with PyTorch tensors, TVM-FFI automatically
   captures PyTorch's current stream for the tensor's device.

A small helper keeps this concise:

.. literalinclude:: ../../examples/kernel_library/tvm_ffi_utils.h
   :language: cpp
   :start-after: [get_stream.begin]
   :end-before: [get_stream.end]

Every kernel function then follows the same two-line pattern:

.. code-block:: cpp

   ffi::CUDADeviceGuard guard(input.device().device_id);
   cudaStream_t stream = get_cuda_stream(input.device());

See :doc:`../concepts/tensor` for details on stream handling and automatic stream
context updates.


Dtype Dispatch
~~~~~~~~~~~~~~

Kernels typically support multiple dtypes. Dispatch on :c:struct:`DLDataType` at
runtime while instantiating templates at compile time:

.. code-block:: cpp

   constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
   constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};

   if (input.dtype() == dl_float32) {
     ScaleKernel<<<blocks, threads, 0, stream>>>(
         static_cast<float*>(output.data_ptr()), ...);
   } else if (input.dtype() == dl_float16) {
     ScaleKernel<<<blocks, threads, 0, stream>>>(
         static_cast<half*>(output.data_ptr()), ...);
   } else {
     TVM_FFI_THROW(TypeError) << "Unsupported dtype: " << input.dtype();
   }

For libraries that support many dtypes, define dispatch macros
(see `FlashInfer's tvm_ffi_utils.h <https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/tvm_ffi_utils.h>`_
for a production example).


Export and Load
---------------

Export and Build
~~~~~~~~~~~~~~~~

**Export.** Use :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` to create a C symbol
that follows the :doc:`TVM-FFI calling convention <../concepts/func_module>`:

.. literalinclude:: ../../examples/kernel_library/scale_kernel.cu
   :language: cpp
   :start-after: [export.begin]
   :end-before: [export.end]

This creates a symbol ``__tvm_ffi_scale`` in the shared library.

**Build.** Compile the kernel into a shared library using GCC/NVCC or CMake
(see :doc:`../packaging/cpp_tooling` for full details):

.. code-block:: bash

   nvcc -shared -O3 scale_kernel.cu -o build/scale_kernel.so \
       -Xcompiler -fPIC,-fvisibility=hidden \
       $(tvm-ffi-config --cxxflags) \
       $(tvm-ffi-config --ldflags) \
       $(tvm-ffi-config --libs)

**Optional arguments.** Wrap any argument type with :cpp:class:`tvm::ffi::Optional`
to accept ``None`` from the Python side:

.. code-block:: cpp

   void MyKernel(TensorView output, TensorView input,
                 Optional<TensorView> bias, Optional<double> scale) {
     if (bias.has_value()) {
       // use bias.value().data_ptr()
     }
     double s = scale.value_or(1.0);
   }

.. code-block:: python

   mod.my_kernel(y, x, None, None)         # no bias, default scale
   mod.my_kernel(y, x, bias_tensor, 2.0)   # with bias and scale


Load from Python
~~~~~~~~~~~~~~~~

Use :py:func:`tvm_ffi.load_module` to load the library and call its functions.
PyTorch tensors (and other framework tensors) are automatically converted to
:cpp:class:`~tvm::ffi::TensorView` at the ABI boundary:

.. literalinclude:: ../../examples/kernel_library/load_scale.py
   :language: python
   :start-after: [load_and_call.begin]
   :end-before: [load_and_call.end]

See :doc:`../get_started/quickstart` for examples with JAX, PaddlePaddle,
NumPy, CuPy, Rust, and pure C++.


Tensor Handling
---------------

TensorView vs Tensor
~~~~~~~~~~~~~~~~~~~~

TVM-FFI provides two tensor types (see :doc:`../concepts/tensor` for full details):

:cpp:class:`~tvm::ffi::TensorView` *(non-owning)*
  A lightweight view of an existing tensor. **Use this for kernel parameters.**
  It adds no reference count overhead and works with all framework tensors.

:cpp:class:`~tvm::ffi::Tensor` *(owning)*
  A reference-counted tensor that manages its own lifetime. Use this only when
  you need to **allocate and return** a tensor from C++.

.. important::

   Prefer :cpp:class:`~tvm::ffi::TensorView` in kernel signatures. It is more
   lightweight, supports more use cases (including XLA buffers that only provide
   views), and avoids unnecessary reference counting.


Tensor Metadata
~~~~~~~~~~~~~~~

Both :cpp:class:`~tvm::ffi::TensorView` and :cpp:class:`~tvm::ffi::Tensor` expose
identical metadata accessors. These are the methods kernel code uses most:
validating inputs, computing launch parameters, and accessing data pointers.

**Shape and elements.**
:cpp:func:`~tvm::ffi::TensorView::ndim` returns the number of dimensions,
:cpp:func:`~tvm::ffi::TensorView::shape` returns the full shape as a
:cpp:class:`~tvm::ffi::ShapeView` (a lightweight ``span``-like view of
``int64_t``), and :cpp:func:`~tvm::ffi::TensorView::size` returns the size of a
single dimension (supports negative indexing, e.g. ``size(-1)`` for the last
dimension). :cpp:func:`~tvm::ffi::TensorView::numel` returns the total element
count — use it for computing grid dimensions:

.. code-block:: cpp

   int64_t n = input.numel();
   int threads = 256;
   int blocks = (n + threads - 1) / threads;

**Dtype.** :cpp:func:`~tvm::ffi::TensorView::dtype` returns a :c:struct:`DLDataType`
with three fields: ``code`` (e.g. ``kDLFloat``, ``kDLBfloat``), ``bits``
(e.g. 16, 32), and ``lanes`` (almost always 1). Compare it against predefined
constants to dispatch on dtype:

.. code-block:: cpp

   constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
   if (input.dtype() == dl_float32) { ... }

**Device.** :cpp:func:`~tvm::ffi::TensorView::device` returns a :c:struct:`DLDevice`
with ``device_type`` (e.g. ``kDLCUDA``) and ``device_id``. Use these for
validation and to set the device guard:

.. code-block:: cpp

   TVM_FFI_ICHECK_EQ(input.device().device_type, kDLCUDA);
   ffi::CUDADeviceGuard guard(input.device().device_id);

**Data pointer.** :cpp:func:`~tvm::ffi::TensorView::data_ptr` returns ``void*``;
cast it to the appropriate typed pointer before passing it to a kernel:

.. code-block:: cpp

   auto* out = static_cast<float*>(output.data_ptr());
   auto* in  = static_cast<float*>(input.data_ptr());

**Strides and contiguity.**
:cpp:func:`~tvm::ffi::TensorView::strides` returns the stride array as a
:cpp:class:`~tvm::ffi::ShapeView`, and
:cpp:func:`~tvm::ffi::TensorView::stride` returns a single dimension's stride.
:cpp:func:`~tvm::ffi::TensorView::IsContiguous` checks whether the tensor is
contiguous in memory. Most kernels require contiguous inputs — the
``CHECK_CONTIGUOUS`` macro shown above enforces this at the top of each function.

.. tip::

   The API is designed to be familiar to PyTorch developers.
   ``dim()``, ``sizes()``, ``size(i)``, ``stride(i)``, and ``is_contiguous()``
   are all available as aliases of their TVM-FFI counterparts.
   See :doc:`../concepts/tensor` for the full API reference.


Tensor Allocation
~~~~~~~~~~~~~~~~~

**Always pre-allocate output tensors on the Python side** and pass them into the
kernel as :cpp:class:`~tvm::ffi::TensorView` parameters. Allocating tensors
inside a kernel function is almost never the right choice:

- it causes **memory fragmentation** from repeated small allocations,
- it **breaks CUDA graph capture**, which requires deterministic memory addresses, and
- it **bypasses the framework's allocator** (caching pools, device placement, memory planning).

The pre-allocation pattern is straightforward:

.. code-block:: python

   # Python: pre-allocate output
   y = torch.empty_like(x)
   mod.scale(y, x, 2.0)

.. code-block:: cpp

   // C++: kernel writes into pre-allocated output
   void Scale(TensorView output, TensorView input, double factor);

If C++-side allocation is truly unavoidable — for example, when the output shape
is data-dependent and cannot be determined before the kernel runs — use
:cpp:func:`tvm::ffi::Tensor::FromEnvAlloc` to at least reuse the host
framework's allocator (e.g., ``torch.empty`` under PyTorch):

.. literalinclude:: ../../examples/kernel_library/tvm_ffi_utils.h
   :language: cpp
   :start-after: [alloc_tensor.begin]
   :end-before: [alloc_tensor.end]

For custom allocators (e.g., ``cudaMalloc``/``cudaFree``), use
:cpp:func:`tvm::ffi::Tensor::FromNDAlloc`. Note that the kernel library must
outlive any tensors allocated this way, since the custom deleter lives in the
library. See :doc:`../concepts/tensor` for details.


Further Reading
---------------

- :doc:`../get_started/quickstart`: End-to-end walkthrough shipping ``add_one`` across frameworks and languages
- :doc:`../packaging/cpp_tooling`: Build toolchain, CMake integration, GCC/NVCC flags, and library distribution
- :doc:`../packaging/python_packaging`: Packaging kernel libraries as Python wheels
- :doc:`../concepts/tensor`: Tensor classes, DLPack interop, stream handling, and allocation APIs
- :doc:`../concepts/func_module`: Function calling convention, modules, and the global registry
- :doc:`../concepts/exception_handling`: Error handling across language boundaries
- :doc:`../concepts/abi_overview`: Low-level C ABI details
