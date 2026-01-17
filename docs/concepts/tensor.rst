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

Tensor and DLPack
=================

At runtime, TVM-FFI often needs to accept tensors from many sources:

* Frameworks (e.g. PyTorch, JAX, PaddlePaddle) via :py:meth:`array_api.array.__dlpack__`;
* C/C++ callers passing :c:struct:`DLTensor* <DLTensor>`;
* Tensors allocated by a library but managed by TVM-FFI itself.

TVM-FFI standardizes on **DLPack as the lingua franca**: tensors are
built on top of DLPack structs with additional C++ convenience methods
and minimal extensions for ownership management.

.. tip::

  Prefer :cpp:class:`tvm::ffi::TensorView` or :cpp:class:`tvm::ffi::Tensor` in C++ code;
  they provide safer and more convenient abstractions over raw DLPack structs.


This tutorial covers common usage patterns, tensor classes, and how tensors flow across ABI boundaries.

Glossary
--------

DLPack
  A cross-library tensor interchange standard defined in the small C header ``dlpack.h``.
  It defines pure C data structures for describing n-dimensional arrays and their memory layout,
  including :c:struct:`DLTensor`, :c:struct:`DLManagedTensorVersioned`, :c:struct:`DLDataType`,
  :c:struct:`DLDevice`, and related types.

View (non-owning)
  A "header" that describes a tensor but does not own its memory. When a consumer
  receives a view, it must respect that the producer owns the underlying storage and controls its
  lifetime. The view is valid only while the producer guarantees it remains valid.

Managed object (owning)
  An object that includes lifetime management, using reference counting or a cleanup callback
  mechanism. This establishes a contract between producer and consumer about when the consumer's ownership ends.

.. note::

  As a loose analogy, think of **view** vs. **managed** as similar to
  ``T*`` (raw pointer) vs. ``std::shared_ptr<T>`` (reference-counted pointer) in C++.

Common Usage
------------

This section introduces the most important APIs for day-to-day use in C++ and Python.

Kernel Signatures
~~~~~~~~~~~~~~~~~

A typical kernel implementation accepts :cpp:class:`TensorView <tvm::ffi::TensorView>` parameters,
validates metadata (dtype, shape, device), and then accesses the data pointer for computation:

.. code-block:: cpp

    #include <tvm/ffi/tvm_ffi.h>

    void MyKernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
      // Validate dtype & device
      if (input.dtype() != DLDataType{kDLFloat, 32, 1})
        TVM_FFI_THROW(TypeError) << "Expect float32 input, but got " << input.dtype();
      if (input.device() != DLDevice{kDLCUDA, 0})
        TVM_FFI_THROW(ValueError) << "Expect input on CUDA:0, but got " << input.device();
      // Access data pointer
      float* input_data_ptr = static_cast<float*>(input.data_ptr());
      float* output_data_ptr = static_cast<float*>(output.data_ptr());
      Kernel<<<...>>>(..., input_data_ptr, output_data_ptr, ...);
    }

On the C++ side, the following APIs are available to query a tensor's metadata:

 :cpp:func:`TensorView::shape() <tvm::ffi::TensorView::shape>` and :cpp:func:`Tensor::shape() <tvm::ffi::Tensor::shape>`
  shape array

 :cpp:func:`TensorView::dtype() <tvm::ffi::TensorView::dtype>` and :cpp:func:`Tensor::dtype() <tvm::ffi::Tensor::dtype>`
  element data type

 :cpp:func:`TensorView::data_ptr() <tvm::ffi::TensorView::data_ptr>` and :cpp:func:`Tensor::data_ptr() <tvm::ffi::Tensor::data_ptr>`
  base pointer to the tensor's data

 :cpp:func:`TensorView::device() <tvm::ffi::TensorView::device>` and :cpp:func:`Tensor::device() <tvm::ffi::Tensor::device>`
  device type and id

 :cpp:func:`TensorView::byte_offset() <tvm::ffi::TensorView::byte_offset>` and :cpp:func:`Tensor::byte_offset() <tvm::ffi::Tensor::byte_offset>`
  byte offset to the first element

 :cpp:func:`TensorView::ndim() <tvm::ffi::TensorView::ndim>` and :cpp:func:`Tensor::ndim() <tvm::ffi::Tensor::ndim>`
  number of dimensions (:cpp:func:`ShapeView::size <tvm::ffi::ShapeView::size>`)

 :cpp:func:`TensorView::numel() <tvm::ffi::TensorView::numel>` and :cpp:func:`Tensor::numel() <tvm::ffi::Tensor::numel>`
  total number of elements (:cpp:func:`ShapeView::Product <tvm::ffi::ShapeView::Product>`)


PyTorch Interop
~~~~~~~~~~~~~~~

On the Python side, :py:class:`tvm_ffi.Tensor` is a managed n-dimensional array that:

* can be created via :py:func:`tvm_ffi.from_dlpack(ext_tensor, ...) <tvm_ffi.from_dlpack>` to import tensors from external frameworks, e.g., :ref:`PyTorch <ship-to-pytorch>`, :ref:`JAX <ship-to-jax>`, :ref:`PaddlePaddle <ship-to-paddle>`, :ref:`NumPy/CuPy <ship-to-numpy>`;
* implements the DLPack protocol so it can be passed back to frameworks without copying, e.g., :py:func:`torch.from_dlpack`.

The following example demonstrates a typical round-trip pattern:

.. code-block:: python

   import tvm_ffi
   import torch

   x_torch = torch.randn(1024, device="cuda")
   x_tvm_ffi = tvm_ffi.from_dlpack(x_torch, require_contiguous=True)
   x_torch_again = torch.from_dlpack(x_tvm_ffi)

In this example, :py:func:`tvm_ffi.from_dlpack` creates ``x_tvm_ffi``, which views the same memory as ``x_torch``.
Similarly, :py:func:`torch.from_dlpack` creates ``x_torch_again``, which shares the underlying buffer with both
``x_tvm_ffi`` and ``x_torch``. No data is copied in either direction.


C++ Allocation
~~~~~~~~~~~~~~

TVM-FFI is not a kernel library and is not linked to any specific device memory allocator or runtime.
However, it provides standardized allocation entry points for kernel library developers by interfacing
with the surrounding framework's allocator - for example, using PyTorch's allocator when running inside
a PyTorch environment.

**Env Allocator.** Use :cpp:func:`Tensor::FromEnvAlloc() <tvm::ffi::Tensor::FromEnvAlloc>` along with C API
:cpp:func:`TVMFFIEnvTensorAlloc` to allocate a tensor using the framework's allocator.

.. code-block:: cpp

  Tensor tensor = Tensor::FromEnvAlloc(
    TVMFFIEnvTensorAlloc,
    /*shape=*/{1, 2, 3},
    /*dtype=*/DLDataType({kDLFloat, 32, 1}),
    /*device=*/DLDevice({kDLCPU, 0})
  );

In a PyTorch environment, this is equivalent to :py:func:`torch.empty`.

.. warning::

  While allocation APIs are available, it is generally **recommended** to avoid allocating tensors
  inside kernels. Instead, prefer pre-allocating outputs and passing them as
  :cpp:class:`tvm::ffi::TensorView` parameters. This approach:

  - avoids memory fragmentation and performance pitfalls,
  - prevents CUDA graph incompatibilities on GPU, and
  - allows the outer framework to control allocation policy (pools, device strategies, etc.).

**Custom Allocator.** Use :cpp:func:`Tensor::FromNDAlloc(custom_alloc, ...) <tvm::ffi::Tensor::FromNDAlloc>`,
or its advanced variant :cpp:func:`Tensor::FromNDAllocStrided(custom_alloc, ...) <tvm::ffi::Tensor::FromNDAllocStrided>`,
to allocate a tensor with a user-provided allocation callback.

The following example uses ``cudaMalloc``/``cudaFree`` as custom allocators for GPU tensors:

.. code-block:: cpp

  struct CUDANDAlloc {
    void AllocData(DLTensor* tensor) {
      size_t data_size = ffi::GetDataSize(*tensor);
      void* ptr = nullptr;
      cudaError_t err = cudaMalloc(&ptr, data_size);
      TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
      tensor->data = ptr;
    }

    void FreeData(DLTensor* tensor) {
      if (tensor->data != nullptr) {
        cudaError_t err = cudaFree(tensor->data);
        TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
        tensor->data = nullptr;
      }
    }
  };

  ffi::Tensor cuda_tensor = ffi::Tensor::FromNDAlloc(
    CUDANDAlloc(),
    /*shape=*/{3, 4, 5},
    /*dtype=*/DLDataType({kDLFloat, 32, 1}),
    /*device=*/DLDevice({kDLCUDA, 0})
  );

C++ Stream Handling
~~~~~~~~~~~~~~~~~~~

Stream context is essential for GPU kernel execution. While CUDA does not have a global context for
default streams, frameworks like PyTorch maintain a "current stream" per device
(:py:func:`torch.cuda.current_stream`), and kernel libraries must read this stream from the embedding environment.

As a hardware-agnostic abstraction layer, TVM-FFI is not linked to any specific stream management library.
However, to ensure GPU kernels launch on the correct stream, it provides standardized APIs to obtain the
stream context from the host framework (e.g., PyTorch).

**Obtain Stream Context.** Use the C API :cpp:func:`TVMFFIEnvGetStream` to obtain the current stream for a given device:

.. code-block:: cpp

  void func(ffi::TensorView input, ...) {
    ffi::DLDevice device = input.device();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(
        TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

This is equivalent to the following PyTorch C++ code:

.. code-block:: cpp

  void func(at::Tensor input, ...) {
    c10::Device device = input.device();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(
        c10::cuda::getCurrentCUDAStream(device.index()).stream());
  }


**Auto-Update Stream Context.** When converting framework tensors via :py:func:`tvm_ffi.from_dlpack`,
TVM-FFI automatically updates the stream context to match the device of the converted tensor.
For example, when converting a PyTorch tensor on ``torch.device('cuda:3')``, TVM-FFI automatically
captures the stream from :py:func:`torch.cuda.current_stream(device='cuda:3')`.

**Set Stream Context.** Use :py:func:`tvm_ffi.use_torch_stream` or :py:func:`tvm_ffi.use_raw_stream`
to manually set the stream context when automatic detection is insufficient.

Tensor Classes
--------------

This section defines each tensor type in the TVM-FFI C++ API and explains its intended usage.
Exact C layout details are covered in :ref:`Tensor Layouts <layout-and-conversion>`.

.. tip::

  On the Python side, only :py:class:`tvm_ffi.Tensor` exists. It strictly follows DLPack semantics for interop and can be converted to PyTorch via :py:func:`torch.from_dlpack`.


DLPack Tensors
~~~~~~~~~~~~~~

DLPack tensors come in two main flavors:

*Non-owning* object, :c:struct:`DLTensor`
 The tensor descriptor is a **view** of the underlying data.
 It describes the device the tensor lives on, its shape, dtype, and data pointer. It does not own the underlying data.

*Owning* object, :c:struct:`DLManagedTensorVersioned`, or its legacy counterpart :c:struct:`DLManagedTensor`
 It is a **managed** variant that wraps a :c:struct:`DLTensor` descriptor with additional fields.
 Notably, it includes a ``deleter`` callback that releases ownership when the consumer is done with the tensor,
 and an opaque ``manager_ctx`` handle used by the producer to store additional context.

TVM-FFI Tensors
~~~~~~~~~~~~~~~

Similarly, TVM-FFI defines two main tensor types in C++:

*Non-owning* object, :cpp:class:`tvm::ffi::TensorView`
 A thin C++ wrapper around :c:struct:`DLTensor` for inspecting metadata and accessing the data pointer.
 It is designed for **kernel authors** to inspect metadata and access the underlying data pointer during a call,
 without taking ownership of the tensor's memory. Being a **view** also means you must ensure the backing tensor remains valid while you use it.

*Owning* object, :cpp:class:`tvm::ffi::TensorObj` and :cpp:class:`tvm::ffi::Tensor`
 :cpp:class:`Tensor <tvm::ffi::Tensor>`, similar to ``std::shared_ptr<TensorObj>``, is the managed class to hold heap-allocated
 :cpp:class:`TensorObj <tvm::ffi::TensorObj>`. Once the reference count drops to zero, the cleanup logic deallocates the descriptor
 and releases ownership of the underlying data buffer.


.. note::

   - For handwritten C++, always use TVM-FFI tensors over DLPack's raw C tensors.

   - For compiler development, DLPack's raw C tensors are recommended because C is easier to target from codegen.

The owning :cpp:class:`Tensor <tvm::ffi::Tensor>` is the recommended interface for passing around managed tensors.
Use owning tensors when you need one or more of the following:

* return a tensor from a function across ABI, which will be converted to :cpp:class:`tvm::ffi::Any`;
* allocate an output tensor as the producer, and hand it to a kernel consumer;
* store a tensor in a long-lived object.

.. admonition:: :cpp:class:`TensorObj <tvm::ffi::TensorObj>` vs :cpp:class:`Tensor <tvm::ffi::Tensor>`
   :class: hint

   :cpp:class:`Tensor <tvm::ffi::Tensor>` is an intrusive pointer of a heap-allocated :cpp:class:`TensorObj <tvm::ffi::TensorObj>`.
   As an analogy to ``std::shared_ptr``, think of

   .. code-block:: cpp

      using Tensor = std::shared_ptr<TensorObj>;

   You can convert between the two types:

   - :cpp:func:`Tensor::get() <tvm::ffi::Tensor::get>` converts it to :cpp:class:`TensorObj* <tvm::ffi::TensorObj>`.
   - :cpp:func:`GetRef\<Tensor\> <tvm::ffi::GetRef>` converts a :cpp:class:`TensorObj* <tvm::ffi::TensorObj>` back to :cpp:class:`Tensor <tvm::ffi::Tensor>`.

.. _layout-and-conversion:

Tensor Layouts
~~~~~~~~~~~~~~

:ref:`Figure 1 <fig:layout-tensor>` summarizes the layout relationships among DLPack tensors and TVM-FFI tensors.
All tensor classes are POD-like; :cpp:class:`tvm::ffi::TensorObj` is also a standard TVM-FFI object, typically
heap-allocated and reference-counted.

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/tvm-ffi/tensor-layout.png
  :alt: Layout of DLPack Tensors and TVM-FFI Tensors
  :align: center
  :name: fig:layout-tensor

  Figure 1. Layout specification of DLPack tensors and TVM-FFI tensors. All the tensor types share :c:struct:`DLTensor` as the common descriptor, while carrying different metadata and ownership semantics.

As demonstrated in the figure, all tensor classes share :c:struct:`DLTensor` as the common descriptor.
In particular,

- :c:struct:`DLTensor` and :cpp:class:`TensorView <tvm::ffi::TensorView>` share the exact same memory layout.
- :c:struct:`DLManagedTensorVersioned` and :cpp:class:`TensorObj <tvm::ffi::TensorObj>` both have a deleter
  callback to manage the lifetime of the underlying data buffer, while :c:struct:`DLTensor` and :cpp:class:`TensorView <tvm::ffi::TensorView>` do not.
- Compared with :cpp:class:`TensorView <tvm::ffi::TensorView>`, :cpp:class:`TensorObj <tvm::ffi::TensorObj>`
  has an extra TVM-FFI object header, making it reference-countable via the standard managed reference :cpp:class:`Tensor <tvm::ffi::Tensor>`.

What Tensor Is Not
~~~~~~~~~~~~~~~~~~

TVM-FFI is not a tensor library. While it provides a unified representation for tensors,
it does not include:

* kernels (e.g., vector addition, matrix multiplication),
* host-device copy or synchronization primitives,
* advanced indexing or slicing, or
* automatic differentiation or computational graph support.

Conversion between :cpp:class:`TVMFFIAny`
-----------------------------------------

At the stable C ABI boundary, TVM-FFI passes values using :cpp:class:`Any <tvm::ffi::Any>` (owning)
or :cpp:class:`AnyView <tvm::ffi::AnyView>` (non-owning). Tensors have two possible representations:

* **Non-owning:** :c:struct:`DLTensor* <DLTensor>` with type index :cpp:enumerator:`TVMFFITypeIndex::kTVMFFIDLTensorPtr`
* **Owning:** :cpp:class:`TensorObj* <tvm::ffi::TensorObj>` with type index :cpp:enumerator:`TVMFFITypeIndex::kTVMFFITensor`

When extracting a tensor from :cpp:class:`TVMFFIAny`, check the :cpp:member:`type_index <TVMFFIAny::type_index>`
to determine the representation before conversion.

.. important::

  An owning tensor can be converted to a non-owning view, but not vice versa.

See :ref:`abi-tensor` for C code examples demonstrating:

- Extracting a :c:struct:`DLTensor` pointer from :cpp:class:`TVMFFIAny`
- Constructing a :cpp:class:`~tvm::ffi::TensorObj` from DLPack
- Exporting a :cpp:class:`~tvm::ffi::TensorObj` to DLPack

Further Reading
---------------

- :doc:`object_and_class`: The object system that backs :cpp:class:`~tvm::ffi::TensorObj`
- :doc:`any`: How tensors are stored in :cpp:class:`~tvm::ffi::Any` containers
- :doc:`abi_overview`: Low-level C ABI details for tensor conversion
- :doc:`../guides/kernel_library_guide`: Best practices for building kernel libraries with TVM-FFI
- :external+dlpack:doc:`DLPack C API <c_api>`: The underlying tensor interchange standard
