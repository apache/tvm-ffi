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

At runtime, TVM-FFI often needs to accept tensors coming from many sources:

* Frameworks (e.g. PyTorch, JAX) via :py:meth:`array_api.array.__dlpack__`;
* C/C++ callers passing a pointer to :c:struct:`DLTensor`;
* Tensors allocated by a library but managed by TVM-FFI itself.

TVM-FFI standardizes tensors on **DLPack as lingua franca**: all tensors are
uniformly described by :c:struct:`DLTensor` (device, dtype, shape, data pointer),
while convenient wrappers and ABI utilities are provided for kernel integration and
compiler/runtime integration.

This tutorial is organized as follows:

* **Tensor Classes**: what each tensor type is and which one you should use.
* **Layout and Conversion**: how the types relate in memory and how tensors flow through ABI.
* **Tensor APIs**: the most important C++ and Python APIs you will use.
* **Allocation**: recommended patterns for allocating outputs.
* **Integration Tips**: practical guidance by audience (kernel developers, compilers, runtimes).

Quick Glossary
--------------

DLPack
  A cross-library tensor interchange standard, written as a small pure C header ``dlpack.h``.
  It defines pure C data structures for describing n-dimensional arrays and their memory layout,
  including :c:struct:`DLTensor`, :c:struct:`DLManagedTensorVersioned`, :c:struct:`DLDataType`,
  :c:struct:`DLDevice`, and related types.

View (non-owning)
  A "header" that describes a tensor but does not own the tensor's memory. When the consumer
  receives a view, it must respect that the producer owns the underlying storage and decides its
  lifetime, and use the view can only be used when the producer guarantees it remains valid.

Managed object (owning)
  An object that includes lifetime management. It involves reference counting or a cleanup callback
  mechanism, which establishes a contract between producer and consumer about when consumer's ownership ends.

Tensor Classes
--------------

This section defines each tensor type you will encounter in TVM-FFI C++ and explains the
*intended* usage. Exact C layout details are covered later in :ref:`layout-and-conversion`.

.. tip::

  Python side, only :py:class:`tvm_ffi.Tensor` exists. It is strictly follows DLPack semantics for interop, and can be converted to PyTorch via :py:func:`torch.from_dlpack`.


DLPack Tensors
~~~~~~~~~~~~~~

:c:struct:`DLTensor` is a **view** and the fundamental tensor descriptor. It describes the device
the tensors lives, its shape, dtype, and data pointer. It does not own the underlying data.

The **managed tensor** is :c:struct:`DLManagedTensorVersioned`, or its legacy counterpart
:c:struct:`DLManagedTensor`. It wraps an non-owning :c:struct:`DLTensor` descriptor with some
extra fields,

* a ``deleter(self)`` callback, the cleanup callback consumer uses to release ownership when done with the tensor, and
* an opaque ``manager_ctx`` handle used by by the producer to store additional information.

A common lifecycle of managed DLPack tensors is:

* the producer creates it;
* the consumer uses it;
* the consumer calls the deleter when done.

TVM-FFI Tensors
~~~~~~~~~~~~~~~

:cpp:class:`tvm::ffi::TensorView` is TVM-FFI's **non-owning** view type, and is strictly equivalent to :c:struct:`DLTensor`.
It is designed for **kernel signatures** and other APIs where you only need to inspect metadata and access the underlying data pointer during the call,
without taking ownership of the tensor's memory. Non-owning also means you must ensure the backing tensor remains valid while you use the view.

:cpp:class:`tvm::ffi::TensorObj` is TVM-FFI's **managed tensor** and it has a corresponding container class :cpp:class:`tvm::ffi::Tensor` (similar to ``std::shared_ptr<TensorObj>``).
:cpp:class:`TensorObj <tvm::ffi::TensorObj>` lives on the heap, contains a reference counter and a :c:struct:`DLTensor` descriptor embedded inside it.
Once the reference count drops to zero, the cleanup logic deallocates both the descriptor, and returns the ownership of the underlying data buffer.

:cpp:class:`Tensor <tvm::ffi::Tensor>` is the recommended interface for passing around managed tensors, and use owning tensors only when you need one or more of the following:

* return a tensor from a function across ABI, which will be converted to :cpp:class:`tvm::ffi::Any`;
* allocate an output tensor as the producer, and hand it to a kernel consumer;
* store a tensor in a long-lived object.

.. important::

  When handwriting C++, it is recommended to always use TVM-FFI tensors over DLPack's raw C tensors.
  Meanwhile, DLPack's raw C tensors are recommended for compilers to target.


.. _layout-and-conversion:

Layout and Conversion
---------------------

This section explains how the tensor types relate in memory and how values flow through
TVM-FFI's stable C ABI.

Layout overview (one-page mental model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key idea:

* :c:struct:`DLTensor` is the universal **descriptor**.
* :cpp:class:`tvm::ffi::TensorView` is a C++ **view** built around that descriptor.
* :cpp:class:`tvm::ffi::Tensor` is a managed handle whose heap object contains a descriptor.
* :c:struct:`DLManagedTensor` and :c:struct:`DLManagedTensorVersioned` are managed wrappers around a descriptor with a deleter.

For introductory users, the most important distinction is **ownership**:

* :c:struct:`DLTensor` and :cpp:class:`tvm::ffi::TensorView` are non-owning views.
* :cpp:class:`tvm::ffi::Tensor` and managed DLPack wrappers provide lifetime management.

Minimal C layout sketches (for orientation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a *simplified* view of layouts you will see. It is sufficient to understand ABI flow
and conversions. (Full details exist in the DLPack and TVM-FFI headers.)

:c:struct:`DLTensor` (descriptor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conceptually::

   struct DLTensor {
     void*     data;
     DLDevice  device;   // { device_type, device_id }
     int       ndim;
     DLDataType dtype;   // { code, bits, lanes }
     int64_t*  shape;    // length = ndim
     /* ... additional layout fields exist in DLPack ... */
   };

.. admonition:: Figure suggestion: "DLTensor header"
   :class: tip

   Draw a box labeled ``DLTensor`` with rows for ``data``, ``device``, ``ndim``, ``dtype``, ``shape``.
   Draw an arrow from ``data`` to a "data buffer" box, and an arrow from ``shape`` to a "shape[]" box.
   (You may optionally add a small note "additional layout fields omitted".)

:c:struct:`DLManagedTensorVersioned` (managed wrapper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conceptually::

   struct DLManagedTensorVersioned {
     DLPackVersion version;  // { major, minor }
     DLTensor      dl_tensor;
     void*         manager_ctx;
     void        (*deleter)(DLManagedTensorVersioned* self);
     uint64_t      flags;
   };

.. admonition:: Figure suggestion: "Managed wrapper + deleter"
   :class: tip

   Draw a heap box labeled ``DLManagedTensorVersioned`` containing a nested ``DLTensor`` region plus
   ``manager_ctx`` and ``deleter``. Add a prominent arrow labeled "consumer calls deleter when done".

:cpp:class:`tvm::ffi::Tensor` (handle → heap object)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At a high level:

* ``Tensor`` is a handle you pass around.
* It points to a heap object (``TensorObj``) that is reference-counted.
* ``TensorObj`` contains a TVM-FFI object header plus an embedded tensor descriptor.

Conceptually::

   Tensor (handle)  --->  TensorObj (heap)
                           [object header][embedded DLTensor][...]

.. admonition:: Figure suggestion: "Tensor = handle → TensorObj"
   :class: tip

   Draw a small ``Tensor`` handle pointing to a heap ``TensorObj``.
   Inside ``TensorObj``, draw two stacked regions:
   (1) "object header (refcount/type/deleter)"
   (2) "embedded DLTensor descriptor"

ABI Representation of Tensors (Any/AnyView)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the stable C ABI boundary, TVM-FFI passes values using an "Any-like" carrier (often referred
to as **Any** or **AnyView**). Conceptually, it holds:

* a ``type_index`` that says what the payload is, and
* a union payload that may contain:
  * object handles (reference-counted pointers), or
  * raw pointers (e.g., :c:struct:`DLTensor` pointer).

For tensors specifically, you will typically encounter two representations:

* **Managed TVM-FFI tensor object** stored as an object handle.
* **Raw DLPack tensor pointer** stored as a raw :c:struct:`DLTensor` pointer.

.. admonition:: Figure suggestion: "Any stores either object-handle or raw pointer"
   :class: tip

   Draw a box labeled ``Any/AnyView`` with fields ``type_index`` and ``payload``.
   Show two branches:
   (1) ``payload.v_obj`` → ``TensorObj*`` (managed)
   (2) ``payload.v_ptr`` → ``DLTensor*`` (borrowed)

Conversion categories
~~~~~~~~~~~~~~~~~~~~~

TVM-FFI integrations commonly perform the following conversions. The main concern in every case is
**lifetime**.

1) AnyView/Any => TensorView / DLTensor* (kernel library / kernel compiler)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal
  Normalize an incoming dynamic value into a descriptor you can use in kernels or codegen.

Typical approach
  *If the value contains a raw :c:struct:`DLTensor*`, use it directly.*
  *If the value contains a tensor object handle, extract a :c:struct:`DLTensor*` from it.*

Pseudo-code sketch (ABI-level)::

  // Input: any_view
  // Output: const DLTensor* t

  if (any_view.type_index == kTVMFFIDLTensorPtr) {
    t = (const DLTensor*)any_view.v_ptr;
  } else if (any_view.type_index == kTVMFFITensor) {
    void* obj = any_view.v_obj;
    t = TVMFFITensorGetDLTensorPtr(obj);  // see ABI headers
  } else {
    error("expected a tensor");
  }

Notes
  * This conversion is view-level: it does not transfer ownership.
  * The returned :c:struct:`DLTensor` pointer must not outlive the value it came from.

2) AnyView/Any => Tensor (runtime management)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal
  Obtain a **managed** :cpp:class:`tvm::ffi::Tensor` you can store or return later.

Key point
  A bare :c:struct:`DLTensor` pointer does not encode ownership. To obtain a managed tensor, you typically
  need either:

  * an existing managed TVM-FFI tensor object (object-handle form), or
  * a managed DLPack wrapper (so the deleter defines lifetime), or
  * allocate a new tensor and copy data.

Practical guidance
  If your runtime needs to keep tensors alive across multiple calls, prefer managed representations
  at the boundary.

3) TensorView / DLTensor* => AnyView (passing tensors into ABI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal
  Package a view (descriptor pointer) into an ABI carrier.

Important lifetime note
  AnyView that stores a raw :c:struct:`DLTensor` pointer is still non-owning. The caller must ensure the
  descriptor remains valid during the callee's use.

4) Tensor => Any (runtime returning a tensor)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Goal
  Return a tensor through the ABI with correct lifetime management.

Typical approach
  Return the tensor object handle form so reference counting preserves lifetime on the receiving side.

Tensor APIs
-----------

This section introduces the most important APIs you will use in C++ and Python. It intentionally
focuses on introductory "daily-driver" methods.

Important C++ types
~~~~~~~~~~~~~~~~~~~

Kernel signatures (recommended)
  Use :cpp:class:`tvm::ffi::TensorView` for inputs (and often outputs) when you only need a view
  within the call.

Return values and storage
  Use :cpp:class:`tvm::ffi::Tensor` when you must return or store a tensor.

Common C++ query methods (Tensor and TensorView)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most commonly used queries are:

* ``ndim()``: number of dimensions
* ``dtype()``: element type
* ``device()``: device type/id
* ``shape()``: shape array (length = ``ndim()``)
* ``data_ptr()``: base pointer to the tensor's data
* ``numel()``: total number of elements (convenience)

A typical pattern inside a kernel:

.. code-block:: cpp

   void MyKernel(tvm::ffi::TensorView x) {
     int ndim = x.ndim();
     auto dtype = x.dtype();
     auto dev = x.device();

     const int64_t* shape = x.shape();
     void* data = x.data_ptr();

     // Validate what your kernel expects.
     // Example: require float32, 2D tensor, etc.
     (void)ndim; (void)dtype; (void)dev;
     (void)shape; (void)data;

     // Launch CPU/GPU work using `data` and `shape`.
   }

.. note::

   Advanced layout details (e.g. non-contiguous views) exist in DLPack and TVM-FFI, but many
   introductory kernels start by requiring a simple layout (e.g. contiguous) and documenting the
   requirement. Add support incrementally when needed.

Python APIs (interop-friendly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python-facing :py:class:`tvm_ffi.Tensor` is a managed n-dimensional array that:

* Can be created via :py:func:`tvm_ffi.from_dlpack(ext_tensor, ...)` to import framework tensors.
* Implements ``__dlpack__`` so it can be passed back to frameworks without copies.

Typical import pattern:

.. code-block:: python

   import tvm_ffi
   import torch

   x = torch.randn(1024, device="cuda")
   t = tvm_ffi.from_dlpack(x, require_contiguous=True)

   # t is a tvm_ffi.Tensor that views the same memory.
   # You can pass t into TVM-FFI-exposed functions.

Allocation
~~~~~~~~~~

TVM-FFI is not a tensor *operation* library, but it does provide well-defined ways to allocate
tensors for outputs.

Common C++ allocation entry points:

* ``Tensor::FromNDAlloc()``:
  Allocate using a custom allocator object you control.

* ``Tensor::FromEnvAlloc()``:
  Allocate using an **environment allocator** provided by the embedding runtime (recommended for
  kernel libraries and plugin modules).

* ``Tensor::FromNDAllocStrided()``:
  Allocate a tensor with a specific non-standard layout (advanced use).

Recommended guidance (intro-friendly)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are writing a kernel library intended to be used by *unknown hosts* (Python runtime, graph
compiler runtime, other embeddings), prefer **FromEnvAlloc** for returned outputs:

* It avoids allocator lifetime coupling to your shared library/module.
* It allows the host environment to pick allocation policy (pools, device memory strategies, etc.).

If you are writing a standalone C++ application or a tightly controlled runtime, **FromNDAlloc**
is often suitable.

Methods
~~~~~~~

These methods are frequently used by introductory users implementing kernels or integrations:

Metadata
  * ``shape()``, ``ndim()``, ``dtype()``, ``device()``, ``numel()``

Data access
  * ``data_ptr()``

Interop
  * ``ToDLPackVersioned()`` / ``ToDLPack()`` (export)
  * ``FromDLPackVersioned(...)`` / ``FromDLPack(...)`` (import)
  * ``GetDLTensorPtr()`` (obtain :c:struct:`DLTensor` pointer for ABI consumers)

What Tensor is not
~~~~~~~~~~~~~~~~~~

TVM-FFI is not a tensor library. While presenting a unified standardized representation to tensors,
it does not come with any of the following:

* kernels, such as vector addition, matrix multiplication;
* host-device copy or synchronization primitives;
* advanced indexing or slicing;
* automatic differentiation or computational graph support.

Integration Tips
----------------

This section provides audience-specific guidance.

Kernel Library and Kernel Compilers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are writing kernels (C++ / CUDA / kernel compilers), the most important rules are:

1) Prefer :cpp:class:`tvm::ffi::TensorView` in function signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Accept inputs as :cpp:class:`tvm::ffi::TensorView`.
* Validate dtype, device, and shape up-front.
* Document any layout assumptions your kernel makes (e.g. contiguous).

2) Allocate outputs with the environment allocator (when returning tensors)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Prefer ``Tensor::FromEnvAlloc(...)`` for outputs that cross module boundaries.
* This lets the embedding runtime decide allocation policy.

3) Respect stream context (GPU correctness)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU kernels should launch on the correct stream. TVM-FFI provides an environment stream table.
The general pattern is:

* Read the stream for the tensor's device.
* Launch kernels on that stream.
* Avoid implicit synchronization unless your API explicitly requires it.

.. admonition:: Figure suggestion: "Stream context flow"
   :class: tip

   Draw a flow diagram:

   * "Framework sets current stream" (Python context manager)
   * → "TVM-FFI environment stream table (device → stream handle)"
   * → "Kernel reads stream via TVMFFIEnvGetStream"
   * → "Kernel launch uses that stream"

Graph Compilers
~~~~~~~~~~~~~~~

Graph compilers typically sit one level above kernel libraries. They produce calls into operators
and must reliably marshal tensor arguments.

Recommended patterns:

1) Normalize tensor arguments from dynamic values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Accept both "Tensor object handle" and "raw :c:struct:`DLTensor` pointer" tensor representations.
* Normalize to a view type (:c:struct:`DLTensor` pointer or :cpp:class:`tvm::ffi::TensorView`) before
  calling kernels.

2) Prefer managed tensors for values that must persist
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Graph compilers often need intermediates to live across multiple operator calls. In that case:

* Store :cpp:class:`tvm::ffi::Tensor` values (managed),
* Avoid holding only raw :c:struct:`DLTensor` pointers across scheduling boundaries.

Runtime
~~~~~~~

If you are implementing a runtime that embeds TVM-FFI modules, your main responsibilities are:

1) Provide allocation policy (optional but recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set an environment allocator so kernel libraries can allocate outputs in a way consistent with
your runtime (device pools, custom memory managers, etc.).

2) Provide stream policy (for GPU environments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Expose or set the current stream for each device so that:

* kernels launch work into the correct stream,
* graph capture or framework interop works correctly,
* implicit synchronization is avoided.

3) Choose appropriate tensor exchange forms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Use managed tensor objects (:cpp:class:`tvm::ffi::Tensor`) when you need refcounted lifetime.
* Use managed DLPack wrappers (:c:struct:`DLManagedTensorVersioned`) when interoperating with
  external frameworks or non-TVM-FFI components.
* Use raw :c:struct:`DLTensor` pointer only for short-lived "during-call" usage.

Final checklist (intro-friendly)
--------------------------------

Before you ship an integration, verify:

* [ ] You never store a :cpp:class:`tvm::ffi::TensorView` beyond the lifetime of its producer.
* [ ] You do not assume a particular layout unless you check/enforce it and document it.
* [ ] You allocate outputs using a strategy appropriate for your deployment (environment allocator
      for plugin modules; custom allocator for controlled runtimes).
* [ ] On GPU, you launch kernels on the correct stream as provided by the environment.
* [ ] Compiler/runtime code accepts both "Tensor object" and "DLTensor*" tensor representations
      where appropriate, and normalizes them consistently.
