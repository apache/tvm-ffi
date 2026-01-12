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

ABI Overview
============

.. hint::

    Authoritative ABI specifications are defined in

    - C header `tvm/ffi/c_api.h <https://github.com/apache/tvm-ffi/blob/main/include/tvm/ffi/c_api.h>`_, which contains the core ABI, and
    - C header `tvm/ffi/extra/c_env_api.h <https://github.com/apache/tvm-ffi/blob/main/include/tvm/ffi/extra/c_env_api.h>`_, which contains extra support features.

The TVM-FFI ABI is designed around the following key principles:

- **Minimal and efficient.** Keep things simple and deliver close-to-metal performance.
- **Stability guarantee.** The ABI remains stable across compiler versions and is independent of host languages or frameworks.
- **Expressive for machine learning.** Native support for tensors, shapes, and data types commonly used in ML workloads.
- **Extensible.** The ABI supports user-defined types and features through a dynamic type registration system.

This tutorial covers common concepts and usage patterns of the TVM-FFI ABI, with low-level C code examples for precise reference.

.. important::
  C code is used for clarity, precision and friendliness to compiler builders.
  And C code can be readily translated into code generators such as LLVM IR builder.

Any and AnyView
---------------

.. seealso::

   :doc:`any` for :cpp:class:`~tvm::ffi::Any` and :cpp:class:`~tvm::ffi::AnyView` usage patterns.

At the core of TVM-FFI is :cpp:class:`TVMFFIAny`, a 16-byte tagged union that can hold any value
recognized by the FFI system. It enables type-erased value passing across language boundaries.

.. dropdown:: C ABI Reference: :cpp:class:`TVMFFIAny`
   :icon: code

   .. literalinclude:: ../../include/tvm/ffi/c_api.h
      :language: c
      :start-after: [TVMFFIAny.begin]
      :end-before: [TVMFFIAny.end]
      :caption: tvm/ffi/c_api.h

**Ownership.** :cpp:class:`TVMFFIAny` struct can represent either an owning or a borrowing reference.
These two ownership patterns are formalized by the C++ wrapper classes :cpp:class:`~tvm::ffi::Any` and :cpp:class:`~tvm::ffi::AnyView`,
which have identical memory layouts but different :ref:`ownership semantics <any-ownership>`:

- **Owning:** :cpp:class:`tvm::ffi::Any` - reference-counted, manages object lifetime
- **Borrowing:** :cpp:class:`tvm::ffi::AnyView` - non-owning view, caller must ensure validity

.. note::
   To convert a borrowing :cpp:class:`~tvm::ffi::AnyView` to an owning :cpp:class:`~tvm::ffi::Any`, use :cpp:func:`TVMFFIAnyViewToOwnedAny`.

**Runtime Type Index.** The ``type_index`` field identifies what kind of value is stored:

- :ref:`Atomic POD types <any-atomic-types>` (``type_index`` < :cpp:enumerator:`kTVMFFIStaticObjectBegin <TVMFFITypeIndex::kTVMFFIStaticObjectBegin>`):
  Stored inline in the payload union without heap allocation or reference counting.
- :ref:`Object types <any-heap-allocated-objects>` (``type_index`` >= :cpp:enumerator:`kTVMFFIStaticObjectBegin <TVMFFITypeIndex::kTVMFFIStaticObjectBegin>`):
  Stored as pointers to heap-allocated, reference-counted TVM-FFI objects.

.. important::
   The TVM-FFI type index system does not rely on C++ RTTI.


Construct Any
~~~~~~~~~~~~~

**From atomic POD types.** The following C code constructs a :cpp:class:`TVMFFIAny` from an integer:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Any_AnyView.FromInt_Float.begin]
  :end-before: [Any_AnyView.FromInt_Float.end]

Set the ``type_index`` from :cpp:enum:`TVMFFITypeIndex` and assign the corresponding payload field.

.. important::

   Always zero the ``zero_padding`` field and any unused bytes in the value union.
   This invariant enables direct byte comparison and hashing of :cpp:class:`TVMFFIAny` values.

**From object types.** The following C code constructs a :cpp:class:`TVMFFIAny` from a heap-allocated object:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Any_AnyView.FromObjectPtr.begin]
  :end-before: [Any_AnyView.FromObjectPtr.end]

When ``IS_OWNING_ANY`` is ``true`` (owning :cpp:class:`~tvm::ffi::Any`), this increments the object's reference count.

.. _abi-destruct-any:

Destruct Any
~~~~~~~~~~~~

The following C code destroys a :cpp:class:`TVMFFIAny`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Any_AnyView.Destroy.begin]
  :end-before: [Any_AnyView.Destroy.end]

When ``IS_OWNING_ANY`` is ``true`` (owning :cpp:class:`~tvm::ffi::Any`), this decrements the object's reference count.

Extract from Any
~~~~~~~~~~~~~~~~

**Extract an atomic POD.** The following C code extracts an integer or float from a :cpp:class:`TVMFFIAny`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Any_AnyView.GetInt_Float.begin]
  :end-before: [Any_AnyView.GetInt_Float.end]

Implicit type conversion may occur. For example, when extracting a float from a :cpp:class:`TVMFFIAny`
that holds an integer, the integer is cast to a float.

**Extract a DLTensor.** A :c:struct:`DLTensor` may originate from either a raw pointer or a heap-allocated :cpp:class:`~tvm::ffi::TensorObj`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Any_AnyView.GetDLTensor.begin]
  :end-before: [Any_AnyView.GetDLTensor.end]

**Extract a TVM-FFI object.** TVM-FFI objects are always heap-allocated and reference-counted,
with ``type_index`` >= :cpp:enumerator:`kTVMFFIStaticObjectBegin <TVMFFITypeIndex::kTVMFFIStaticObjectBegin>`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Any_AnyView.GetObject.begin]
  :end-before: [Any_AnyView.GetObject.end]

To take ownership of the returned value, increment the reference count via :cpp:func:`TVMFFIObjectIncRef`.
Release ownership later via :cpp:func:`TVMFFIObjectDecRef`.

.. _abi-object:

Object
------

.. seealso::

   :doc:`object_and_class` for the object system and reflection.

TVM-FFI Object (:cpp:class:`TVMFFIObject`) is the cornerstone of TVM-FFI's stable yet extensible type system.

.. dropdown:: C ABI Reference: :cpp:class:`TVMFFIObject`
   :icon: code

   .. literalinclude:: ../../include/tvm/ffi/c_api.h
      :language: c
      :start-after: [TVMFFIObject.begin]
      :end-before: [TVMFFIObject.end]
      :caption: tvm/ffi/c_api.h

All TVM-FFI objects share these characteristics:

- Heap-allocated and reference-counted
- Layout-stable 24-byte header containing reference counts, type index, and deleter callback
- Type index >= :cpp:enumerator:`kTVMFFIStaticObjectBegin <TVMFFITypeIndex::kTVMFFIStaticObjectBegin>`

**Dynamic Type System.** Classes can be registered at runtime via :cpp:func:`TVMFFITypeGetOrAllocIndex`,
with support for single inheritance. See :ref:`type-checking-and-casting` for usage details.

A small **static section** between :cpp:enumerator:`kTVMFFIStaticObjectBegin <TVMFFITypeIndex::kTVMFFIStaticObjectBegin>`
and :cpp:enumerator:`kTVMFFIDynObjectBegin <TVMFFITypeIndex::kTVMFFIDynObjectBegin>`
is reserved for static object types, for example,

- Strings (:cpp:enumerator:`kTVMFFIStr <TVMFFITypeIndex::kTVMFFIStr>`) and Bytes (:cpp:enumerator:`kTVMFFIBytes <TVMFFITypeIndex::kTVMFFIBytes>`): Section :ref:`abi-string-and-byte`
- Errors (:cpp:enumerator:`kTVMFFIError <TVMFFITypeIndex::kTVMFFIError>`): Section :ref:`abi-exception`.
- Functions (:cpp:enumerator:`kTVMFFIFunction <TVMFFITypeIndex::kTVMFFIFunction>`): Section :ref:`abi-function`.
- Tensors (:cpp:enumerator:`kTVMFFITensor <TVMFFITypeIndex::kTVMFFITensor>`): Section :ref:`abi-tensor`.
- Miscellaneous:
  Modules (:cpp:enumerator:`kTVMFFIModule <TVMFFITypeIndex::kTVMFFIModule>`),
  Arrays (:cpp:enumerator:`kTVMFFIArray <TVMFFITypeIndex::kTVMFFIArray>`),
  Maps (:cpp:enumerator:`kTVMFFIMap <TVMFFITypeIndex::kTVMFFIMap>`),
  Shapes (:cpp:enumerator:`kTVMFFIShape <TVMFFITypeIndex::kTVMFFIShape>`),
  Opaque Python objects (:cpp:enumerator:`kTVMFFIOpaquePyObject <TVMFFITypeIndex::kTVMFFIOpaquePyObject>`).

.. _abi-object-ownership:

Ownership Management
~~~~~~~~~~~~~~~~~~~~

Ownership is managed via reference counting, which includes both strong and weak references.
Two C APIs manage strong reference counting:

- :cpp:func:`TVMFFIObjectIncRef`: Acquire strong ownership by incrementing the reference count
- :cpp:func:`TVMFFIObjectDecRef`: Release strong ownership by decrementing the reference count

The ``deleter`` callback (:cpp:member:`TVMFFIObject::deleter`) executes when the strong or weak count reaches zero with different flags.
See :ref:`object-reference-counting` for details.

**Move ownership from Any/AnyView.** The following C code transfers ownership from an owning :cpp:class:`~tvm::ffi::Any` to an object pointer:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Object.MoveFromAny.begin]
  :end-before: [Object.MoveFromAny.end]

Since :cpp:class:`~tvm::ffi::AnyView` is non-owning (``IS_OWNING_ANY`` is ``false``),
acquiring ownership requires explicitly incrementing the reference count.

**Release ownership.** The following C code releases ownership of a TVM-FFI object:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :name: ABI.Object.Destroy
  :start-after: [Object.Destroy.begin]
  :end-before: [Object.Destroy.end]

Inheritance Checking
~~~~~~~~~~~~~~~~~~~~

TVM-FFI models single inheritance as a tree where each node points to its parent.
Each type has a unique type index, and the system tracks ancestors, inheritance depth, and other metadata.
This information is available via :cpp:func:`TVMFFIGetTypeInfo`.

The following C code checks whether a type is a subclass of another:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Object.IsInstance.begin]
  :end-before: [Object.IsInstance.end]

.. _abi-tensor:

Tensor
------

.. seealso::

   :doc:`tensor` for details about TVM-FFI tensors and DLPack interoperability.

TVM-FFI provides :cpp:class:`tvm::ffi::TensorObj`, a DLPack-native tensor class that is also a standard TVM-FFI object.
This means tensors can be managed using the same reference counting mechanisms as other objects.

.. dropdown:: C ABI Reference: :cpp:class:`tvm::ffi::TensorObj`
   :icon: code

   .. code-block:: cpp
    :caption: tvm/ffi/container/tensor.h

     class TensorObj : public Object, public DLTensor {
      // no other members besides those from Object and DLTensor
     };


Access Tensor Metadata
~~~~~~~~~~~~~~~~~~~~~~

The following C code obtains a :c:struct:`DLTensor` pointer from a :cpp:class:`~tvm::ffi::TensorObj`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Tensor.AccessDLTensor.begin]
  :end-before: [Tensor.AccessDLTensor.end]

The :c:struct:`DLTensor` pointer provides access to shape, dtype, device, data pointer, and other tensor metadata.

Construct Tensor from DLPack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following C code constructs a :cpp:class:`~tvm::ffi::TensorObj` from a :c:struct:`DLManagedTensorVersioned`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Tensor_FromDLPack.begin]
  :end-before: [Tensor_FromDLPack.end]

.. hint::
   TVM-FFI's Python API automatically wraps framework tensors (e.g., :py:class:`torch.Tensor`) as :cpp:class:`~tvm::ffi::TensorObj`,
   so manual conversion is typically unnecessary.

Destruct Tensor
~~~~~~~~~~~~~~~

As a standard TVM-FFI object, :cpp:class:`~tvm::ffi::TensorObj` follows the :ref:`standard destruction pattern <ABI.Object.Destroy>`.
When the reference count reaches zero, the deleter callback (:cpp:member:`TVMFFIObject::deleter`) executes.

Export Tensor to DLPack
~~~~~~~~~~~~~~~~~~~~~~~

To share a :cpp:class:`~tvm::ffi::TensorObj` with other frameworks, export it as a :c:struct:`DLManagedTensorVersioned`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Tensor_ToDLPackVersioned.begin]
  :end-before: [Tensor_ToDLPackVersioned.end]

Note that the caller takes ownership of the returned :c:struct:`DLManagedTensorVersioned* <DLManagedTensorVersioned>`
and must call its ``deleter`` to release the tensor.

.. _abi-function:

Function
--------

.. seealso::

   :ref:`sec:function` for a detailed description of TVM-FFI functions.

All functions in TVM-FFI follow a unified C calling convention that enables ABI-stable,
type-erased, and cross-language function calls, defined by :cpp:type:`TVMFFISafeCallType`.

**Calling convention.** The signature includes:

- ``handle`` (``void*``): Optional resource handle passed to the callee; typically ``NULL`` for exported symbols
- ``args`` (``TVMFFIAny*``) and ``num_args`` (``int``): Array of non-owning :cpp:class:`~tvm::ffi::AnyView` input arguments
- ``result`` (``TVMFFIAny*``): Owning :cpp:class:`~tvm::ffi::Any` output value
- Return value: ``0`` for success; ``-1`` or ``-2`` for errors (see :ref:`sec:exception`)

See :ref:`sec:function-calling-convention` for more details.

**Memory layout.** The :cpp:class:`~tvm::ffi::FunctionObj` stores call pointers after the object header.

.. dropdown:: C ABI Reference: :cpp:class:`TVMFFIFunctionCell`
   :icon: code

   .. literalinclude:: ../../include/tvm/ffi/c_api.h
      :language: c
      :start-after: [TVMFFIFunctionCell.begin]
      :end-before: [TVMFFIFunctionCell.end]
      :caption: tvm/ffi/c_api.h

Construct and Destroy
~~~~~~~~~~~~~~~~~~~~~

.. important::
   Dynamic function creation is useful for passing lambdas or closures across language boundaries.

The following C code constructs a :cpp:class:`~tvm::ffi::FunctionObj` from a :cpp:type:`TVMFFISafeCallType` and a ``deleter`` callback.
The ``deleter`` cleans up resources owned by the function; for global symbols, it is typically ``NULL``.

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Function.Construct.begin]
  :end-before: [Function.Construct.end]

Release a :cpp:class:`~tvm::ffi::FunctionObj` using the :ref:`standard destruction pattern <ABI.Object.Destroy>`.

Global Registry
~~~~~~~~~~~~~~~

**Retrieve a global function.** The following C code uses :cpp:func:`TVMFFIFunctionGetGlobal` to retrieve a function by name from the global registry:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Function.GetGlobal.begin]
  :end-before: [Function.GetGlobal.end]

.. note::
   :cpp:func:`TVMFFIFunctionGetGlobal` returns an owning handle.
   The caller must release it by calling :cpp:func:`TVMFFIObjectDecRef` when it's no longer needed.

**Register a global function.** The following C code uses :cpp:func:`TVMFFIFunctionSetGlobal` to register a function by name in the global registry:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Function.SetGlobal.begin]
  :end-before: [Function.SetGlobal.end]

Call Function
~~~~~~~~~~~~~

The following C code invokes a :cpp:class:`~tvm::ffi::FunctionObj` with arguments:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Function.Call.begin]
  :end-before: [Function.Call.end]


.. _abi-exception:

Exception
---------

.. seealso::

   :ref:`sec:exception` for detailed exception handling patterns.

Exceptions are a central part of TVM-FFI's ABI and calling convention.
When errors occur, they are stored as objects with a :cpp:class:`TVMFFIErrorCell` payload.

.. dropdown:: C ABI Reference: :cpp:class:`TVMFFIErrorCell`
   :icon: code

   .. literalinclude:: ../../include/tvm/ffi/c_api.h
      :language: c
      :start-after: [TVMFFIErrorCell.begin]
      :end-before: [TVMFFIErrorCell.end]
      :caption: tvm/ffi/c_api.h

.. important::
  Errors from all languages (e.g. Python, C++) will be properly translated into the TVM-FFI error object.


Retrieve Error Object
~~~~~~~~~~~~~~~~~~~~~

When a function returns ``-1``, an error object is stored in thread-local storage (TLS).
Retrieve it with :cpp:func:`TVMFFIErrorMoveFromRaised`, which returns a :cpp:class:`tvm::ffi::ErrorObj`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Error.HandleReturnCode.begin]
  :end-before: [Error.HandleReturnCode.end]

This function transfers ownership to the caller and clears the TLS slot.
Call :cpp:func:`TVMFFIObjectDecRef` when done to avoid memory leaks.

**Frontend errors (-2).** Error code ``-2`` is reserved for frontend errors.
It is returned when :cpp:func:`TVMFFIEnvCheckSignals` detects a pending Python signal.
In this case, do not retrieve the error from TLS; instead, consult the frontend's error mechanism.

.. admonition:: Print Error Message
  :class: hint

  The error payload is a :cpp:type:`TVMFFIErrorCell` structure containing the error kind, message, and backtrace.
  Access it by skipping the :cpp:type:`TVMFFIObject` header via pointer arithmetic.

  .. literalinclude:: ../../examples/abi_overview/example_code.c
    :language: c
    :start-after: [Error.Print.begin]
    :end-before: [Error.Print.end]

  This prints the error message along with its backtrace.

Raise Exception
~~~~~~~~~~~~~~~

The following C code sets the TLS error and returns ``-1`` via :cpp:func:`TVMFFIErrorSetRaisedFromCStr`:

.. literalinclude:: ../../examples/abi_overview/example_code.c
  :language: c
  :start-after: [Error.RaiseException.begin]
  :end-before: [Error.RaiseException.end]

For non-null-terminated strings, use :cpp:func:`TVMFFIErrorSetRaisedFromCStrParts`, which accepts explicit string lengths.

.. note::
   You rarely need to create a :cpp:class:`~tvm::ffi::ErrorObj` directly.
   The C APIs :cpp:func:`TVMFFIErrorSetRaisedFromCStr` and :cpp:func:`TVMFFIErrorSetRaisedFromCStrParts` handle this internally.

.. _abi-string-and-byte:

🚧 String and Bytes
-------------------

.. warning::
   This section is under construction.

The ABI supports strings and bytes as first-class citizens. A string can take multiple forms that are identified by
its ``type_index``.

- ``kTVMFFIRawStr``: raw C string terminated by ``\0``.
- ``kTVMFFISmallStr``: small string, the length is stored in ``small_str_len`` and data is stored in ``v_bytes``.
- ``kTVMFFIStr``: on-heap string object for strings that are longer than 7 characters.

The following code shows the layout of the on-heap string object.

.. code-block:: cpp

  // span-like data structure to store header and length
  typedef struct {
    const char* data;
    size_t size;
  } TVMFFIByteArray;

  // showcase the layout of the on-heap string.
  class StringObj : public ffi::Object, public TVMFFIByteArray {
  };


The following code shows how to read a string from :cpp:class:`TVMFFIAny`

.. code-block:: cpp

  TVMFFIByteArray ReadString(const TVMFFIAny *value) {
    TVMFFIByteArray ret;
    if (value->type_index == kTVMFFIRawStr) {
      ret.data = value->v_c_str;
      ret.size = strlen(ret.data);
    } else if (value->type_index == kTVMFFISmallStr) {
      ret.data = value->v_bytes;
      ret.size = value->small_str_len;
    } else {
      assert(value->type_index == kTVMFFIStr);
      ret = *reinterpret_cast<TVMFFIByteArray*>(
        reinterpret_cast<char*>(value->v_obj) + sizeof(TVMFFIObject));
    }
    return ret;
  }


Similarly, we have type indices to represent bytes. The C++ API provides classes
:cpp:class:`~tvm::ffi::String` and :cpp:class:`~tvm::ffi::Bytes` to enable the automatic conversion of these values with Any storage format.

**Rationales**. Separate string and bytes enable clear mappings from the Python side. Small string allows us to
store short names on-stack. To favor 8-byte alignment (v_bytes) and keep things simple, we did not further
pack characters into the ``small_len`` field.

Further Reading
---------------

- :doc:`any`: High-level C++ usage of :cpp:class:`~tvm::ffi::Any` and :cpp:class:`~tvm::ffi::AnyView`
- :doc:`object_and_class`: The object system and reflection
- :doc:`tensor`: Tensor classes and DLPack interoperability
- :doc:`func_module`: Functions, exceptions, and modules
- :doc:`../get_started/stable_c_abi`: Quick introduction to the stable C ABI
