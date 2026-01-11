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

Any and AnyView
===============

TVM-FFI has :cpp:class:`tvm::ffi::Any` and :cpp:class:`tvm::ffi::AnyView`,
type-erased containers that hold any supported value and transport it across
C, C++, Python, and Rust boundaries through a stable ABI.

Similar to ``std::any``, :cpp:class:`~tvm::ffi::Any` is a tagged union that stores
values of a wide variety of types, including primitives, objects, and strings.
Unlike ``std::any``, it is designed for zero-copy inter-language exchange without RTTI,
featuring a fixed 16-byte layout with built-in reference counting and ownership semantics.

This tutorial covers everything you need to know about :cpp:class:`~tvm::ffi::Any` and :cpp:class:`~tvm::ffi::AnyView`:
common usage patterns, ownership semantics, and memory layout.


Common Usage
------------

Function Signatures
~~~~~~~~~~~~~~~~~~~

Use :cpp:class:`~tvm::ffi::AnyView` for function parameters
to avoid reference count overhead and unnecessary copies.
Use :cpp:class:`~tvm::ffi::Any` for return values to transfer ownership to the caller.

.. code-block:: cpp

   ffi::Any func_cpp_signature(ffi::AnyView arg0, ffi::AnyView arg1) {
    ffi::Any result = arg0.cast<int>() + arg1.cast<int>();
    return result;
   }

   // Variant: variadic function
   void func_cpp_variadic(PackedArgs args, Any* ret) {
     int32_t num_args = args.size();
     int x0 = args[0].cast<int>();
     int x1 = args[1].cast<int>();
     int y = x0 + x1;
     *ret = y;
   }

   // Variant: variadic function with C ABI signature
   int func_c_abi_variadic(void*, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* ret) {
     TVM_FFI_SAFE_CALL_BEGIN();
     int x0 = reinterpret_cast<const AnyView*>(args)[0].cast<int>();
     int x1 = reinterpret_cast<const AnyView*>(args)[1].cast<int>();
     int y = x0 + x1;
     reinterpret_cast<Any*>(ret)[0] = y;
     TVM_FFI_SAFE_CALL_END();
   }


Container Storage
~~~~~~~~~~~~~~~~~

:cpp:class:`~tvm::ffi::Any` can be stored in containers like :cpp:class:`~tvm::ffi::Map` and :cpp:class:`~tvm::ffi::Array`:

.. code-block:: cpp

   ffi::Map<ffi::String, ffi::Any> config;
   config.Set("learning_rate", 0.001);
   config.Set("batch_size", 32);
   config.Set("device", DLDevice{kDLCUDA, 0});

Extracting Values
~~~~~~~~~~~~~~~~~

Three methods extract values from :cpp:class:`~tvm::ffi::Any`
and :cpp:class:`~tvm::ffi::AnyView`, each with different levels of strictness:

.. list-table::
   :header-rows: 1
   :widths: 20 45 30

   * - Method
     - Behavior
     - Use When
   * - :cpp:func:`cast\<T\>() <tvm::ffi::Any::cast>`
     - Returns ``T`` or throws :cpp:class:`tvm::ffi::Error`
     - When you know the expected type and want an exception on mismatch
   * - :cpp:func:`try_cast\<T\>() <tvm::ffi::Any::try_cast>`
     - Returns ``std::optional<T>``
     - When you want graceful failure and allow type conversions (e.g., int to double)
   * - :cpp:func:`as\<T\>() <tvm::ffi::Any::as>`
     - Returns ``std::optional<T>`` or ``const T*`` (for TVM-FFI object types)
     - When you need an exact type match with no conversions

.. dropdown:: Example of :cpp:func:`cast\<T\>() <tvm::ffi::Any::cast>`

  :cpp:func:`cast\<T\>() <tvm::ffi::Any::cast>` is the workhorse. It returns the value or throws:

  .. code-block:: cpp

    ffi::Any value = 42;
    int x = value.cast<int>();       // OK: 42
    double y = value.cast<double>(); // OK: 42.0 (int → double)

    try {
      ffi::String s = value.cast<ffi::String>();  // Throws TypeError
    } catch (const ffi::Error& e) {
      // "Cannot convert from type `int` to `ffi.Str`"
    }

.. dropdown:: Example of :cpp:func:`try_cast\<T\>() <tvm::ffi::Any::try_cast>`

  :cpp:func:`try_cast\<T\>() <tvm::ffi::Any::try_cast>` allows type coercion:

  .. code-block:: cpp

    ffi::Any value = 42;

    std::optional<double> opt_float = value.try_cast<double>();
    // opt_float.has_value() == true, *opt_float == 42.0

    std::optional<bool> opt_bool = value.try_cast<bool>();
    // opt_bool.has_value() == true, *opt_bool == true


.. dropdown:: Example of :cpp:func:`as\<T\>() <tvm::ffi::Any::as>`

  :cpp:func:`as\<T\>() <tvm::ffi::Any::as>` is strict - it succeeds only if the stored type matches exactly:

  .. code-block:: cpp

    ffi::Any value = 42;

    std::optional<int64_t> opt_int = value.as<int64_t>();
    // opt_int.has_value() == true

    std::optional<double> opt_float = value.as<double>();
    // opt_float.has_value() == false (int stored, not float)

    ffi::Any str_value = ffi::String("hello, world!");
    if (const ffi::Object* obj = str_value.as<ffi::Object>()) {
      // Use obj without copying
    }


Nullability Checks
~~~~~~~~~~~~~~~~~~

Compare with ``nullptr`` to check for ``None``:

.. code-block:: cpp

   ffi::Any value = std::nullopt;
   if (value == nullptr) {
     // Handle None case
   } else {
     // Process value
   }


Ownership
---------

The core distinction between :cpp:class:`tvm::ffi::Any` and
:cpp:class:`tvm::ffi::AnyView` is **ownership**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - :cpp:class:`~tvm::ffi::AnyView`
     - :cpp:class:`~tvm::ffi::Any`
   * - Ownership
     - Non-owning (like ``std::string_view``)
     - Owning (like ``std::string``)
   * - Reference counting
     - No reference count changes on copy
     - Increments reference count on copy; decrements on destroy
   * - Lifetime
     - Valid only while source lives
     - Extends object lifetime
   * - Primary use
     - Function inputs
     - Return values, storage

Code Examples
~~~~~~~~~~~~~~

:cpp:class:`~tvm::ffi::AnyView` is a lightweight, non-owning view. Copying it simply
copies 16 bytes with no reference count updates, making it ideal for passing arguments without overhead:

.. code-block:: cpp

   void process(ffi::AnyView value) {}

:cpp:class:`~tvm::ffi::Any` is an owning container. Copying an :cpp:class:`~tvm::ffi::Any` that holds an object
increments the reference count; destroying it decrements the count:

.. code-block:: cpp

   ffi::Any create_value() {
     ffi::Any result;
     {
       ffi::String str = "hello"; // refcount = 1 (str created)
       result = str;              // refcount 1 -> 2 (result owns str)
     }                            // refcount 2 -> 1 (str is destroyed)
     return result;               // refcount = 1 (result returns to caller)
   }


ABI Boundary
~~~~~~~~~~~~

TVM-FFI's function calling convention follows two simple rules:

- **Inputs are non-owning**: Arguments are passed as :cpp:class:`~tvm::ffi::AnyView`. The caller
  retains ownership, and the callee borrows them for the duration of the call.
- **Outputs are owning**: Return values are passed as :cpp:class:`~tvm::ffi::Any`. Ownership transfers
  to the caller, who becomes responsible for managing the value's lifetime.

.. code-block:: cpp

   // TVM-FFI C ABI
   int32_t tvm_ffi_c_abi(
     void* handle,
     const AnyView* args,   // (Non-owning) args: AnyView[num_args]
     int32_t num_args,
     Any* result,           // (Owning) result: Any (caller takes ownership)
   );

Destruction Semantics in C
~~~~~~~~~~~~~~~~~~~~~~~~~~

In C, which lacks RAII, you must manually destroy :cpp:class:`~tvm::ffi::Any` objects
by calling :cpp:func:`TVMFFIObjectDecRef` for heap-allocated objects.

.. code-block:: cpp

   void destroy_any(TVMFFIAny* any) {
     if (any->type_index >= kTVMFFIStaticObjectBegin) {
       // Decrement the reference count of the heap-allocated object
       TVMFFIObjectDecRef(any->v_obj);
     }
     *any = (TVMFFIAny){0};
   }

In contrast, destroying an :cpp:class:`~tvm::ffi::AnyView` is effectively a no-op - just clear its contents.

.. code-block:: cpp

   void destroy_any_view(TVMFFIAny* any_view) {
     *any_view = (TVMFFIAny){0};
   }


Layout
------

Tagged Union
~~~~~~~~~~~~

At C ABI level, every value lives in a :cpp:class:`TVMFFIAny`:

.. code-block:: cpp

   typedef struct TVMFFIAny {
     int32_t type_index;      // Bytes 0-3: identifies the stored type
     union {
       uint32_t zero_padding; // Bytes 4-7: must be zero (or small_str_len)
       uint32_t small_str_len;
     };
     union {                  // Bytes 8-15: the actual value
       int64_t v_int64;
       double v_float64;
       void* v_ptr;
       DLDataType v_dtype;
       DLDevice v_device;
       TVMFFIObject* v_obj;
       // ... other union members
     };
   } TVMFFIAny;


.. tip::

   Think of :cpp:class:`TVMFFIAny` as the "layout format", and :cpp:class:`~tvm::ffi::Any`/:cpp:class:`~tvm::ffi::AnyView`
   as a thin "application layer" that adds type safety, RAII, and ergonomic APIs, which has no change to the layout.

.. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/tvm-ffi/stable-c-abi-layout-any.svg
   :alt: Layout of the 128-bit Any tagged union
   :align: center

   Figure 1. Layout of the :cpp:class:`TVMFFIAny` tagged union in C ABI. :cpp:class:`~tvm::ffi::Any`/:cpp:class:`~tvm::ffi::AnyView`
   shares the same layout as :cpp:class:`TVMFFIAny`, but adds extra C++ APIs on top of it for type safety, RAII, and ergonomics.

It is effectively a layout-stable 16-byte tagged union.

* The first 4 bytes (:cpp:member:`TVMFFIAny::type_index`) serve as a tag identifying the stored type.
* The last 8 bytes hold the actual value - either stored inline for atomic types (e.g., ``int64_t``, ``float64``, ``void*``) or as a pointer to a heap-allocated object.

Atomic Types
~~~~~~~~~~~~

Primitive values - integers, floats, booleans, devices, and raw pointers - are stored
directly in the 8-byte payload with no heap allocation and no reference counting.

.. list-table:: Figure 2. Common atomic types stored directly in :cpp:class:`TVMFFIAny`
   :header-rows: 1
   :name: atomic-types-table
   :widths: 40 40 30

   * - Type
     - type_index
     - Payload Field
   * - ``None`` / ``nullptr``
     - :cpp:enumerator:`kTVMFFINone <TVMFFITypeIndex::kTVMFFINone>` = 0
     - :cpp:member:`~TVMFFIAny::v_int64` (must be 0)
   * - ``int64_t``
     - :cpp:enumerator:`kTVMFFIInt <TVMFFITypeIndex::kTVMFFIInt>` = 1
     - :cpp:member:`~TVMFFIAny::v_int64`
   * - ``bool``
     - :cpp:enumerator:`kTVMFFIBool <TVMFFITypeIndex::kTVMFFIBool>` = 2
     - :cpp:member:`~TVMFFIAny::v_int64` (0 or 1)
   * - ``float64_t``
     - :cpp:enumerator:`kTVMFFIFloat <TVMFFITypeIndex::kTVMFFIFloat>` = 3
     - :cpp:member:`~TVMFFIAny::v_float64`
   * - ``void*`` (opaque pointer)
     - :cpp:enumerator:`kTVMFFIOpaquePtr <TVMFFITypeIndex::kTVMFFIOpaquePtr>` = 4
     - :cpp:member:`~TVMFFIAny::v_ptr`
   * - :c:struct:`DLDataType <DLDataType>`
     - :cpp:enumerator:`kTVMFFIDataType <TVMFFITypeIndex::kTVMFFIDataType>` = 5
     - :cpp:member:`~TVMFFIAny::v_dtype`
   * - :c:struct:`DLDevice <DLDevice>`
     - :cpp:enumerator:`kTVMFFIDevice <TVMFFITypeIndex::kTVMFFIDevice>` = 6
     - :cpp:member:`~TVMFFIAny::v_device`
   * - :c:struct:`DLTensor* <DLTensor>`
     - :cpp:enumerator:`kTVMFFIDLTensorPtr <TVMFFITypeIndex::kTVMFFIDLTensorPtr>` = 7
     - :cpp:member:`~TVMFFIAny::v_ptr`
   * - ``const char*`` (raw string)
     - :cpp:enumerator:`kTVMFFIRawStr <TVMFFITypeIndex::kTVMFFIRawStr>` = 8
     - :cpp:member:`~TVMFFIAny::v_c_str`
   * - :cpp:class:`TVMFFIByteArray* <TVMFFIByteArray>`
     - :cpp:enumerator:`kTVMFFIByteArrayPtr <TVMFFITypeIndex::kTVMFFIByteArrayPtr>` = 9
     - :cpp:member:`~TVMFFIAny::v_ptr`

:ref:`Figure 2 <atomic-types-table>` shows common atomic types stored in-place inside the
:cpp:class:`TVMFFIAny` payload.

.. code-block:: cpp

   AnyView int_val = 42;                   // v_int64 = 42
   AnyView float_val = 3.14;               // v_float64 = 3.14
   AnyView bool_val = true;                // v_int64 = 1
   AnyView device = DLDevice{kDLCUDA, 0};  // v_device
   DLTensor tensor;
   AnyView view = &tensor;                 // v_ptr = &tensor

Note that raw pointers like :c:struct:`DLTensor* <DLTensor>` and ``char*`` also fit here.
These pointers carry no ownership, so the caller must ensure the pointed-to data outlives
the :cpp:class:`~tvm::ffi::AnyView` or :cpp:class:`~tvm::ffi::Any`.

Heap-Allocated Objects
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Figure 3. Common TVM-FFI object types stored as pointers in :cpp:member:`TVMFFIAny::v_obj`.
   :header-rows: 1
   :widths: 40 40 30

   * - Type
     - type_index
     - Payload Field
   * - :cpp:class:`ErrorObj* <tvm::ffi::ErrorObj>`
     - :cpp:enumerator:`kTVMFFIError <TVMFFITypeIndex::kTVMFFIError>` = 67
     - :cpp:member:`~TVMFFIAny::v_obj`
   * - :cpp:class:`FunctionObj* <tvm::ffi::FunctionObj>`
     - :cpp:enumerator:`kTVMFFIFunction <TVMFFITypeIndex::kTVMFFIFunction>` = 68
     - :cpp:member:`~TVMFFIAny::v_obj`
   * - :cpp:class:`TensorObj* <tvm::ffi::TensorObj>`
     - :cpp:enumerator:`kTVMFFITensor <TVMFFITypeIndex::kTVMFFITensor>` = 70
     - :cpp:member:`~TVMFFIAny::v_obj`
   * - :cpp:class:`ArrayObj* <tvm::ffi::ArrayObj>`
     - :cpp:enumerator:`kTVMFFIArray <TVMFFITypeIndex::kTVMFFIArray>` = 71
     - :cpp:member:`~TVMFFIAny::v_obj`
   * - :cpp:class:`MapObj* <tvm::ffi::MapObj>`
     - :cpp:enumerator:`kTVMFFIMap <TVMFFITypeIndex::kTVMFFIMap>` = 72
     - :cpp:member:`~TVMFFIAny::v_obj`
   * - :cpp:class:`ModuleObj* <tvm::ffi::ModuleObj>`
     - :cpp:enumerator:`kTVMFFIModule <TVMFFITypeIndex::kTVMFFIModule>` = 73
     - :cpp:member:`~TVMFFIAny::v_obj`


Heap-allocated objects - :cpp:class:`~tvm::ffi::String`, :cpp:class:`~tvm::ffi::Function`, :cpp:class:`~tvm::ffi::Tensor`, :cpp:class:`~tvm::ffi::Array`, :cpp:class:`~tvm::ffi::Map`, and custom types - are
stored as pointers to reference-counted :cpp:class:`TVMFFIObject` headers:

.. code-block:: cpp

   ffi::String str = "hello world";
   ffi::Any any_str = str;  // v_obj points to StringObj

   // Object layout in memory:
   // [TVMFFIObject header (24 bytes)][object-specific data]


Caveats
-------

Small String Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Strings and byte arrays receive special treatment: values of 7 bytes or fewer are stored
inline using **small string optimization**, avoiding heap allocation entirely:

.. code-block:: cpp

   ffi::Any small = "hello";                    // kTVMFFISmallStr, in v_bytes
   ffi::Any large = "this is a longer string";  // kTVMFFIStr, heap allocated


Further Reading
---------------

- **Object system**: :doc:`object_and_class` covers how TVM-FFI objects work, including reference counting and type checking
- **Function system**: :doc:`func_module` covers function calling conventions and the global registry
- **C examples**: :doc:`../get_started/stable_c_abi` demonstrates working with :cpp:class:`TVMFFIAny` directly in C
- **Tensor conversions**: :doc:`tensor` covers how tensors flow through :cpp:class:`~tvm::ffi::Any` and :cpp:class:`~tvm::ffi::AnyView`
