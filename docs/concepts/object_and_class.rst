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

Object and Class
================

TVM-FFI provides a unified object system that enables cross-language interoperability
between C++, Python, and Rust. The object system is built around :cpp:class:`tvm::ffi::Object`
and :cpp:class:`tvm::ffi::ObjectRef`, which together form the foundation for:

- **Type-safe runtime type identification** without relying on C++ RTTI
- **Intrusive reference counting** for smart memory management
- **Reflection-based class exposure** across programming languages
- **Serialization and deserialization** via reflection metadata

This tutorial covers defining, using, and extending TVM-FFI objects across languages.

Glossary
--------

:cpp:class:`tvm::ffi::Object`
  A heap-allocated, reference-counted container.
  All TVM-FFI objects inherit from this base class and share a common 24-byte header
  that stores reference counts, type index, and a deleter callback.

:cpp:class:`tvm::ffi::ObjectRef`
  An intrusive pointer that manages an :cpp:class:`~tvm::ffi::Object`'s lifetime through reference counting.
  Its subclasses provide type-safe access to specific object types. In its low-level implementation, it is
  equivalent to a normal C++ pointer to a heap-allocated :cpp:class:`~tvm::ffi::Object`.

Type index and type key
  Type index is an integer that uniquely identifies each object type.
  Built-in types have statically assigned indices defined in :cpp:enum:`TVMFFITypeIndex`,
  while user-defined types receive indices at startup when first accessed.
  Type key is a unique string identifier (e.g., ``"my_ext.MyClass"``) that names an object type.
  It is used for registration, serialization, and cross-language mapping.

Common Usage
------------

Define a Class in C++
~~~~~~~~~~~~~~~~~~~~~

To define a custom object class in normal C++, inherit it from :cpp:class:`tvm::ffi::Object` or its subclasses,
and then add one of the following macros that declares its metadata:

 :c:macro:`TVM_FFI_DECLARE_OBJECT_INFO(TypeKey, TypeName, ParentType) <TVM_FFI_DECLARE_OBJECT_INFO>`
   Declare an object type that can be subclassed. Type index is assigned dynamically.

 :c:macro:`TVM_FFI_DECLARE_OBJECT_INFO_FINAL(TypeKey, TypeName, ParentType) <TVM_FFI_DECLARE_OBJECT_INFO_FINAL>`
   Declare a final object type (no subclasses). Enables faster type checking.

**Example**. The code below shows a minimal example of defining a TVM-FFI object class.
It declares a class ``MyObjectObj`` that inherits from :cpp:class:`~tvm::ffi::Object`.

.. code-block:: cpp

   #include <tvm/ffi/tvm_ffi.h>

   namespace ffi = tvm::ffi;

   class MyObjectObj : public ffi::Object {
    public:
     // Normal C++ code: Declare fields, methods, constructor, destructor, etc.
     int64_t value;
     ffi::String name;

     MyObjectObj(int64_t value, ffi::String name) : value(value), name(std::move(name)) {}

     int64_t GetValue() const { return value; }

     void AddToValue(int64_t other) { value += other; }

     // Declare object type info
     TVM_FFI_DECLARE_OBJECT_INFO(
       /*type_key=*/"my_ext.MyObject",
       /*type_name=*/MyObjectObj,
       /*parent_type=*/ffi::Object);
   };

**Managed reference**. Optionally, a managed reference class can be defined by inheriting
from :cpp:class:`~tvm::ffi::ObjectRef` and using one of the following macros to define the methods.
Define its constructor by wrapping the :cpp:func:`tvm::ffi::make_object` function.

 :c:macro:`TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TypeName, ParentType, ObjectName) <TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE>`
   Define a nullable reference class.

 :c:macro:`TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TypeName, ParentType, ObjectName) <TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE>`
   Define a non-nullable reference class.

For example, a non-nullable reference class ``MyObject`` can be defined as follows:

.. code-block:: cpp

   class MyObject : public ffi::ObjectRef {
    public:
     MyObject(int64_t value, ffi::String name)
         : ObjectRef(ffi::make_object<MyObjectObj>(value, std::move(name))) {}
     TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(
        /*type_name=*/MyObject,
        /*parent_type=*/ffi::ObjectRef,
        /*object_name=*/MyObjectObj);
   };

   // Create a managed object
   MyObject obj = MyObject(42, "hello");

   // Access fields via operator->
   std::cout << obj->value << std::endl;  // -> 42


Expose a Class in Python
~~~~~~~~~~~~~~~~~~~~~~~~

**Reflection**. The object's metadata is used for reflection. Use :cpp:class:`tvm::ffi::reflection::ObjectDef`
to register an object's constructor, fields, and methods.

.. code-block:: cpp

   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = tvm::ffi::reflection;
     refl::ObjectDef<MyObjectObj>()
         // Register constructor with signature
         .def(refl::init<int64_t, ffi::String>())
         // Register read-write fields
         .def_rw("value", &MyObjectObj::value, "The integer value")
         .def_rw("name", &MyObjectObj::name, "The name string")
         // Register methods
         .def("get_value", &MyObjectObj::GetValue, "Returns the value");
   }

**Python binding**. After registration, the object is automatically available in Python. Use
:py:func:`tvm_ffi.register_object` to bind a Python class to a registered C++ type:

.. code-block:: python

   import tvm_ffi
   from typing import TYPE_CHECKING

   @tvm_ffi.register_object("my_ext.MyObject")
   class MyObject(tvm_ffi.Object):
       # tvm-ffi-stubgen(begin): object/my_ext.MyObject
       value: int
       name: str

       if TYPE_CHECKING:
         def __init__(self, value: int, name: str) -> None: ...
         def get_value(self) -> int: ...
       # tvm-ffi-stubgen(end)

   # Create and use objects
   obj = MyObject(42, "hello")
   print(obj.value)        # -> 42
   print(obj.get_value())  # -> 42
   obj.value = 100         # Mutable field access

The decorator looks up the type key ``"my_ext.MyObject"`` in the C++ type registry and binds the Python class to it.
Fields and methods registered via :cpp:class:`~tvm::ffi::reflection::ObjectDef` are automatically available on the Python class.

The tool ``tvm-ffi-stubgen`` automatically generates the Python type stubs (the code between the markers)
from reflection metadata. See :ref:`Stub Generation Tool <sec-stubgen>` for details.


.. _type-checking-and-casting:

Type Checking and Casting
~~~~~~~~~~~~~~~~~~~~~~~~~

**Type checking**. Use :cpp:func:`Object::IsInstance\<T\>() <tvm::ffi::Object::IsInstance>`
for runtime type checking:

.. code-block:: cpp

   bool CheckType(const ffi::ObjectRef& obj) {
     if (obj->IsInstance<MyObjectObj>()) {
       // obj is a MyObjectObj or subclass
       return true;
     }
     return false;
   }

**Type casting**. Use :cpp:func:`ObjectRef::as\<T\>() <tvm::ffi::ObjectRef::as>` for safe downcasting:

.. code-block:: cpp

   ffi::ObjectRef obj = ...;

   // as<ObjectType>() returns a pointer (nullptr if type doesn't match)
   if (const MyObjectObj* ptr = obj.as<MyObjectObj>()) {
     std::cout << ptr->value << std::endl;
   }

   // as<ObjectRefType>() returns std::optional
   if (auto opt = obj.as<MyObject>()) {
     std::cout << opt->get()->value << std::endl;
   }

**Type info**. Type index is available via :cpp:func:`ObjectRef::type_index() <tvm::ffi::ObjectRef::type_index>`
and type key is available via :cpp:func:`ObjectRef::GetTypeKey() <tvm::ffi::ObjectRef::GetTypeKey>`. These methods
can be used to safely identify object types without relying on C++ RTTI.

.. note::
  C++ RTTI (e.g. ``typeid``, ``dynamic_cast``) is strictly not useful in TVM-FFI-based approaches.

Miscellaneous APIs
~~~~~~~~~~~~~~~~~~

**C++ Serialization**. Use :cpp:func:`tvm::ffi::ToJSONGraph` to serialize an object to a JSON value,
and :cpp:func:`tvm::ffi::FromJSONGraph` to deserialize a JSON value to an object.

.. code-block:: cpp

   #include <tvm/ffi/extra/serialization.h>

   // Serialize to JSON
   ffi::Any obj = ...;
   ffi::json::Value json = ffi::ToJSONGraph(obj);

   // Deserialize from JSON
   ffi::Any restored = ffi::FromJSONGraph(json);

**Python Serialization**. Pickle is overloaded in Python to support TVM-FFI object serialization.
Or explicitly use the :py:func:`tvm_ffi.serialization.to_json_graph_str`
and :py:func:`tvm_ffi.serialization.from_json_graph_str` to serialize and deserialize an object to a JSON string.

.. code-block:: python

   import pickle

   obj = MyObject(42, "test")
   data = pickle.dumps(obj)
   restored = pickle.loads(data)

**Convert between raw and managed references**. Use :cpp:func:`tvm::ffi::GetRef` to convert a raw object pointer to a managed reference,
and :cpp:func:`tvm::ffi::ObjectRef::get` to convert a managed reference to a raw object pointer.

ABI and Layout
--------------

**Stable C Layout**. All subclasses of :cpp:class:`tvm::ffi::Object` share a common 24-byte header (:cpp:class:`TVMFFIObject`)
containing reference counts, type index, and a deleter callback.
See :ref:`abi-object` for the C struct definition. :cpp:class:`tvm::ffi::ObjectRef` and :cpp:class:`tvm::ffi::ObjectPtr` are smart pointers
equivalent to a single ``void*`` pointer.


.. _object-reference-counting:

Reference Counting
~~~~~~~~~~~~~~~~~~

.. seealso::

   :ref:`abi-object-ownership` for C code examples.

**Intrusive reference counting**. The reference count is stored directly in the object header, not in a separate control block.
This design reduces memory overhead and improves cache locality. The :cpp:member:`TVMFFIObject::combined_ref_count`
field packs both strong (lower 32 bits) and weak (upper 32 bits) reference counts in a single 64-bit integer.

C APIs are provided to manipulate the reference count:

- :cpp:func:`TVMFFIObjectIncRef` to increase the strong reference count
- :cpp:func:`TVMFFIObjectDecRef` to decrease the strong reference count

**Deleter**. When an object is managed by :cpp:class:`~tvm::ffi::ObjectRef`, the ``deleter`` callback is invoked:

- When strong reference count reaches zero: the object's destructor is called.
- When weak reference count reaches zero: the memory is freed.

The flags in :cpp:enum:`TVMFFIObjectDeleterFlagBitMask` indicate which action to perform.

.. _object-conversion-with-any:

**Conversion with Any**. At the stable C ABI boundary, TVM-FFI passes values using :cpp:class:`Any <tvm::ffi::Any>`
(owning) or :cpp:class:`AnyView <tvm::ffi::AnyView>` (non-owning). Object handles are stored in the
:cpp:member:`TVMFFIAny::v_obj` field with a type index >= :cpp:enumerator:`kTVMFFIStaticObjectBegin <TVMFFITypeIndex::kTVMFFIStaticObjectBegin>`.

See :ref:`abi-object-ownership` for C code examples demonstrating:

- Extracting an object handle from :cpp:class:`TVMFFIAny`
- Storing an object handle into :cpp:class:`~tvm::ffi::Any` or :cpp:class:`~tvm::ffi::AnyView`
- Managing ownership via reference counting

Object Type Registry
--------------------

TVM-FFI maintains a global type registry that keeps track of all registered object types,
their inheritance relationships, and their reflection metadata.

Inheritance and Type Casting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
    Only single inheritance is supported in TVM-FFI Object system.

TVM-FFI implements its own runtime type system that enables type-safe operations
without relying on C++ RTTI. Every object carries a runtime type index in its header.

**Example**. Code below shows a minimal example of defining a base class and a derived class.

.. code-block:: cpp

   class MyBaseObj : public ffi::Object {
    public:
     TVM_FFI_DECLARE_OBJECT_INFO("my_ext.MyBase", MyBaseObj, ffi::Object);
   };

   class MyDerivedObj : public MyBaseObj {
    public:
     // Final class: no subclasses allowed
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("my_ext.MyDerived", MyDerivedObj, MyBaseObj);
   };


Registration happens automatically on first access. The :c:macro:`TVM_FFI_DECLARE_OBJECT_INFO`
and :c:macro:`TVM_FFI_DECLARE_OBJECT_INFO_FINAL` macros use :cpp:func:`TVMFFITypeGetOrAllocIndex`
internally to allocate a type index.

See :ref:`type-checking-and-casting` for how to use the type system.

Reflect Fields and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

The reflection system enables cross-language exposure of C++ classes, their fields,
and methods. Use :cpp:class:`ObjectDef\<T\> <tvm::ffi::reflection::ObjectDef>`
to register reflection metadata for object type ``T``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``.def(init<Args...>())``
     - Register a constructor with the given argument types
   * - ``.def_ro("name", &T::field)``
     - Register a read-only field
   * - ``.def_rw("name", &T::field)``
     - Register a read-write field
   * - ``.def("name", &T::method)``
     - Register a member method
   * - ``.def_static("name", &func)``
     - Register a static method

**Example**. Code below shows a minimal example of registering a class with reflection metadata.

.. code-block:: cpp

   class IntPairObj : public ffi::Object {
    public:
     int64_t a;
     int64_t b;

     IntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

     int64_t Sum() const { return a + b; }

     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("my_ext.IntPair", IntPairObj, ffi::Object);
   };

   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = tvm::ffi::reflection;
     refl::ObjectDef<IntPairObj>()
         .def(refl::init<int64_t, int64_t>())
         .def_rw("a", &IntPairObj::a, "the first field")
         .def_rw("b", &IntPairObj::b, "the second field")
         .def("sum", &IntPairObj::Sum, "compute a + b");
   }


**Metadata and Documentation**. Add documentation strings and custom metadata to fields and methods:

.. code-block:: cpp

   // The following example uses MyObjectObj defined earlier to show
   // how to add documentation and metadata.
   refl::ObjectDef<MyObjectObj>()
       .def_rw("value", &MyObjectObj::value,
               "The numeric value",                    // docstring
               refl::DefaultValue(0),                  // default value
               refl::Metadata{{"min", 0}, {"max", 100}})  // custom metadata
       .def("add_to_value", &MyObjectObj::AddToValue,
            "Add a value to the object's value field");


Python Interoperability
~~~~~~~~~~~~~~~~~~~~~~~

**Cross-language lifetime**. Each Python :py:class:`tvm_ffi.Object` instance holds a C handle
(``void*``) that references the underlying C++ object. The Python wrapper increments the
reference count when constructed and decrements when garbage collected.

.. code-block:: python

   obj = MyObject(42, "test")    # C++ object created, C++ refcount = 1
   obj2 = obj                    # Python alias created, C++ refcount unchanged
   del obj                       # Python alias removed, C++ refcount unchanged
   del obj2                      # Last Python reference gone, C++ refcount -> 0, object destroyed


Further Reading
---------------

- :doc:`any`: How objects are stored in :cpp:class:`~tvm::ffi::Any` containers
- :doc:`func_module`: Function objects and the global registry
- :doc:`tensor`: Tensor objects and DLPack interoperability
- :doc:`abi_overview`: Low-level C ABI details for the object system
- :doc:`../packaging/python_packaging`: Packaging C++ objects for Python wheels
