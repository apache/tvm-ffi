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

.. _dataclass-reflection:

Dataclass-Style Reflection
==========================

TVM-FFI's reflection system provides Python-dataclass-style features for C++
classes: auto-generated constructors, default values, keyword-only parameters,
repr, hashing, comparison, and deep copy. These features are enabled by
per-field and per-class traits registered via
:cpp:class:`~tvm::ffi::reflection::ObjectDef`.

This guide assumes familiarity with :doc:`export_func_cls` and
:doc:`../concepts/object_and_class`.


Quick Start
-----------

Define a C++ object with fields, register it with traits, and use it from
Python with full dataclass semantics:

.. code-block:: cpp

   #include <tvm/ffi/tvm_ffi.h>

   namespace ffi = tvm::ffi;

   class PointObj : public ffi::Object {
    public:
     int64_t x;
     int64_t y;
     ffi::String label;

     static constexpr bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("my_ext.Point", PointObj, ffi::Object);
   };

   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = ffi::reflection;
     refl::ObjectDef<PointObj>()
         .def_rw("x", &PointObj::x)
         .def_rw("y", &PointObj::y)
         .def_rw("label", &PointObj::label, refl::default_(""));
   }

No ``refl::init<>()`` call is needed — the reflection system auto-generates a
packed ``__ffi_init__`` from the reflected fields:

.. code-block:: python

   import tvm_ffi

   @tvm_ffi.register_object("my_ext.Point")
   class Point(tvm_ffi.Object):
       x: int
       y: int
       label: str

   p = Point(1, 2)                # positional args
   p = Point(1, 2, label="origin")  # keyword arg with default


.. _auto-init:

Auto-Generated Constructors
----------------------------

When no explicit ``refl::init<Args...>()`` is registered, ``ObjectDef``
auto-generates a packed constructor (``__ffi_init__``) from the reflected
fields. The generated signature follows Python conventions:

1. **Required positional** parameters come first.
2. **Optional positional** parameters (those with defaults) come next.
3. **Keyword-only** parameters follow after a ``*`` separator.

Field Traits for Init
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Trait
     - Effect on auto-init
   * - ``refl::default_(value)``
     - Makes the parameter optional with a literal default value.
   * - ``refl::default_factory(fn)``
     - Makes the parameter optional; calls ``fn()`` each time a default is
       needed. Use for mutable defaults (e.g. ``Array``, ``Dict``).
   * - ``refl::kw_only(true)``
     - Moves the parameter after the ``*`` separator (keyword-only).
   * - ``refl::init(false)``
     - Excludes the field from the constructor entirely. The field must have
       a default value or be initialized by a base-class constructor.

.. code-block:: cpp

   refl::ObjectDef<ConfigObj>()
       .def_rw("batch_size", &ConfigObj::batch_size)
       .def_rw("lr", &ConfigObj::lr, refl::default_(0.001))
       .def_rw("device", &ConfigObj::device, refl::kw_only(true),
               refl::default_("cpu"))
       .def_rw("_cache", &ConfigObj::_cache, refl::init(false),
               refl::default_factory([] { return ffi::Dict(); }));

The generated Python signature is:

.. code-block:: python

   def __init__(self, batch_size, lr=0.001, *, device="cpu"):
       ...
   # _cache is excluded from __init__ but initialized to Dict()

Suppressing Auto-Init
~~~~~~~~~~~~~~~~~~~~~

Pass ``refl::init(false)`` at the class level to suppress auto-init entirely:

.. code-block:: cpp

   refl::ObjectDef<InternalObj>(refl::init(false))
       .def_rw("x", &InternalObj::x)
       .def_rw("y", &InternalObj::y);

The object will have no ``__ffi_init__`` method. Construction must happen
through a custom factory or from C++.

Explicit Constructors
~~~~~~~~~~~~~~~~~~~~~

Use ``refl::init<Args...>()`` to register an explicit typed constructor
instead of auto-init:

.. code-block:: cpp

   refl::ObjectDef<IntPairObj>()
       .def(refl::init<int64_t, int64_t>())
       .def_ro("a", &IntPairObj::a)
       .def_ro("b", &IntPairObj::b);

This calls ``IntPairObj(int64_t, int64_t)`` directly. Auto-init is
automatically suppressed when an explicit constructor is registered.


Default Values
--------------

Literal Defaults
~~~~~~~~~~~~~~~~

``refl::default_(value)`` stores a literal default. The value is captured once
at registration time:

.. code-block:: cpp

   .def_rw("threshold", &Obj::threshold, refl::default_(0.5))

Factory Defaults
~~~~~~~~~~~~~~~~

``refl::default_factory(fn)`` calls ``fn()`` each time a default is needed.
Use this for mutable containers to avoid aliasing:

.. code-block:: cpp

   .def_rw("items", &Obj::items,
           refl::default_factory([] { return ffi::List<ffi::String>(); }))

.. note::

   ``refl::default_`` and ``refl::default_factory`` are the preferred names for
   new code. The original names ``refl::DefaultValue`` and
   ``refl::DefaultFactory`` are retained for backward compatibility.


.. _field-traits-ref:

Field Traits Reference
----------------------

All traits are passed as extra arguments to ``def_ro`` or ``def_rw``.
Multiple traits can be combined on a single field:

.. code-block:: cpp

   .def_rw("name", &Obj::name,
           refl::default_("unnamed"),
           refl::kw_only(true),
           refl::repr(false))

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Trait
     - Description
   * - ``refl::init(bool)``
     - Include/exclude field from auto-generated ``__ffi_init__``.
   * - ``refl::kw_only(bool)``
     - Mark field as keyword-only in auto-init.
   * - ``refl::default_(value)``
     - Literal default value for the field.
   * - ``refl::default_factory(fn)``
     - Factory function ``() -> Any`` for mutable defaults.
   * - ``refl::repr(bool)``
     - Include/exclude field from repr output.
   * - ``refl::hash(bool)``
     - Include/exclude field from recursive hashing.
   * - ``refl::compare(bool)``
     - Include/exclude field from recursive comparison.
   * - ``refl::Metadata({...})``
     - Attach custom key-value metadata to the field.


.. _dataclass-operations:

Dataclass Operations
--------------------

Once a class is registered with ``ObjectDef``, several dataclass operations are
available automatically. These are defined in
``include/tvm/ffi/extra/dataclass.h``:

.. code-block:: cpp

   #include <tvm/ffi/extra/dataclass.h>

   ffi::Any value = ...;
   ffi::Any copy = ffi::DeepCopy(value);
   ffi::String repr = ffi::ReprPrint(value);
   int64_t h = ffi::RecursiveHash(value);
   bool eq = ffi::RecursiveEq(a, b);

All operations use iterative DFS with an explicit stack (no recursion), so they
are safe for deep object graphs.

Deep Copy
~~~~~~~~~

``DeepCopy(value)`` recursively copies an object and all reachable objects in
its graph. Objects that register a copy constructor via ``ObjectDef``
automatically support deep copy (through the ``__ffi_shallow_copy__`` type
attribute).

- **Immutable leaves** (returned as-is): primitives, ``String``, ``Bytes``,
  ``Shape``
- **Recursively copied**: ``Array``, ``List``, ``Map``, ``Dict``, and
  reflected objects

Repr
~~~~

``ReprPrint(value)`` produces a human-readable string representation using
field names and values from reflection metadata. It handles cycles (prints
``...``) and DAG structures (caches repr strings).

Exclude a field from repr output with ``refl::repr(false)``:

.. code-block:: cpp

   .def_ro("internal_state", &Obj::internal_state, refl::repr(false))

Hashing
~~~~~~~

``RecursiveHash(value)`` computes a deterministic recursive hash. The hash is
consistent with equality: if ``RecursiveEq(a, b)`` then
``RecursiveHash(a) == RecursiveHash(b)``.

Exclude a field from hashing with ``refl::hash(false)``:

.. code-block:: cpp

   .def_ro("cache_key", &Obj::cache_key, refl::hash(false))

Comparison
~~~~~~~~~~

``RecursiveEq(a, b)`` tests structural equality. Ordering comparisons
(``RecursiveLt``, ``RecursiveLe``, ``RecursiveGt``, ``RecursiveGe``) provide
lexicographic field-by-field ordering.

Exclude a field from comparison with ``refl::compare(false)``:

.. code-block:: cpp

   .def_ro("timestamp", &Obj::timestamp, refl::compare(false))

In Python, these are wired up as ``__eq__``, ``__lt__``, ``__le__``, ``__gt__``,
``__ge__``, and ``__hash__`` on classes created with ``c_class``.


Custom Hooks
------------

Override the default behavior for repr, hash, or comparison by registering
type-level attributes via :cpp:class:`~tvm::ffi::reflection::TypeAttrDef`:

.. code-block:: cpp

   namespace refl = ffi::reflection;

   // Custom hash: only hash the "key" field
   refl::TypeAttrDef<MyObj>().def(
       refl::type_attr::kHash,
       [](const Object* self, const Function& fn_hash) -> int64_t {
         auto* obj = static_cast<const MyObj*>(self);
         return fn_hash(AnyView(obj->key)).cast<int64_t>();
       });

   // Custom equality: only compare "key" fields
   refl::TypeAttrDef<MyObj>().def(
       refl::type_attr::kEq,
       [](const Object* lhs, const Object* rhs, const Function& fn_eq) -> bool {
         auto* a = static_cast<const MyObj*>(lhs);
         auto* b = static_cast<const MyObj*>(rhs);
         return fn_eq(AnyView(a->key), AnyView(b->key)).cast<bool>();
       });

   // Custom three-way comparison
   refl::TypeAttrDef<MyObj>().def(
       refl::type_attr::kCompare,
       [](const Object* lhs, const Object* rhs, const Function& fn_cmp) -> int32_t {
         auto* a = static_cast<const MyObj*>(lhs);
         auto* b = static_cast<const MyObj*>(rhs);
         return fn_cmp(AnyView(a->key), AnyView(b->key)).cast<int32_t>();
       });

   // Custom repr
   refl::TypeAttrDef<MyObj>().def(
       refl::type_attr::kRepr,
       [](const Object* self, const Function& fn_repr) -> String {
         auto* obj = static_cast<const MyObj*>(self);
         return "MyObj(key=" + fn_repr(AnyView(obj->key)).cast<String>() + ")";
       });

.. list-table:: Type attribute hooks
   :header-rows: 1
   :widths: 30 30 40

   * - Attribute
     - Constant
     - Signature
   * - ``__ffi_hash__``
     - ``type_attr::kHash``
     - ``(const Object* self, Function fn_hash) -> int64_t``
   * - ``__ffi_eq__``
     - ``type_attr::kEq``
     - ``(const Object* lhs, const Object* rhs, Function fn_eq) -> bool``
   * - ``__ffi_compare__``
     - ``type_attr::kCompare``
     - ``(const Object* lhs, const Object* rhs, Function fn_cmp) -> int32_t``
   * - ``__ffi_repr__``
     - ``type_attr::kRepr``
     - ``(const Object* self, Function fn_repr) -> String``

The ``fn_hash``, ``fn_eq``, ``fn_cmp``, and ``fn_repr`` callbacks are provided
by the framework for recursing into child values.


Python ``c_class`` Decorator
-----------------------------

On the Python side, ``tvm_ffi.dataclasses.c_class`` provides equivalent
functionality to Python's ``@dataclass`` for C++-backed objects:

.. code-block:: python

   from tvm_ffi.dataclasses import c_class

   @c_class("my_ext.Point")
   class Point(tvm_ffi.Object):
       x: int
       y: int
       label: str

The decorator:

1. Looks up the C++ type info registered by ``ObjectDef<PointObj>``.
2. Matches Python annotations to C++ fields.
3. Generates ``__init__`` from ``__ffi_init__``, respecting ``kw_only``,
   defaults, and ``init(false)`` settings from C++.
4. Installs ``__copy__``, ``__deepcopy__``, ``__eq__``, ``__hash__``,
   ``__repr__``, and comparison operators.

.. note::

   ``@tvm_ffi.register_object`` can also be used, which delegates to
   ``c_class`` internally for objects with reflected fields.


Inheritance
-----------

Dataclass traits compose across inheritance. A child class inherits the
parent's fields and adds its own:

.. code-block:: cpp

   class ParentObj : public ffi::Object {
    public:
     int64_t parent_required;
     int64_t parent_default;

     static constexpr bool _type_mutable = true;
     static constexpr uint32_t _type_child_slots = 1;
     TVM_FFI_DECLARE_OBJECT_INFO("my_ext.Parent", ParentObj, ffi::Object);
   };

   class ChildObj : public ParentObj {
    public:
     int64_t child_required;
     int64_t child_kw_only;

     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("my_ext.Child", ChildObj, ParentObj);
   };

   TVM_FFI_STATIC_INIT_BLOCK() {
     namespace refl = ffi::reflection;

     refl::ObjectDef<ParentObj>()
         .def_rw("parent_required", &ParentObj::parent_required)
         .def_rw("parent_default", &ParentObj::parent_default,
                 refl::default_(int64_t{5}));

     refl::ObjectDef<ChildObj>()
         .def_rw("child_required", &ChildObj::child_required)
         .def_rw("child_kw_only", &ChildObj::child_kw_only, refl::kw_only(true));
   }

In Python, the child's auto-init includes all fields:

.. code-block:: python

   # Generated signature:
   # def __init__(self, parent_required, child_required, parent_default=5, *, child_kw_only):
   child = Child(1, 2, child_kw_only=3)
   assert child.parent_required == 1
   assert child.child_required == 2
   assert child.parent_default == 5    # uses default
   assert child.child_kw_only == 3


Further Reading
---------------

- :doc:`export_func_cls`: Basic function and class export guide
- :doc:`../concepts/object_and_class`: Object system fundamentals, type registry, and ABI
- :ref:`sec-stubgen`: Auto-generating Python type stubs from reflection metadata
- :doc:`cpp_lang_guide`: Full C++ API guide covering ``Any``, ``Function``, containers
