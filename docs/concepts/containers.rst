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

Containers
==========

TVM-FFI provides five built-in container types for storing and exchanging
collections of values across C++, Python, and Rust. They are all
heap-allocated, reference-counted objects that can be stored in
:cpp:class:`~tvm::ffi::Any` and passed through the FFI boundary.

The containers split into two categories: **immutable** containers that use
copy-on-write semantics, and **mutable** containers that use shared-reference
semantics.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 14 20 20 16 30

   * - Type
     - C++ Class
     - Python Class
     - Mutability
     - Semantics
   * - Array
     - :cpp:class:`Array\<T\> <tvm::ffi::Array>`
     - :py:class:`tvm_ffi.Array`
     - Immutable
     - Homogeneous sequence with copy-on-write
   * - List
     - :cpp:class:`List\<T\> <tvm::ffi::List>`
     - :py:class:`tvm_ffi.List`
     - Mutable
     - Homogeneous sequence with shared-reference
   * - Tuple
     - :cpp:class:`Tuple\<Ts...\> <tvm::ffi::Tuple>`
     - (backed by :py:class:`tvm_ffi.Array`)
     - Immutable
     - Heterogeneous fixed-size sequence (backed by ArrayObj)
   * - Map
     - :cpp:class:`Map\<K, V\> <tvm::ffi::Map>`
     - :py:class:`tvm_ffi.Map`
     - Immutable
     - Homogeneous key-value mapping with copy-on-write
   * - Dict
     - :cpp:class:`Dict\<K, V\> <tvm::ffi::Dict>`
     - :py:class:`tvm_ffi.Dict`
     - Mutable
     - Homogeneous key-value mapping with shared-reference

Immutable Containers (Copy-on-Write)
-------------------------------------

Array
~~~~~

``Array<T>`` is an immutable homogeneous sequence backed by
:cpp:class:`~tvm::ffi::ArrayObj`. It implements **copy-on-write** semantics:
when a mutation method is called in C++ (e.g. ``push_back``, ``Set``), the
array checks whether the backing storage is uniquely owned. If it is shared
with other handles, it copies the data first so that existing handles are
unaffected.

.. code-block:: cpp

   ffi::Array<int> a = {1, 2, 3};
   ffi::Array<int> b = a;       // b shares the same ArrayObj
   a.push_back(4);              // copy-on-write: a gets a new backing storage
   assert(a.size() == 4);
   assert(b.size() == 3);       // b is unchanged

In Python, :py:class:`tvm_ffi.Array` implements ``collections.abc.Sequence``
(read-only). When a Python ``list`` or ``tuple`` is passed to an FFI function,
it is automatically converted to ``Array``.

Tuple
~~~~~

``Tuple<T1, T2, ...>`` is an immutable heterogeneous fixed-size sequence. It is
backed by the same :cpp:class:`~tvm::ffi::ArrayObj` as ``Array``, but provides
compile-time type safety for each element position via C++ variadic templates.

.. code-block:: cpp

   ffi::Tuple<int, ffi::String, bool> t(42, "hello", true);
   int x = t.get<0>();              // 42
   ffi::String s = t.get<1>();      // "hello"

In Python, ``Tuple`` does not have a separate class -- Python tuples passed
through the FFI are converted to ``Array``.

Map
~~~

``Map<K, V>`` is an immutable homogeneous key-value mapping backed by
:cpp:class:`~tvm::ffi::MapObj`. It implements **copy-on-write** semantics
(same principle as ``Array``). Insertion order is preserved.

.. code-block:: cpp

   ffi::Map<ffi::String, int> m = {{"Alice", 100}, {"Bob", 95}};
   ffi::Map<ffi::String, int> m2 = m;  // m2 shares the same MapObj
   m.Set("Charlie", 88);               // copy-on-write
   assert(m.size() == 3);
   assert(m2.size() == 2);             // m2 is unchanged

In Python, :py:class:`tvm_ffi.Map` implements ``collections.abc.Mapping``
(read-only). When a Python ``dict`` is passed to an FFI function, it is
automatically converted to ``Map``.

Mutable Containers (Shared Reference)
--------------------------------------

List
~~~~

``List<T>`` is a mutable homogeneous sequence backed by
:cpp:class:`~tvm::ffi::ListObj`. Unlike ``Array``, it does **not** use
copy-on-write. Mutations happen directly on the underlying shared object, and
**all handles** sharing the same ``ListObj`` see the mutations immediately.

.. code-block:: cpp

   ffi::List<int> a = {1, 2, 3};
   ffi::List<int> b = a;       // b shares the same ListObj
   a.push_back(4);             // in-place mutation
   assert(a.size() == 4);
   assert(b.size() == 4);      // b sees the mutation

In Python, :py:class:`tvm_ffi.List` implements
``collections.abc.MutableSequence`` and supports ``append``, ``insert``,
``__setitem__``, ``__delitem__``, ``pop``, ``reverse``, ``clear``, and
``extend``.

Dict
~~~~

``Dict<K, V>`` is a mutable homogeneous key-value mapping backed by
:cpp:class:`~tvm::ffi::DictObj`. Like ``List``, mutations happen directly on
the shared object with no copy-on-write.

.. code-block:: cpp

   ffi::Dict<ffi::String, int> d = {{"Alice", 100}};
   ffi::Dict<ffi::String, int> d2 = d;  // d2 shares the same DictObj
   d.Set("Bob", 95);                    // in-place mutation
   assert(d.size() == 2);
   assert(d2.size() == 2);              // d2 sees the mutation

In Python, :py:class:`tvm_ffi.Dict` implements
``collections.abc.MutableMapping`` and supports ``__setitem__``,
``__delitem__``, ``pop``, ``clear``, and ``update``.

When to Use Each Type
---------------------

.. list-table::
   :header-rows: 1
   :widths: 50 20

   * - Use Case
     - Container
   * - Immutable snapshot of a sequence (e.g. function arguments)
     - Array
   * - Building up or modifying a sequence in-place
     - List
   * - Fixed heterogeneous collection (e.g. a multi-typed return value)
     - Tuple
   * - Immutable key-value lookup (e.g. configuration)
     - Map
   * - Mutable key-value store (e.g. accumulating results)
     - Dict

Thread Safety
-------------

**Immutable containers** (``Array``, ``Tuple``, ``Map``) can be safely shared
across threads for **read-only** access. Copy-on-write mutations are
thread-safe because they create a new backing object when the storage is shared.

**Mutable containers** (``List``, ``Dict``) are **NOT** thread-safe. If
multiple threads need to read or write the same ``List`` or ``Dict``, external
synchronization (e.g. a mutex) is required.

Further Reading
---------------

- :doc:`../guides/cpp_lang_guide`: C++ examples for all container types
- :doc:`../guides/python_lang_guide`: Python examples and conversion rules
- :doc:`any`: How containers are stored in the type-erased :cpp:class:`~tvm::ffi::Any` value
- :doc:`object_and_class`: The object system underlying all container types
