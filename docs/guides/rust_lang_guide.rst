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

Rust Guide
==========

.. note::

   The Rust support is currently in an experimental stage.

This guide demonstrates how to use TVM FFI from Rust applications.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

The Rust support depends on ``libtvm_ffi``. First, install the ``tvm-ffi`` Python package:

.. code-block:: bash

   pip install apache-tvm-ffi

From a source checkout, install it in editable mode with ``pip install -v -e .`` instead.

Confirm that ``tvm-ffi-config`` is available:

.. code-block:: bash

   tvm-ffi-config --libdir

Adding to Your Project
~~~~~~~~~~~~~~~~~~~~~~

Add to your ``Cargo.toml``:

.. code-block:: toml

   [dependencies]
   tvm-ffi = { path = "path/to/tvm-ffi/rust/tvm-ffi" }

For a published version, use the version of the installed ``apache-tvm-ffi``
package:

.. code-block:: toml

   [dependencies]
   tvm-ffi = "<version>"

Environment Setup
~~~~~~~~~~~~~~~~~

Set the library path so ``libtvm_ffi`` can be found at runtime:

.. code-block:: bash

   export LD_LIBRARY_PATH=$(tvm-ffi-config --libdir):$LD_LIBRARY_PATH

Basic Usage
-----------

Any and AnyView
~~~~~~~~~~~~~~~

|rs-any|_ owns a dynamically typed value, while |rs-anyview|_ borrows a value.
An object stored in ``Any`` carries a strong reference; creating an ``AnyView`` does not
increment its reference count. Rust tracks the view's lifetime so it cannot outlive
the value it borrows. A number and a built-in array illustrate owning and borrowed access:

.. code-block:: rust

   use tvm_ffi::{Any, AnyView, Array};

   let number = Any::from(3_i64);
   assert_eq!(number.downcast_ref::<i64>(), Some(&3)); // Strict: an i64 is stored.
   assert_eq!(f64::try_from(&number).unwrap(), 3.0); // Converting, as C++ `Any::cast`.

   let value = Any::from(Array::from_vec(vec![1_i64, 2, 3]));
   let view = AnyView::from(&value); // Borrow the value without a new reference.
   let borrowed: &Array<i64> = view.downcast_ref::<Array<i64>>().unwrap();
   assert_eq!(borrowed.len(), 3);

   let array = Array::<i64>::try_from(&value).unwrap(); // Acquire another strong reference.
   assert_eq!(array.len(), 3);

Both ``Any`` and ``AnyView`` provide ``downcast_ref::<T>() -> Option<&T>``, which borrows
the value in place as a managed reference such as ``Array<i64>``, a value such as
``tvm_ffi::String`` or ``i64``, or an object type such as ``ArrayObj``. The check is
strict and never converts; it returns ``None`` for other types. The returned
reference borrows its source without consuming it or incrementing the reference
count.
In contrast, ``TryFrom`` (``T::try_from(&value)`` or ``value.try_into()``) returns an
owned value and converts as C++ ``Any::cast`` does: an integer reads as a float or a
boolean. For a managed reference such as ``Array<i64>`` it retains the object
independently of the source ``Any`` or ``AnyView``; from an owned ``Any`` it moves the
object out instead.

Object and ObjectRef
~~~~~~~~~~~~~~~~~~~~

The object representation and its owning handle are separate. |rs-arrayobj|_, the
representation of an array, implements |rs-objectcore|_; ``Object`` is the root
representation, and ``ObjectRef`` is the erased managed reference.

**Named managed reference**. A named managed reference follows the newtype pattern around
|rs-objectarc|_, which manages the underlying object's lifetime through reference counting.
Define methods and associated functions on the named type, and implement traits to provide
shared interfaces. The named type can also impose additional constraints on the underlying object.
For example, |rs-array|_ holds an ``ObjectArc<ArrayObj>`` while constraining the array's
element type to ``T``. Named reference types implement |rs-objectrefcore|_ for shared reference operations:

.. code-block:: rust

   use tvm_ffi::collections::array::ArrayObj;
   use tvm_ffi::{Array, ObjectRefCore};

   let array = Array::from_vec(vec![1_i64, 2, 3]);
   assert_eq!(array.downcast_ref::<ArrayObj>().unwrap().size, 3);
   assert_eq!(array.get(1).unwrap(), 2); // A method of the named type.

Upcasts are ``From``/``Into``. Each upcast also gives the strict downcast back as
``TryFrom``, so ``let array: Array<i64> = object.try_into()?;`` checks the object's type
and returns a ``TypeError`` naming both types; between references it never converts.
``downcast::<T>()`` takes the reference and hands it back unchanged as ``Err``, so another
type can be tried, as std's ``Box<dyn Any>::downcast`` does, and ``downcast_ref::<T>()``
borrows it in place, with the same rules as ``Any`` and ``AnyView``. ``try_from`` an
``Any`` may convert by the FFI's conversion rules, such as an integer into a float, while
``try_from`` a reference only checks.

``AnyCompatible`` is the root contract for crossing into ``Any``; ``#[derive(ObjectRef)]``
is its object-reference default, and ``ObjectRefAnyCompatible`` customizes it.

Loading a Module
~~~~~~~~~~~~~~~~

Load a compiled TVM FFI module and call its functions:

.. code-block:: rust

   use tvm_ffi::{Module, Result};

   fn main() -> Result<()> {
       // Load compiled module
       let module = Module::load_from_file("build/add_one_cpu.so")?;

       // Get function by name
       let add_fn = module.get_function("add_one_cpu")?;

       Ok(())
   }

A module also reports whether it implements a function, and reads the
metadata and doc string a function was exported with, as
``Module.implements_function``, ``Module.get_function_metadata`` and
``Module.get_function_doc`` do in Python. Their ``*_with_imports`` variants
also look in the modules it imports:

.. code-block:: rust

   let module = Module::load_from_file("build/add_one_cpu.so")?;
   if module.implements_function("add_one_cpu")? {
       // A JSON object such as {"type_schema": "..."}, if it was exported
       let metadata = module.get_function_metadata("add_one_cpu")?;
       let doc = module.get_function_doc("add_one_cpu")?;
   }

Working with Tensors
~~~~~~~~~~~~~~~~~~~~

Create and manipulate tensors:

.. code-block:: rust

   use tvm_ffi::Tensor;

   // Create a tensor from a slice
   let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
   let tensor = Tensor::from_slice(&data, &[2, 3])?;

Calling Functions
~~~~~~~~~~~~~~~~~

Call functions with tensors:

.. code-block:: rust

   use tvm_ffi::{Module, Tensor, Result};

   fn run_example() -> Result<()> {
       let module = Module::load_from_file("build/add_one_cpu.so")?;
       let func = module.get_function("add_one_cpu")?;

       // Create input and output tensors
       let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])?;
       let output = Tensor::from_slice(&[0.0f32; 4], &[4])?;

       // Call function
       func.call_tuple((&input, &output))?;

       Ok(())
   }

Advanced Topics
---------------

Global Functions
~~~~~~~~~~~~~~~~

Register and access global functions:

.. code-block:: rust

   use tvm_ffi::{Any, AnyView, Function, Result};

   // Register a new global function
   let my_func = Function::from_packed(|_args: &[AnyView]| -> Result<Any> {
       // Function implementation
       Ok(Any::default())
   });
   Function::register_global("my_custom_func", my_func)?;

   // Get a global function
   let func = Function::get_global("my_custom_func")?;

   // Register a typed function with a doc string
   Function::register_global_typed(
       "my_add",
       |x: i64, y: i64| -> Result<i64> { Ok(x + y) },
       "Add two integers.",
   )?;

A typed function's metadata records its type schema, so
``tvm_ffi.get_global_func_metadata`` and the stub generator see its parameter and
return types. ``Function::register_global`` also records the schema of an untyped
function.

Exporting Functions from a Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Rust ``cdylib`` exports a typed function under the ``__tvm_ffi_<name>`` symbol
that ``Module::load_from_file`` and ``tvm_ffi.load_module`` look up:

.. code-block:: rust

   use tvm_ffi::*;

   fn add(a: i64, b: i64) -> Result<i64> {
       Ok(a + b)
   }

   tvm_ffi_dll_export_typed_func!(add, add);
   tvm_ffi_dll_export_typed_func_doc!(add, "Add two integers and return the sum.");

With the ``export-metadata`` feature of the ``tvm-ffi`` crate, the export also writes the
function's type schema as ``__tvm_ffi__metadata_<name>``, and
``tvm_ffi_dll_export_typed_func_doc!`` writes its doc string as
``__tvm_ffi__doc_<name>``, which ``Module.get_function_metadata`` and
``Module.get_function_doc`` read. The feature is off by default, and without
it ``tvm_ffi_dll_export_typed_func_doc!`` exports nothing:

.. code-block:: toml

   [dependencies]
   tvm-ffi = { version = "...", features = ["export-metadata"] }

Reflected Type Methods
~~~~~~~~~~~~~~~~~~~~~~

Methods registered on a type through reflection live in a per-type method table
rather than the global function table. Resolve them by type key (or type
index) and method name; a registered constructor is reachable under the
reserved name ``__ffi_init__``. ``testing.TestIntPair`` is registered by
``libtvm_ffi_testing``, next to ``libtvm_ffi``, once it is loaded, for example
with ``Module::load_from_file``:

.. code-block:: rust

   use tvm_ffi::reflection;
   use tvm_ffi::AnyView;

   // Resolve the reflected constructor and construct an instance
   let ctor = reflection::get_method("testing.TestIntPair", "__ffi_init__")?;
   let pair = ctor.call_tuple((1i64, 2i64))?;

   // Resolve an instance method; the first packed argument is the object itself
   let sum = reflection::get_method("testing.TestIntPair", "sum")?;
   let result = sum.call_packed(&[AnyView::from(&pair)])?;
   assert_eq!(i64::try_from(result)?, 3);

``reflection::get_method`` mirrors C++ ``reflection::GetMethod``;
``reflection::get_method_by_index(type_index, name)`` performs the same lookup when
the type index is already known (e.g. from ``Any::type_index``).

Defining Reflected Object Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``#[rs_object]`` defines an object type in Rust from its object struct, like
Python's ``py_class``. The type registers when the library that defines it
loads, as a C++ type does:

.. code-block:: rust

   use tvm_ffi::derive::ObjectRef;
   use tvm_ffi::reflection::ObjectDef;
   use tvm_ffi::{rs_object, Object, ObjectArc, Result};

   #[rs_object(type_key = "my.Point", s_eq_hash_kind = "TreeNode", reflection = PointObj::reflect_methods)]
   pub struct PointObj {
       base_: Object,
       /// The x coordinate.
       pub x: i64,
       #[reflect(writable, seq_hash_ignore, doc = "A label.")]
       pub label: tvm_ffi::String,
   }

   #[derive(ObjectRef, Clone)]
   pub struct Point {
       data: ObjectArc<PointObj>,
   }

   impl PointObj {
       fn reflect_methods(def: &mut ObjectDef<Self>) {
           def.def("x_abs", |p: Point| -> Result<i64> { Ok(p.x.abs()) }, "The |x|.")
               .def_static("zero", || -> Result<i64> { Ok(0) }, "");
       }
   }

The first field, ``base_``, is the parent's object. Every other field is
reflected: it gets a C ABI getter and setter, its doc string and its type
schema, so structural equality and hashing, ``ffi.ReprPrint``, ``ffi.DeepCopy``
and Python see it. The flags in ``#[reflect(...)]`` name
``tvm_ffi::reflection::FieldFlags`` constants; ``name = "..."`` registers the field under
another name, and ``doc = "..."`` sets its doc string, otherwise the field's
``///`` doc comment. Without ``s_eq_hash_kind``, a type keeps its nearest
reflected ancestor's kind. The type also implements ``ShallowCopy`` and
registers it as ``__ffi_shallow_copy__``, which requires the parent to
implement ``ShallowCopy`` and the other fields to be ``Clone``.
``immutable`` declares that shared objects of the type are never mutated
(``ImmutableObject``), so its references cross threads when its fields can; its
fields cannot be ``writable``.

``reflection = path`` names a function that registers the type's
methods with ``def`` (the object is the first parameter) and ``def_static``,
each with its type schema and doc
string, where ``reflection::get_method`` and Python find them. It runs
while the type registers, under the registration lock, so it must not call
``type_index()`` of the type or of another Rust type that is not registered
yet, nor register type attributes or call ``TypeAttrColumn::ensure``, which
take the same lock. Method names, like
reflected field names, must be unique in the type.

``#[derive(Object)]`` instead binds a type that another library, usually C++,
registers; ``tvm-ffi-stubgen`` generates such bindings.

Deriving Object Types
~~~~~~~~~~~~~~~~~~~~~

``#[derive(ObjectRef)]`` declares the reference, and ``#[object_ref(parent = Point)]``
names the reference to the parent's object:

.. code-block:: rust

   use tvm_ffi::derive::ObjectRef;
   use tvm_ffi::{rs_object, Object, ObjectArc};

   #[rs_object(type_key = "my.Point", s_eq_hash_kind = "TreeNode")]
   pub struct PointObj {
       base_: Object,
       pub x: i64,
       #[reflect(writable, seq_hash_ignore)]
       pub label: tvm_ffi::String,
   }

   #[derive(Clone, ObjectRef)]
   pub struct Point {
       data: ObjectArc<PointObj>,
   }

   #[rs_object(type_key = "my.Point3", final)]
   pub struct Point3Obj {
       base_: PointObj,
       pub z: i64,
   }

   #[derive(Clone, ObjectRef)]
   #[object_ref(parent = Point)]
   pub struct Point3 {
       data: ObjectArc<Point3Obj>,
   }

   impl Point3 {
       pub fn new(x: i64, label: &str, z: i64) -> Self {
           Self::from_object(Point3Obj::new(PointObj::new(Object::new(), x, label), z))
       }
   }

   let point3 = Point3::new(1, "p", 3);
   assert_eq!((point3.x, point3.z), (1, 3));
   let point: Point = point3.into();

``#[rs_object]`` adds ``#[repr(C)]``, ``Deref`` to the parent's object and
``Point3Obj::new``, which takes the parent's object, then the fields: a scalar
(``bool``, an integer or a float) as its own type, so that a literal infers as it,
and any other field as ``impl Into<T>``. A constructor on the reference takes an
object reference as ``impl IntoArg<T>``, as the ones ``tvm-ffi-stubgen``
generates do. The reference dereferences to the object
(``#[object_ref(skip_deref)]`` opts out), and its private ``from_object`` wraps a new
object for the type's constructors. The reference also converts into ``parent``,
whose object the reference's object must embed, directly or through its own
parents, which the derive checks at compile time. A parent that promises more than
its object type, a generic one such as ``Array<T>`` or one with a custom
``ObjectRefAnyCompatible``, is written ``unsafe parent = ...``: the caller promises
that every value of the reference is a valid parent. Every reference converts into
``ObjectRef``; ``impl_object_ref_upcast!`` declares conversions into further
ancestors with the same check, such as ``impl_object_ref_upcast!(Point4 => Point)``
for a ``Point4`` whose parent is ``Point3``, and ``impl_object_ref_upcast!(unsafe
...)`` without it, for an ancestor that promises more than its object type.

The type also registers a creator, and from it ``__ffi_new__`` and ``__ffi_init__``
(unless ``reflection = path`` defines ``__ffi_init__``), so that a JSON graph reads
the type back and Python constructs it from its fields. The creator makes the
object with every field undefined, and reflection then sets the fields. A Rust reference is never null, so an
undefined field holds all-zero bytes, which are no value of its type
(``AnyCompatible::is_undefined``): the field setter overwrites them without
dropping, the field getter reads them as None and the object's deleter skips
them. An ``init_off`` field stays all-zero after ``__ffi_init__``, so its type must
take all-zero bytes as a value, as a scalar, an ``Option`` of a reference or an
``Optional<T>`` does; a reference or
a string does not, and fails the type's registration. The parent's object must
implement ``ShallowCopy`` and ``UnsafeInit``, as ``rs_object`` types do, and so
does a binding with ``#[object(rs_parent)]`` on its ``derive(Object)``, such as a
non-final type bound by ``tvm-ffi-stubgen``.

Converting Borrowed Values into ``Any``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Any::from(value)`` takes ownership of ``value``. Use it when you own the value.
When you only have a shared reference—for example, a field accessed through
``&Point``—you cannot move the field into an ``Any``. Call
``AnyCompatible::to_any()`` instead:

.. code-block:: rust

   use tvm_ffi::{Any, AnyCompatible};

   fn label_of(point: &Point) -> Any {
       point.label.to_any()
   }

``to_any()`` creates an owned ``Any`` while leaving the borrowed value usable. For
object-backed values, it retains the object by incrementing its reference
count. It is equivalent to ``Any::from(point.label.clone())``, without requiring an
explicit clone at the call site.

Type-Erased Functions
~~~~~~~~~~~~~~~~~~~~~

Create functions from Rust closures:

.. code-block:: rust

   use tvm_ffi::{Function, Any, AnyView, Result};

   // From packed closure
   let func = Function::from_packed(|args: &[AnyView]| -> Result<Any> {
       // Process args and return result
       Ok(Any::default())
   });

   // From typed closure
   let typed_func = Function::from_typed(|x: i64, y: i64| -> Result<i64> {
       Ok(x + y)
   });

Error Handling
~~~~~~~~~~~~~~

TVM FFI uses standard Rust ``Result`` types:

.. code-block:: rust

   use tvm_ffi::{Error, Module, Result, VALUE_ERROR};

   fn may_fail(value: i32) -> Result<()> {
       // Operations that may fail
       let module = Module::load_from_file("path.so")?;

       // Custom errors
       if value < 0 {
           return Err(Error::new(VALUE_ERROR, "Value must be non-negative"));
       }

       Ok(())
   }

Matching and Dispatch
~~~~~~~~~~~~~~~~~~~~~

Matching Types
^^^^^^^^^^^^^^

An ``Any`` or ``AnyView`` can hold any type, so code that receives one often
needs to ask which type it holds. |rs-match_any|_ answers this the way ``match``
does for an enum. Each *arm* is a pattern followed by ``=>`` and an expression; the first
arm whose type matches the value runs:

.. code-block:: rust

   use tvm_ffi::{match_any, Any, Array};

   fn kind(value: &Any) -> String {
       match_any! {
           value {
               i64(n) => format!("integer {n}"),
               tvm_ffi::String(ref s) => format!("string {}", s.as_str()),
               Array<Any>(ref arr) => format!("array of {} items", arr.len()),
               _ => "something else".to_string(),
           }
       }
   }

   for value in [
       Any::from(42_i64),
       Any::from(tvm_ffi::String::from("hi")),
       Any::from(Array::from_vec(vec![1_i64, 2, 3])),
       Any::from(1.5_f64),
   ] {
       println!("{}", kind(&value));
   }

The output is:

.. code-block:: text

   integer 42
   string hi
   array of 3 items
   something else

``Type(name)`` binds an owned value. ``Type(ref name)`` instead borrows the value
in place, as the string and array arms above do, without changing any reference
count; prefer it when the arm only reads. An arm can also name the object type,
as in ``ArrayObj(ref node)``, to borrow the object node. The final ``_`` arm is required:
new types can be registered at any time, so no list of arms is exhaustive.

Dispatching
^^^^^^^^^^^

A common need is to dispatch over many kinds of values: run different logic
depending on which type a value has. ``match_any!`` provides the machinery,
and |rs-dispatch|_ **makes such dispatchers simpler to build**: write one
method per type, and the attribute generates the match that picks between
them. Let's describe nested arrays, with a separator between items:

.. code-block:: rust

   use tvm_ffi::{dispatch, Any, AnyView, Array};

   struct Describer;

   #[dispatch(describe, (AnyView<'_>, &str), String)]
   impl Describer {
       fn describe_array(&self, arr: &Array<Any>, sep: &str) -> String {
           let items: Vec<String> = arr
               .iter()
               .map(|item| {
                   let item = AnyView::from(item);
                   self.dispatch(item, sep).unwrap_or_else(|| "?".into())
               })
               .collect();
           format!("[{}]", items.join(sep))
       }

       fn describe_int(&self, value: i64) -> String {
           value.to_string()
       }

       fn describe_str(&self, value: &tvm_ffi::String) -> String {
           value.to_string()
       }
   }

   let value = Any::from(Array::from_vec(vec![
       Any::from(1_i64),
       Any::from(tvm_ffi::String::from("a")),
       Any::from(Array::from_vec(vec![2_i64, 3])),
   ]));
   let text = Describer.dispatch(AnyView::from(&value), ", ").unwrap();
   println!("{text}");

The output is:

.. code-block:: text

   [1, "a", [2, 3]]

The ``#[dispatch(describe, (AnyView<'_>, &str), String)]`` attribute makes the
``describe_*`` methods handlers and generates
``dispatch(value, sep) -> Option<String>``. The first element of the tuple is
the value to match; the rest are passed to the handlers. ``describe_array``
calls ``self.dispatch`` on each item, so nested arrays recurse, and turns
``None``, which means no handler matched, into ``"?"``. ``tvm_ffi::String``
displays with quotes, which keeps strings apart from numbers in the output.

**The attribute is sugar for a match.** Simplified, ``#[dispatch]`` generates
a ``match_any!`` and a |rs-dispatcher|_ implementation:

.. code-block:: rust

   impl Describer {
       pub fn dispatch(&self, value: AnyView<'_>, sep: &str) -> Option<String> {
           match_any! {
               value {
                   Array<Any>(ref arr) => Some(self.describe_array(arr, sep).into()),
                   i64(value) => Some(self.describe_int(value).into()),
                   tvm_ffi::String(ref value) => Some(self.describe_str(value).into()),
                   _ => None,
               }
           }
       }
   }

   impl Dispatcher<(AnyView<'_>, &str), Option<String>> for &Describer {
       fn dispatch_tuple(self, (value, sep): (AnyView<'_>, &str)) -> Option<String> {
           Describer::dispatch(self, value, sep)
       }
   }

Each handler becomes one arm, in source order, and its parameter type selects
the values it matches; a ``&Type`` parameter becomes a ``Type(ref value)`` arm,
which borrows the value in place. The attribute appends ``_ => None``, so a type without a handler returns
``None`` instead of failing. The ``Dispatcher`` implementation lets generic
code call the same match through one trait.

**A handler takes only the arguments it needs.** It receives the matched value
and a leading prefix of the remaining arguments, in declaration order:
``describe_array`` takes ``sep``, while ``describe_int`` and ``describe_str``
omit it.

.. note::

   A handler's result converts into the declared return type with ``Into``. A
   handler taking ``AnyView<'_>`` matches everything, so it must come last.
   Handlers taking ``&mut self`` make ``dispatch`` take ``&mut self``, and the
   ``Dispatcher`` implementation is then for ``&mut Describer``. As in an
   ordinary method, elided lifetimes in the tuple are independent and an elided
   lifetime in the return type borrows ``self``; to return a borrow of an
   input, name the same lifetime in both.

The structural utilities below accept the same kind of dispatcher: walk,
visit, map and mutate each call it on the values they reach. The prefix only
names the methods; the utility that calls the dispatcher decides what happens
to the result.

Structural Walk and Visit
~~~~~~~~~~~~~~~~~~~~~~~~~

|rs-structural_walk|_ visits every value reachable from a root and accepts a
dispatcher, as described in Dispatching above:
``#[dispatch(walk, (AnyView<'_>, DefRegionKind), Result<WalkResult>)]`` groups
``walk_*`` handlers, which return ``Result<WalkResult>`` to advance, skip
children, or interrupt traversal.

Walk handlers may take an optional trailing ``DefRegionKind`` argument:

.. code-block:: rust

   use tvm_ffi::{dispatch, structural_walk, Array, Result, WalkOrder, WalkResult};
   use tvm_ffi::DefRegionKind;

   #[derive(Default)]
   struct Probe {
       total: i64,
       floats: usize,
   }

   #[dispatch(walk, (tvm_ffi::AnyView<'_>, DefRegionKind), Result<WalkResult>)]
   impl Probe {
       fn walk_integer(&mut self, value: i64) -> Result<WalkResult> {
           self.total += value;
           Ok(WalkResult::Advance)
       }

       fn walk_float(&mut self, _value: f64, _kind: DefRegionKind) -> Result<WalkResult> {
           self.floats += 1;
           Ok(WalkResult::Advance)
       }
   }

   let values = Array::from_vec(vec![1_i64, 2, 3]);
   let mut probe = Probe::default();
   structural_walk(&values, &mut probe, WalkOrder::PreOrder)?;
   assert_eq!(probe.total, 6);

For ``Map`` and ``Dict``, structural walk treats keys as structural anchors: it
visits container values but does not pass keys to handlers. The map or dict
object itself is still visited normally.

Lambdas also work — pass a single typed lambda, or a tuple of them tried in
order with the first matching argument type winning. A flat tuple holds up to 12
lambdas; a tuple is itself a link, so nest ``(a, b, (c, d, ...))`` to chain
more — order stays the flattened left-to-right order. Unmatched values
simply advance; an ``AnyView<'_>`` lambda acts as a catch-all and must come
last, since links after an always-matching one never run. Each lambda may
take a trailing ``DefRegionKind`` argument:

.. code-block:: rust

   use tvm_ffi::{structural_walk, Array, Object, WalkOrder, WalkResult};
   use tvm_ffi::DefRegionKind;

   let values = Array::from_vec(vec![1_i64, 2, 3]);

   let mut total = 0;
   structural_walk(
       &values,
       |value: i64| {
           total += value;
           WalkResult::Advance
       },
       WalkOrder::PreOrder,
   )?;
   assert_eq!(total, 6);

   let mut evens = 0;
   let mut objects = 0;
   structural_walk(
       &values,
       (
           |value: i64| {
               if value % 2 == 0 {
                   evens += 1;
               }
               WalkResult::Advance
           },
           |_object: &Object, _kind: DefRegionKind| {
               objects += 1;
               WalkResult::Advance
           },
       ),
       WalkOrder::PreOrder,
   )?;
   assert_eq!((evens, objects), (1, 1));

Both entry points return ``Result<Option<VisitInterrupt>>``: ``Ok(None)`` means
the whole graph was visited, and a handler stops the walk early by returning
``WalkResult::InterruptWith(payload)``, which comes back to the caller as
``Ok(Some(interrupt))``. Handlers may also return ``Result<WalkResult>`` and
propagate errors with ``?``:

.. code-block:: rust

   use tvm_ffi::{structural_walk, Array, WalkOrder, WalkResult};

   let values = Array::from_vec(vec![1_i64, 2, 3]);
   let found = structural_walk(
       &values,
       |value: i64| {
           if value == 2 {
               return WalkResult::InterruptWith(value.into());
           }
           WalkResult::Advance
       },
       WalkOrder::PreOrder,
   )?;
   assert_eq!(found.map(|i| i64::try_from(&i.value).unwrap()), Some(2));

``structural_visit`` accepts a dispatcher, owned or borrowed. Pass state belongs
on ``self``; matching handlers control recursion through ``StructuralVisitorView<'_, Self>``.
Unmatched values use default child traversal. Walk continues to accept both
dispatchers and closures.

.. code-block:: rust

   use tvm_ffi::{dispatch, structural_visit, Array, Result, VisitInterrupt};
   use tvm_ffi::StructuralVisitorView;

   #[derive(Default)]
   struct Stats {
       total: i64,
   }

   #[dispatch(
       visit,
       (tvm_ffi::AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
       Result<Option<VisitInterrupt>>
   )]
   impl Stats {
       fn visit_integer(&mut self, value: i64, _visitor: &mut StructuralVisitorView<'_, Self>)
           -> Result<Option<VisitInterrupt>> {
           self.total += value;
           Ok(None)
       }
   }

   let values = Array::from_vec(vec![1_i64, 2]);
   let mut visitor = Stats::default();
   structural_visit(&values, &mut visitor)?;
   assert_eq!(visitor.total, 3);

A handler calls ``visitor.default_visit(self, value)`` to descend through its input,
including a borrowed object node, without acquiring ownership.
``visitor.visit(self, &child)`` re-enters matching for a child. Both operations
inherit the definition region established by the defining node or reflected
field; handlers query it with ``visitor.def_region_kind()``. Ordinary recursion
does not need a region scope. Calls reborrow the same dispatcher, so state can be
read and updated before and after recursion. Interrupt values must be
returned explicitly because ``?`` only propagates errors.

An inherent ``#[dispatch]`` impl groups typed ``visit_*`` methods. Its declared
argument tuple and return type supply the engine signature directly:

.. code-block:: rust

   use tvm_ffi::{dispatch, structural_visit, AnyView, Array, Result, VisitInterrupt};
   use tvm_ffi::StructuralVisitorView;

   #[derive(Default)]
   struct Depth {
       max: usize,
       current: usize,
   }

   #[dispatch(
       visit,
       (tvm_ffi::AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
       Result<Option<VisitInterrupt>>
   )]
   impl Depth {
       fn visit_any(&mut self, value: AnyView<'_>, visitor: &mut StructuralVisitorView<'_, Self>)
           -> Result<Option<VisitInterrupt>> {
           self.current += 1;
           self.max = self.max.max(self.current);
           let outcome = visitor.default_visit(self, &value);
           self.current -= 1;
           outcome
       }
   }

   let values = Array::from_vec(vec![1_i64, 2]);
   let mut depth = Depth::default();
   structural_visit(&values, &mut depth)?;
   assert_eq!(depth.max, 2);

The engine owns a temporary FFI runtime object and lends each handler an explicit
context for that invocation. Contexts do not own or retain a second reference to
the dispatcher. They cannot escape the callback or move to another thread.

Default descent uses registered structural hooks, then reflection for types
without a hook. A hook uses the visitor only through its arguments, during the
call, and must not keep it or call it outside them.

``AnyView<'_>`` is passed by value to catch-all visit, walk, and map callbacks;
mutation dispatch consumes ``MutationView<'_>`` instead. Copying an ``AnyView``
preserves a borrow without incrementing a reference count. Bind the view
before calling ``downcast_ref::<Node>()`` when the node
reference is used across statements; that reference borrows the view.
Use ``Handle::try_from(value)`` for an owned typed handle or ``Any::from(value)`` to
retain an erased value beyond the callback. Mapping callbacks return
replacement values; their input view does not grant in-place mutation
permission.

Structural Mapping and Mutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``structural_map`` is the transforming counterpart to ``structural_walk``. An
inherent ``#[dispatch(map, (AnyView<'_>, DefRegionKind), Result<UnchangedOr<Any>>)]`` impl groups
``map_*`` methods returning the engine result ``Result<UnchangedOr<Any>>``. Methods are tested in source order, the
first matching argument type wins, and an unmatched value is preserved. A
method may take an optional trailing ``DefRegionKind``; an ``AnyView<'_>`` method is a
catch-all and should therefore come last:

.. code-block:: rust

   use tvm_ffi::{
       dispatch, structural_map, Any, Array, AnyView, Result, WalkOrder, UnchangedOr, Unchanged,
   };
   use tvm_ffi::DefRegionKind;

   #[derive(Default)]
   struct Increment {
       integers: usize,
   }

   #[dispatch(map, (tvm_ffi::AnyView<'_>, DefRegionKind), Result<UnchangedOr<Any>>)]
   impl Increment {
       fn map_integer(&mut self, value: i64, _kind: DefRegionKind) -> Result<UnchangedOr<Any>> {
           self.integers += 1;
           Ok((value + 1).into())
       }

       fn map_other(&mut self, _value: AnyView<'_>) -> Result<UnchangedOr<Any>> {
           Ok(Unchanged.into())
       }
   }

   let mut increment = Increment::default();
   let mapped = structural_map(
       Array::from_vec(vec![1_i64, 2]),
       &mut increment,
       WalkOrder::PostOrder,
   )?;
   let mapped = Array::<i64>::try_from(mapped)?;
   assert_eq!(mapped.iter().copied().collect::<Vec<_>>(), vec![2, 3]);

A single typed closure and an ordered tuple of up to twelve closures are also
accepted; a tuple is itself a link, so ``(a, b, (c, d, ...))`` nests beyond that
without changing the flattened order. Tuple dispatch is first-match, not broadcast: later closures do not
run after an earlier argument type matches. As with generated dispatch, a
``AnyView<'_>`` catch-all belongs last. Numeric handlers claim the complete FFI
``Int`` or ``Float`` tag and then use Rust ``as`` conversion semantics; prefer ``i64``
or ``f64`` unless narrowing is deliberate.

``WalkOrder::PreOrder`` runs the callback before default child mapping. If it
returns a different root, that replacement root is not passed to the callback
again, but its children are still mapped. ``WalkOrder::PostOrder`` maps children
first and passes the resulting value to the callback; a replacement returned
there is final. ``Array`` and ``List`` elements are mapped in order. ``Map`` and
``Dict`` keys are identity anchors and are never mapped; only their values are.

The root is consumed. A uniquely owned built-in container may reuse its
storage in place; passing ``root.clone()`` keeps the source shared and selects
copy-on-write behavior. The engine rechecks uniqueness after a pre-order
callback, so retaining an owning ``Any::from(value)`` alias before the
container's children are mapped forces the non-in-place path. Reflected
objects must provide ``__ffi_shallow_copy__``; the copy is validated before
fields are mapped and discarded if no structural field changes.

Within one ``structural_map`` call, callbacks run at every occurrence; their
results are not cached. Default recursion manages identity remapping, with
the same semantics in every language binding.

Default recursion uses the registered type attributes: it calls
``__s_maybe_inplace_mutate__`` for a uniquely owned object when available, or
falls back to ``__s_mutate__``. Each hook receives the active Rust-backed
mutator and can recurse through its language-independent vtable. This lets the
implementation that registered the type own its storage and mutation rules.
A hook uses the mutator only through its arguments, during the call, and must
not keep it or call it outside them. When a type has no hook, object-backed values use reflected fields.

Callbacks may return ``Result<T>`` for any ``T`` convertible into ``Any``; for
example, an integer handler can return ``Result<i64>`` to report failures and use
``?``. Errors propagate with object or reflected-field context. In-place changes
completed before a later error are not rolled back, and the consumed root is
not returned on error.

Handlers of a ``#[dispatch]`` struct return ``Result<UnchangedOr<Any>>``, the
same type that ``mutate`` and ``default_mutate`` return, so a handler can return
the result of recursion directly. Return ``Unchanged.into()`` to keep the value,
or ``value.into()`` to replace it with a managed reference or scalar; wrap an
``Any`` replacement with ``UnchangedOr::changed``. Closures may also return
``Unchanged``, ``UnchangedOr<T>``, or a replacement value, optionally wrapped in
``Result``. A pre-order map still maps the original value's children when a
callback returns ``Unchanged``; a post-order map keeps the result of child
mapping. Owning-value helpers and top-level functions resolve the marker to the
original value.

``structural_mutate`` accepts an owned or borrowed dispatcher. Its handlers keep
mutable pass state on ``self`` and receive ``StructuralMutatorView<'_, Self>`` for recursion.
Map continues to accept both dispatchers and closures. A native base, a
``StructuralVisitorBase`` or ``StructuralMutatorBase``, customizes default descent:
``StructuralVisitEngine::<Base>::visit_root(root, dispatcher)`` and
``StructuralMutateEngine::<Base>::mutate_root(root, dispatcher)`` run on it. A base
delegates to the default operation of ``StructuralVisitorObj`` or
``StructuralMutatorObj`` for ordinary hooks and reflection. The views, bases,
engines, ``DefRegionKind`` and ``InplaceMode`` live in
``tvm_ffi::extra::structural_visit`` and ``tvm_ffi::extra::structural_mutate``.

.. code-block:: rust

   use tvm_ffi::{dispatch, structural_mutate, Any, Array, MutationView, Result, UnchangedOr};
   use tvm_ffi::StructuralMutatorView;

   #[derive(Default)]
   struct Stats {
       integers: usize,
   }

   #[dispatch(mutate, (tvm_ffi::MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
   impl Stats {
       fn mutate_integer(&mut self, value: i64, _mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
           self.integers += 1;
           Ok((value + 1).into())
       }
       fn mutate_any(&mut self, value: MutationView<'_>, mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
           mutator.default_mutate(self, value)
       }
   }

   let mut mutator = Stats::default();
   let mutated = structural_mutate(Array::from_vec(vec![1_i64, 2]), &mut mutator)?;
   let mutated = Array::<i64>::try_from(mutated)?;
   assert_eq!(mutated.iter().copied().collect::<Vec<_>>(), vec![2, 3]);
   assert_eq!(mutator.integers, 2);

``mutator.mutate(self, value)`` re-enters dispatch with a ``MutationView``, while
``mutator.default_mutate(self, value)`` runs default descent. Both return
``Result<UnchangedOr<Any>>`` and preserve the capability's permission. For a local owning
``Any``, pass ``&mut owned`` and resolve the returned change with
``change.value_or_unchanged(owned)`` after the borrow ends. These calls inherit the definition
region established by the defining node or reflected field; handlers query it
with ``mutator.def_region_kind()``. Reserve ``with_def_region_kind`` on either view
for hooks that establish a definition region according to their node's semantics.
An enclosing Pattern region wins, and the prior region is restored even on error
or panic. State remains available after the recursive call.

Typed ``MutationView<'_, T>`` inputs work with the same recursive methods.
Use ``value.narrow_inplace_mode(InplaceMode::Disallow)`` to restrict permission.
Borrow the value in place through ``value.downcast_ref::<Node>()``, as with
``AnyView``, or other payloads through ``value.as_any_view()``; finish using those
borrows before consuming the capability. A typed object capability also dereferences to its shared node.
Use ``value.field(|node| &node.a)`` to borrow a direct ``Any`` or owning object-handle
field for recursive mutation. The child capability exclusively borrows its parent
and inherits permission only while that parent is unique; child uniqueness is
checked again during mutation. Field projection does not grant parent-field
writeback: preserve ``Unchanged`` or reconstruct the parent from replacement results.
``MutationView::borrowed(view)`` grants
only copy permission. Read permission from ``value.inplace_mode()``; see
``MutationView`` for ownership requirements.

An inherent ``#[dispatch]`` impl declares the engine's required mutation tuple
and ``Result<UnchangedOr<Any>>`` return type. An unmatched value follows default mutation with
its current permission. A handler may omit the context argument even when no
other handler consumes it:

.. code-block:: rust

   use tvm_ffi::{dispatch, structural_mutate, Any, Array, Result, UnchangedOr};
   use tvm_ffi::StructuralMutatorView;

   #[derive(Default)]
   struct Increment {
       integers: usize,
   }

   #[dispatch(mutate, (tvm_ffi::MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
   impl Increment {
       fn mutate_integer(&mut self, value: i64, _mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
           self.integers += 1;
           Ok((value + 1).into())
       }
   }

   let mut increment = Increment::default();
   let mutated = structural_mutate(Array::from_vec(vec![1_i64, 2]), &mut increment)?;
   let mutated = Array::<i64>::try_from(mutated)?;
   assert_eq!(mutated.iter().copied().collect::<Vec<_>>(), vec![2, 3]);
   assert_eq!(increment.integers, 2);

``MutationView::borrowed`` wraps an ``AnyView`` with ``Disallow``; an exclusive
``&mut Any`` borrow permits ``Allow``. Explicit modes can only restrict permission. Default mutation
rechecks uniqueness before writing. A catch-all can select copy-only recursion
while retaining typed scalar replacements:

.. code-block:: rust

   use tvm_ffi::{dispatch, structural_mutate, Any, Array, MutationView, Result, UnchangedOr};
   use tvm_ffi::{InplaceMode, StructuralMutatorView};

   struct Increment;

   #[dispatch(mutate, (tvm_ffi::MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
   impl Increment {
       fn mutate_integer(&mut self, value: i64, _mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
           Ok((value + 1).into())
       }
       fn mutate_any(&mut self, value: MutationView<'_>, mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
           mutator.default_mutate(self, value.narrow_inplace_mode(InplaceMode::Disallow))
       }
   }

   let mutated = structural_mutate(Array::from_vec(vec![1_i64, 2]), Increment)?;
   let mutated = Array::<i64>::try_from(mutated)?;
   assert_eq!(mutated.iter().copied().collect::<Vec<_>>(), vec![2, 3]);

The context's ``var_remap_get`` and ``var_remap_set`` methods use state local to one
invocation, preserving completed ``FreeVar`` and ``DAGNode`` substitutions without
adding a field to the dispatcher. Explicit recursion and default descent share
that invocation state.

Examples
--------

The repository includes a complete example in ``rust/tvm-ffi/examples/load_library.rs``.

Run it with:

.. code-block:: bash

   cd rust
   cargo run --example load_library --features example

Building the Workspace
----------------------

Build the entire Rust workspace:

.. code-block:: bash

   cd rust
   cargo build

Run tests:

.. code-block:: bash

   cargo test

API Reference
-------------

For detailed API documentation, see the :doc:`Rust API Reference <../reference/rust/index>`.

Related Resources
-----------------

- :doc:`Quick Start Guide <../get_started/quickstart>` - General TVM FFI introduction
- :doc:`C++ Guide <cpp_lang_guide>` - C++ API usage
- :doc:`Python Guide <python_lang_guide>` - Python API usage

.. Rust API reference links, generated with BUILD_RUST_DOCS=1.

.. |rs-any| replace:: ``Any``
.. _rs-any: ../reference/rust/generated/tvm_ffi/any/struct.Any.html
.. |rs-anyview| replace:: ``AnyView<'a>``
.. _rs-anyview: ../reference/rust/generated/tvm_ffi/any/struct.AnyView.html
.. |rs-arrayobj| replace:: ``ArrayObj``
.. _rs-arrayobj: ../reference/rust/generated/tvm_ffi/collections/array/struct.ArrayObj.html
.. |rs-objectcore| replace:: ``ObjectCore``
.. _rs-objectcore: ../reference/rust/generated/tvm_ffi/object/trait.ObjectCore.html
.. |rs-objectarc| replace:: ``ObjectArc<T>``
.. _rs-objectarc: ../reference/rust/generated/tvm_ffi/object/struct.ObjectArc.html
.. |rs-array| replace:: ``Array<T>``
.. _rs-array: ../reference/rust/generated/tvm_ffi/collections/array/struct.Array.html
.. |rs-objectrefcore| replace:: ``ObjectRefCore``
.. _rs-objectrefcore: ../reference/rust/generated/tvm_ffi/object/trait.ObjectRefCore.html
.. |rs-match_any| replace:: ``match_any!``
.. _rs-match_any: ../reference/rust/generated/tvm_ffi/macro.match_any.html
.. |rs-dispatch| replace:: ``#[dispatch]``
.. _rs-dispatch: ../reference/rust/generated/tvm_ffi/attr.dispatch.html
.. |rs-dispatcher| replace:: ``Dispatcher``
.. _rs-dispatcher: ../reference/rust/generated/tvm_ffi/extra/dispatch/trait.Dispatcher.html
.. |rs-structural_walk| replace:: ``structural_walk``
.. _rs-structural_walk: ../reference/rust/generated/tvm_ffi/extra/structural_walk/fn.structural_walk.html
