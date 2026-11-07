/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
//! The derive macros of object types and object references.

/// Derive [`ObjectCore`](crate::ObjectCore) for a `#[repr(C)]` object struct
/// whose first field is its parent, so that a pointer to the object is one to
/// its parent.
///
/// `#[type_key = "..."]` names the type, and `#[type_index(unsafe ...)]` gives
/// it a static index; the `unsafe` marks the caller's promise that the runtime
/// gives that index to this type. Otherwise a type defined in Rust declares
/// `#[type_register]`: it is registered under its parent when its library
/// loads, as a C++ type is in a static init block, or takes the index its key
/// already has, as C++ `TVM_FFI_DECLARE_OBJECT_INFO` does. Under a binding
/// whose library is not loaded yet, it registers on its first `type_index()`
/// after that library loads instead. Without `#[type_register]`, the type
/// binds a key its defining library, usually C++, has registered, and
/// `type_index()` panics if that library is not loaded yet, so a binding
/// never registers a key with the wrong child slots. Such a panic is final:
/// `type_index()` of the type, and of every type under it, keeps panicking
/// after the library loads, so call it only once the library is loaded. Either way `type_index()` panics unless the registered
/// key has the type's parent and depth and, when the type table records a
/// size, that size is the type's, or for a binding at least the type's. A
/// binding that declares only part of its object must not be constructed:
/// keep its fields private.
///
/// `#[type_final]`, and with `#[type_register]` `#[type_child_slots = N]` and
/// `#[type_child_slots_can_overflow = bool]`, mirror C++ `_type_final`,
/// `_type_child_slots` (default 0) and `_type_child_slots_can_overflow`
/// (default true).
///
/// With `#[type_register]`, fields marked `#[reflect]` are registered with the
/// type table when the type is, as C++ `refl::ObjectDef<T>().def_ro(...)`
/// does: a getter, a setter, the doc string and the type schema per field,
/// the metadata, and the `__ffi_shallow_copy__` and `__ffi_type_final__`
/// type attributes. C++ structural equality and hashing, `repr`, deep copy
/// and Python then see the fields. `#[reflect(writable, seq_hash_ignore,
/// ...)]` adds the [`FieldFlags`](crate::reflection::FieldFlags) of those
/// names, `name = "..."` registers the field under another name, and
/// `doc = "..."` sets its doc string, otherwise the field's `///` doc comment.
/// `#[type_s_eq_hash_kind = "TreeNode"]` names the
/// [`SEqHashKind`](crate::reflection::SEqHashKind); without it the type keeps
/// its nearest reflected ancestor's, as a C++ subclass inherits
/// `_type_s_eq_hash_kind`. `#[type_reflection(path)]` names a function
/// `fn(&mut ObjectDef<Self>)` that registers the type's methods, as C++
/// `refl::ObjectDef<T>().def(...)` does; see
/// [`ObjectDef`](crate::reflection::ObjectDef) for what it must not do. A type
/// with any of these attributes also gets `ShallowCopy`, which shallow-copies
/// the parent and clones every other field, so the parent must implement
/// `ShallowCopy` (`Object` and reflected Rust types do) and the fields must be
/// `Clone`; a reflected field's type must implement
/// [`AnyCompatible`](crate::AnyCompatible). As a C++ `def_rw` field, a
/// `writable` field can be replaced through reflection, such as from Python,
/// while the object is shared.
///
/// A key registered more than once, by `#[type_register]` types or bindings
/// in different libraries, or by another language, must describe the same
/// type each time: `type_index()` checks the parent, depth and size, not the
/// fields.
///
/// Registrations from one library hold that library's lock, so they do not
/// race each other; libraries that each link tvm-ffi have a lock each. The
/// runtime's type table is not locked otherwise, and its readers take no
/// lock: as for C++, types register during static initialization, before
/// other threads use them or load a library that registers types. A type
/// that registers lazily, under a binding loaded later, must take its first
/// `type_index()` under the same rule.
pub use tvm_ffi_macros::Object;

/// Declare an object reference: the managed, reference-counted handle to an
/// object type, as `Arc<FooObj>` is to `FooObj`.
///
/// The struct holds one field, `ObjectArc<FooObj>`. The derive implements
/// [`ObjectRefCore`](crate::ObjectRefCore), the default
/// [`ObjectRefAnyCompatible`](crate::ObjectRefAnyCompatible) (conversions to
/// and from [`Any`](crate::Any), in-place downcasts, typed function
/// arguments), the upcast to [`ObjectRef`](crate::object::ObjectRef),
/// `Deref<Target = FooObj>` for field reads, `Debug`, `Display` (the runtime
/// `repr`, as [`AnyView`](crate::AnyView) formats it), and a private
/// `fn from_object(object: FooObj) -> Self` for the type's constructors.
///
/// # Options
///
/// `#[object_ref(...)]` takes:
///
/// * `parent = Ref` - The reference to the parent's object, or to a further
///   ancestor's; `Self` converts into it. The object type must embed that
///   object type through its chain of parents
///   ([`ObjectDerivesFrom`](crate::object::ObjectDerivesFrom)), so a wrong parent does not
///   compile. Any other ancestor takes
///   [`impl_object_ref_upcast!`](crate::impl_object_ref_upcast).
/// * `unsafe parent = Ref` - As `parent`, for a reference that promises more
///   than its object type: a generic one (`Array<T>`, `Map<K, V>`,
///   `Optional<T>`) or one with a custom `ObjectRefAnyCompatible`. The caller
///   promises that every value of `Self` is a valid `Ref`; `parent` alone
///   does not compile for such a reference.
/// * `skip_deref` - No `Deref`, for a reference whose object is internal.
/// * `skip_debug` - `Debug` is written by hand.
/// * `skip_display` - `Display` is written by hand.
/// * `skip_any_compatible` - `ObjectRefAnyCompatible` is written by hand,
///   for a typed reference with a stricter check or fallback conversions.
/// * `crate = path` - Where the generated code finds tvm-ffi, as for
///   [`rs_object`](macro@crate::rs_object).
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::ObjectRef;
/// use tvm_ffi::{rs_object, Object, ObjectArc};
///
/// #[rs_object("my_ext.MyPair")]
/// pub struct MyPairObj {
///     base_: Object,
///     pub first: i64,
///     pub second: i64,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// pub struct MyPair {
///     data: ObjectArc<MyPairObj>,
/// }
///
/// #[rs_object("my_ext.MyTriple", final)]
/// pub struct MyTripleObj {
///     base_: MyPairObj,
///     pub third: i64,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// #[object_ref(parent = MyPair)]
/// pub struct MyTriple {
///     data: ObjectArc<MyTripleObj>,
/// }
///
/// impl MyTriple {
///     pub fn new(first: i64, second: i64, third: i64) -> Self {
///         let pair = MyPairObj::new(Object::new(), first, second);
///         Self::from_object(MyTripleObj::new(pair, third))
///     }
/// }
///
/// let triple = MyTriple::new(1, 2, 3);
/// assert_eq!(triple.first + triple.third, 4);
/// let _pair: MyPair = triple.into();
/// ```
///
/// A `MyPairObj` does not embed an `ArrayObj`, so it cannot claim `Array` as
/// its parent:
///
/// ```compile_fail,E0277
/// use tvm_ffi::derive::ObjectRef;
/// use tvm_ffi::{rs_object, Array, Object, ObjectArc};
///
/// #[rs_object("my_ext.MyPair")]
/// pub struct MyPairObj {
///     base_: Object,
///     pub first: i64,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// #[object_ref(unsafe parent = Array<i64>)]
/// pub struct MyPair {
///     data: ObjectArc<MyPairObj>,
/// }
/// ```
pub use tvm_ffi_macros::ObjectRef;
