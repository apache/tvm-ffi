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

/// Bind an object type that another library, usually C++, registers: derive
/// [`ObjectCore`](crate::ObjectCore) for a `#[repr(C)]` object struct whose
/// first field is its parent, so that a pointer to the object is one to its
/// parent. Define a new type in Rust with [`rs_object`](macro@crate::rs_object)
/// instead.
///
/// Without `register` or `type_index`, `type_index()` takes the index the
/// defining library registered for the key, and panics if that library is not
/// loaded yet, so a binding never registers a key with the wrong child slots.
/// Such a panic is final: `type_index()` of the type, and of every type under
/// it, keeps panicking after the library loads, so call it only once the
/// library is loaded. It also panics unless the registered key has the type's
/// parent and depth and, when the type table records a size, a size no
/// smaller than the type's.
///
/// A binding that declares only part of its object must not be constructed:
/// keep its fields private.
///
/// # Options
///
/// `#[object(...)]` takes a comma-separated list, as in
/// `#[object(type_key = "ffi.String", unsafe type_index = TypeIndex::kTVMFFIStr)]`:
///
/// * `type_key = "..."` - The type key; required.
/// * `unsafe type_index = expr` - A static index instead of the registered
///   one; the `unsafe` marks the caller's promise that the runtime gives that
///   index to this type.
/// * `final` - The type has no subtypes, as C++ `_type_final`.
/// * `immutable` - Implement
///   [`ImmutableObject`](crate::object::ImmutableObject), as for a C++ type
///   not registered as `_type_mutable`; every subtype must be immutable too.
/// * `rs_parent` - Implement `ShallowCopy` and
///   [`UnsafeInit`](crate::reflection::UnsafeInit), so an `rs_object` can
///   derive from the type. Each field other than the parent must be `Clone`
///   and implement [`AnyCompatible`](crate::AnyCompatible), be an
///   [`Any`](crate::Any) or have no drop glue.
/// * `register` - The type is defined here: register its key under its
///   parent when the library loads, as `rs_object` does.
/// * `child_slots = N`, `child_slots_can_overflow = bool` - Type indices kept
///   for subtypes, as C++ `_type_child_slots` and
///   `_type_child_slots_can_overflow`; 0 and `true` by default. Needs
///   `register`.
/// * `s_eq_hash_kind = "TreeNode"` - The
///   [`SEqHashKind`](crate::reflection::SEqHashKind) of structural equality;
///   by default the parent's.
/// * `reflection = path` - A `fn(&mut ObjectDef<Self>)` that registers the
///   type's methods.
///
/// `s_eq_hash_kind`, `reflection` and field `#[reflect(...)]`, as
/// [`rs_object`](macro@crate::rs_object) takes it, need `register` and no
/// `type_index`.
pub use tvm_ffi_macros::Object;

/// Declare an object reference: the managed, reference-counted handle to an
/// object type, as `Arc<FooObj>` is to `FooObj`.
///
/// The struct holds one field, `ObjectArc<FooObj>`. The derive implements
/// [`ObjectRefCore`](crate::ObjectRefCore), the default
/// [`ObjectRefAnyCompatible`](crate::object::ObjectRefAnyCompatible) (conversions to
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
