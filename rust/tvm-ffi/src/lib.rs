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
//! Rust bindings of TVM FFI, the open ABI and FFI for machine learning
//! systems.
//!
//! Values cross the boundary as [`Any`] and [`AnyView`]. Objects, such as
//! [`Array`], [`Map`], [`Tensor`] and types defined with [`rs_object`], are
//! reference counted and shared with C++ and Python. A [`Function`] is
//! callable from every language, and an [`Error`] raised in one arrives in
//! the others.
//!
//! # Examples
//!
//! ```
//! use tvm_ffi::{Array, Function, Result};
//!
//! let sum = |xs: Array<i64>| -> Result<i64> { Ok(xs.iter().sum()) };
//! Function::register_global_typed("my_ext.sum", sum, "Sum the elements.")?;
//!
//! let sum = Function::get_global("my_ext.sum")?;
//! let total = sum.call_tuple((Array::from_vec(vec![1_i64, 2, 3]),))?;
//! assert_eq!(i64::try_from(total)?, 6);
//! # Ok::<(), tvm_ffi::Error>(())
//! ```
//!
//! # FFI contracts
//!
//! Some guarantees rest on contracts every language keeps, which Rust cannot
//! check:
//!
//! * An `ffi.String` holds UTF-8, so [`String::as_str`] does not validate it.
//! * Types, functions and attributes register during static initialization,
//!   such as in a [`static_init_block!`], before other threads read them:
//!   readers of the runtime's tables take no lock.
//! * A type key describes the same type in every library that declares it;
//!   see [`rs_object`].
//! * A [`TensorView`] received as a function argument is valid only during
//!   the call, and so is an [`Any`] holding a `DLTensor*` converted from it
//!   or from an [`AnyView`].
#![warn(missing_docs)]

// Lets macro-generated `::tvm_ffi` paths resolve inside this crate too.
extern crate self as tvm_ffi;

pub mod any;
pub mod big_int;
pub mod collections;
pub mod derive;
pub mod device;
pub mod dtype;
pub mod enum_ref;
pub mod error;
pub mod extra;
pub mod function;
#[doc(hidden)]
pub mod function_internal;
#[doc(hidden)]
pub mod macros;
#[doc(hidden)]
pub mod match_any_internal;
pub mod object;
pub mod optional;
pub mod reflection;
pub mod rvalue_ref;
pub mod static_init;
pub mod string;
pub mod type_traits;
mod util;
pub use tvm_ffi_sys;

pub use crate::any::{Any, AnyView};
pub use crate::big_int::BigInt;
pub use crate::collections::array::Array;
pub use crate::collections::list::List;
pub use crate::collections::map::Map;
pub use crate::collections::shape::Shape;
pub use crate::collections::tensor::{CPUNDAlloc, NDAllocator, Tensor, TensorView};
pub use crate::device::{current_stream, with_stream};
pub use crate::dtype::DLDataTypeExt;
pub use crate::enum_ref::{EnumObj, EnumRef};
pub use crate::error::{Error, ErrorKind, Result};
pub use crate::error::{
    ATTRIBUTE_ERROR, INDEX_ERROR, INTERNAL_ERROR, KEY_ERROR, OVERFLOW_ERROR, RUNTIME_ERROR,
    TYPE_ERROR, VALUE_ERROR, ZERO_DIVISION_ERROR,
};
pub use crate::extra::dispatch::{Dispatcher, IntoDispatcher};
pub use crate::extra::module::Module;
pub use crate::extra::structural_eq_hash::{
    get_first_structural_mismatch, get_first_structural_mismatch_with, structural_equal,
    structural_equal_with, structural_hash, structural_hash_with, StructuralOptions,
};
pub use crate::extra::structural_map::structural_map;
pub use crate::extra::structural_mutate::{
    structural_mutate, InplaceMode, MutationView, StructuralMutateEngine, StructuralMutatorBase,
    StructuralMutatorObj, StructuralMutatorView,
};
pub use crate::extra::structural_visit::{
    structural_visit, DefRegionKind, StructuralVisitEngine, StructuralVisitorBase,
    StructuralVisitorObj, StructuralVisitorView, VisitInterrupt, VisitInterruptObj,
};
pub use crate::extra::structural_walk::{structural_walk, IntoWalkResult, WalkOrder, WalkResult};
pub use crate::extra::unchanged::{Unchanged, UnchangedOr};
pub use crate::function::Function;
pub use crate::function_internal::AsPackedCallable;
pub use crate::object::{
    ImmutableObject, IntoObjectRef, Object, ObjectArc, ObjectCore, ObjectCoreWithExtraItems,
    ObjectIdentity, ObjectRefAnyCompatible, ObjectRefCore,
};
pub use crate::optional::{Optional, OptionalCompatible};
pub use crate::rvalue_ref::RValueRef;
pub use crate::static_init::run_static_init_blocks;
pub use crate::string::{Bytes, String};
pub use crate::type_traits::{AnyCompatible, TypeSubsumes};
/// Generate runtime type dispatch from the `prefix_*` methods of an impl.
///
/// `#[dispatch(prefix, (First, Args...), Ret)]` on an inherent impl makes its
/// `prefix_*` methods handlers and adds a `dispatch(first, args...) ->
/// Option<Ret>` method. `dispatch` calls the first handler, in source order,
/// whose first parameter matches the runtime type of `first`, and returns
/// `None` when no handler matches. The macro also implements [`Dispatcher`]
/// for `&Self` or `&mut Self`, so the type can drive [`structural_walk`] and
/// the other traversals, whose engines call it and run their own default
/// recursion; no prefix has a reserved meaning. Handlers without an `inline`
/// attribute are marked `#[inline(always)]`.
///
/// # Arguments
///
/// * `prefix` - The handler name prefix; `describe` selects `describe_int`,
///   `describe_str` and so on.
/// * `(First, Args...)` - The complete input tuple. `First`, the value to
///   match, is `AnyView<'_>` or [`MutationView<'_>`], including aliases.
///   `Args` are passed through: each handler takes a leading prefix of them
///   without cloning, and need not take all of them.
/// * `Ret` - The result type. A handler's result is converted with
///   `Into<Ret>`. When both the handler's result and `Ret` are written
///   `Result<..>`, the success value is converted instead, so a
///   `Result<UnchangedOr<MyPair>>` handler fits a `Result<UnchangedOr<Any>>`
///   dispatcher.
///
/// # Handlers
///
/// A handler's first parameter selects what it matches, as for
/// [`match_any!`]: an owned FFI value such as `i64` or `Array<Any>`, converted
/// strictly; `&Type` (such as `&Array<i64>` or `&tvm_ffi::String`) or
/// `&NodeObj`, which borrow the value or the object node in place without
/// changing a reference count; `MutationView<'_, T>`,
/// which keeps the in-place mode; or `AnyView<'_>`, which
/// matches every value and must be the last handler.
///
/// Only the outer `None` means that no handler matched: a handler error, or a
/// `None` inside `Ret`, is final. Handlers share one receiver, `&self` or
/// `&mut self`, and the same method generics and bounds; shared generics must
/// occur in the receiver, the input tuple or `Ret`. Use the same lifetime name
/// in an input and `Ret` to return a borrow of that input; elided output
/// lifetimes borrow the receiver. Handler-level `cfg` attributes are not
/// supported; configure the whole impl instead. Generate one dispatcher per
/// type; when inference is ambiguous, give the expected result type.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{dispatch, AnyView};
///
/// struct Describer;
///
/// #[dispatch(describe, (AnyView<'_>,), String)]
/// impl Describer {
///     fn describe_int(&self, value: i64) -> String {
///         format!("integer {value}")
///     }
///
///     fn describe_str(&self, value: &tvm_ffi::String) -> String {
///         format!("string {}", value.as_str())
///     }
/// }
///
/// let text = tvm_ffi::String::from("hi");
/// assert_eq!(Describer.dispatch(AnyView::from(&1_i64)).as_deref(), Some("integer 1"));
/// assert_eq!(Describer.dispatch(AnyView::from(&text)).as_deref(), Some("string hi"));
/// assert_eq!(Describer.dispatch(AnyView::from(&1.5_f64)), None);
/// ```
///
/// [`MutationView<'_>`]: crate::MutationView
/// [`Dispatcher`]: crate::Dispatcher
/// [`structural_walk`]: crate::structural_walk
pub use tvm_ffi_macros::dispatch;

/// Match the runtime type of a value, the way `match` matches an enum.
///
/// The value is an [`Any`], an [`AnyView`], a managed reference or a
/// [`MutationView`]. It is evaluated once and taken by value; write `&value`
/// to keep it. Arms are tried in order, and the first arm whose type matches
/// runs; a guard that is false moves on to the next arm. The last arm must be
/// an unguarded `_ => ...`, since new types can be registered at any time.
///
/// * `Type(name)` binds an owned value of `Type`, such as `i64` or
///   `Array<Any>`.
/// * `Type(ref name)` borrows a managed reference, and `NodeObj(ref name)`
///   borrows the object node, without changing its reference count.
/// * `MutationView<Type>(name)` moves the in-place mode into a
///   typed view.
///
/// Matching is strict: an `i64` arm does not match a stored `f64`, and a value
/// is never truncated. Type aliases and qualified paths may name `Type`. Typed
/// `MutationView` arms do not match a `Map`, including a nested one; use
/// `MutationView<Any>` for those. Ordinary `Map<K, V>` arms do match.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{match_any, Any, Array};
///
/// fn kind(value: &Any) -> String {
///     match_any! {
///         value {
///             i64(n) => format!("integer {n}"),
///             tvm_ffi::String(s) => format!("string {}", s.as_str()),
///             Array<Any>(arr) => format!("array of {} items", arr.len()),
///             _ => "something else".to_string(),
///         }
///     }
/// }
///
/// assert_eq!(kind(&Any::from(42_i64)), "integer 42");
/// assert_eq!(kind(&Any::from(Array::from_vec(vec![1_i64, 2, 3]))), "array of 3 items");
/// assert_eq!(kind(&Any::from(1.5_f64)), "something else");
/// ```
pub use tvm_ffi_macros::match_any;

/// Define an object type in Rust that every language can use.
///
/// `#[rs_object("type.Key", options...)]` registers the struct as an object
/// type with the reflection of every field, so that structural equality and
/// hashing, `repr`, deep copy and Python see the fields. The first field,
/// `base_`, is the parent type's object, embedded at the start of the struct:
/// `base_: Object` for a type under `ffi.Object`; the trailing `_` leaves
/// `base` free for a reflected field. Declare the managed reference with
/// [`#[derive(ObjectRef)]`](derive::ObjectRef).
///
/// The macro adds to the struct:
///
/// * `#[repr(C)]` and `Deref` to the parent's object, so `obj.field` reads the
///   fields of both.
/// * `MyPairObj::new(base_, fields...)`, which takes the fields in order: a
///   scalar (`bool`, an integer or a float) as its own type, so that a literal
///   infers as it, and any other field as `impl Into<T>`. This is the
///   object-level allocator, which sees only tokens and so cannot tell an
///   object reference from a value; a constructor on the reference takes an
///   object reference as [`impl IntoObjectRef<T>`](IntoObjectRef), as the
///   ones `tvm-ffi-stubgen` generates do.
///
/// # Arguments
///
/// * `"type.Key"` - The type key.
/// * `final` - The type has no subtypes.
/// * `s_eq_hash_kind = "TreeNode"` - The [`SEqHashKind`](crate::reflection::SEqHashKind) of structural
///   equality; by default the parent's.
/// * `child_slots = N`, `child_slots_can_overflow = bool` - Type indices kept
///   for subtypes; 0 and `true` by default.
/// * `reflection = path` - A `fn(&mut ObjectDef<Self>)` that registers the
///   type's methods.
/// * `crate = path` - Where the generated code finds tvm-ffi, for a crate that
///   depends on it only through a re-export; by default `tvm_ffi`.
///
/// A field takes `#[reflect(writable, seq_hash_ignore, ...)]` for the
/// [`FieldFlags`](crate::reflection::FieldFlags) of those names, and `name = "..."` or `doc = "..."` to
/// register it under another name or with a doc string, by default its `///`
/// comment. As a C++ `def_rw` field, a `writable` field can be replaced
/// through reflection, such as from Python, while the object is shared.
///
/// # Requirements
///
/// A field's type must implement `AnyCompatible` and `Clone`; an `init_off`
/// field's type must take all-zero bytes as a value. An optional field is an
/// `Option` of an object reference or an [`Optional`] of any other value. The parent's object must
/// implement `ShallowCopy` and `UnsafeInit`, as `rs_object` types and non-final
/// types bound by `tvm-ffi-stubgen` do. The type registers on its first
/// `type_index()`, with its fields, methods, `__ffi_new__` and, unless the
/// methods define it, `__ffi_init__`.
///
/// A key already registered, by another library or a binding, must describe
/// the same type: `type_index()` checks its parent, depth and size, not its
/// fields.
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::ObjectRef;
/// use tvm_ffi::{rs_object, Object, ObjectArc};
///
/// #[rs_object("my_ext.MyPair", final)]
/// pub struct MyPairObj {
///     base_: Object,
///     pub first: i64,
///     #[reflect(seq_hash_ignore)]
///     pub second: tvm_ffi::String,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// pub struct MyPair {
///     data: ObjectArc<MyPairObj>,
/// }
///
/// let pair = MyPair::from_object(MyPairObj::new(Object::new(), 1, "one"));
/// assert_eq!(pair.first, 1);
/// assert_eq!(pair.second.as_str(), "one");
/// ```
pub use tvm_ffi_macros::rs_object;

pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
pub use tvm_ffi_sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, TVMFFIAny, TVMFFIObject, TVMFFIStreamHandle,
};
