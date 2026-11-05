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
// Lets macro-generated `::tvm_ffi` paths resolve inside this crate too.
extern crate self as tvm_ffi;

pub mod any;
pub mod big_int;
pub mod collections;
pub mod derive;
pub mod device;
pub mod dtype;
pub mod error;
pub mod extra;
pub mod function;
pub mod function_internal;
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
pub use crate::error::{
    ATTRIBUTE_ERROR, INDEX_ERROR, INTERNAL_ERROR, KEY_ERROR, OVERFLOW_ERROR, RUNTIME_ERROR,
    TYPE_ERROR, VALUE_ERROR, ZERO_DIVISION_ERROR,
};
pub use crate::error::{Error, ErrorKind, Result};
pub use crate::extra::dispatch::{Dispatcher, IntoDispatcher};
pub use crate::extra::module::Module;
pub use crate::extra::structural_map::structural_map;
pub use crate::extra::structural_mutate::{
    InplaceMode, MutationView, StructuralMutateEngine, StructuralMutatorBase, StructuralMutatorObj,
    StructuralMutatorView, StructuralVarRemap, structural_mutate,
};
pub use crate::extra::structural_visit::{
    DefRegionKind, StructuralVisitEngine, StructuralVisitorBase, StructuralVisitorObj,
    StructuralVisitorView, VisitInterrupt, VisitInterruptObj, structural_visit,
};
pub use crate::extra::structural_walk::{IntoWalkResult, WalkOrder, WalkResult, structural_walk};
pub use crate::extra::unchanged::{Unchanged, UnchangedOr};
pub use crate::function::Function;
pub use crate::object::ObjectRefCast;
pub use crate::object::{
    Object, ObjectArc, ObjectCore, ObjectCoreWithExtraItems, ObjectIdentity, ObjectRefCore,
};
pub use crate::optional::{Optional, OptionalCompatible};
pub use crate::reflection::{
    FieldFlags, FieldGetter, ObjectDef, SEqHashKind, ShallowCopy, TypeAttrColumn, UnsafeInit,
    get_type_attr,
};
pub use crate::rvalue_ref::RValueRef;
pub use crate::static_init::run_static_init_blocks;
pub use crate::string::{Bytes, String};
pub use crate::type_traits::AnyCompatible;
/// Generate runtime type dispatch from the `prefix_*` methods of an impl.
///
/// `#[dispatch(prefix, (First, Args...), Ret)]` on an inherent impl makes its
/// `prefix_*` methods handlers and adds a `dispatch(first, args...) ->
/// Option<Ret>` method. `dispatch` calls the first handler, in source order,
/// whose first parameter matches the runtime type of `first`, and returns
/// `None` when no handler matches. Other methods of the impl are unchanged.
///
/// # Arguments
///
/// - `prefix`: the handler name prefix; `describe` selects `describe_int`,
///   `describe_array` and so on.
/// - `(First, Args...)`: the complete input tuple. `First` is the value to
///   match, an `AnyView<'_>`, or a [`MutationView<'_>`] for handlers that may
///   mutate in place. `Args` are passed through: each handler takes a leading
///   prefix of them, so a handler may leave out the trailing ones.
/// - `Ret`: the result type. A handler's result is converted with `Into<Ret>`.
///   When both the handler's result and `Ret` are written `Result<..>`, the
///   success value is converted instead, so a `Result<UnchangedOr<Expr>>`
///   handler fits a `Result<UnchangedOr<Any>>` dispatcher.
///
/// A handler's first parameter selects what it matches:
///
/// - an owned value such as `i64`, [`String`] or `Array<Any>`, converted
///   strictly, as [`match_any!`] does;
/// - `&NodeObj`, which borrows the object node without retaining it;
/// - `MutationView<'_, T>`, which keeps the permission to mutate in place;
/// - `AnyView<'_>`, which matches every value and must be the last handler.
///
/// Every handler takes the same receiver, `&self` or `&mut self`. The macro also
/// implements [`Dispatcher`] for `&Self` or `&mut Self`, so the type can be
/// passed to [`structural_walk`], [`structural_map`] and the other traversals.
/// A handler error, or a `None` inside `Ret`, is final; only the outer `None`
/// means that no handler matched. Handler-level `cfg` attributes are not
/// supported; configure the whole impl instead.
///
/// # Examples
///
/// Describe nested arrays, with a separator between items:
///
/// ```
/// use tvm_ffi::{dispatch, Any, AnyView, Array};
///
/// struct Describer;
///
/// #[dispatch(describe, (AnyView<'_>, &str), String)]
/// impl Describer {
///     fn describe_array(&self, arr: Array<Any>, sep: &str) -> String {
///         let items: Vec<String> = arr
///             .iter()
///             .map(|item| {
///                 let item = AnyView::from(&item);
///                 self.dispatch(item, sep).unwrap_or_else(|| "?".into())
///             })
///             .collect();
///         format!("[{}]", items.join(sep))
///     }
///
///     fn describe_int(&self, value: i64) -> String {
///         value.to_string()
///     }
///
///     fn describe_str(&self, value: tvm_ffi::String) -> String {
///         value.to_string()
///     }
/// }
///
/// let value = Any::from(Array::new(vec![
///     Any::from(1_i64),
///     Any::from(tvm_ffi::String::from("a")),
///     Any::from(Array::new(vec![2_i64, 3])),
/// ]));
/// let text = Describer.dispatch(AnyView::from(&value), ", ");
/// assert_eq!(text.as_deref(), Some(r#"[1, "a", [2, 3]]"#));
///
/// // No handler matches a float.
/// assert_eq!(Describer.dispatch(AnyView::from(&1.5_f64), ", "), None);
/// ```
///
/// A handler can return a typed `Result`, which converts into the declared
/// erased one:
///
/// ```
/// use tvm_ffi::{bail, dispatch, Any, AnyView, Result, UnchangedOr, VALUE_ERROR};
///
/// struct Double;
///
/// #[dispatch(double, (AnyView<'_>,), Result<UnchangedOr<Any>>)]
/// impl Double {
///     fn double_int(&self, value: i64) -> Result<UnchangedOr<i64>> {
///         if value < 0 {
///             bail!(VALUE_ERROR, "negative value {}", value);
///         }
///         Ok(UnchangedOr::changed(value * 2))
///     }
///
///     fn double_any(&self, _value: AnyView<'_>) -> Result<UnchangedOr<Any>> {
///         Ok(UnchangedOr::unchanged())
///     }
/// }
///
/// let doubled = Double.dispatch(AnyView::from(&21_i64)).unwrap()?;
/// assert_eq!(doubled.into_option().unwrap().try_as::<i64>(), Some(42));
/// assert!(Double.dispatch(AnyView::from(&-1_i64)).unwrap().is_err());
/// assert!(Double.dispatch(AnyView::from(&1.5_f64)).unwrap()?.is_unchanged());
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
///
/// [`MutationView<'_>`]: crate::MutationView
/// [`Dispatcher`]: crate::Dispatcher
/// [`structural_walk`]: crate::structural_walk
/// [`structural_map`]: crate::structural_map
pub use tvm_ffi_macros::dispatch;

/// Match the runtime type of a value, the way `match` matches an enum.
///
/// ```text
/// match_any! {
///     value {
///         Type(binding) if guard => expr,
///         ...
///         _ => fallback,
///     }
/// }
/// ```
///
/// `value` is an [`Any`], an [`AnyView`], a managed reference or a
/// [`MutationView`]. Arms are tried in order, and the first arm whose type
/// matches runs; the last arm must be an unguarded `_ => ...`, since new types
/// can be registered at any time. A guard that is false moves on to the next
/// arm. Each arm's pattern selects its binding:
///
/// - `Type(name)` binds an owned value of `Type`, such as `i64`, [`String`]
///   or `Array<Any>`.
/// - `Type(ref name)` borrows a managed reference, and `NodeObj(ref name)`
///   borrows the object node, without changing its reference count.
/// - `MutationView<Type>(name)` moves the permission to mutate in place into a
///   typed view.
///
/// `value` is evaluated once and taken by value; write `&value` to keep it.
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
///             i64(n) if n < 0 => format!("negative integer {n}"),
///             i64(n) => format!("integer {n}"),
///             tvm_ffi::String(s) => format!("string {}", s.as_str()),
///             Array<Any>(arr) => format!("array of {} items", arr.len()),
///             _ => "something else".to_string(),
///         }
///     }
/// }
///
/// assert_eq!(kind(&Any::from(42_i64)), "integer 42");
/// assert_eq!(kind(&Any::from(-1_i64)), "negative integer -1");
/// assert_eq!(kind(&Any::from(tvm_ffi::String::from("hi"))), "string hi");
/// assert_eq!(kind(&Any::from(Array::new(vec![1_i64, 2, 3]))), "array of 3 items");
/// assert_eq!(kind(&Any::from(1.5_f64)), "something else");
/// ```
///
/// An object arm borrows the node and reads its fields:
///
/// ```
/// use tvm_ffi::collections::array::ArrayObj;
/// use tvm_ffi::{match_any, Array};
///
/// let array = Array::new(vec![1_i64, 2, 3]);
/// let size = match_any! {
///     &array {
///         ArrayObj(ref node) => node.size,
///         _ => 0,
///     }
/// };
/// assert_eq!(size, 3);
/// ```
pub use tvm_ffi_macros::match_any;

/// Define an object type in Rust from its object struct, as Python's
/// `py_class` does.
///
/// `#[rs_object("type.Key", options...)]` registers the type key with the
/// reflection of every field, so that C++ structural equality and hashing,
/// `repr`, deep copy and Python see the fields.
///
/// # Examples
///
/// ```
/// use tvm_ffi::object::ObjectRef;
/// use tvm_ffi::{def_object_ref, rs_object, Object, ObjectRefCore};
///
/// /// A named shape, under `ffi.Object`.
/// #[rs_object("my_ext.Shape2D")]
/// pub struct Shape2DObj {
///     base: Object,
///     pub name: tvm_ffi::String,
/// }
/// def_object_ref!(pub Shape2D, ObjectRef, Shape2DObj);
///
/// /// A rectangle, under `Shape2D`.
/// #[rs_object("my_ext.Rect", final)]
/// pub struct RectObj {
///     base: Shape2DObj,
///     pub width: f64,
///     pub height: f64,
///     #[reflect(seq_hash_ignore)]
///     pub note: tvm_ffi::String,
/// }
/// def_object_ref!(pub Rect, Shape2D, RectObj);
///
/// let rect = Rect::from(RectObj::new(Shape2DObj::new(Object::new(), "box"), 2.0, 3.0, ""));
/// assert_eq!(rect.width * rect.height, 6.0);
/// // The object dereferences to its parent's fields.
/// assert_eq!(rect.name.as_str(), "box");
///
/// let shape = Shape2D::from(rect);
/// assert_eq!(shape.downcast_ref::<RectObj>().unwrap().width, 2.0);
/// ```
///
/// The first field, `base`, is the parent's object, as the base class of a C++
/// object is: `base: Object` for a type under `ffi.Object`. The parent's object
/// must implement `ShallowCopy` and [`UnsafeInit`], as `rs_object` types and
/// non-final C++ types bound by `tvm-ffi-stubgen` do. Every other field is
/// reflected; `#[reflect(writable, seq_hash_ignore, ...)]` gives a field the
/// [`FieldFlags`] of those names, and `name` and `doc` as for
/// `derive(Object)`. The macro adds to the struct:
///
/// - `#[repr(C)]` and `Deref` to the parent's object, so `obj.field` reads the
///   fields of both.
/// - `PairObj::new(base, ...)`, which builds the object from the parent's
///   object and the fields in order: a scalar (`bool`, an integer or a float)
///   as its own type, so that a literal infers as it, and any other field from
///   anything that converts into it, as the `new` of a type bound by
///   `tvm-ffi-stubgen` takes them.
///
/// [`def_object_ref!`] defines the reference to it and its conversion into the
/// reference to its parent.
///
/// After the type key:
///
/// - `final`, `s_eq_hash_kind = "TreeNode"`, `child_slots = N`,
///   `child_slots_can_overflow = bool` and `reflection = path` are
///   `derive(Object)`'s `#[type_final]`, `#[type_s_eq_hash_kind]`,
///   `#[type_child_slots]`, `#[type_child_slots_can_overflow]` and
///   `#[type_reflection]`.
/// - `crate = path` is where the generated code finds tvm-ffi
///   (`crate = ::my_lib::tvm_ffi`), for a crate that depends on it only through
///   a re-export; by default `tvm_ffi`.
///
/// The type is a `derive(Object)` type with `#[type_register]`: it registers on
/// its first `type_index()`, with the fields, the methods,
/// `__ffi_shallow_copy__` and `__ffi_type_final__`, and a creator that makes
/// the object with every field undefined for reflection to set, as C++ makes one
/// with `UnsafeInit`, from which it registers `__ffi_new__` and, unless the
/// methods define it, `__ffi_init__`; the macro implements [`UnsafeInit`] for
/// the object. A field's type must implement `AnyCompatible` and `Clone`; an
/// `init_off` field's type must take all-zero bytes as a value, as
/// `__ffi_init__` leaves the field so.
///
/// [`FieldFlags`]: crate::reflection::FieldFlags
/// [`UnsafeInit`]: crate::reflection::UnsafeInit
pub use tvm_ffi_macros::rs_object;

/// Define the managed reference to an object type, as C++
/// `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE` does.
///
/// `def_object_ref!(TypeName, ParentType, ObjectName)` defines `TypeName`, which
/// holds an `ObjectName`.
///
/// # Examples
///
/// ```
/// use tvm_ffi::object::ObjectRef;
/// use tvm_ffi::{def_object_ref, rs_object, Object};
///
/// #[rs_object("my_ext.Pair", final)]
/// pub struct PairObj {
///     base: Object,
///     pub first: i64,
///     pub second: i64,
/// }
///
/// def_object_ref!(
///     /// A pair of integers.
///     pub Pair,
///     ObjectRef,
///     PairObj
/// );
///
/// let pair = Pair::from(PairObj::new(Object::new(), 1, 2));
/// assert_eq!(pair.first + pair.second, 3);
/// let object = ObjectRef::from(pair);
/// assert!(Pair::try_from(tvm_ffi::Any::from(object)).is_ok());
/// ```
///
///
/// `Pair` holds a `PairObj` and is a `derive(ObjectRef)` type. It dereferences
/// to `PairObj`, and a `PairObj` converts into it.
///
/// `ParentType` is the reference to the parent's object: `Base` here, and
/// `tvm_ffi::object::ObjectRef` for a type under `ffi.Object`. `Pair` converts
/// into it by `impl_object_upcast!`, so its object must be a registered base of
/// `PairObj`, as that macro requires. A conversion into the reference to a
/// further ancestor takes its own `impl_object_upcast!`.
///
/// Attributes and visibility before the name go on the struct. `crate = path`
/// after the object type is where the generated code finds tvm-ffi, as for
/// [`rs_object`](macro@crate::rs_object).
pub use tvm_ffi_macros::def_object_ref;

pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
pub use tvm_ffi_sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, TVMFFIAny, TVMFFIObject, TVMFFIStreamHandle,
};
