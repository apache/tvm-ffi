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
/// `#[dispatch(prefix, (First, Args...), Ret)]` adds a `dispatch(first,
/// args...) -> Option<Ret>` method, which calls the first `prefix_*` handler
/// whose first parameter matches the runtime type of `first`, or returns
/// `None`. It also implements [`Dispatcher`], so the type can drive
/// [`structural_walk`] and the other traversals.
///
/// # Arguments
///
/// - `prefix`: the handler name prefix.
/// - `(First, Args...)`: the input tuple. `First`, the value to match, is
///   `AnyView<'_>` or [`MutationView<'_>`]; each handler takes a leading
///   prefix of `Args`.
/// - `Ret`: the result type, converted from each handler's result with
///   `Into`; for a `Result` handler, its success value is converted.
///
/// A handler's first parameter is matched as by [`match_any!`]: an owned
/// value such as `i64`, `&NodeObj` to borrow the node, `MutationView<'_, T>`,
/// or `AnyView<'_>` as a last catch-all.
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
/// }
///
/// let doubled = Double.dispatch(AnyView::from(&21_i64)).unwrap()?;
/// assert_eq!(doubled.into_option().unwrap().try_as::<i64>(), Some(42));
/// assert!(Double.dispatch(AnyView::from(&-1_i64)).unwrap().is_err());
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
///
/// [`MutationView<'_>`]: crate::MutationView
/// [`Dispatcher`]: crate::Dispatcher
/// [`structural_walk`]: crate::structural_walk
pub use tvm_ffi_macros::dispatch;

/// Match the runtime type of a value, the way `match` matches an enum.
///
/// The value is an [`Any`], [`AnyView`], managed reference or
/// [`MutationView`]. Arms are tried in order, and the last must be `_ => ...`,
/// since new types can be registered at any time. Matching is strict: an `i64`
/// arm does not match a stored `f64`.
///
/// - `Type(name)` binds an owned value, such as `i64` or `Array<Any>`.
/// - `NodeObj(ref name)` borrows the object node without retaining it.
/// - `MutationView<Type>(name)` narrows a mutation view to `Type`.
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

/// Define an object type in Rust, as Python's `py_class` does.
///
/// `#[rs_object("type.Key", options...)]` registers the struct as an object
/// type. Its first field is the parent's object, `base: Object` for a type
/// under `ffi.Object`; the other fields are reflected, so C++ structural
/// equality, `repr` and Python see them. The macro adds
/// `MyPairObj::new(base, fields...)` and `Deref` to the parent's object;
/// define the managed reference with [`def_object_ref!`].
///
/// # Arguments
///
/// - `"type.Key"`: the type key.
/// - `final`: the type has no subtypes.
/// - `s_eq_hash_kind = "TreeNode"`: the [`SEqHashKind`]
///   of structural equality; by default the parent's.
/// - `reflection = path`: a `fn(&mut ObjectDef<Self>)` that registers methods.
/// - `crate = path`: where the generated code finds tvm-ffi, for a crate that
///   uses it through a re-export.
///
/// A field takes `#[reflect(writable, seq_hash_ignore, ...)]` for its
/// [`FieldFlags`]. Field types must implement
/// `AnyCompatible` and `Clone`.
///
/// # Examples
///
/// ```
/// use tvm_ffi::object::ObjectRef;
/// use tvm_ffi::{def_object_ref, rs_object, Object};
///
/// #[rs_object("my_ext.MyPair", final)]
/// pub struct MyPairObj {
///     base: Object,
///     pub first: i64,
///     #[reflect(seq_hash_ignore)]
///     pub second: tvm_ffi::String,
/// }
/// def_object_ref!(pub MyPair, ObjectRef, MyPairObj);
///
/// let pair = MyPair::from(MyPairObj::new(Object::new(), 1, "one"));
/// assert_eq!(pair.first, 1);
/// assert_eq!(pair.second.as_str(), "one");
/// ```
pub use tvm_ffi_macros::rs_object;

/// Define the managed reference to an object type, as C++
/// `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE` does.
///
/// `def_object_ref!(TypeName, ParentType, ObjectName)` defines `TypeName`,
/// which holds an `ObjectName`, dereferences to it, and converts into
/// `ParentType`, the reference to the parent's object (`ObjectRef` for a type
/// under `ffi.Object`). Attributes and visibility before `TypeName` go on the
/// struct; `crate = path` at the end is as for [`rs_object`](macro@rs_object).
///
/// # Examples
///
/// ```
/// use tvm_ffi::object::ObjectRef;
/// use tvm_ffi::{def_object_ref, rs_object, Any, Object};
///
/// #[rs_object("my_ext.MyPair", final)]
/// pub struct MyPairObj {
///     base: Object,
///     pub first: i64,
///     pub second: i64,
/// }
///
/// def_object_ref!(
///     /// A pair of integers.
///     pub MyPair,
///     ObjectRef,
///     MyPairObj
/// );
///
/// let pair = MyPair::from(MyPairObj::new(Object::new(), 1, 2));
/// assert_eq!(pair.first + pair.second, 3);
/// let object = ObjectRef::from(pair);
/// assert!(MyPair::try_from(Any::from(object)).is_ok());
/// ```
pub use tvm_ffi_macros::def_object_ref;

pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
pub use tvm_ffi_sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, TVMFFIAny, TVMFFIObject, TVMFFIStreamHandle,
};
