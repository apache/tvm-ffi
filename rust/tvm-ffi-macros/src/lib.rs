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

use proc_macro::TokenStream;
use proc_macro_error::proc_macro_error;

mod def_object_ref;
mod dispatch;
mod match_any;
mod object_macros;
mod rs_class;
mod utils;

/// Generate source-ordered runtime type dispatch from `prefix_*` handlers.
///
/// `#[dispatch(prefix, (Carrier, Arg, ...), Ret)]` declares the complete input
/// tuple and result type. The first input is `AnyView<'_>` or `MutationView<'_>`
/// (including aliases); subsequent inputs are passed to matching handlers.
/// The macro implements `Dispatcher<(Carrier, Arg, ...), Option<Ret>>` for the
/// receiver reference (`&T` or `&mut T`). The inherent `.dispatch(value, ...)`
/// method owns matching; `.dispatch_tuple((value, ...))` unpacks and forwards.
/// Both methods always inline. A matched result is converted with `Into<Ret>`
/// and wrapped in `Some`. Only the outer `None` means no match; matched errors
/// and inner `None` are final. Conversion uses ordinary trait implementations,
/// without recursively converting results or wrapping values in `Ok`.
///
/// Handlers match an owned FFI value, a borrowed object node, or an optional
/// by-value `AnyView` catch-all, which must be last. Each handler receives its
/// typed first value and may consume an initial prefix of the remaining declared
/// arguments without cloning. Neither arguments nor results are deduced from a
/// handler signature, and no handler needs to consume every declared argument.
///
/// Handlers share the same receiver and method generic declarations and bounds.
/// Pattern-only, unbounded lifetimes may differ between handlers. Shared method
/// generics must occur in the receiver, declared tuple or declared return type.
/// Named lifetimes come from those shared declarations or the enclosing impl;
/// use the same name in input and output to return a borrow of that input.
/// Elided input lifetimes are independent; elided output lifetimes borrow the
/// receiver. `Self` in the declared types denotes the original implementor.
/// Handler-level `cfg`/`cfg_attr` attributes are unsupported; configure the whole
/// impl instead. A consuming `MutationView` retains mutation permission; its
/// typed misses recover the same handle, and `AnyView` cannot grant it.
///
/// Matching invokes the public `match_any!` macro and its exact-final-type
/// lookup for eligible larger handler sets. Dispatch also matches scalar payloads
/// and keeps owned argument conversions strict. Parameterized containers and
/// other non-exact matchers retain ordered, fully checked conversion.
///
/// The macro operates on inherent impls only. Traversal engines explicitly
/// call the dispatcher and configure their own default recursion;
/// no method prefix has a reserved traversal meaning.
///
/// Generate one dispatcher per type. Method generics can select different
/// argument and return types; use an expected result or explicit type arguments
/// when inference is ambiguous.
#[proc_macro_error]
#[proc_macro_attribute]
pub fn dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    dispatch::dispatch(attr, item)
}

/// Match payloads in `Any`, `AnyView`, object references or `MutationView`.
///
/// Write `Type(value)` to bind an owned or scalar result, `NodeObj(ref node)`
/// to borrow an object node, or `MutationView<Type>(capability)` to transfer a
/// mutation capability. Rust type aliases and qualified types are accepted.
/// Arms are tried in order; end with an unguarded `_ => ...` fallback.
///
/// The input is evaluated and taken by value once. Use `&value` to retain its
/// owner. Matching is strict: it does not coerce between value categories or
/// truncate stored numeric values. Guards may reject a match and try the next arm.
///
/// Typed mutation capabilities do not support Map payloads, including nested
/// Maps; use `MutationView<Any>` for those. Ordinary `Map<K, V>` arms are supported.
#[proc_macro_error]
#[proc_macro]
pub fn match_any(input: TokenStream) -> TokenStream {
    match_any::expand(input)
}

/// Derive `ObjectCore` for an object struct whose first field is its parent.
///
/// `#[type_key = "..."]` names the type, and `#[type_index(...)]` gives it a
/// static index. Otherwise a type defined in Rust declares `#[type_register]`:
/// it is registered under its parent on first use of `type_index()`, or takes
/// the index its key already has, as C++ `TVM_FFI_DECLARE_OBJECT_INFO` does.
/// Without `#[type_register]`, the type binds a key its defining library,
/// usually C++, has registered, and `type_index()` panics if that library is
/// not loaded yet, so a binding never registers a key with the wrong child
/// slots.
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
/// ...)]` adds the [`FieldFlags`] of those names, `name = "..."` registers
/// the field under another name, and `doc = "..."` sets its doc string,
/// otherwise the field's `///` doc comment. `#[type_s_eq_hash_kind =
/// "TreeNode"]` names the [`SEqHashKind`]; without it the type keeps its
/// nearest reflected ancestor's, as a C++ subclass inherits
/// `_type_s_eq_hash_kind`. `#[type_reflection(path)]` names a function
/// `fn(&mut ObjectDef<Self>)` that registers the type's methods, as C++
/// `refl::ObjectDef<T>().def(...)` does. A type with any of these attributes
/// also gets `ShallowCopy`, which shallow-copies the parent and clones every
/// other field, so the parent must implement `ShallowCopy` (`Object` and
/// reflected Rust types do) and the fields must be `Clone`; a reflected
/// field's type must implement `AnyCompatible`.
///
/// [`FieldFlags`]: ../tvm_ffi/reflection/struct.FieldFlags.html
/// [`SEqHashKind`]: ../tvm_ffi/reflection/enum.SEqHashKind.html
///
/// Registrations from Rust hold one lock, so they do not race each other. The
/// runtime's type table is not locked otherwise: as for C++, a type must not
/// be registered while another thread loads a library that registers types.
/// Calling `type_index()` before starting such threads registers a type
/// eagerly.
#[proc_macro_error]
#[proc_macro_derive(
    Object,
    attributes(
        type_key,
        type_index,
        type_final,
        type_register,
        type_child_slots,
        type_child_slots_can_overflow,
        type_s_eq_hash_kind,
        type_reflection,
        reflect
    )
)]
pub fn derive_object(input: TokenStream) -> TokenStream {
    object_macros::derive_object(input)
}

#[proc_macro_error]
#[proc_macro_derive(ObjectRef, attributes(type_key, type_index))]
pub fn derive_object_ref(input: TokenStream) -> TokenStream {
    object_macros::derive_object_ref(input)
}

/// Define an object type in Rust from its object struct, as Python's
/// `py_class` does: the type registers its key with the reflection of every
/// field, so that C++ structural equality and hashing, `repr`, deep copy and
/// Python see the fields.
///
/// ```ignore
/// use tvm_ffi::{def_object_ref, rs_class};
///
/// /// A pair of values under a `Base`.
/// #[rs_class("my_ext.Pair", final)]
/// pub struct PairObj {
///     base: BaseObj,
///     pub first: tvm_ffi::String,
///     pub second: tvm_ffi::String,
///     #[reflect(seq_hash_ignore)]
///     pub note: tvm_ffi::String,
/// }
///
/// def_object_ref!(pub Pair, Base, PairObj);
///
/// let pair = Pair::from(PairObj::new(BaseObj::new(/* ... */), "a", "b", ""));
/// let base = Base::from(pair);
/// ```
///
/// The first field, `base`, is the parent's object, as the base class of a C++
/// object is: `base: Object` for a type under `ffi.Object`. The parent's object
/// must implement `ShallowCopy` and [`UnsafeInit`], as `rs_class` types and
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
/// [`FieldFlags`]: ../tvm_ffi/reflection/struct.FieldFlags.html
/// [`UnsafeInit`]: ../tvm_ffi/reflection/trait.UnsafeInit.html
#[proc_macro_error]
#[proc_macro_attribute]
pub fn rs_class(args: TokenStream, item: TokenStream) -> TokenStream {
    rs_class::rs_class(args, item)
}

/// Define the reference to an object type, as C++
/// `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE` does:
/// `def_object_ref!(TypeName, ParentType, ObjectName)`.
///
/// ```ignore
/// use tvm_ffi::def_object_ref;
///
/// def_object_ref!(
///     /// A pair of values.
///     pub Pair,
///     Base,
///     PairObj
/// );
/// ```
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
/// [`rs_class`].
#[proc_macro_error]
#[proc_macro]
pub fn def_object_ref(input: TokenStream) -> TokenStream {
    def_object_ref::def_object_ref(input)
}
