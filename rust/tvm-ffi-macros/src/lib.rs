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

mod dispatch;
mod match_any;
mod object_macros;
mod rs_object;
mod utils;

// Documented on the `tvm_ffi::dispatch` re-export, whose examples run as
// tvm-ffi doctests.
#[proc_macro_error]
#[proc_macro_attribute]
pub fn dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    dispatch::dispatch(attr, item)
}

// Documented on the `tvm_ffi::match_any` re-export, whose examples run as
// tvm-ffi doctests.
#[proc_macro_error]
#[proc_macro]
pub fn match_any(input: TokenStream) -> TokenStream {
    match_any::expand(input)
}

/// Derive `ObjectCore` for a `#[repr(C)]` object struct whose first field is
/// its parent, so that a pointer to the object is one to its parent.
///
/// `#[type_key = "..."]` names the type, and `#[type_index(unsafe ...)]` gives
/// it a static index; the `unsafe` marks the caller's promise that the runtime
/// gives that index to this type. Otherwise a type defined in Rust declares
/// `#[type_register]`: it is registered under its parent when its library
/// loads, as a C++ type is in a static init block, or takes the index its key
/// already has, as C++
/// `TVM_FFI_DECLARE_OBJECT_INFO` does. Under a binding whose library is not
/// loaded yet, it registers on its first `type_index()` instead.
/// Without `#[type_register]`, the type binds a key its defining library,
/// usually C++, has registered, and `type_index()` panics if that library is
/// not loaded yet, so a binding never registers a key with the wrong child
/// slots. Either way `type_index()` panics unless the registered key has the
/// type's parent and depth and, when the type table records a size, the
/// type's size, or for a binding at least it. A binding that declares only part
/// of its object must not be constructed: keep its fields private.
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
/// field's type must implement `AnyCompatible`. As a C++ `def_rw` field, a
/// `writable` field can be replaced through reflection, such as from Python,
/// while the object is shared.
///
/// A key registered more than once, by `#[type_register]` types or bindings
/// in different libraries, must describe the same type each time:
/// `type_index()` checks the parent, depth and size, not the fields.
///
/// [`FieldFlags`]: ../tvm_ffi/reflection/struct.FieldFlags.html
/// [`SEqHashKind`]: ../tvm_ffi/reflection/enum.SEqHashKind.html
///
/// Registrations from Rust hold one lock, so they do not race each other. The
/// runtime's type table is not locked otherwise, and its readers take no
/// lock: as for C++, types register during static initialization, before
/// other threads use them or load a library that registers types.
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

// Documented on the `tvm_ffi::derive::ObjectRef` re-export, whose examples
// run as tvm-ffi doctests.
#[proc_macro_error]
#[proc_macro_derive(ObjectRef, attributes(object_ref))]
pub fn derive_object_ref(input: TokenStream) -> TokenStream {
    object_macros::derive_object_ref(input)
}

// Documented on the `tvm_ffi::rs_object` re-export, whose examples run as
// tvm-ffi doctests.
#[proc_macro_error]
#[proc_macro_attribute]
pub fn rs_object(args: TokenStream, item: TokenStream) -> TokenStream {
    rs_object::rs_object(args, item)
}
