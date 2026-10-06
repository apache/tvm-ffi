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
mod utils;

/// Generate `walk`, `map`, `visit`, or `mutate` dispatch from an inherent impl.
///
/// `#[dispatch(visit, policy = expr)]` and `#[dispatch(mutate, policy = expr)]`
/// use a `ContextPolicy<Self>` or `MutContextPolicy<Self>` for default recursion.
/// The expression runs at each default descent and may read `self` for configuration;
/// mutable traversal state stays on `self`. Policy tuples compose as usual.
#[proc_macro_error]
#[proc_macro_attribute]
pub fn dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    dispatch::dispatch(attr, item)
}

/// Match object-backed values carried by an Any-compatible scrutinee.
///
/// The scrutinee may be an owned object handle, `Any`, or `AnyView`. Convert an
/// already-borrowed object handle to `AnyView` before invoking the macro.
///
/// Non-object values skip the typed patterns and use the `_` fallback.
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
        type_child_slots_can_overflow
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
