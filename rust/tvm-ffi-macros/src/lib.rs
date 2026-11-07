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

// Documented on the `tvm_ffi::derive::Object` re-export, whose examples run
// as tvm-ffi doctests.
#[proc_macro_error]
#[proc_macro_derive(
    Object,
    attributes(
        type_key,
        type_index,
        type_final,
        type_immutable,
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
