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
use proc_macro2::TokenStream;
use quote::quote;
use std::env;

pub(crate) fn get_tvm_ffi_crate() -> TokenStream {
    if env::var("CARGO_PKG_NAME").unwrap() == "tvm-ffi" {
        quote!(crate)
    } else {
        quote!(tvm_ffi)
    }
}

pub(crate) fn get_attr<'a>(
    derive_input: &'a syn::DeriveInput,
    name: &str,
) -> Option<&'a syn::Attribute> {
    derive_input.attrs.iter().find(|a| a.path().is_ident(name))
}

pub(crate) fn attr_to_str(attr: &syn::Attribute) -> syn::LitStr {
    match &attr.meta {
        syn::Meta::NameValue(syn::MetaNameValue {
            value:
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }),
            ..
        }) => s.clone(),
        _ => panic!("Expected #[attr = \"string\"] attribute"),
    }
}

pub(crate) fn attr_to_expr(attr: &syn::Attribute) -> syn::Result<syn::Expr> {
    attr.parse_args::<syn::Expr>()
}
