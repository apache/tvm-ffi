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

/// Get the tvm-rt crate name
/// \return The tvm-rt crate name
pub(crate) fn get_tvm_ffi_crate() -> TokenStream {
    if env::var("CARGO_PKG_NAME").unwrap() == "tvm-ffi" {
        quote!(crate)
    } else {
        quote!(tvm_ffi)
    }
}

/// Get an attribute by name from a derive input
///
/// # Arguments
/// * `derive_input` - The derive input to get the attribute from
/// * `name` - The name of the attribute to get
///
/// # Returns
/// * `Option<&syn::Attribute>` - The attribute if it exists
pub(crate) fn get_attr<'a>(
    derive_input: &'a syn::DeriveInput,
    name: &str,
) -> Option<&'a syn::Attribute> {
    derive_input.attrs.iter().find(|a| a.path.is_ident(name))
}

/// Convert an attribute to a string
///
/// # Arguments
/// * `attr` - The attribute to convert
///
/// # Returns
/// * `syn::LitStr` - The string value of the attribute
pub(crate) fn attr_to_str(attr: &syn::Attribute) -> syn::LitStr {
    match attr.parse_meta() {
        Ok(syn::Meta::NameValue(syn::MetaNameValue {
            lit: syn::Lit::Str(s),
            ..
        })) => s,
        Ok(_m) => panic!("Expected a string literal, got"),
        Err(e) => panic!("{}", e),
    }
}

/// Convert an attribute to an integer
///
/// # Arguments
/// * `attr` - The attribute to convert
///
/// # Returns
/// * `syn::Result<syn::Expr>` - The integer value of the attribute
pub(crate) fn attr_to_expr(attr: &syn::Attribute) -> syn::Result<syn::Expr> {
    let parser = |input: syn::parse::ParseStream| {
        input.parse::<syn::Expr>() // parse expression after '='
    };
    syn::parse::Parser::parse2(parser, attr.tokens.clone())
}
