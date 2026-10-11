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

/// Return the path used to reference the `tvm-ffi` crate in generated code.
pub(crate) fn get_tvm_ffi_crate() -> TokenStream {
    if env::var("CARGO_CRATE_NAME").unwrap() == "tvm_ffi" {
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
    derive_input.attrs.iter().find(|a| a.path().is_ident(name))
}

/// Convert an attribute to a string
///
/// # Arguments
/// * `attr` - The attribute to convert
///
/// # Returns
/// * `syn::LitStr` - The string value of the attribute
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
        _ => panic!("Expected a string literal"),
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
    match &attr.meta {
        syn::Meta::NameValue(value) => Ok(value.value.clone()),
        syn::Meta::List(_) => attr.parse_args(),
        _ => Err(syn::Error::new_spanned(
            attr,
            "expected `name = expression`",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    #[test]
    fn attribute_expressions() {
        for attr in [
            syn::parse_quote!(#[type_index(42)]),
            syn::parse_quote!(#[type_index = 42]),
        ] {
            assert_eq!(
                attr_to_expr(&attr).unwrap().to_token_stream().to_string(),
                "42"
            );
        }
        assert!(attr_to_expr(&syn::parse_quote!(#[type_index])).is_err());
        assert_eq!(
            attr_to_str(&syn::parse_quote!(#[type_key = "example.Key"])).value(),
            "example.Key"
        );
    }
}
