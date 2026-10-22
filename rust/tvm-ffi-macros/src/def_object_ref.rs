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
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Attribute, DeriveInput, Ident, Path, Token, Type, Visibility};

use crate::object_macros::expand_object_ref;
use crate::utils::get_tvm_ffi_crate;

pub(crate) fn def_object_ref(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as DefObjectRefArgs);
    expand(&args)
        .unwrap_or_else(|error| error.to_compile_error())
        .into()
}

/// `def_object_ref!(#[attrs] pub TypeName, ParentType, ObjectName, crate = path)`.
struct DefObjectRefArgs {
    attrs: Vec<Attribute>,
    vis: Visibility,
    type_name: Ident,
    parent_type: Type,
    object_name: Type,
    tvm_ffi_crate: Option<Path>,
}

impl Parse for DefObjectRefArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse()?;
        let type_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let parent_type = input.parse()?;
        input.parse::<Token![,]>()?;
        let object_name = input.parse()?;
        let mut tvm_ffi_crate = None;
        if input.parse::<Option<Token![,]>>()?.is_some() && input.peek(Token![crate]) {
            input.parse::<Token![crate]>()?;
            input.parse::<Token![=]>()?;
            tvm_ffi_crate = Some(Path::parse_mod_style(input)?);
            input.parse::<Option<Token![,]>>()?;
        }
        if !input.is_empty() {
            return Err(input.error("expected `crate = <path>` after the object type"));
        }
        Ok(DefObjectRefArgs {
            attrs,
            vis,
            type_name,
            parent_type,
            object_name,
            tvm_ffi_crate,
        })
    }
}

fn expand(args: &DefObjectRefArgs) -> syn::Result<TokenStream2> {
    let tvm_ffi = match &args.tvm_ffi_crate {
        Some(path) => quote!(#path),
        None => get_tvm_ffi_crate(),
    };
    let DefObjectRefArgs {
        attrs,
        vis,
        type_name,
        parent_type,
        object_name,
        ..
    } = args;
    let ref_input: DeriveInput = syn::parse2(quote! {
        #vis struct #type_name {
            base: #tvm_ffi::object::ObjectArc<#object_name>,
        }
    })?;
    let ref_impl = expand_object_ref(&ref_input, &tvm_ffi);
    let doc = if attrs.iter().any(|a| a.path.is_ident("doc")) {
        None
    } else {
        let doc = format!(
            "The reference to [`{}`].",
            quote!(#object_name).to_string().replace(' ', "")
        );
        Some(quote!(#[doc = #doc]))
    };

    Ok(quote! {
        #doc
        #(#attrs)*
        #[repr(C)]
        #[derive(::core::clone::Clone)]
        #vis struct #type_name {
            base: #tvm_ffi::object::ObjectArc<#object_name>,
        }

        #ref_impl

        impl ::core::ops::Deref for #type_name {
            type Target = #object_name;
            #[inline]
            fn deref(&self) -> &#object_name {
                &self.base
            }
        }

        impl ::core::convert::From<#object_name> for #type_name {
            #[inline]
            fn from(object: #object_name) -> Self {
                Self {
                    base: #tvm_ffi::object::ObjectArc::new(object),
                }
            }
        }

        #tvm_ffi::impl_object_upcast!(#type_name => #parent_type);
    })
}
