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
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Data, DeriveInput, Fields, LitBool, LitInt, LitStr, Path, Token};

use crate::object_macros::expand_object;
use crate::utils::get_tvm_ffi_crate;

pub(crate) fn rs_class(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as RsClassArgs);
    let item = parse_macro_input!(item as DeriveInput);
    expand(&args, &item)
        .unwrap_or_else(|error| error.to_compile_error())
        .into()
}

/// `#[rs_class("type.key", final, ...)]`.
struct RsClassArgs {
    type_key: LitStr,
    type_final: bool,
    s_eq_hash_kind: Option<LitStr>,
    child_slots: Option<LitInt>,
    child_slots_can_overflow: Option<LitBool>,
    reflection: Option<Path>,
    tvm_ffi_crate: Option<Path>,
}

impl Parse for RsClassArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let type_key = input.parse::<LitStr>().map_err(|error| {
            syn::Error::new(
                error.span(),
                "expected the type key first, as in `#[rs_class(\"my.Point\")]`",
            )
        })?;
        let mut args = RsClassArgs {
            type_key,
            type_final: false,
            s_eq_hash_kind: None,
            child_slots: None,
            child_slots_can_overflow: None,
            reflection: None,
            tvm_ffi_crate: None,
        };
        let mut seen = Vec::new();
        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            // `final` and `crate` are keywords.
            let name = input.call(syn::Ident::parse_any)?;
            let key = name.to_string();
            if seen.contains(&key) {
                return Err(syn::Error::new(name.span(), format!("duplicate `{key}`")));
            }
            if key != "final" {
                input.parse::<Token![=]>()?;
            }
            match key.as_str() {
                "final" => args.type_final = true,
                "s_eq_hash_kind" => args.s_eq_hash_kind = Some(input.parse()?),
                "child_slots" => args.child_slots = Some(input.parse()?),
                "child_slots_can_overflow" => args.child_slots_can_overflow = Some(input.parse()?),
                "reflection" => args.reflection = Some(input.parse()?),
                "crate" => args.tvm_ffi_crate = Some(Path::parse_mod_style(input)?),
                _ => {
                    return Err(syn::Error::new(
                        name.span(),
                        "expected `final`, `s_eq_hash_kind = \"<kind>\"`, `child_slots = <n>`, \
                         `child_slots_can_overflow = <bool>`, `reflection = <path>` or \
                         `crate = <path>`",
                    ));
                }
            }
            seen.push(key);
        }
        if args.type_final
            && (args.child_slots.is_some() || args.child_slots_can_overflow.is_some())
        {
            return Err(syn::Error::new(
                args.type_key.span(),
                "a final type has no children to reserve slots for",
            ));
        }
        Ok(args)
    }
}

const EXPECTED_BASE: &str =
    "the first field holds the parent's object: `base: Object` at the root, else `base: ParentObj`";

fn expand(args: &RsClassArgs, item: &DeriveInput) -> syn::Result<TokenStream2> {
    if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &item.generics,
            "an `rs_class` cannot be generic",
        ));
    }
    let Data::Struct(data) = &item.data else {
        return Err(syn::Error::new(
            item.ident.span(),
            "`rs_class` expects a struct",
        ));
    };
    let fields: Vec<&syn::Field> = match &data.fields {
        Fields::Named(fields) => fields.named.iter().collect(),
        Fields::Unit => Vec::new(),
        Fields::Unnamed(fields) => {
            return Err(syn::Error::new_spanned(
                fields,
                "`rs_class` expects named fields",
            ));
        }
    };
    let (base, fields) = match fields.split_first() {
        Some((base, fields)) if base.ident.as_ref().is_some_and(|id| id.unraw() == "base") => {
            (*base, fields)
        }
        Some((field, _)) => return Err(syn::Error::new_spanned(field, EXPECTED_BASE)),
        None => return Err(syn::Error::new(item.ident.span(), EXPECTED_BASE)),
    };
    if let Some(attr) = base.attrs.iter().find(|a| a.path.is_ident("reflect")) {
        return Err(syn::Error::new_spanned(
            attr,
            "`base` is the parent's object, whose fields the parent reflects",
        ));
    }
    let base_id = &base.ident;
    let base_ty = &base.ty;
    // `new` takes a scalar as its own type, so that a literal infers as it, and any
    // other field as `impl Into`.
    let params = fields.iter().map(|field| {
        let (name, ty) = (&field.ident, &field.ty);
        if is_exact_param(ty) {
            quote!(#name: #ty)
        } else {
            quote!(#name: impl ::core::convert::Into<#ty>)
        }
    });
    let inits = fields.iter().map(|field| {
        let name = &field.ident;
        if is_exact_param(&field.ty) {
            quote!(#name)
        } else {
            quote!(#name: ::core::convert::Into::into(#name))
        }
    });

    let tvm_ffi = match &args.tvm_ffi_crate {
        Some(path) => quote!(#path),
        None => get_tvm_ffi_crate(),
    };
    let attrs = &item.attrs;
    let vis = &item.vis;
    let obj = &item.ident;
    let key = &args.type_key;

    // `derive(Object)` reads the type from these attributes; the struct written out
    // carries neither them nor `#[reflect]`, which no derive on it would take.
    let mut derive_attrs = vec![quote!(#[type_key = #key]), quote!(#[type_register])];
    if args.type_final {
        derive_attrs.push(quote!(#[type_final]));
    }
    if let Some(kind) = &args.s_eq_hash_kind {
        derive_attrs.push(quote!(#[type_s_eq_hash_kind = #kind]));
    }
    if let Some(slots) = &args.child_slots {
        derive_attrs.push(quote!(#[type_child_slots = #slots]));
    }
    if let Some(overflow) = &args.child_slots_can_overflow {
        derive_attrs.push(quote!(#[type_child_slots_can_overflow = #overflow]));
    }
    if let Some(reflection) = &args.reflection {
        derive_attrs.push(quote!(#[type_reflection(#reflection)]));
    }
    let reflected_fields = fields.iter().map(|field| {
        if field.attrs.iter().any(|a| a.path.is_ident("reflect")) {
            quote!(#field)
        } else {
            quote!(#[reflect] #field)
        }
    });
    let object_input: DeriveInput = syn::parse2(quote! {
        #(#derive_attrs)*
        #vis struct #obj {
            #base,
            #(#reflected_fields,)*
        }
    })?;
    let object_impl = expand_object(&object_input, &tvm_ffi, true);
    let plain_fields = fields.iter().map(|field| {
        let mut field = (*field).clone();
        field.attrs.retain(|a| !a.path.is_ident("reflect"));
        field
    });

    Ok(quote! {
        #(#attrs)*
        #[repr(C)]
        #vis struct #obj {
            #base,
            #(#plain_fields,)*
        }

        #object_impl

        impl ::core::ops::Deref for #obj {
            type Target = #base_ty;
            #[inline]
            fn deref(&self) -> &#base_ty {
                &self.#base_id
            }
        }

        impl #obj {
            /// The object from the parent's object, then the fields in order: a
            /// scalar as its own type, any other from anything that converts into it.
            #[allow(clippy::too_many_arguments)]
            pub fn new(#base_id: #base_ty, #(#params),*) -> Self {
                Self { #base_id, #(#inits,)* }
            }
        }
    })
}

/// The scalars `new` takes as their own type, as the allocator of a type bound by
/// tvm-ffi-stubgen does (`RUST_EXACT_PARAMS` there): a literal infers only from
/// its exact type.
const EXACT_PARAMS: &[&str] = &[
    "bool", "i8", "i16", "i32", "i64", "isize", "u8", "u16", "u32", "u64", "usize", "f32", "f64",
];

fn is_exact_param(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(path) if path.qself.is_none() => path
            .path
            .get_ident()
            .is_some_and(|ident| EXACT_PARAMS.iter().any(|name| ident == name)),
        _ => false,
    }
}
