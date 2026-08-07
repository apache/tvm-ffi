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
use quote::{quote, quote_spanned};
use syn::{parse_macro_input, FnArg, ImplItem, ImplItemMethod, ItemImpl, Meta, NestedMeta, Type};

use crate::utils::get_tvm_ffi_crate;

pub(crate) fn dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as DispatchArgs);
    let item_impl = parse_macro_input!(item as ItemImpl);

    match expand(&item_impl, args.mode) {
        Ok(generated) => quote!(#item_impl #generated).into(),
        Err(error) => {
            let error = error.to_compile_error();
            quote!(#item_impl #error).into()
        }
    }
}

struct DispatchArgs {
    mode: DispatchMode,
}

#[derive(Clone, Copy)]
enum DispatchMode {
    Visit,
    Map,
}

impl DispatchMode {
    fn name(self) -> &'static str {
        match self {
            Self::Visit => "visit",
            Self::Map => "map",
        }
    }

    fn handler_prefix(self) -> &'static str {
        match self {
            Self::Visit => "visit_",
            Self::Map => "map_",
        }
    }

    fn value_type(self) -> &'static str {
        match self {
            Self::Visit => "VisitValue",
            Self::Map => "MapValue",
        }
    }
}

impl syn::parse::Parse for DispatchArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mode: syn::Ident = input.parse()?;
        let mode = if mode == "visit" {
            DispatchMode::Visit
        } else if mode == "map" {
            DispatchMode::Map
        } else {
            return Err(syn::Error::new(
                mode.span(),
                "expected `dispatch(visit)` or `dispatch(map)`",
            ));
        };
        if !input.is_empty() {
            return Err(input.error(format!(
                "`dispatch({})` takes no further arguments; a handler that needs the \
                 definition-region state declares a trailing `DefRegionKind` argument",
                mode.name()
            )));
        }
        Ok(DispatchArgs { mode })
    }
}

fn expand(item_impl: &ItemImpl, mode: DispatchMode) -> syn::Result<TokenStream2> {
    if item_impl.trait_.is_some() {
        return Err(syn::Error::new_spanned(
            item_impl,
            format!("`dispatch({})` requires an inherent impl", mode.name()),
        ));
    }

    let handler_prefix = mode.handler_prefix();
    let handlers = item_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Method(method)
                if method.sig.ident.to_string().starts_with(handler_prefix) =>
            {
                Some(parse_handler(method, mode))
            }
            _ => None,
        })
        .collect::<syn::Result<Vec<_>>>()?;

    if handlers.is_empty() {
        return Err(syn::Error::new_spanned(
            item_impl,
            format!(
                "`dispatch({})` found no `{}*` methods",
                mode.name(),
                handler_prefix
            ),
        ));
    }
    let tvm_ffi = get_tvm_ffi_crate();
    let into_result = match mode {
        DispatchMode::Visit => quote! {
            #tvm_ffi::extra::structural_visit::IntoVisitResult::into_visit_result
        },
        DispatchMode::Map => quote! {
            #tvm_ffi::extra::structural_mutate::IntoMapResult::into_map_result
        },
    };

    let links = handlers.iter().map(|handler| {
        let method = &handler.method;
        let attrs = &handler.cfg_attrs;
        // A handler opts into the definition-region state by declaring a
        // trailing argument; the generated dispatch forwards by arity, like
        // the corresponding C++ structural callback overloads.
        let kind_arg = if handler.wants_def_region {
            quote!(, def_region_kind)
        } else {
            quote!()
        };
        let invoke = match &handler.argument {
            HandlerArgument::Value => quote! {
                return Some(
                    #into_result(self.#method(value #kind_arg))
                );
            },
            HandlerArgument::BorrowedNode(node_type) => quote! {
                if let Some(node) = value.as_node::<#node_type>() {
                    return Some(
                        #into_result(self.#method(node #kind_arg))
                    );
                }
            },
            HandlerArgument::Owned(value_type) => quote! {
                if let Some(node) = value.cast::<#value_type>() {
                    return Some(
                        #into_result(self.#method(node #kind_arg))
                    );
                }
            },
        };
        quote! {
            #(#[#attrs])*
            {
                #invoke
            }
        }
    });
    let self_type = &item_impl.self_ty;
    let (impl_generics, _, where_clause) = item_impl.generics.split_for_impl();
    let impl_cfg_attrs = presence_attrs(&item_impl.attrs)?;
    let ordering_errors = handlers
        .iter()
        .enumerate()
        .filter(|(_, handler)| matches!(&handler.argument, HandlerArgument::Value))
        .flat_map(|(index, handler)| {
            handlers[index + 1..].iter().map(|later| {
                let span = handler.method.span();
                let handler_attrs = &handler.cfg_attrs;
                let later_attrs = &later.cfg_attrs;
                let value_type = mode.value_type();
                quote_spanned! {span=>
                    #(#[#impl_cfg_attrs])*
                    #(#[#handler_attrs])*
                    #(#[#later_attrs])*
                    compile_error!(concat!(
                        "the `&", #value_type,
                        "` catch-all handler must be last among enabled handlers"
                    ));
                }
            })
        });

    let dispatch_impl = match mode {
        DispatchMode::Visit => quote! {
            impl #impl_generics #tvm_ffi::extra::structural_visit::VisitDispatch
                for #self_type #where_clause
            {
                #[allow(unreachable_code, unused_variables)]
                fn dispatch_visit(
                    &mut self,
                    value: &#tvm_ffi::extra::structural_visit::VisitValue,
                    def_region_kind: #tvm_ffi::extra::structural_visit::DefRegionKind,
                ) -> Option<#tvm_ffi::extra::structural_visit::VisitResult> {
                    #(#links)*
                    None
                }
            }
        },
        DispatchMode::Map => quote! {
            impl #impl_generics #tvm_ffi::extra::structural_mutate::MapDispatch
                for #self_type #where_clause
            {
                #[allow(unreachable_code, unused_variables)]
                fn dispatch_map(
                    &mut self,
                    value: &#tvm_ffi::extra::structural_mutate::MapValue,
                    def_region_kind: #tvm_ffi::extra::structural_visit::DefRegionKind,
                ) -> Option<#tvm_ffi::extra::structural_mutate::MapResult> {
                    #(#links)*
                    None
                }
            }
        },
    };

    Ok(quote! {
        #(#ordering_errors)*

        #(#[#impl_cfg_attrs])*
        #dispatch_impl
    })
}

struct Handler {
    method: syn::Ident,
    argument: HandlerArgument,
    /// The handler declared a trailing `DefRegionKind` argument.
    wants_def_region: bool,
    cfg_attrs: Vec<Meta>,
}

enum HandlerArgument {
    Value,
    BorrowedNode(Type),
    Owned(Type),
}

fn parse_handler(method: &ImplItemMethod, mode: DispatchMode) -> syn::Result<Handler> {
    let inputs = &method.sig.inputs;
    let receiver_is_mut = matches!(
        inputs.first(),
        Some(FnArg::Receiver(receiver))
            if receiver.reference.is_some() && receiver.mutability.is_some()
    );
    if !receiver_is_mut || !(inputs.len() == 2 || inputs.len() == 3) {
        return Err(syn::Error::new_spanned(
            &method.sig,
            format!(
                "{} handlers must take `&mut self`, a node, and optionally a trailing \
                 `DefRegionKind` argument",
                mode.name()
            ),
        ));
    }
    let wants_def_region = inputs.len() == 3;

    let value_type = match inputs.iter().nth(1) {
        Some(FnArg::Typed(value)) => (*value.ty).clone(),
        _ => unreachable!("the second argument cannot be a receiver"),
    };
    let argument = match &value_type {
        Type::Reference(reference) if reference.mutability.is_none() => {
            if is_dispatch_value(reference.elem.as_ref(), mode) {
                HandlerArgument::Value
            } else {
                HandlerArgument::BorrowedNode((*reference.elem).clone())
            }
        }
        Type::Reference(_) => {
            return Err(syn::Error::new_spanned(
                value_type,
                format!(
                    "{} handler values cannot be mutable references",
                    mode.name()
                ),
            ));
        }
        _ => HandlerArgument::Owned(value_type),
    };
    let cfg_attrs = presence_attrs(&method.attrs)?;
    Ok(Handler {
        method: method.sig.ident.clone(),
        argument,
        wants_def_region,
        cfg_attrs,
    })
}

fn presence_attrs(attrs: &[syn::Attribute]) -> syn::Result<Vec<Meta>> {
    attrs
        .iter()
        .filter(|attr| attr.path.is_ident("cfg") || attr.path.is_ident("cfg_attr"))
        .map(|attr| attr.parse_meta().map(presence_meta))
        .filter_map(Result::transpose)
        .collect()
}

fn presence_meta(meta: Meta) -> Option<Meta> {
    if meta.path().is_ident("cfg") {
        return Some(meta);
    }
    let Meta::List(mut list) = meta else {
        return None;
    };
    if !list.path.is_ident("cfg_attr") {
        return None;
    }

    let mut items = list.nested.into_iter();
    let condition = items.next()?;
    let mut retained = syn::punctuated::Punctuated::new();
    retained.push(condition);
    for item in items {
        if let NestedMeta::Meta(meta) = item {
            if let Some(meta) = presence_meta(meta) {
                retained.push(NestedMeta::Meta(meta));
            }
        }
    }
    if retained.len() == 1 {
        None
    } else {
        list.nested = retained;
        Some(Meta::List(list))
    }
}

fn is_dispatch_value(value_type: &Type, mode: DispatchMode) -> bool {
    let Type::Path(path) = value_type else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == mode.value_type())
}
