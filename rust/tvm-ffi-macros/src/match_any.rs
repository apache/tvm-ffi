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

use proc_macro2::{Delimiter, Ident, Span, TokenStream, TokenTree};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{braced, Expr, Pat, Result, Token, Type};

use crate::utils::get_tvm_ffi_crate;

// Avoid call-site table setup for small matches.
const MIN_LOOKUP_TABLE_ARMS: usize = 20;

struct MatchAnyInput {
    scrutinee: Expr,
    arms: Vec<TypedArm>,
    fallback: Expr,
}

struct TypedArm {
    matcher: Type,
    binding: Pat,
    guard: Option<Expr>,
    body: Expr,
}

impl Parse for MatchAnyInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let scrutinee = input.call(Expr::parse_without_eager_brace)?;
        let content;
        braced!(content in input);

        let mut arms = Vec::new();
        let mut fallback = None;
        while !content.is_empty() {
            if fallback.is_some() {
                return Err(content.error("the `_` fallback must be the final arm"));
            }

            if content.peek(Token![_]) {
                content.parse::<Token![_]>()?;
                if content.peek(Token![if]) {
                    return Err(content.error("the `_` fallback cannot have a guard"));
                }
                content.parse::<Token![=>]>()?;
                fallback = Some(content.parse::<Expr>()?);
            } else {
                // The last parenthesized group is the binding, so unit, pointer,
                // qualified and generic Rust types retain their ordinary syntax.
                let mut pattern = Vec::new();
                while !content.is_empty() && !content.peek(Token![if]) && !content.peek(Token![=>])
                {
                    pattern.push(content.parse::<TokenTree>()?);
                }
                let binding = match pattern.pop() {
                    Some(TokenTree::Group(group))
                        if group.delimiter() == Delimiter::Parenthesis =>
                    {
                        syn::parse2::<Pat>(group.stream())?
                    }
                    _ => return Err(content.error("expected `Type(binding)`")),
                };
                let matcher = syn::parse2::<Type>(pattern.into_iter().collect())?;
                let guard = if content.peek(Token![if]) {
                    content.parse::<Token![if]>()?;
                    Some(content.parse::<Expr>()?)
                } else {
                    None
                };
                content.parse::<Token![=>]>()?;
                let body = content.parse::<Expr>()?;
                arms.push(TypedArm {
                    matcher,
                    binding,
                    guard,
                    body,
                });
            }

            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            } else if !content.is_empty() {
                return Err(content.error("expected `,` between match_any! arms"));
            }
        }

        let fallback = fallback
            .ok_or_else(|| content.error("match_any! requires a final `_` fallback arm"))?;
        Ok(Self {
            scrutinee,
            arms,
            fallback,
        })
    }
}

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as MatchAnyInput);
    expand_match_any(input).into()
}

fn expand_match_any(input: MatchAnyInput) -> TokenStream {
    let tvm_ffi = get_tvm_ffi_crate();
    let span = Span::mixed_site();
    let source = Ident::new("__tvm_ffi_match_any_source", span);
    let probe = Ident::new("__tvm_ffi_match_any_source_probe", span);
    let view = Ident::new("__tvm_ffi_match_any_view", span);
    let lookup = Ident::new("__tvm_ffi_match_any_lookup", span);
    let scrutinee = input.scrutinee;
    let fallback = input.fallback;
    let arms = input.arms;
    let can_lookup = arms.len() >= MIN_LOOKUP_TABLE_ARMS
        && arms
            .iter()
            .all(|arm| arm.guard.is_none() && is_simple_binding(&arm.binding));
    let lookup_setup = if can_lookup {
        let table = expand_leaf_table_lookup(&tvm_ffi, &arms, &view);
        quote! {
            let #lookup = {
                let #view = #tvm_ffi::match_any_internal::SourceView::view(&#source);
                if #view.type_index() >= #tvm_ffi::TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                    #table
                } else { ::core::result::Result::Err(()) }
            };
        }
    } else {
        quote!()
    };
    let dispatch = arms.iter().enumerate().rev().fold(quote!({ #fallback }), |next, (id, arm)| {
        let id = id as u32;
        let matcher = &arm.matcher;
        let binding = &arm.binding;
        let body = &arm.body;
        let guard = arm.guard.as_ref().map(|guard| quote!(if #guard));
        let conversion = if shared_ref_binding(binding).is_some() {
            quote!({
                use #tvm_ffi::match_any_internal::RefPatternConversion as _;
                let probe = #tvm_ffi::match_any_internal::PatternConversionProbe::<#matcher>::new();
                (&&&probe).try_convert_ref(&mut #source)
            })
        } else {
            quote!(<#matcher as #tvm_ffi::match_any_internal::MatchPattern<_>>::try_match(&mut #source))
        };
        let conversion = if can_lookup {
            quote!(if #lookup.is_err() || #lookup == ::core::result::Result::Ok(::core::option::Option::Some(#id)) {
                #conversion
            } else { ::core::option::Option::None })
        } else { conversion };
        let rejected = Ident::new("__tvm_ffi_match_any_rejected", span);
        // Drop a rejected value before later arms check uniqueness.
        let drop_rejected = quote!(#[allow(clippy::drop_non_drop)] ::core::mem::drop(#rejected););
        if let Some(binding) = shared_ref_binding(binding) {
            let holder = Ident::new("__tvm_ffi_match_any_ref_holder", span);
            let converted = Ident::new("__tvm_ffi_match_any_ref_converted", span);
            let accepted = Ident::new("__tvm_ffi_match_any_ref_accepted", span);
            let test_guard = arm.guard.as_ref().map(|guard| quote!(#guard)).unwrap_or_else(|| quote!(true));
            quote! {{
                let #converted = #conversion;
                let #accepted = if let ::core::option::Option::Some(#holder) = &#converted {
                    #[allow(unused_variables)]
                    let #binding: &#matcher = ::core::borrow::Borrow::<#matcher>::borrow(#holder);
                    #test_guard
                } else { false };
                if #accepted {
                    let #holder = #converted.as_ref().unwrap();
                    #[allow(unused_variables)]
                    let #binding: &#matcher = ::core::borrow::Borrow::<#matcher>::borrow(#holder);
                    #body
                } else {
                    let #rejected = #converted;
                    #drop_rejected
                    #next
                }
            }}
        } else {
            quote! {
                match #conversion {
                    ::core::option::Option::Some(#binding) #guard => { #body },
                    #rejected => {
                        #drop_rejected
                        #next
                    }
                }
            }
        }
    });
    quote! {{
        use #tvm_ffi::match_any_internal::PrepareSource as _;
        let mut #probe = #tvm_ffi::match_any_internal::SourceProbe::new(#scrutinee);
        let mut #source = #probe.prepare();
        #lookup_setup
        #dispatch
    }}
}

fn shared_ref_binding(binding: &Pat) -> Option<&Ident> {
    match binding {
        Pat::Ident(binding)
            if binding.by_ref.is_some()
                && binding.mutability.is_none()
                && binding.subpat.is_none() =>
        {
            Some(&binding.ident)
        }
        _ => None,
    }
}

fn is_simple_binding(binding: &Pat) -> bool {
    match binding {
        Pat::Ident(binding) => binding.subpat.is_none(),
        Pat::Wild(_) => true,
        _ => false,
    }
}

fn expand_leaf_table_lookup(tvm_ffi: &TokenStream, arms: &[TypedArm], view: &Ident) -> TokenStream {
    let span = Span::mixed_site();
    let probe = Ident::new("__tvm_ffi_match_any_probe", span);
    let pattern_list_id = Ident::new("__tvm_ffi_match_any_leaf_pattern_list_id", span);
    let type_indices = Ident::new("__tvm_ffi_match_any_type_indices", span);
    let static_table = Ident::new("__TVM_FFI_MATCH_ANY_LEAF_TABLE", span);
    let table = Ident::new("__tvm_ffi_match_any_leaf_table", span);
    let arm_count = arms.len();
    let pattern_probe = arms.iter().rev().fold(
        quote!(#tvm_ffi::match_any_internal::LeafPatternProbe::<()>::new()),
        |tail, arm| {
            let matcher = &arm.matcher;
            let metadata = if shared_ref_binding(&arm.binding).is_some() {
                quote!({
                    use #tvm_ffi::match_any_internal::RefPatternMetadata as _;
                    let probe = #tvm_ffi::match_any_internal::PatternConversionProbe::<#matcher>::new();
                    (&&&probe).leaf_pattern()
                })
            } else {
                quote!(::core::marker::PhantomData::<#matcher>)
            };
            quote!(#tail.prepend(#metadata))
        },
    );
    quote! {{
        use #tvm_ffi::match_any_internal::LeafPatternMetadata as _;
        let #probe = #pattern_probe;
        match (&#probe).leaf_pattern_list_id() {
            ::core::option::Option::Some(#pattern_list_id) => {
                static #static_table: ::std::sync::OnceLock<#tvm_ffi::match_any_internal::LeafLookupTable> = ::std::sync::OnceLock::new();
                let #table = #static_table.get_or_init(|| {
                    let mut #type_indices = [0_i32; #arm_count];
                    (&#probe).fill_leaf_type_indices(&mut #type_indices);
                    #tvm_ffi::match_any_internal::LeafLookupTable::build(#pattern_list_id, &#type_indices)
                });
                #table.lookup(#pattern_list_id, #view.type_index())
            }
            ::core::option::Option::None => ::core::result::Result::Err(()),
        }
    }}
}
