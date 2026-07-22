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

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{braced, parenthesized, Expr, Pat, Path, Result, Token};

use crate::utils::get_tvm_ffi_crate;

struct MatchObjectInput {
    scrutinee: Expr,
    arms: Vec<TypedArm>,
    fallback: Expr,
}

struct TypedArm {
    matcher: Path,
    binding: Pat,
    guard: Option<Expr>,
    body: Expr,
}

impl Parse for MatchObjectInput {
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
                let matcher = content.parse::<Path>()?;
                let binding_content;
                parenthesized!(binding_content in content);
                let binding = binding_content.parse::<Pat>()?;
                if !binding_content.is_empty() {
                    return Err(binding_content.error("expected one binding pattern"));
                }
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
                return Err(content.error("expected `,` between match_object! arms"));
            }
        }

        let fallback = fallback
            .ok_or_else(|| content.error("match_object! requires a final `_` fallback arm"))?;
        Ok(Self {
            scrutinee,
            arms,
            fallback,
        })
    }
}

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as MatchObjectInput);
    expand_match_object(input).into()
}

fn expand_match_object(input: MatchObjectInput) -> TokenStream {
    let tvm_ffi = get_tvm_ffi_crate();
    let span = Span::mixed_site();
    let source = Ident::new("__tvm_ffi_match_object_source", span);
    let converted = Ident::new("__tvm_ffi_match_object_converted", span);
    let view = Ident::new("__tvm_ffi_match_object_view", span);
    let rejected = Ident::new("__tvm_ffi_match_object_rejected", span);
    let scrutinee = input.scrutinee;
    let fallback = input.fallback;
    let dispatch_fallback = fallback.clone();
    let arms = input.arms;
    let dispatch = arms
        .into_iter()
        .rev()
        .fold(quote!({ #dispatch_fallback }), |next, arm| {
            let matcher = arm.matcher;
            let binding = arm.binding;
            let body = arm.body;
            let matched = if let Some(guard) = arm.guard {
                quote!(::core::option::Option::Some(#binding) if #guard)
            } else {
                quote!(::core::option::Option::Some(#binding))
            };

            quote! {
                match <#matcher as #tvm_ffi::ObjectPattern>::try_match(#view) {
                    #matched => { #body },
                    #rejected => {
                        ::core::mem::drop(#rejected);
                        #next
                    },
                }
            }
        });

    quote! {
        {
            let #source = &(#scrutinee);
            let #converted: ::core::result::Result<
                #tvm_ffi::AnyView<'_>,
                ::core::convert::Infallible,
            > = ::core::convert::TryInto::<#tvm_ffi::AnyView<'_>>::try_into(#source);
            let #view = match #converted {
                ::core::result::Result::Ok(view) => view,
                ::core::result::Result::Err(error) => match error {},
            };
            if #view.type_index()
                >= #tvm_ffi::TypeIndex::kTVMFFIStaticObjectBegin as i32
            {
                #dispatch
            } else {
                #fallback
            }
        }
    }
}
