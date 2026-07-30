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

// O3 hot-loop benchmarks across multiple shuffle seeds show the first stable
// win for uniformly distributed hits at seven arms. This is a retestable
// heuristic; smaller matches keep the ordered path.
const MIN_LOOKUP_TABLE_ARMS: usize = 7;

struct MatchAnyInput {
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
    let scrutinee = input.scrutinee;
    let fallback = input.fallback;
    let arms = input.arms;
    let can_attempt_leaf_lookup = arms.len() >= MIN_LOOKUP_TABLE_ARMS
        && arms
            .iter()
            .all(|arm| arm.guard.is_none() && is_simple_binding(&arm.binding));

    if can_attempt_leaf_lookup {
        expand_leaf_lookup_match(&tvm_ffi, &scrutinee, &arms, &fallback)
    } else {
        expand_ordered_match(&tvm_ffi, &scrutinee, &arms, &fallback)
    }
}

fn expand_ordered_match(
    tvm_ffi: &TokenStream,
    scrutinee: &Expr,
    arms: &[TypedArm],
    fallback: &Expr,
) -> TokenStream {
    let span = Span::mixed_site();
    let source = Ident::new("__tvm_ffi_match_any_source", span);
    let converted = Ident::new("__tvm_ffi_match_any_converted", span);
    let view = Ident::new("__tvm_ffi_match_any_view", span);
    let rejected = Ident::new("__tvm_ffi_match_any_rejected", span);
    let dispatch = expand_ordered_dispatch(tvm_ffi, arms, fallback, &view, &rejected);

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

fn expand_ordered_dispatch(
    tvm_ffi: &TokenStream,
    arms: &[TypedArm],
    fallback: &Expr,
    view: &Ident,
    rejected: &Ident,
) -> TokenStream {
    expand_ordered_try_into_chain(
        tvm_ffi,
        arms,
        quote!({ #fallback }),
        view,
        rejected,
        |_, arm| {
            let binding = &arm.binding;
            let body = &arm.body;
            if let Some(guard) = &arm.guard {
                quote!(::core::result::Result::Ok(#binding) if #guard => { #body })
            } else {
                quote!(::core::result::Result::Ok(#binding) => { #body })
            }
        },
    )
}

fn expand_ordered_try_into_chain<F>(
    tvm_ffi: &TokenStream,
    arms: &[TypedArm],
    fallback: TokenStream,
    view: &Ident,
    rejected: &Ident,
    mut matched_arm: F,
) -> TokenStream
where
    F: FnMut(usize, &TypedArm) -> TokenStream,
{
    arms.iter()
        .enumerate()
        .rev()
        .fold(fallback, |next, (arm_id, arm)| {
            let matcher = &arm.matcher;
            let matched = matched_arm(arm_id, arm);
            let conversion = expand_pattern_conversion(tvm_ffi, matcher, view);

            quote! {
                match #conversion {
                    #matched,
                    #rejected => {
                        ::core::mem::drop(#rejected);
                        #next
                    }
                }
            }
        })
}

fn expand_pattern_conversion(tvm_ffi: &TokenStream, matcher: &Path, view: &Ident) -> TokenStream {
    let span = Span::mixed_site();
    let probe = Ident::new("__tvm_ffi_match_any_conversion_probe", span);
    let converted = Ident::new("__tvm_ffi_match_any_pattern_conversion", span);

    quote! {
        {
            use #tvm_ffi::match_any_internal::PatternConversion as _;

            let #probe =
                #tvm_ffi::match_any_internal::PatternConversionProbe::<#matcher>::new();
            let #converted: ::core::result::Result<#matcher, ()> =
                (&#probe).try_convert(#view);
            #converted
        }
    }
}

fn expand_exact_pattern_conversion(
    tvm_ffi: &TokenStream,
    matcher: &Path,
    view: &Ident,
) -> TokenStream {
    let span = Span::mixed_site();
    let probe = Ident::new("__tvm_ffi_match_any_conversion_probe", span);
    let converted = Ident::new("__tvm_ffi_match_any_pattern_conversion", span);

    quote! {
        {
            use #tvm_ffi::match_any_internal::PatternConversion as _;

            let #probe =
                #tvm_ffi::match_any_internal::PatternConversionProbe::<#matcher>::new();
            let #converted: ::core::result::Result<#matcher, ()> = unsafe {
                (&#probe).try_convert_after_exact_match(#view)
            };
            #converted
        }
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
    let pattern_list = arms
        .iter()
        .map(|arm| &arm.matcher)
        .rev()
        .fold(quote!(()), |tail, matcher| quote!((#matcher, #tail)));

    quote! {
        {
            use #tvm_ffi::match_any_internal::LeafPatternMetadata as _;

            let #probe =
                #tvm_ffi::match_any_internal::LeafPatternProbe::<#pattern_list>::new();
            match (&#probe).leaf_pattern_list_id() {
                ::core::option::Option::Some(#pattern_list_id) => {
                    static #static_table:
                        #tvm_ffi::match_any_internal::LeafLookupTableCell<#arm_count> =
                            #tvm_ffi::match_any_internal::LeafLookupTableCell::new();
                    let #table = #static_table.try_get_or_init(|| {
                        let mut #type_indices = [0_i32; #arm_count];
                        (&#probe).fill_leaf_type_indices(&mut #type_indices);
                        #tvm_ffi::match_any_internal::LeafLookupTable::build(
                            #pattern_list_id,
                            #type_indices,
                        )
                    });
                    match #table {
                        ::core::option::Option::Some(#table) => {
                            #table.lookup(#pattern_list_id, #view.type_index())
                        }
                        ::core::option::Option::None => {
                            ::core::result::Result::Err(())
                        }
                    }
                }
                ::core::option::Option::None => {
                    ::core::result::Result::Err(())
                }
            }
        }
    }
}

fn expand_direct_leaf_selection(
    tvm_ffi: &TokenStream,
    arms: &[TypedArm],
    arm_constants: &[Ident],
    arm_variants: &[Ident],
    arm_id: &Ident,
    view: &Ident,
    rejected: &Ident,
    selected_enum: &Ident,
    selected_value: &Ident,
) -> TokenStream {
    let selections = arms.iter().enumerate().map(|(arm_id, arm)| {
        let matcher = &arm.matcher;
        let variant = &arm_variants[arm_id];
        let arm_constant = &arm_constants[arm_id];
        let conversion = expand_exact_pattern_conversion(tvm_ffi, matcher, view);

        quote! {
            #arm_constant => {
                match #conversion {
                    ::core::result::Result::Ok(#selected_value) => {
                        #selected_enum::#variant(#selected_value)
                    }
                    #rejected => {
                        ::core::mem::drop(#rejected);
                        ::core::panic!(
                            "match_any! leaf lookup selected an incompatible arm"
                        )
                    }
                }
            }
        }
    });

    quote! {
        match #arm_id {
            #(#selections,)*
            _ => ::core::unreachable!(),
        }
    }
}

fn expand_leaf_body_dispatch(
    arms: &[TypedArm],
    arm_variants: &[Ident],
    selected_enum: &Ident,
    selected: &Ident,
    fallback_variant: &Ident,
    fallback: &Expr,
) -> TokenStream {
    let body_arms = arms.iter().enumerate().map(|(arm_id, arm)| {
        let binding = &arm.binding;
        let body = &arm.body;
        let variant = &arm_variants[arm_id];

        quote! {
            #selected_enum::#variant(#binding) => {
                #body
            }
        }
    });

    quote! {
        match #selected {
            #(#body_arms,)*
            #selected_enum::#fallback_variant => {
                #fallback
            }
        }
    }
}

fn expand_leaf_lookup_match(
    tvm_ffi: &TokenStream,
    scrutinee: &Expr,
    arms: &[TypedArm],
    fallback: &Expr,
) -> TokenStream {
    let span = Span::mixed_site();
    let source = Ident::new("__tvm_ffi_match_any_source", span);
    let converted = Ident::new("__tvm_ffi_match_any_converted", span);
    let view = Ident::new("__tvm_ffi_match_any_view", span);
    let rejected = Ident::new("__tvm_ffi_match_any_rejected", span);
    let arm_id = Ident::new("__tvm_ffi_match_any_arm_id", span);
    let selected = Ident::new("__tvm_ffi_match_any_selected", span);
    let selected_value = Ident::new("__tvm_ffi_match_any_selected_value", span);
    let selected_enum = Ident::new("__TvmFfiMatchAnyArm", span);
    let fallback_variant = Ident::new("Fallback", span);
    let arm_count = arms.len();
    let arm_types = (0..arm_count)
        .map(|arm_id| Ident::new(&format!("__TvmFfiMatchAnyType{arm_id}"), span))
        .collect::<Vec<_>>();
    let arm_variants = (0..arm_count)
        .map(|arm_id| Ident::new(&format!("Arm{arm_id}"), span))
        .collect::<Vec<_>>();
    let arm_constants = (0..arm_count)
        .map(|arm_id| Ident::new(&format!("__TVM_FFI_MATCH_ANY_ARM_{arm_id}"), span))
        .collect::<Vec<_>>();
    let arm_constant_definitions =
        arm_constants
            .iter()
            .enumerate()
            .map(|(arm_id, arm_constant)| {
                quote! {
                    const #arm_constant: #tvm_ffi::match_any_internal::ArmId =
                        #arm_id;
                }
            });
    let lookup_arm_id = expand_leaf_table_lookup(tvm_ffi, arms, &view);
    let ordered_selection = expand_ordered_try_into_chain(
        tvm_ffi,
        arms,
        quote!(#selected_enum::#fallback_variant),
        &view,
        &rejected,
        |arm_id, _| {
            let variant = &arm_variants[arm_id];
            quote!(
                ::core::result::Result::Ok(#selected_value) => {
                    #selected_enum::#variant(#selected_value)
                }
            )
        },
    );

    let direct_selection = expand_direct_leaf_selection(
        tvm_ffi,
        arms,
        &arm_constants,
        &arm_variants,
        &arm_id,
        &view,
        &rejected,
        &selected_enum,
        &selected_value,
    );
    let body_dispatch = expand_leaf_body_dispatch(
        arms,
        &arm_variants,
        &selected_enum,
        &selected,
        &fallback_variant,
        fallback,
    );

    quote! {
        {
            enum #selected_enum<#(#arm_types),*> {
                #(#arm_variants(#arm_types),)*
                #fallback_variant,
            }

            #(#arm_constant_definitions)*

            let #source = &(#scrutinee);
            let #converted: ::core::result::Result<
                #tvm_ffi::AnyView<'_>,
                ::core::convert::Infallible,
            > = ::core::convert::TryInto::<#tvm_ffi::AnyView<'_>>::try_into(#source);
            let #view = match #converted {
                ::core::result::Result::Ok(view) => view,
                ::core::result::Result::Err(error) => match error {},
            };
            let #selected =
                if #view.type_index()
                    >= #tvm_ffi::TypeIndex::kTVMFFIStaticObjectBegin as i32
                {
                    match #lookup_arm_id {
                        ::core::result::Result::Ok(
                            ::core::option::Option::Some(#arm_id),
                        ) => {
                            #direct_selection
                        }
                        ::core::result::Result::Ok(
                            ::core::option::Option::None,
                        ) => {
                            #selected_enum::#fallback_variant
                        }
                        ::core::result::Result::Err(()) => {
                            #ordered_selection
                        }
                    }
                } else {
                    #selected_enum::#fallback_variant
                };
            #body_dispatch
        }
    }
}

fn is_simple_binding(binding: &Pat) -> bool {
    match binding {
        Pat::Ident(binding) => binding.subpat.is_none(),
        Pat::Wild(_) => true,
        _ => false,
    }
}
