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
use proc_macro2::{Span, TokenStream as TokenStream2, TokenTree};
use quote::{format_ident, quote, quote_spanned, ToTokens};
use syn::{
    parse_macro_input, FnArg, GenericParam, ImplItem, ImplItemMethod, ItemImpl, Meta, NestedMeta,
    Token, Type,
};

use crate::utils::get_tvm_ffi_crate;

pub(crate) fn dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as DispatchArgs);
    let item_impl = parse_macro_input!(item as ItemImpl);
    match expand(&item_impl, &args) {
        Ok(generated) => quote!(#item_impl #generated).into(),
        Err(error) => {
            let error = error.to_compile_error();
            let cfg = presence_attrs(&item_impl.attrs).unwrap_or_default();
            quote!(#item_impl #(#[#cfg])* #error).into()
        }
    }
}

struct DispatchArgs {
    prefix: syn::Ident,
    arguments: syn::TypeTuple,
    output: Type,
}

impl syn::parse::Parse for DispatchArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let prefix = input.parse()?;
        input.parse::<Token![,]>()?;
        let arguments = match input.parse::<Type>()? {
            Type::Tuple(tuple) if !tuple.elems.is_empty() => tuple,
            ty => return Err(syn::Error::new_spanned(
                ty,
                "dispatch requires a nonempty argument tuple, starting with AnyView or MutationView",
            )),
        };
        input.parse::<Token![,]>()?;
        let output = input.parse::<Type>()?;
        if matches!(output, Type::Infer(_) | Type::ImplTrait(_)) {
            return Err(syn::Error::new_spanned(
                output,
                "dispatch requires an explicit return type",
            ));
        }
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }
        Ok(Self {
            prefix,
            arguments,
            output,
        })
    }
}

struct Handler<'a> {
    method: &'a ImplItemMethod,
    generics: syn::Generics,
    local_lifetimes: Vec<syn::Ident>,
    argument: Argument,
    capability: Option<Type>,
    trailing: usize,
}

fn expand(item: &ItemImpl, args: &DispatchArgs) -> syn::Result<TokenStream2> {
    if item.trait_.is_some() {
        return Err(syn::Error::new_spanned(
            item,
            "dispatch requires an inherent impl",
        ));
    }
    let method_prefix = format!("{}_", args.prefix);
    let handlers = item
        .items
        .iter()
        .filter_map(|member| match member {
            ImplItem::Method(method)
                if method.sig.ident.to_string().starts_with(&method_prefix) =>
            {
                Some(parse_handler(method, args))
            }
            _ => None,
        })
        .collect::<syn::Result<Vec<_>>>()?;
    let Some(first) = handlers.first() else {
        return Err(syn::Error::new_spanned(
            item,
            format!("no `{method_prefix}*` handlers"),
        ));
    };
    for handler in handlers.iter().take(handlers.len() - 1) {
        let catch_all = matches!(handler.argument, Argument::Value)
            || handler.capability.as_ref().is_some_and(|ty| {
                matches!(ty, Type::Path(path) if path.path.segments.last().is_some_and(|s| s.ident == "Any"))
            });
        if catch_all {
            return Err(syn::Error::new_spanned(
                &handler.method.sig.ident,
                "the dispatch catch-all must be last",
            ));
        }
    }
    let tvm_ffi = get_tvm_ffi_crate();
    // Only binder declarations and the receiver are shared between handlers.
    // The attribute alone supplies the argument and result contract.
    let mut generics = item.generics.clone();
    generics.params.extend(first.generics.params.clone());
    if let Some(method_where) = &first.generics.where_clause {
        generics
            .make_where_clause()
            .predicates
            .extend(method_where.predicates.clone());
    }
    let mut lifetimes = Lifetimes::new(&generics);
    let FnArg::Receiver(receiver) = &first.method.sig.inputs[0] else {
        unreachable!()
    };
    let receiver_lifetime = receiver
        .reference
        .as_ref()
        .and_then(|(_, l)| l.clone())
        .filter(|l| l.ident != "_")
        .unwrap_or_else(|| lifetimes.fresh());
    let mut arguments = args.arguments.clone();
    for ty in &mut arguments.elems {
        lifetimes.normalize(ty, None);
    }
    let mut output = args.output.clone();
    // As in an ordinary method, result lifetime elision borrows the receiver.
    lifetimes.normalize(&mut output, Some(&receiver_lifetime));
    let carrier = &arguments.elems[0];
    generics
        .make_where_clause()
        .predicates
        .push(syn::parse_quote!(
            #carrier: #tvm_ffi::extra::dispatch::DispatchInput
        ));
    let source = quote!(<#carrier as #tvm_ffi::extra::dispatch::DispatchInput>::Source);
    let mut borrow_name = "__tvm_ffi_match_borrow".to_owned();
    while lifetimes.used.contains(&borrow_name) {
        borrow_name.push('_');
    }
    let borrow = syn::Lifetime::new(&format!("'{borrow_name}"), Span::mixed_site());
    let mut patterns = Vec::new();
    for handler in &handlers {
        let FnArg::Typed(value) = &handler.method.sig.inputs[1] else {
            unreachable!()
        };
        let mut ty: Type = syn::parse2(elide_lifetimes(
            value.ty.to_token_stream(),
            &handler.local_lifetimes,
        ))?;
        if matches!(handler.argument, Argument::Owned)
            && handler.capability.is_none()
            && has_pattern_lifetime(&ty)
        {
            lifetimes.normalize(&mut ty, None);
            // Fix recovery independently of the arm-local mutable source borrow.
            let recovery =
                quote!(<#ty as #tvm_ffi::extra::dispatch::DispatchRecovery<#source>>::Recovery);
            generics.make_where_clause().predicates.push(syn::parse_quote!(
                for<#borrow> #ty: #tvm_ffi::match_any_internal::MatchPattern<&#borrow mut #source, Recovery = #recovery>
            ));
        }
        patterns.push(ty);
    }
    generics.params.extend(
        lifetimes
            .added
            .into_iter()
            .map(|l| -> GenericParam { syn::parse_quote!(#l) }),
    );
    // Lifetimes must precede type and const parameters in impl declarations.
    let mut params: Vec<_> = generics.params.into_iter().collect();
    params.sort_by_key(|param| !matches!(param, GenericParam::Lifetime(_)));
    generics.params = params.into_iter().collect();
    // Self in the contract means the implementor, not its receiver reference.
    let arguments: syn::TypeTuple =
        syn::parse2(replace_self(arguments.into_token_stream(), &item.self_ty))?;
    let output: Type = syn::parse2(replace_self(output.into_token_stream(), &item.self_ty))?;
    let where_clause = generics
        .where_clause
        .take()
        .map(|clause| syn::parse2(replace_self(clause.into_token_stream(), &item.self_ty)))
        .transpose()?;
    generics = syn::parse2(replace_self(generics.into_token_stream(), &item.self_ty))?;
    generics.where_clause = where_clause;
    let mut constrained = quote!(#arguments #output).into_iter().collect::<Vec<_>>();
    constrained.extend(item.self_ty.to_token_stream());
    for param in &generics.params {
        let ident = match param {
            GenericParam::Type(param) => &param.ident,
            GenericParam::Const(param) => &param.ident,
            GenericParam::Lifetime(_) => continue,
        };
        if !contains_ident(constrained.iter().cloned().collect(), ident) {
            return Err(syn::Error::new_spanned(param,
                "dispatch generic parameters must occur in the receiver type, argument tuple or return type; move an unused parameter to an ordinary helper method"));
        }
    }
    let (impl_generics, _, where_clause) = generics.split_for_impl();
    let self_type = &item.self_ty;
    let mutable = &receiver.mutability;
    let value = format_ident!("__tvm_ffi_value", span = Span::mixed_site());
    let bound = format_ident!("__tvm_ffi_matched", span = Span::mixed_site());
    let names: Vec<_> = (0..arguments.elems.len() - 1)
        .map(|i| format_ident!("__tvm_ffi_arg_{i}", span = Span::mixed_site()))
        .collect();
    let types: Vec<_> = arguments.elems.iter().collect();
    let dispatch_self = format_ident!("self", span = Span::mixed_site());
    let mut errors = Vec::new();
    let mut arms = Vec::new();
    for (handler, pattern_type) in handlers.iter().zip(&patterns) {
        let method = &handler.method.sig.ident;
        let error = if first.generics != handler.generics
            || first.method.sig.inputs[0] != handler.method.sig.inputs[0]
        {
            Some("dispatch handlers must use the same receiver and method generics")
        } else if handler.trailing > names.len() {
            Some("dispatch handler has more trailing arguments than the dispatch signature")
        } else {
            None
        };
        if let Some(error) = error {
            errors.push(quote_spanned! {method.span()=> compile_error!(#error); });
            continue;
        }
        let forwarded = &names[..handler.trailing];
        let type_args: Vec<_> = handler
            .method
            .sig
            .generics
            .params
            .iter()
            .filter_map(|param| match param {
                GenericParam::Type(param) => {
                    let ident = &param.ident;
                    Some(quote!(#ident))
                }
                GenericParam::Const(param) => {
                    let ident = &param.ident;
                    Some(quote!(#ident))
                }
                GenericParam::Lifetime(_) => None,
            })
            .collect();
        let type_args = if type_args.is_empty() {
            quote!()
        } else {
            quote!(::<#(#type_args),*>)
        };
        let invocation = quote_spanned! {method.span()=> #dispatch_self.#method #type_args (#bound #(, #forwarded)*) };
        // `Into` cannot convert between two `Result`s, so a `Result` handler of a
        // `Result` dispatcher converts its success value instead.
        let returns_result = matches!(
            &handler.method.sig.output,
            syn::ReturnType::Type(_, ty) if is_result(ty)
        );
        let call = if returns_result && is_result(&output) {
            quote_spanned! {method.span()=> ::core::option::Option::Some(::core::result::Result::map(#invocation, ::core::convert::Into::into)) }
        } else {
            quote_spanned! {method.span()=> ::core::option::Option::Some(::core::convert::Into::<#output>::into(#invocation)) }
        };
        let pattern = match &handler.argument {
            Argument::BorrowedNode => {
                let Type::Reference(reference) = pattern_type else {
                    unreachable!()
                };
                let ty = &reference.elem;
                quote!(#ty(ref #bound))
            }
            Argument::Value | Argument::Owned => quote!(#pattern_type(#bound)),
        };
        arms.push(replace_self(quote!(#pattern => #call), &item.self_ty));
    }
    let matching = quote! {
        #tvm_ffi::match_any!(#value {
            #(#arms,)*
            _ => ::core::option::Option::None,
        })
    };
    // Handler generics belong to the inherent method, not its enclosing impl.
    // The tuple adapter keeps them in its trait arguments and forwards directly.
    let param_name = |param: &GenericParam| match param {
        GenericParam::Lifetime(param) => param.lifetime.ident.to_string(),
        GenericParam::Type(param) => param.ident.to_string(),
        GenericParam::Const(param) => param.ident.to_string(),
    };
    let mut method_generics = generics.clone();
    method_generics.params = method_generics
        .params
        .into_iter()
        .filter(|param| {
            !item.generics.params.iter().any(|original| {
                std::mem::discriminant(original) == std::mem::discriminant(param)
                    && param_name(original) == param_name(param)
            })
        })
        .collect();
    let type_args: Vec<_> = method_generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(param) => Some(param.ident.to_token_stream()),
            GenericParam::Const(param) => Some(param.ident.to_token_stream()),
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    let type_args = if type_args.is_empty() {
        quote!()
    } else {
        quote!(::<#(#type_args),*>)
    };
    let (method_generics, _, method_where) = method_generics.split_for_impl();
    let (inherent_generics, _, inherent_where) = item.generics.split_for_impl();
    let value_type = types[0];
    let trailing_types = &types[1..];
    let impl_cfg = presence_attrs(&item.attrs)?;
    Ok(quote! {
        #(#[#impl_cfg])*
        impl #inherent_generics #self_type #inherent_where {
            /// Dispatch one value, returning `None` only when no handler matches.
            #[inline(always)]
            #[allow(unreachable_code, unused_variables, unused_mut)]
            pub fn dispatch #method_generics (
                &#receiver_lifetime #mutable #dispatch_self,
                #value: #value_type #(, #names: #trailing_types)*
            ) -> ::core::option::Option<#output> #method_where {
                #(#errors)*
                #matching
            }
        }

        #(#[#impl_cfg])*
        impl #impl_generics #tvm_ffi::Dispatcher<(#(#types,)*), ::core::option::Option<#output>>
            for &#receiver_lifetime #mutable #self_type #where_clause
        {
            #[inline(always)]
            fn dispatch_tuple(#dispatch_self, (#value, #(#names,)*): (#(#types,)*))
                -> ::core::option::Option<#output>
            {
                <#self_type>::dispatch #type_args (#dispatch_self, #value #(, #names)*)
            }
        }
    })
}

fn parse_handler<'a>(method: &'a ImplItemMethod, args: &DispatchArgs) -> syn::Result<Handler<'a>> {
    if let Some(attribute) = method
        .attrs
        .iter()
        .find(|attribute| attribute.path.is_ident("cfg") || attribute.path.is_ident("cfg_attr"))
    {
        return Err(syn::Error::new_spanned(
            attribute,
            "conditional dispatch handlers are unsupported; configure the whole impl instead",
        ));
    }
    let signature = &method.sig;
    if !matches!(signature.inputs.first(), Some(FnArg::Receiver(r)) if r.reference.is_some())
        || signature.inputs.len() < 2
        || signature.asyncness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || signature.variadic.is_some()
    {
        return Err(syn::Error::new_spanned(signature,
            "dispatch handlers must be safe synchronous methods taking &self or &mut self and a value"));
    }
    let FnArg::Typed(value) = &signature.inputs[1] else {
        unreachable!()
    };
    let capability = if is_capability(&value.ty) {
        let Type::Path(path) = &*value.ty else {
            unreachable!()
        };
        let segment = path.path.segments.last().unwrap();
        let ty = if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
            args.args.iter().find_map(|arg| match arg {
                syn::GenericArgument::Type(ty) => Some(ty.clone()),
                _ => None,
            })
        } else {
            None
        };
        let tvm_ffi = get_tvm_ffi_crate();
        Some(ty.unwrap_or_else(|| syn::parse_quote!(#tvm_ffi::Any)))
    } else {
        None
    };
    let receiver = &signature.inputs[0];
    let arguments = &args.arguments;
    let output = &args.output;
    let where_clause = &signature.generics.where_clause;
    let mut visible = quote!(#receiver #arguments #output #where_clause);
    for param in &signature.generics.params {
        match param {
            GenericParam::Lifetime(param) => param.bounds.to_tokens(&mut visible),
            param => param.to_tokens(&mut visible),
        }
    }
    // Lifetimes absent from the explicit contract and shared bounds stay local
    // to each handler's matching pattern.
    let local_lifetimes: Vec<_> = signature
        .generics
        .lifetimes()
        .filter(|param| {
            param.bounds.is_empty() && !contains_lifetime(visible.clone(), &param.lifetime.ident)
        })
        .map(|param| param.lifetime.ident.clone())
        .collect();
    let mut generics = signature.generics.clone();
    generics.params = generics.params.into_iter().filter(|param| {
        !matches!(param, GenericParam::Lifetime(param) if local_lifetimes.contains(&param.lifetime.ident))
    }).collect();
    if generics.params.is_empty() {
        generics.lt_token = None;
        generics.gt_token = None;
    }
    Ok(Handler {
        method,
        generics,
        local_lifetimes,
        argument: parse_argument(&value.ty)?,
        capability,
        trailing: signature.inputs.len() - 2,
    })
}

/// Whether `ty` is written as a `Result<..>` path.
fn is_result(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.path.segments.last().is_some_and(|s| s.ident == "Result"))
}

fn is_capability(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.path.segments.last().is_some_and(|s| s.ident == "MutationView"))
}

fn has_pattern_lifetime(ty: &Type) -> bool {
    let Type::Path(path) = ty else { return false };
    path.qself
        .as_ref()
        .is_some_and(|qualified| has_pattern_lifetime(&qualified.ty))
        || path.path.segments.iter().any(|segment| {
            let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
                return false;
            };
            args.args.iter().any(|arg| match arg {
                syn::GenericArgument::Lifetime(lifetime) => lifetime.ident != "_",
                syn::GenericArgument::Type(ty) => has_pattern_lifetime(ty),
                _ => false,
            })
        })
}

fn contains_ident(tokens: TokenStream2, ident: &syn::Ident) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(found) => found == *ident,
        TokenTree::Group(group) => contains_ident(group.stream(), ident),
        _ => false,
    })
}

fn contains_lifetime(tokens: TokenStream2, ident: &syn::Ident) -> bool {
    let mut lifetime = false;
    for token in tokens {
        match &token {
            TokenTree::Ident(found) if lifetime && *found == *ident => return true,
            TokenTree::Group(group) if contains_lifetime(group.stream(), ident) => return true,
            _ => {}
        }
        lifetime = matches!(token, TokenTree::Punct(p) if p.as_char() == '\'');
    }
    false
}

fn elide_lifetimes(tokens: TokenStream2, lifetimes: &[syn::Ident]) -> TokenStream2 {
    let mut output = TokenStream2::new();
    let mut lifetime = false;
    for token in tokens {
        match token {
            TokenTree::Ident(ident) if lifetime && lifetimes.contains(&ident) => {
                output.extend([TokenTree::Ident(syn::Ident::new("_", ident.span()))]);
                lifetime = false;
            }
            TokenTree::Group(group) => {
                let mut rewritten = TokenTree::Group(proc_macro2::Group::new(
                    group.delimiter(),
                    elide_lifetimes(group.stream(), lifetimes),
                ));
                rewritten.set_span(group.span());
                output.extend([rewritten]);
                lifetime = false;
            }
            token => {
                lifetime = matches!(&token, TokenTree::Punct(p) if p.as_char() == '\'');
                output.extend([token]);
            }
        }
    }
    output
}

fn replace_self(tokens: TokenStream2, self_type: &Type) -> TokenStream2 {
    let mut output = TokenStream2::new();
    let mut tokens = tokens.into_iter().peekable();
    while let Some(token) = tokens.next() {
        match token {
            TokenTree::Ident(ident) if ident == "Self" => {
                if matches!(tokens.peek(), Some(TokenTree::Punct(p))
                    if p.as_char() == ':' && p.spacing() == proc_macro2::Spacing::Joint)
                {
                    output.extend(quote!(<#self_type>));
                } else {
                    self_type.to_tokens(&mut output);
                }
            }
            TokenTree::Group(group) => {
                let mut new = proc_macro2::Group::new(
                    group.delimiter(),
                    replace_self(group.stream(), self_type),
                );
                new.set_span(group.span());
                output.extend([TokenTree::Group(new)]);
            }
            token => output.extend([token]),
        }
    }
    output
}

struct Lifetimes {
    used: Vec<String>,
    added: Vec<syn::Lifetime>,
}

impl Lifetimes {
    fn new(generics: &syn::Generics) -> Self {
        Self {
            used: generics
                .lifetimes()
                .map(|p| p.lifetime.ident.to_string())
                .collect(),
            added: Vec::new(),
        }
    }

    fn fresh(&mut self) -> syn::Lifetime {
        let mut index = self.used.len();
        loop {
            let name = format!("__tvm_ffi_dispatch_{index}");
            if !self.used.contains(&name) {
                self.used.push(name.clone());
                let lifetime = syn::Lifetime::new(&format!("'{name}"), Span::mixed_site());
                self.added.push(lifetime.clone());
                return lifetime;
            }
            index += 1;
        }
    }

    fn elided(&mut self, output: Option<&syn::Lifetime>) -> syn::Lifetime {
        output.cloned().unwrap_or_else(|| self.fresh())
    }

    fn path(&mut self, path: &mut syn::Path, output: Option<&syn::Lifetime>) {
        for segment in &mut path.segments {
            if matches!(segment.arguments, syn::PathArguments::None)
                && matches!(
                    segment.ident.to_string().as_str(),
                    "AnyView" | "MutationView"
                )
            {
                let lifetime = self.elided(output);
                segment.arguments =
                    syn::PathArguments::AngleBracketed(syn::parse_quote!(<#lifetime>));
            }
            if let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments {
                for arg in &mut args.args {
                    match arg {
                        syn::GenericArgument::Lifetime(l) => {
                            if l.ident == "_" {
                                *l = self.elided(output);
                            }
                        }
                        syn::GenericArgument::Type(ty) => self.normalize(ty, output),
                        syn::GenericArgument::Binding(binding) => {
                            self.normalize(&mut binding.ty, output)
                        }
                        _ => (),
                    }
                }
                if matches!(
                    segment.ident.to_string().as_str(),
                    "AnyView" | "MutationView"
                ) && !matches!(args.args.first(), Some(syn::GenericArgument::Lifetime(_)))
                {
                    let lifetime = self.elided(output);
                    args.args
                        .insert(0, syn::GenericArgument::Lifetime(lifetime));
                }
            }
            // Parenthesized Fn trait and nested fn types bind their own elided
            // lifetimes; do not turn those higher-ranked borrows into impl params.
        }
    }

    fn normalize(&mut self, ty: &mut Type, output: Option<&syn::Lifetime>) {
        match ty {
            Type::Reference(reference) => {
                if reference.lifetime.as_ref().is_none_or(|l| l.ident == "_") {
                    reference.lifetime = Some(self.elided(output));
                }
                self.normalize(&mut reference.elem, output);
            }
            Type::Path(path) => {
                if let Some(qself) = &mut path.qself {
                    self.normalize(&mut qself.ty, output);
                }
                self.path(&mut path.path, output);
            }
            Type::Array(array) => self.normalize(&mut array.elem, output),
            Type::Slice(slice) => self.normalize(&mut slice.elem, output),
            Type::Ptr(pointer) => self.normalize(&mut pointer.elem, output),
            Type::Paren(paren) => self.normalize(&mut paren.elem, output),
            Type::Group(group) => self.normalize(&mut group.elem, output),
            Type::Tuple(tuple) => {
                for elem in &mut tuple.elems {
                    self.normalize(elem, output);
                }
            }
            Type::TraitObject(object) => {
                for bound in &mut object.bounds {
                    match bound {
                        syn::TypeParamBound::Lifetime(l) => {
                            if l.ident == "_" {
                                *l = self.elided(output);
                            }
                        }
                        syn::TypeParamBound::Trait(t) if t.lifetimes.is_none() => {
                            self.path(&mut t.path, output)
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        }
    }
}

pub(crate) enum Argument {
    Value,
    BorrowedNode,
    Owned,
}

pub(crate) fn parse_argument(ty: &Type) -> syn::Result<Argument> {
    match ty {
        Type::Reference(reference) if reference.mutability.is_none() => Ok(Argument::BorrowedNode),
        Type::Reference(_) => Err(syn::Error::new_spanned(
            ty,
            "dispatch values cannot be mutable references",
        )),
        ty if is_dispatch_value(ty) => Ok(Argument::Value),
        _ => Ok(Argument::Owned),
    }
}

pub(crate) fn presence_attrs(attrs: &[syn::Attribute]) -> syn::Result<Vec<Meta>> {
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

pub(crate) fn is_dispatch_value(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.path.segments.last().is_some_and(|s| s.ident == "AnyView"))
}
