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
use proc_macro_error::abort;
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::DeriveInput;

use crate::utils::*;

/// Derive Object trait for a struct to generate boilerplate code
pub fn derive_object(input: proc_macro::TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(input as DeriveInput);
    TokenStream::from(expand_object(&derive_input, &get_tvm_ffi_crate(), false))
}

/// The `ObjectCore` impl of an object struct, and its `ShallowCopy` when it
/// registers reflection, naming the tvm-ffi crate `tvm_ffi_crate`.
///
/// `rs_object` registers the reflection of a type without `#[reflect]` fields
/// or `#[type_s_eq_hash_kind]` too, with the creator of its `UnsafeInit`, as
/// `#[rs_object]` does.
pub(crate) fn expand_object(
    derive_input: &DeriveInput,
    tvm_ffi_crate: &TokenStream2,
    rs_object: bool,
) -> TokenStream2 {
    let struct_name = derive_input.ident.clone();

    let type_key = get_attr(derive_input, "type_key")
        .map(attr_to_str)
        .expect("Expect #[type_key = \"<my_type_key>\"] attribute");
    let type_final = match get_attr(derive_input, "type_final") {
        Some(attr) if matches!(attr.parse_meta(), Ok(syn::Meta::Path(_))) => true,
        Some(_) => panic!("Expect #[type_final] attribute"),
        None => false,
    };

    // Whether this type is defined here, so that it registers its key, as C++
    // `TVM_FFI_DECLARE_OBJECT_INFO` does. Without it, the type binds a key
    // that its defining library, usually C++, has registered.
    let type_register = match get_attr(derive_input, "type_register") {
        Some(attr) if matches!(attr.parse_meta(), Ok(syn::Meta::Path(_))) => true,
        Some(_) => panic!("Expect #[type_register] attribute"),
        None => false,
    };

    // Reserved child slots and whether children may exceed them, as C++
    // `_type_child_slots` and `_type_child_slots_can_overflow` declare them.
    let child_slots = match get_attr(derive_input, "type_child_slots").map(|a| a.parse_meta()) {
        Some(Ok(syn::Meta::NameValue(syn::MetaNameValue {
            lit: syn::Lit::Int(n),
            ..
        }))) => n
            .base10_parse::<i32>()
            .expect("Expect #[type_child_slots = <non-negative integer>]"),
        Some(_) => panic!("Expect #[type_child_slots = <non-negative integer>]"),
        None => 0,
    };
    let child_slots_can_overflow =
        match get_attr(derive_input, "type_child_slots_can_overflow").map(|a| a.parse_meta()) {
            Some(Ok(syn::Meta::NameValue(syn::MetaNameValue {
                lit: syn::Lit::Bool(b),
                ..
            }))) => b.value,
            Some(_) => panic!("Expect #[type_child_slots_can_overflow = <bool>]"),
            None => true,
        };
    assert!(
        child_slots >= 0,
        "Expect #[type_child_slots = <non-negative integer>]"
    );
    assert!(
        !(type_final && child_slots > 0),
        "a final object type cannot reserve child slots"
    );
    assert!(
        type_register
            || (get_attr(derive_input, "type_child_slots").is_none()
                && get_attr(derive_input, "type_child_slots_can_overflow").is_none()),
        "#[type_child_slots] and #[type_child_slots_can_overflow] need #[type_register]"
    );
    // we expect base always to be the first field
    let base_ty = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().map(|f| f.ty.clone()),
        _ => None,
    }
    .expect("First field must be `<base_name>: <ObjectCoreType>`");

    let reflection = Reflection::parse(derive_input, rs_object);
    assert!(
        reflection.is_none() || (type_register && get_attr(derive_input, "type_index").is_none()),
        "#[reflect], #[type_s_eq_hash_kind] and #[type_reflection] need #[type_register] and no \
         #[type_index]"
    );
    let (register_reflection, shallow_copy_impl) = match &reflection {
        Some(reflection) => (
            reflection.register_tokens(&struct_name, &type_key, tvm_ffi_crate),
            reflection.shallow_copy_tokens(&struct_name, &base_ty, tvm_ffi_crate),
        ),
        None => (quote! {}, quote! {}),
    };

    // A type with a static index has it; one defined here registers its key
    // under its parent on first use, or takes the index the key already has,
    // as C++ `TVM_FFI_DECLARE_OBJECT_INFO` does; any other type binds the
    // index its defining library registered.
    let type_index_tokens = match get_attr(derive_input, "type_index").map(attr_to_expr) {
        Some(type_index) => {
            let type_index_expr =
                type_index.expect("Expect #[type_index(TypeIndex::<my_type_index>)] attribute");
            quote! {
                #[inline]
                fn type_index() -> i32 {
                    #type_index_expr as i32
                }
            }
        }
        None if type_register => {
            quote! {
                #[inline]
                fn type_index() -> i32 {
                    static TYPE_INDEX: #tvm_ffi_crate::extra::cold_init_lazy_lock::ColdInitLazyLock<i32> = #tvm_ffi_crate::extra::cold_init_lazy_lock::ColdInitLazyLock::new(||
                        unsafe {
                            // The parent first, since registering it takes the
                            // same lock.
                            let parent =
                                <#base_ty as #tvm_ffi_crate::object::ObjectCore>::type_index();
                            let _registering = #tvm_ffi_crate::object::TYPE_REGISTRATION
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            let type_key_arg =
                                 #tvm_ffi_crate::tvm_ffi_sys::TVMFFIByteArray::from_str(#type_key);
                            let tindex = #tvm_ffi_crate::tvm_ffi_sys::TVMFFITypeGetOrAllocIndex(
                                &type_key_arg,
                                -1,
                                <#struct_name as #tvm_ffi_crate::object::ObjectCore>::TYPE_DEPTH,
                                #child_slots,
                                #child_slots_can_overflow as i32,
                                parent,
                            );
                            if tindex < 0 {
                                panic!(
                                    "Failed to get or allocate type index for type key: {}",
                                    #type_key
                                );
                            }
                            #register_reflection
                            tindex
                        }
                    );
                    *TYPE_INDEX
                }
            }
        }
        None => {
            quote! {
                #[inline]
                fn type_index() -> i32 {
                    static TYPE_INDEX: #tvm_ffi_crate::extra::cold_init_lazy_lock::ColdInitLazyLock<i32> = #tvm_ffi_crate::extra::cold_init_lazy_lock::ColdInitLazyLock::new(||
                        unsafe {
                            let type_key_arg =
                                 #tvm_ffi_crate::tvm_ffi_sys::TVMFFIByteArray::from_str(#type_key);
                            let mut tindex = 0;
                            let ret = #tvm_ffi_crate::tvm_ffi_sys::TVMFFITypeKeyToIndex(
                                &type_key_arg, &mut tindex
                            );
                            if ret != 0 {
                                panic!(
                                    "Type key {} is not registered: load the library that defines \
                                     it first, or add #[type_register] if this crate defines it",
                                    #type_key
                                );
                            }
                            tindex
                        }
                    );
                    *TYPE_INDEX
                }
            }
        }
    };
    // search for field name base and derive the base type
    // we expect base always to be the first field
    let final_parent_check = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().map(|f| {
            let base_ty = f.ty.clone();
            quote! {
                const _: () = {
                    ::core::assert!(
                        !<#base_ty as #tvm_ffi_crate::object::ObjectCore>::TYPE_FINAL,
                        "an object type cannot derive from a final parent"
                    );
                };
            }
        }),
        _ => panic!("First field must be `<base_name>: <ObjectCoreType>`"),
    };
    let base_def_tokens = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().and_then(|f| {
            let (base_id, base_ty) = (f.ident.clone()?, f.ty.clone());
            // The transitive case of subtyping
            Some(quote! {
                const TYPE_DEPTH: i32 =
                    <#base_ty as #tvm_ffi_crate::object::ObjectCore>::TYPE_DEPTH + 1;

                #[inline]
                unsafe fn object_header_mut(
                    this: &mut Self
                ) -> &mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIObject {
                    const _: () = {
                        fn assert_impl<T: #tvm_ffi_crate::object::ObjectCore>() {}
                        let _ = assert_impl::<#base_ty>;
                    };
                    <#base_ty as #tvm_ffi_crate::object::ObjectCore>::object_header_mut(
                        &mut this.#base_id
                    )
                }
            })
        }),
        _ => panic!("First field must be `<base_name>: <ObjectCoreType>`"),
    };

    quote! {
        #final_parent_check

        unsafe impl #tvm_ffi_crate::object::ObjectCore for #struct_name {
            const TYPE_KEY: &'static str = #type_key;
            const TYPE_FINAL: bool = #type_final;

            #type_index_tokens

            #base_def_tokens
        }

        impl<'a> ::core::convert::From<&'a #struct_name> for #tvm_ffi_crate::AnyView<'a> {
            #[inline(always)]
            fn from(value: &'a #struct_name) -> Self {
                Self::from_object(value)
            }
        }

        #shallow_copy_impl
    }
}

/// The reflection a `#[type_register]` type declares: its `#[reflect]` fields,
/// its `#[type_s_eq_hash_kind]` and its `#[type_reflection]` function.
struct Reflection {
    /// The structural kind; `None` inherits the parent's.
    kind: Option<syn::Ident>,
    /// The function `#[type_reflection(path)]` names, which registers methods.
    methods: Option<syn::Path>,
    /// The parent's field name and every other field's name.
    base_id: syn::Ident,
    field_ids: Vec<syn::Ident>,
    /// The reflected fields.
    reflected: Vec<ReflectedField>,
    /// Whether the type registers the creator of its `UnsafeInit`, as an
    /// `#[rs_object]` type does.
    unsafe_init: bool,
}

/// A `#[reflect]` field: its identifier, type, flag names, and the name and
/// doc string registered for it.
struct ReflectedField {
    id: syn::Ident,
    ty: syn::Type,
    flags: Vec<syn::Ident>,
    name: String,
    doc: String,
}

const EXPECTED_REFLECT: &str =
    "Expect #[reflect] or #[reflect(<flag>, ..., name = \"<name>\", doc = \"<doc>\")]";

impl ReflectedField {
    /// The field's reflection from its `#[reflect]` attribute, whose `name`
    /// and `doc` default to the field's name and its `///` doc comment.
    fn parse(field: &syn::Field, attr: &syn::Attribute) -> Self {
        let id = field
            .ident
            .clone()
            .expect("#[derive(Object)] needs named fields");
        let mut flags = Vec::new();
        let mut name = None;
        let mut doc = None;
        let nested = match attr.parse_meta() {
            Ok(syn::Meta::Path(_)) => Vec::new(),
            Ok(syn::Meta::List(list)) => list.nested.into_iter().collect(),
            _ => panic!("{EXPECTED_REFLECT}"),
        };
        for item in nested {
            match item {
                syn::NestedMeta::Meta(syn::Meta::Path(path)) => {
                    flags.push(path.get_ident().cloned().expect(EXPECTED_REFLECT));
                }
                syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                    path,
                    lit: syn::Lit::Str(value),
                    ..
                })) if path.is_ident("name") || path.is_ident("doc") => {
                    let slot = if path.is_ident("name") {
                        &mut name
                    } else {
                        &mut doc
                    };
                    assert!(
                        slot.is_none(),
                        "duplicate `{}` in #[reflect] of field `{id}`",
                        path.get_ident().unwrap()
                    );
                    *slot = Some(value.value());
                }
                _ => panic!("{EXPECTED_REFLECT}"),
            }
        }
        Self {
            name: name.unwrap_or_else(|| id.unraw().to_string()),
            doc: doc.unwrap_or_else(|| doc_comment(&field.attrs)),
            id,
            ty: field.ty.clone(),
            flags,
        }
    }
}

/// The `///` doc comment of an item, as rustdoc reads it: the lines of its
/// `#[doc = "..."]` attributes without their leading space, joined by newlines.
fn doc_comment(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|attr| attr.path.is_ident("doc"))
        .filter_map(|attr| match attr.parse_meta() {
            Ok(syn::Meta::NameValue(syn::MetaNameValue {
                lit: syn::Lit::Str(line),
                ..
            })) => Some(line.value()),
            _ => None,
        })
        .map(|line| line.strip_prefix(' ').unwrap_or(&line).to_string())
        .collect();
    lines.join("\n").trim().to_string()
}

impl Reflection {
    /// The type's reflection, or `None` when it declares none and is not an
    /// `rs_object`.
    fn parse(derive_input: &DeriveInput, rs_object: bool) -> Option<Self> {
        let kind = get_attr(derive_input, "type_s_eq_hash_kind").map(|attr| {
            let kind = attr_to_str(attr);
            syn::Ident::new(&kind.value(), kind.span())
        });
        let methods = get_attr(derive_input, "type_reflection").map(|attr| {
            attr.parse_args::<syn::Path>()
                .expect("Expect #[type_reflection(<path to fn(&mut ObjectDef<Self>)>)]")
        });
        let syn::Data::Struct(s) = &derive_input.data else {
            return None;
        };
        let mut fields = s.fields.iter();
        let base = fields.next()?;
        assert!(
            !base.attrs.iter().any(|a| a.path.is_ident("reflect")),
            "the parent field cannot be #[reflect]; the parent registers its own fields"
        );
        let mut field_ids = Vec::new();
        let mut reflected: Vec<ReflectedField> = Vec::new();
        for field in fields {
            let id = field
                .ident
                .clone()
                .expect("#[derive(Object)] needs named fields");
            field_ids.push(id);
            if let Some(attr) = field.attrs.iter().find(|a| a.path.is_ident("reflect")) {
                let field = ReflectedField::parse(field, attr);
                if let Some(other) = reflected.iter().find(|f| f.name == field.name) {
                    panic!(
                        "fields `{}` and `{}` of `{}` both reflect as `{}`",
                        other.id, field.id, derive_input.ident, field.name
                    );
                }
                reflected.push(field);
            }
        }
        if kind.is_none() && methods.is_none() && reflected.is_empty() && !rs_object {
            return None;
        }
        Some(Self {
            kind,
            methods,
            base_id: base
                .ident
                .clone()
                .expect("First field must be `<base_name>: <ObjectCoreType>`"),
            field_ids,
            reflected,
            unsafe_init: rs_object,
        })
    }

    /// Registers the reflection under `tindex`, inside the registration of
    /// the type index, which holds the registration lock.
    fn register_tokens(
        &self,
        struct_name: &syn::Ident,
        type_key: &syn::LitStr,
        tvm_ffi_crate: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let reflection = quote!(#tvm_ffi_crate::reflection);
        let kind = match &self.kind {
            Some(kind) => quote!(Some(#reflection::SEqHashKind::#kind)),
            None => quote!(None),
        };
        let fields = self.reflected.iter().map(|field| {
            let ReflectedField {
                id,
                ty,
                flags,
                name,
                doc,
            } = field;
            // `#[reflect(seq_hash_ignore)]` names `FieldFlags::SEQ_HASH_IGNORE`.
            let flags = flags.iter().map(|flag| {
                format_ident!("{}", flag.to_string().to_uppercase(), span = flag.span())
            });
            quote! {
                .field::<#ty>(
                    #name,
                    #doc,
                    ::core::mem::offset_of!(#struct_name, #id),
                    #reflection::FieldFlags::NONE #(| #reflection::FieldFlags::#flags)*,
                )?
            }
        });
        // The `#[type_reflection]` function registers the methods after the
        // fields, as a C++ `ObjectDef` chains `def` after `def_ro`.
        let methods = self.methods.as_ref().map(|path| {
            // The registration is a `static` initializer, where `Self` does
            // not resolve, so `Self::f` names the struct.
            let mut path = path.clone();
            if let Some(first) = path.segments.first_mut() {
                if first.ident == "Self" {
                    first.ident = struct_name.clone();
                }
            }
            quote! {
                let mut def = def;
                #path(&mut def);
            }
        });
        let creator = if self.unsafe_init {
            quote!(Some(#reflection::unsafe_init_creator::<#struct_name>))
        } else {
            quote!(None)
        };
        quote! {
            let registered = (|| -> #tvm_ffi_crate::Result<()> {
                let def = #reflection::ObjectDef::<#struct_name>::registering(
                    tindex, #kind, #creator,
                )?
                    #(#fields)*
                    .shallow_copy()?;
                #methods
                def.finish()
            })();
            if let Err(error) = registered {
                panic!("Failed to register the reflection of type key {}: {}", #type_key, error);
            }
        }
    }

    /// `ShallowCopy`: the parent's shallow copy and a clone of every other field.
    fn shallow_copy_tokens(
        &self,
        struct_name: &syn::Ident,
        base_ty: &syn::Type,
        tvm_ffi_crate: &proc_macro2::TokenStream,
    ) -> proc_macro2::TokenStream {
        let base_id = &self.base_id;
        let field_ids = &self.field_ids;
        quote! {
            impl #tvm_ffi_crate::reflection::ShallowCopy for #struct_name {
                fn shallow_copy(&self) -> Self {
                    Self {
                        #base_id: <#base_ty as #tvm_ffi_crate::reflection::ShallowCopy>
                            ::shallow_copy(&self.#base_id),
                        #(#field_ids: ::core::clone::Clone::clone(&self.#field_ids),)*
                    }
                }
            }
        }
    }
}

/// Derive ObjectRef trait for a struct to generate boilerplate code
pub fn derive_object_ref(input: proc_macro::TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(input as DeriveInput);
    TokenStream::from(expand_object_ref(&derive_input, &get_tvm_ffi_crate()))
}

/// The `ObjectRefCore` and `AnyCompatible` impls of a reference struct,
/// naming the tvm-ffi crate `tvm_ffi_crate`.
pub(crate) fn expand_object_ref(
    derive_input: &DeriveInput,
    tvm_ffi_crate: &TokenStream2,
) -> TokenStream2 {
    let struct_name = derive_input.ident.clone();

    // The `ObjectArc<T>` slot is the first field; its name is the struct's choice
    // (`data` in this crate, `base` in stubgen output, so that reflected fields
    // named `data` keep their name).
    let (data_id, data_ty) = match &derive_input.data {
        syn::Data::Struct(s) => s
            .fields
            .iter()
            .next()
            .and_then(|f| Some((f.ident.clone()?, f.ty.clone()))),
        _ => panic!("derive only works for structs"),
    }
    .expect("First field must be `<name>: ObjectArc<T>`");

    let mut expanded = quote! {
        unsafe impl #tvm_ffi_crate::object::ObjectRefCore for #struct_name {
            type ContainerType = <#data_ty as std::ops::Deref>::Target;
            #[inline]
            fn data(this: &Self) -> &#tvm_ffi_crate::object::ObjectArc<Self::ContainerType> {
                &this.#data_id
            }
            #[inline]
            fn into_data(this: Self) -> #tvm_ffi_crate::object::ObjectArc<Self::ContainerType> {
                this.#data_id
            }
            #[inline]
            unsafe fn from_data(
                data: #tvm_ffi_crate::object::ObjectArc<Self::ContainerType>
            ) -> Self {
                Self { #data_id: data }
            }
        }

        // The handle is exactly its object pointer, which an FFI cell's
        // payload word holds, so an `Array` cell can lend `&Self` in place.
        const _: () = {
            type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>::ContainerType;
            assert!(
                ::core::mem::size_of::<#struct_name>()
                    == ::core::mem::size_of::<#tvm_ffi_crate::object::ObjectArc<ContainerType>>()
                    && ::core::mem::offset_of!(#struct_name, #data_id) == 0,
                "an ObjectRef must consist of its ObjectArc alone"
            );
        };

        unsafe impl #tvm_ffi_crate::type_traits::RefFromAnyUnchecked for #struct_name {
            #[inline(always)]
            unsafe fn ref_from_any_unchecked(cell: &#tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny) -> &Self {
                &*(::core::ptr::addr_of!(cell.data_union) as *const Self)
            }
        }

        // As `str`, `Path` and `[T]` do, so `impl AsRef<Base>` bounds accept the
        // base itself; upcasts add `AsRef` to each base.
        impl ::core::convert::AsRef<#struct_name> for #struct_name {
            #[inline(always)]
            fn as_ref(&self) -> &Self {
                self
            }
        }

        unsafe impl #tvm_ffi_crate::type_traits::DowncastRefFromAny for #struct_name {
            #[inline(always)]
            unsafe fn downcast_ref_from_any(
                cell: &#tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny,
            ) -> ::core::option::Option<&Self> {
                #tvm_ffi_crate::type_traits::downcast_ref_from_any_in_place::<Self>(cell)
            }
        }

        unsafe impl #tvm_ffi_crate::type_traits::DowncastRefFromObject for #struct_name {
            #[inline(always)]
            unsafe fn downcast_ref_from_object(
                place: &*mut #tvm_ffi_crate::tvm_ffi_sys::TVMFFIObject,
            ) -> ::core::option::Option<&Self> {
                #tvm_ffi_crate::type_traits::downcast_ref_from_object_in_place::<Self>(place)
            }
        }

        impl #tvm_ffi_crate::object::IntoObjectRef<#struct_name> for #struct_name {
            type Output = Self;
            #[inline(always)]
            fn into_object_ref(self) -> Self {
                self
            }
        }

        impl<'a> #tvm_ffi_crate::object::IntoObjectRef<#struct_name> for &'a #struct_name {
            type Output = &'a #struct_name;
            #[inline(always)]
            fn into_object_ref(self) -> &'a #struct_name {
                self
            }
        }

        impl ::std::convert::From<&#struct_name> for #struct_name {
            #[inline]
            fn from(value: &#struct_name) -> Self {
                value.clone()
            }
        }

        // The default `AnyCompatible`: the object type check alone, which
        // `match_any!` may route by.
        unsafe impl #tvm_ffi_crate::object::ObjectRefAnyCompatible for #struct_name {
            const MATCH_ANY_EXACT: bool = true;
        }
    };
    // The runtime type key and object address, so `Result<_, Self>` from
    // `downcast` can be unwrapped. `#[object_ref(skip_debug)]` keeps a
    // hand-written one.
    if !object_ref_options(derive_input).skip_debug {
        expanded.extend(quote! {
        impl ::core::fmt::Debug for #struct_name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    #tvm_ffi_crate::object::fmt_object_ref(self, f)
                }
            }
        });
    }
    // skip ObjectRef since it can create circular dependency with any.rs
    if struct_name != "ObjectRef" {
        expanded.extend(quote! {
            #tvm_ffi_crate::impl_try_from_any!(#struct_name);
            #tvm_ffi_crate::impl_arg_into_ref!(#struct_name);
            #tvm_ffi_crate::impl_into_arg_holder_default!(#struct_name);
            // Every object reference upcasts to the root `ObjectRef`.
            #tvm_ffi_crate::unsafe_impl_object_upcast!(#struct_name => #tvm_ffi_crate::object::ObjectRef);
        });
    }
    expanded
}

/// The options of `#[object_ref(...)]` on an `ObjectRef` derive.
#[derive(Default)]
struct ObjectRefOptions {
    skip_debug: bool,
}

fn object_ref_options(derive_input: &DeriveInput) -> ObjectRefOptions {
    let mut options = ObjectRefOptions::default();
    for attr in derive_input
        .attrs
        .iter()
        .filter(|a| a.path.is_ident("object_ref"))
    {
        let nested = match attr.parse_meta() {
            Ok(syn::Meta::List(list)) => list.nested,
            _ => abort!(attr, "expected `#[object_ref(...)]`"),
        };
        for item in nested {
            match item {
                syn::NestedMeta::Meta(syn::Meta::Path(path)) if path.is_ident("skip_debug") => {
                    options.skip_debug = true
                }
                other => abort!(other, "unknown `object_ref` option; expected `skip_debug`"),
            }
        }
    }
    options
}
