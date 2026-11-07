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
    // Upcasts reinterpret a pointer to the object as one to its parent, which
    // only `repr(C)` keeps at offset zero.
    if !derive_input.attrs.iter().any(is_repr_c) {
        abort!(struct_name, "#[derive(Object)] needs #[repr(C)]");
    }
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
    // under its parent on its first `type_index()`, which a static init block
    // calls at load, or takes the index the key already has,
    // as C++ `TVM_FFI_DECLARE_OBJECT_INFO` does; any other type binds the
    // index its defining library registered.
    let type_index_tokens = match get_attr(derive_input, "type_index") {
        Some(attr) => {
            // A static index is trusted: the type must be the one the runtime
            // gives that index.
            let type_index_expr = attr
                .parse_args_with(|input: syn::parse::ParseStream| {
                    input.parse::<syn::Token![unsafe]>()?;
                    input.parse::<syn::Expr>()
                })
                .unwrap_or_else(|error| {
                    abort!(
                        attr,
                        "expected #[type_index(unsafe TypeIndex::...)]: {}",
                        error
                    )
                });
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
                            let _registering =
                                #tvm_ffi_crate::object::lock_type_registration();
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
                                ::core::panic!(
                                    "Failed to get or allocate type index for type key: {}",
                                    #type_key
                                );
                            }
                            #tvm_ffi_crate::object::assert_registered_type_matches::<#struct_name>(
                                tindex, parent, false,
                            );
                            #register_reflection
                            tindex
                        }
                    );
                    *TYPE_INDEX
                }

                fn type_index_available() -> bool {
                    <#base_ty as #tvm_ffi_crate::object::ObjectCore>::type_index_available()
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
                                ::core::panic!(
                                    "Type key {} is not registered: load the library that defines \
                                     it first, or add #[type_register] if this crate defines it",
                                    #type_key
                                );
                            }
                            #tvm_ffi_crate::object::assert_registered_type_matches::<#struct_name>(
                                tindex,
                                <#base_ty as #tvm_ffi_crate::object::ObjectCore>::type_index(),
                                true,
                            );
                            tindex
                        }
                    );
                    *TYPE_INDEX
                }

                fn type_index_available() -> bool {
                    let mut tindex = 0;
                    // SAFETY: the key and the index outlive the call.
                    unsafe {
                        let type_key_arg =
                            #tvm_ffi_crate::tvm_ffi_sys::TVMFFIByteArray::from_str(#type_key);
                        #tvm_ffi_crate::tvm_ffi_sys::TVMFFITypeKeyToIndex(&type_key_arg, &mut tindex)
                            == 0
                    }
                }
            }
        }
    };
    // A type defined here registers when its library loads, as a C++ type does
    // through `refl::ObjectDef` in a static init block; a binding stays lazy,
    // and so does a type under a binding whose library is not loaded yet.
    let register_at_load = if type_register && get_attr(derive_input, "type_index").is_none() {
        if !derive_input.generics.params.is_empty() {
            abort!(
                struct_name,
                "#[type_register] needs a struct without generic parameters"
            );
        }
        // The key is in the text of the block, which identifies the block, so
        // two libraries that differ only in their keys both register.
        quote! {
            #tvm_ffi_crate::static_init_block! {
                let _: &str = #type_key;
                if <#struct_name as #tvm_ffi_crate::object::ObjectCore>::type_index_available() {
                    let _ = <#struct_name as #tvm_ffi_crate::object::ObjectCore>::type_index();
                }
            }
        }
    } else {
        quote! {}
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
                type Parent = #base_ty;

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

    let base_id = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().and_then(|f| f.ident.clone()),
        _ => None,
    }
    .expect("First field must be `<base_name>: <ObjectCoreType>`");

    quote! {
        #final_parent_check

        const _: () = ::core::assert!(
            ::core::mem::offset_of!(#struct_name, #base_id) == 0,
            "the parent object must be the first field"
        );

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

        #register_at_load
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
            Some(kind) => quote!(::core::option::Option::Some(#reflection::SEqHashKind::#kind)),
            None => quote!(::core::option::Option::None),
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
                let def = def.field::<#ty>(
                    #name,
                    #doc,
                    ::core::mem::offset_of!(#struct_name, #id),
                    #reflection::FieldFlags::NONE #(| #reflection::FieldFlags::#flags)*,
                )?;
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
            quote!(#path)
        });
        let methods = methods.unwrap_or_else(|| quote!(|_| {}));
        let creator = if self.unsafe_init {
            quote!(::core::option::Option::Some(#reflection::unsafe_init_creator::<#struct_name>))
        } else {
            quote!(::core::option::Option::None)
        };
        quote! {
            let registered = #reflection::register_object::<#struct_name>(
                tindex,
                #kind,
                #creator,
                |def| {
                    #(#fields)*
                    ::core::result::Result::Ok(def)
                },
                #methods,
            );
            if let ::core::result::Result::Err(error) = registered {
                ::core::panic!("Failed to register the reflection of type key {}: {}", #type_key, error);
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

/// The impls of a reference struct, as configured by `#[object_ref(...)]`,
/// naming the tvm-ffi crate `tvm_ffi_crate`.
pub(crate) fn expand_object_ref(
    derive_input: &DeriveInput,
    tvm_ffi_crate: &TokenStream2,
) -> TokenStream2 {
    let struct_name = derive_input.ident.clone();
    let options = object_ref_options(derive_input);
    let tvm_ffi_crate = &match &options.tvm_ffi_crate {
        Some(path) => quote!(#path),
        None => tvm_ffi_crate.clone(),
    };

    // The `ObjectArc<T>` slot is the first field; its name is the struct's choice
    // (`data` in this crate, `data_` in stubgen output, so that reflected fields
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
    // Spell the object type as written, so rustdoc shows `Target = FooObj`.
    let object_ty = object_arc_argument(&data_ty)
        .unwrap_or_else(|| quote!(<#data_ty as ::core::ops::Deref>::Target));

    let mut expanded = quote! {
        unsafe impl #tvm_ffi_crate::object::ObjectRefCore for #struct_name {
            type ContainerType = #object_ty;
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

        impl #struct_name {
            /// The reference to a new object, for the type's constructors.
            #[allow(dead_code)]
            #[inline(always)]
            fn from_object(object: #object_ty) -> Self {
                Self { #data_id: #tvm_ffi_crate::object::ObjectArc::new(object) }
            }
        }

        // The handle is exactly its object pointer, which an FFI cell's
        // payload word holds, so an `Array` cell can lend `&Self` in place.
        const _: () = {
            type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>::ContainerType;
            ::core::assert!(
                ::core::mem::size_of::<#struct_name>()
                    == ::core::mem::size_of::<#tvm_ffi_crate::object::ObjectArc<ContainerType>>()
                    && ::core::mem::offset_of!(#struct_name, #data_id) == 0,
                "an ObjectRef must consist of its ObjectArc alone"
            );
        };

        // SAFETY: one `ObjectArc` (asserted above), checked by `check_any_strict`.
        #tvm_ffi_crate::impl_object_ref_downcast_ref!(unsafe #struct_name);

        // As `str`, `Path` and `[T]` do, so `impl AsRef<Base>` bounds accept the
        // base itself; upcasts add `AsRef` to each base.
        impl ::core::convert::AsRef<#struct_name> for #struct_name {
            #[inline(always)]
            fn as_ref(&self) -> &Self {
                self
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
                ::core::clone::Clone::clone(value)
            }
        }
    };
    // The default `AnyCompatible`: the object type check alone, which
    // `match_any!` may route by.
    if !options.skip_any_compatible {
        expanded.extend(quote! {
            unsafe impl #tvm_ffi_crate::object::ObjectRefAnyCompatible for #struct_name {
                const MATCH_ANY_EXACT: bool = true;
            }
        });
        if derive_input.generics.params.is_empty() {
            expanded.extend(quote! {
                unsafe impl #tvm_ffi_crate::object::PlainObjectRef for #struct_name {}
            });
        }
    }
    // Field reads through the reference, as C++ `operator->`.
    if !options.skip_deref {
        expanded.extend(quote! {
            impl ::core::ops::Deref for #struct_name {
                type Target = #object_ty;
                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    &self.#data_id
                }
            }
        });
    }
    // The runtime type key and object address, so `Result<_, Self>` from
    // `downcast` can be unwrapped.
    if !options.skip_debug {
        expanded.extend(quote! {
            impl ::core::fmt::Debug for #struct_name {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    #tvm_ffi_crate::object::fmt_object_ref(self, f)
                }
            }
        });
    }
    // The runtime `repr`, as `AnyView` formats it.
    if !options.skip_display {
        expanded.extend(quote! {
            impl ::core::fmt::Display for #struct_name {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    let object = &**#tvm_ffi_crate::object::ObjectRefCore::data(self);
                    ::core::fmt::Display::fmt(&#tvm_ffi_crate::AnyView::from_object(object), f)
                }
            }
        });
    }
    // `ObjectRef` itself skips these: `any.rs` implements them for it.
    if struct_name != "ObjectRef" {
        expanded.extend(quote! {
            #tvm_ffi_crate::impl_any_compatible_conversions!(#struct_name);
            // Every object reference upcasts to the root `ObjectRef`.
            #tvm_ffi_crate::impl_object_ref_upcast!(
                unsafe #struct_name => #tvm_ffi_crate::object::ObjectRef
            );
        });
    }
    // The root upcast is already above.
    if let Some(parent) = options.parent.filter(|parent| !is_object_ref_root(parent)) {
        if !options.unsafe_parent {
            expanded.extend(quote! {
                const _: () = {
                    fn plain<P: #tvm_ffi_crate::object::PlainObjectRef>() {}
                    let _ = plain::<#parent>;
                };
            });
        }
        // The upcast is sound only if the object type derives from the parent's.
        expanded.extend(quote! {
            const _: () = {
                fn derives_from<T: #tvm_ffi_crate::object::ObjectDerivesFrom<A, I>, A, I>() {}
                let _ = derives_from::<
                    <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>::ContainerType,
                    <#parent as #tvm_ffi_crate::object::ObjectRefCore>::ContainerType,
                    _,
                >;
            };
            #tvm_ffi_crate::impl_object_ref_upcast!(unsafe #struct_name => #parent);
        });
    }
    expanded
}

/// Whether `attr` is a `#[repr(...)]` that includes `C`.
fn is_repr_c(attr: &syn::Attribute) -> bool {
    let Ok(syn::Meta::List(list)) = attr.parse_meta() else {
        return false;
    };
    list.path.is_ident("repr")
        && list.nested.iter().any(|item| {
            matches!(item, syn::NestedMeta::Meta(syn::Meta::Path(path)) if path.is_ident("C"))
        })
}

/// `T` of a field type spelled `ObjectArc<T>`.
fn object_arc_argument(ty: &syn::Type) -> Option<TokenStream2> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let last = path.path.segments.last()?;
    if last.ident != "ObjectArc" {
        return None;
    }
    match &last.arguments {
        syn::PathArguments::AngleBracketed(args) => match args.args.first()? {
            syn::GenericArgument::Type(object) if args.args.len() == 1 => Some(quote!(#object)),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `ty` names the root `ObjectRef`.
fn is_object_ref_root(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == "ObjectRef"),
        _ => false,
    }
}

/// The options of `#[object_ref(...)]` on an `ObjectRef` derive.
#[derive(Default)]
struct ObjectRefOptions {
    tvm_ffi_crate: Option<syn::Path>,
    parent: Option<syn::Type>,
    unsafe_parent: bool,
    skip_deref: bool,
    skip_debug: bool,
    skip_display: bool,
    skip_any_compatible: bool,
}

/// One `#[object_ref(...)]` item: a flag, `[unsafe] parent = Type` or `crate = path`.
enum ObjectRefOption {
    Flag(syn::Ident),
    Parent(Box<syn::Type>, bool),
    Crate(syn::Path),
}

impl syn::parse::Parse for ObjectRefOption {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let is_unsafe = input.parse::<Option<syn::Token![unsafe]>>()?.is_some();
        let name: syn::Ident = input.call(syn::Ident::parse_any)?;
        if name == "parent" {
            input.parse::<syn::Token![=]>()?;
            return Ok(ObjectRefOption::Parent(Box::new(input.parse()?), is_unsafe));
        }
        if is_unsafe {
            return Err(syn::Error::new(name.span(), "only `parent` takes `unsafe`"));
        }
        if name == "crate" {
            input.parse::<syn::Token![=]>()?;
            return Ok(ObjectRefOption::Crate(syn::Path::parse_mod_style(input)?));
        }
        Ok(ObjectRefOption::Flag(name))
    }
}

fn object_ref_options(derive_input: &DeriveInput) -> ObjectRefOptions {
    let mut options = ObjectRefOptions::default();
    for attr in derive_input
        .attrs
        .iter()
        .filter(|a| a.path.is_ident("object_ref"))
    {
        let items = attr
            .parse_args_with(
                syn::punctuated::Punctuated::<ObjectRefOption, syn::Token![,]>::parse_terminated,
            )
            .unwrap_or_else(|error| abort!(error.span(), "{}", error));
        for item in items {
            match item {
                ObjectRefOption::Parent(parent, is_unsafe) => {
                    options.parent = Some(*parent);
                    options.unsafe_parent = is_unsafe;
                }
                ObjectRefOption::Crate(path) => options.tvm_ffi_crate = Some(path),
                ObjectRefOption::Flag(flag) if flag == "skip_deref" => options.skip_deref = true,
                ObjectRefOption::Flag(flag) if flag == "skip_debug" => options.skip_debug = true,
                ObjectRefOption::Flag(flag) if flag == "skip_display" => {
                    options.skip_display = true
                }
                ObjectRefOption::Flag(flag) if flag == "skip_any_compatible" => {
                    options.skip_any_compatible = true
                }
                ObjectRefOption::Flag(flag) => abort!(
                    flag,
                    "unknown `object_ref` option; expected `[unsafe] parent = ...`, `skip_deref`, \
                     `skip_debug`, `skip_display`, `skip_any_compatible` or `crate = ...`"
                ),
            }
        }
    }
    options
}
