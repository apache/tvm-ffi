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
use quote::quote;
use syn::DeriveInput;

use crate::utils::*;

/// Derive Object trait for a struct to generate boilerplate code
pub fn derive_object(input: proc_macro::TokenStream) -> TokenStream {
    let tvm_ffi_crate = get_tvm_ffi_crate();
    let derive_input = syn::parse_macro_input!(input as DeriveInput);
    let struct_name = derive_input.ident.clone();

    let type_key = get_attr(&derive_input, "type_key")
        .map(attr_to_str)
        .expect("Expect #[type_key = \"<my_type_key>\"] attribute");
    let type_final = match get_attr(&derive_input, "type_final") {
        Some(attr) if matches!(attr.parse_meta(), Ok(syn::Meta::Path(_))) => true,
        Some(_) => panic!("Expect #[type_final] attribute"),
        None => false,
    };

    // Whether this type is defined here, so that it registers its key, as C++
    // `TVM_FFI_DECLARE_OBJECT_INFO` does. Without it, the type binds a key
    // that its defining library, usually C++, has registered.
    let type_register = match get_attr(&derive_input, "type_register") {
        Some(attr) if matches!(attr.parse_meta(), Ok(syn::Meta::Path(_))) => true,
        Some(_) => panic!("Expect #[type_register] attribute"),
        None => false,
    };

    // Reserved child slots and whether children may exceed them, as C++
    // `_type_child_slots` and `_type_child_slots_can_overflow` declare them.
    let child_slots = match get_attr(&derive_input, "type_child_slots").map(|a| a.parse_meta()) {
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
        match get_attr(&derive_input, "type_child_slots_can_overflow").map(|a| a.parse_meta()) {
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
            || (get_attr(&derive_input, "type_child_slots").is_none()
                && get_attr(&derive_input, "type_child_slots_can_overflow").is_none()),
        "#[type_child_slots] and #[type_child_slots_can_overflow] need #[type_register]"
    );
    // we expect base always to be the first field
    let base_ty = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().map(|f| f.ty.clone()),
        _ => None,
    }
    .expect("First field must be `<base_name>: <ObjectCoreType>`");

    // The reflection a type defined here registers along with its key, as
    // C++ `refl::ObjectDef` does: its fields marked `#[def_ro]` or
    // `#[def_rw]`, then what the function `#[type_reflection(path)]` names
    // registers. `#[def_rw]` needs `#[type_mutable]`, as C++ `def_rw` needs
    // `_type_mutable`.
    let type_mutable = match get_attr(&derive_input, "type_mutable") {
        Some(attr) if matches!(attr.parse_meta(), Ok(syn::Meta::Path(_))) => true,
        Some(_) => panic!("Expect #[type_mutable] attribute"),
        None => false,
    };
    let type_reflection = get_attr(&derive_input, "type_reflection").map(|attr| {
        attr.parse_args::<syn::Path>()
            .expect("Expect #[type_reflection(<path to fn(&mut ObjectDef<Self>)>)]")
    });
    let fields = reflected_fields(&derive_input);
    assert!(
        type_register || (fields.is_empty() && type_reflection.is_none()),
        "#[def_ro], #[def_rw] and #[type_reflection] need #[type_register]: only the \
         library that defines a type registers its reflection"
    );
    assert!(
        type_mutable || fields.iter().all(|field| !field.writable),
        "#[def_rw] needs #[type_mutable]"
    );
    let reflection_tokens = if fields.is_empty() && type_reflection.is_none() {
        quote! {}
    } else {
        let field_tokens = fields.iter().map(|field| {
            let (ident, ty, name, doc, writable) = (
                &field.ident,
                &field.ty,
                &field.name,
                &field.doc,
                field.writable,
            );
            quote! {
                def.field::<#ty>(
                    #name,
                    #doc,
                    ::core::mem::offset_of!(#struct_name, #ident),
                    #writable,
                );
            }
        });
        let hook_tokens = type_reflection.map(|path| quote! { #path(&mut def); });
        quote! {
            // Unless another copy of this type, in another library,
            // registered it.
            if let Some(mut def) =
                #tvm_ffi_crate::reflection::ObjectDef::<#struct_name>::begin(tindex)
            {
                #(#field_tokens)*
                #hook_tokens
                def.finish();
            }
        }
    };

    // A type with a static index has it; one defined here registers its key
    // under its parent on first use, or takes the index the key already has,
    // as C++ `TVM_FFI_DECLARE_OBJECT_INFO` does; any other type binds the
    // index its defining library registered.
    let type_index_tokens = match get_attr(&derive_input, "type_index").map(attr_to_expr) {
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
                    static TYPE_INDEX: std::sync::LazyLock<i32> = std::sync::LazyLock::new(||
                        unsafe {
                            // The parent first, since registering it takes the
                            // same lock.
                            let parent =
                                <#base_ty as #tvm_ffi_crate::object::ObjectCore>::type_index();
                            let tindex = {
                                let _registering = #tvm_ffi_crate::object::TYPE_REGISTRATION
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                let type_key_arg =
                                    #tvm_ffi_crate::tvm_ffi_sys::TVMFFIByteArray::from_str(
                                        #type_key
                                    );
                                #tvm_ffi_crate::tvm_ffi_sys::TVMFFITypeGetOrAllocIndex(
                                    &type_key_arg,
                                    -1,
                                    <#struct_name as #tvm_ffi_crate::object::ObjectCore>
                                        ::TYPE_DEPTH,
                                    #child_slots,
                                    #child_slots_can_overflow as i32,
                                    parent,
                                )
                            };
                            if tindex < 0 {
                                panic!(
                                    "Failed to get or allocate type index for type key: {}",
                                    #type_key
                                );
                            }
                            // Each registration takes the lock, so the
                            // registering function may register other types.
                            #reflection_tokens
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
                    static TYPE_INDEX: std::sync::LazyLock<i32> = std::sync::LazyLock::new(||
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
                    #base_ty::object_header_mut(&mut this.#base_id)
                }
            })
        }),
        _ => panic!("First field must be `<base_name>: <ObjectCoreType>`"),
    };

    let expanded = quote! {
        #final_parent_check

        unsafe impl #tvm_ffi_crate::object::ObjectCore for #struct_name {
            const TYPE_KEY: &'static str = #type_key;
            const TYPE_FINAL: bool = #type_final;

            #type_index_tokens

            #base_def_tokens
        }
    };
    TokenStream::from(expanded)
}

/// Derive ObjectRef trait for a struct to generate boilerplate code
pub fn derive_object_ref(input: proc_macro::TokenStream) -> TokenStream {
    let tvm_ffi_crate = get_tvm_ffi_crate();
    let derive_input = syn::parse_macro_input!(input as DeriveInput);
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
            fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
                &this.#data_id
            }
            #[inline]
            fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
                this.#data_id
            }
            #[inline]
            unsafe fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
                Self { #data_id: data }
            }
        }

        impl ::std::convert::From<&#struct_name> for #struct_name {
            #[inline]
            fn from(value: &#struct_name) -> Self {
                value.clone()
            }
        }

        // implement AnyCompatible for #struct_name
        unsafe impl #tvm_ffi_crate::type_traits::AnyCompatible for #struct_name {
            const FIELD_STATIC_TYPE_INDEX: i32 =
                #tvm_ffi_crate::tvm_ffi_sys::TVMFFITypeIndex::kTVMFFIObject as i32;

            const MATCH_ANY_EXACT: bool = {
                type ContainerType =
                    <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>::ContainerType;
                <ContainerType as #tvm_ffi_crate::object::ObjectCore>::TYPE_FINAL
            };

            #[inline]
            fn match_any_exact_type_index() -> i32 {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                <ContainerType as #tvm_ffi_crate::object::ObjectCore>::type_index()
            }

            fn type_str() -> std::string::String {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                <ContainerType as #tvm_ffi_crate::object::ObjectCore>::TYPE_KEY.into()
            }

            #[inline(always)]
            unsafe fn copy_to_any_view(
                src: &Self,
                data: &mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let data_ptr = #tvm_ffi_crate::object::ObjectArc::<ContainerType>::as_raw(
                    &src.#data_id
                );
                let object_ptr =
                    data_ptr as *mut ContainerType as *mut #tvm_ffi_crate::tvm_ffi_sys::TVMFFIObject;
                data.type_index = (*object_ptr).type_index;
                data.small_str_len = 0;
                data.data_union.v_obj = object_ptr;
            }

            #[inline(always)]
            unsafe fn check_any_strict(
                data: & #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) -> bool {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                #tvm_ffi_crate::object::is_instance_of::<ContainerType>(data.type_index)
            }

            unsafe fn copy_from_any_view_after_check(
                data: & #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) -> Self {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let data_ptr = data.data_union.v_obj;
                // need to increase ref because original weak ptr
                // do not own the code
                #tvm_ffi_crate::object::unsafe_::inc_ref(
                    data_ptr as *mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIObject
                );
                Self {
                    #data_id: #tvm_ffi_crate::object::ObjectArc::from_raw(
                        data_ptr as *mut ContainerType
                    )
                }
            }

            #[inline(always)]
            unsafe fn move_to_any(
                src: Self,
                data: &mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let data_ptr = #tvm_ffi_crate::object::ObjectArc::into_raw(
                    src.#data_id
                );
                let object_ptr =
                    data_ptr as *mut ContainerType as *mut #tvm_ffi_crate::tvm_ffi_sys::TVMFFIObject;
                data.type_index = (*object_ptr).type_index;
                data.small_str_len = 0;
                data.data_union.v_obj = object_ptr;
            }

            #[inline(always)]
            unsafe fn move_from_any_after_check(
                data: &mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) -> Self {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let data_ptr = data.data_union.v_obj as *mut ContainerType;
                Self {
                    #data_id: #tvm_ffi_crate::object::ObjectArc::<ContainerType>::from_raw(data_ptr)
                }
            }

            unsafe fn try_cast_from_any_view(
                data: & #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) -> Result<Self, ()> {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                if #tvm_ffi_crate::object::is_instance_of::<ContainerType>(data.type_index) {
                    Ok(Self::copy_from_any_view_after_check(data))
                } else {
                    Err(())
                }
            }
        }
    };
    // skip ObjectRef since it can create circular dependency with any.rs
    if struct_name != "ObjectRef" {
        expanded.extend(quote! {
            #tvm_ffi_crate::impl_try_from_any!(#struct_name);
            #tvm_ffi_crate::impl_arg_into_ref!(#struct_name);
            #tvm_ffi_crate::impl_into_arg_holder_default!(#struct_name);
        });
    }
    TokenStream::from(expanded)
}

/// A field marked `#[def_ro]` or `#[def_rw]`.
struct ReflectedField {
    ident: syn::Ident,
    ty: syn::Type,
    name: String,
    doc: String,
    writable: bool,
}

/// The fields marked `#[def_ro]` or `#[def_rw]`, which may set the reflected
/// `name` (by default the field's) and `doc`: `#[def_ro(name = "...", doc = "...")]`.
fn reflected_fields(derive_input: &DeriveInput) -> Vec<ReflectedField> {
    let fields = match &derive_input.data {
        syn::Data::Struct(s) => &s.fields,
        _ => return Vec::new(),
    };
    let mut reflected = Vec::new();
    for (position, field) in fields.iter().enumerate() {
        let attrs: Vec<_> = field
            .attrs
            .iter()
            .filter(|a| a.path.is_ident("def_ro") || a.path.is_ident("def_rw"))
            .collect();
        let attr = match attrs.as_slice() {
            [] => continue,
            [attr] => attr,
            _ => panic!("a field takes one of #[def_ro] and #[def_rw]"),
        };
        assert!(
            position != 0,
            "the first field, the parent, cannot be reflected"
        );
        let ident = field
            .ident
            .clone()
            .expect("#[def_ro] and #[def_rw] need a named field");
        let mut name = syn::ext::IdentExt::unraw(&ident).to_string();
        let mut doc = String::new();
        match attr.parse_meta() {
            Ok(syn::Meta::Path(_)) => {}
            Ok(syn::Meta::List(list)) => {
                for item in list.nested {
                    match item {
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path,
                            lit: syn::Lit::Str(value),
                            ..
                        })) if path.is_ident("name") => name = value.value(),
                        syn::NestedMeta::Meta(syn::Meta::NameValue(syn::MetaNameValue {
                            path,
                            lit: syn::Lit::Str(value),
                            ..
                        })) if path.is_ident("doc") => doc = value.value(),
                        _ => panic!("Expect #[def_ro(name = \"...\", doc = \"...\")]"),
                    }
                }
            }
            _ => panic!("Expect #[def_ro(name = \"...\", doc = \"...\")]"),
        }
        reflected.push(ReflectedField {
            ident,
            ty: field.ty.clone(),
            name,
            doc,
            writable: attr.path.is_ident("def_rw"),
        });
    }
    reflected
}
