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

    // type index can be optional
    // for now we make it required for static index
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
        None => {
            quote! {
                #[inline]
                fn type_index() -> i32 {
                    static TYPE_INDEX: std::sync::LazyLock<i32> = std::sync::LazyLock::new(||
                        unsafe {
                            let type_key_arg =
                                 #tvm_ffi_crate::tvm_ffi_sys::TVMFFIByteArray::from_str(#type_key);
                            let mut tindex = 0;
                            let ret =  #tvm_ffi_crate::tvm_ffi_sys::TVMFFITypeKeyToIndex(
                                &type_key_arg, &mut tindex
                            );
                            if ret != 0 {
                                proc_macro_error::abort!("Failed to get type index for type key: {}", #type_key);
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
    let base_def_tokens = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().and_then(|f| {
            let (base_id, base_ty) = (f.ident.clone()?, f.ty.clone());
            // The transitive case of subtyping
            Some(quote! {
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
        unsafe impl #tvm_ffi_crate::object::ObjectCore for #struct_name {
            const TYPE_KEY: &'static str = #type_key;

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

    // search for field name base and derive the base type
    // we expect base always to be the first field
    let data_ty = match &derive_input.data {
        syn::Data::Struct(s) => s.fields.iter().next().and_then(|f| {
            let (data_id, data_ty) = (f.ident.clone()?, f.ty.clone());
            if data_id == "data" {
                // The transitive case of subtyping
                Some(data_ty)
            } else {
                None
            }
        }),
        _ => panic!("derive only works for structs"),
    }
    .expect("First field must be `data: ObjectArc<T>`");

    let mut expanded = quote! {
        unsafe impl #tvm_ffi_crate::object::ObjectRefCore for #struct_name {
            type ContainerType = <#data_ty as std::ops::Deref>::Target;
            #[inline]
            fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
                &this.data
            }
            #[inline]
            fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
                this.data
            }
            #[inline]
            fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
                Self { data}
            }
        }

        // implement AnyCompatible for #struct_name
        unsafe impl #tvm_ffi_crate::type_traits::AnyCompatible for #struct_name {
            fn type_str() -> String {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                <ContainerType as #tvm_ffi_crate::object::ObjectCore>::TYPE_KEY.into()
            }

            unsafe fn copy_to_any_view(
                src: &Self,
                data: &mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let type_index =
                    <ContainerType as #tvm_ffi_crate::object::ObjectCore>::type_index();
                data.type_index = type_index as i32;
                data.small_str_len = 0;
                let data_ptr = #tvm_ffi_crate::object::ObjectArc::<ContainerType>::as_raw(
                    &src.data
                );
                data.data_union.v_obj =
                    data_ptr as *mut ContainerType as *mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIObject;
            }

            unsafe fn check_any_strict(data: & #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny) -> bool {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let type_index =
                    <ContainerType as #tvm_ffi_crate::object::ObjectCore>::type_index();
                data.type_index == type_index as i32
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
                    data : #tvm_ffi_crate::object::ObjectArc::from_raw(
                        data_ptr as *mut ContainerType
                    )
                }
            }

            unsafe fn move_to_any(
                src: Self,
                data: &mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let type_index =
                    <ContainerType as #tvm_ffi_crate::object::ObjectCore>::type_index();
                data.type_index = type_index as i32;
                data.small_str_len = 0;
                let data_ptr = #tvm_ffi_crate::object::ObjectArc::into_raw(
                    src.data
                );
                data.data_union.v_obj =
                    data_ptr as *mut ContainerType as *mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIObject;
            }

            unsafe fn move_from_any_after_check(
                data: &mut  #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) -> Self {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let data_ptr = data.data_union.v_obj as *mut ContainerType;
                Self {
                    data : #tvm_ffi_crate::object::ObjectArc::<ContainerType>::from_raw(data_ptr)
                }
            }

            unsafe fn try_cast_from_any_view(
                data: & #tvm_ffi_crate::tvm_ffi_sys::TVMFFIAny
            ) -> Result<Self, ()> {
                type ContainerType = <#struct_name as #tvm_ffi_crate::object::ObjectRefCore>
                    ::ContainerType;
                let type_index =
                    <ContainerType as #tvm_ffi_crate::object::ObjectCore>::type_index();
                if data.type_index == type_index as i32 {
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
