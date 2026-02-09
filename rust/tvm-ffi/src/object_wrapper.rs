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

use crate::any::Any;
use crate::object::{Object, ObjectArc, ObjectRef, ObjectRefCore};
use crate::type_traits::AnyCompatible;
use std::marker::PhantomData;
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldGetter, TVMFFIGetTypeInfo, TVMFFIObject,
    TVMFFITypeKeyToIndex,
};

/// Runtime support for stubgen-generated object wrappers.
///
/// This module is intended for code emitted by the Rust stub generator and is
/// not meant as a general-purpose user-facing API.
pub trait ObjectWrapper: Clone {
    const TYPE_KEY: &'static str;
    fn from_object(inner: ObjectRef) -> Self;
    fn as_object_ref(&self) -> &ObjectRef;
    fn into_object_ref(self) -> ObjectRef;
}

struct FieldGetterInner {
    offset: usize,
    getter: TVMFFIFieldGetter,
}

impl FieldGetterInner {
    fn get_any(&self, obj: &ObjectRef) -> crate::Result<Any> {
        unsafe {
            let arc = <ObjectRef as ObjectRefCore>::data(obj);
            let raw = ObjectArc::as_raw(arc) as *mut TVMFFIObject;
            if raw.is_null() {
                crate::bail!(crate::error::ATTRIBUTE_ERROR, "Null object for field access");
            }
            let field_ptr = (raw as *mut u8).add(self.offset) as *mut std::ffi::c_void;
            let mut out = TVMFFIAny::new();
            crate::check_safe_call!((self.getter)(field_ptr, &mut out))?;
            Ok(Any::from_raw_ffi_any(out))
        }
    }
}

pub struct FieldGetter<T> {
    inner: FieldGetterInner,
    _marker: PhantomData<T>,
}

// FieldGetter stores only reflection metadata, not values of T.
// It is safe to share across threads regardless of T's Send/Sync.
unsafe impl<T> Send for FieldGetter<T> {}
unsafe impl<T> Sync for FieldGetter<T> {}

impl<T> FieldGetter<T> {
    pub fn new(type_key: &'static str, field_name: &'static str) -> crate::Result<Self> {
        let inner = resolve_field_by_type_key(type_key, field_name)?;
        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    pub fn get_any(&self, obj: &ObjectRef) -> crate::Result<Any> {
        self.inner.get_any(obj)
    }
}

impl<T> FieldGetter<T>
where
    T: TryFrom<Any>,
    T::Error: Into<crate::Error>,
{
    pub fn get(&self, obj: &ObjectRef) -> crate::Result<T> {
        self.inner.get_any(obj)?.try_into().map_err(Into::into)
    }
}

fn resolve_field_by_type_key(
    type_key: &'static str,
    field_name: &'static str,
) -> crate::Result<FieldGetterInner> {
    unsafe {
        let key = TVMFFIByteArray::from_str(type_key);
        let mut type_index = 0i32;
        crate::check_safe_call!(TVMFFITypeKeyToIndex(&key, &mut type_index))?;
        resolve_field_by_type_index(type_index, field_name)
    }
}

fn resolve_field_by_type_index(
    type_index: i32,
    field_name: &'static str,
) -> crate::Result<FieldGetterInner> {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            crate::bail!(
                crate::error::ATTRIBUTE_ERROR,
                "Type info missing for field {}",
                field_name
            );
        }
        let info = &*info;
        if info.fields.is_null() || info.num_fields <= 0 {
            crate::bail!(
                crate::error::ATTRIBUTE_ERROR,
                "Type {} has no fields",
                info.type_key.as_str()
            );
        }
        let fields = std::slice::from_raw_parts(info.fields, info.num_fields as usize);
        for field in fields {
            if field.name.as_str() != field_name {
                continue;
            }
            let getter = match field.getter {
                Some(getter) => getter,
                None => {
                    crate::bail!(
                        crate::error::ATTRIBUTE_ERROR,
                        "Field {} has no getter",
                        field_name
                    );
                }
            };
            if field.offset < 0 {
                crate::bail!(
                    crate::error::ATTRIBUTE_ERROR,
                    "Field {} has invalid offset",
                    field_name
                );
            }
            return Ok(FieldGetterInner {
                offset: field.offset as usize,
                getter,
            });
        }
        crate::bail!(
            crate::error::ATTRIBUTE_ERROR,
            "Field {} not found",
            field_name
        );
    }
}

fn type_index_for_key(type_key: &'static str) -> Option<i32> {
    let key = unsafe { TVMFFIByteArray::from_str(type_key) };
    let mut index = 0i32;
    let code = unsafe { TVMFFITypeKeyToIndex(&key, &mut index) };
    if code == 0 {
        Some(index)
    } else {
        None
    }
}

unsafe fn is_instance_type(type_index: i32, target_index: i32) -> bool {
    if type_index == target_index {
        return true;
    }
    let info = TVMFFIGetTypeInfo(type_index);
    if info.is_null() {
        return false;
    }
    let info = &*info;
    let ancestors = info.type_acenstors;
    if ancestors.is_null() {
        return false;
    }
    for depth in 0..info.type_depth {
        let ancestor = *ancestors.add(depth as usize);
        if !ancestor.is_null() && (*ancestor).type_index == target_index {
            return true;
        }
    }
    false
}

unsafe impl<T: ObjectWrapper> AnyCompatible for T {
    fn type_str() -> String {
        T::TYPE_KEY.to_string()
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        let obj = src.as_object_ref();
        let arc = <ObjectRef as ObjectRefCore>::data(obj);
        let raw = ObjectArc::as_raw(arc) as *mut TVMFFIObject;
        data.type_index = (*raw).type_index;
        data.small_str_len = 0;
        data.data_union.v_obj = raw;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        let obj = src.into_object_ref();
        let arc = <ObjectRef as ObjectRefCore>::into_data(obj);
        let raw = ObjectArc::into_raw(arc) as *mut TVMFFIObject;
        data.type_index = (*raw).type_index;
        data.small_str_len = 0;
        data.data_union.v_obj = raw;
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        let Some(target_index) = type_index_for_key(T::TYPE_KEY) else {
            return false;
        };
        is_instance_type(data.type_index, target_index)
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *mut TVMFFIObject;
        crate::object::unsafe_::inc_ref(ptr);
        let arc = ObjectArc::from_raw(ptr as *mut Object);
        let obj = <ObjectRef as ObjectRefCore>::from_data(arc);
        T::from_object(obj)
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *mut TVMFFIObject;
        let arc = ObjectArc::from_raw(ptr as *mut Object);
        data.type_index = crate::TypeIndex::kTVMFFINone as i32;
        data.data_union.v_int64 = 0;
        let obj = <ObjectRef as ObjectRefCore>::from_data(arc);
        T::from_object(obj)
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if Self::check_any_strict(data) {
            Ok(Self::copy_from_any_view_after_check(data))
        } else {
            Err(())
        }
    }
}
