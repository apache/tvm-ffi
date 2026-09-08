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

//! Safe access to object reflection metadata.

use std::ffi::c_void;
use std::ptr::NonNull;

use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldGetter, TVMFFIFieldInfo, TVMFFIGetTypeAttrColumn,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex,
};
use crate::{Any, AnyView, Error, ObjectCore, Result, TYPE_ERROR};

/// A registry-owned type-attribute column indexed by runtime type.
///
/// [`TypeAttrColumn::get`] returns owning copies. Registration must not race
/// with reads.
#[derive(Clone, Copy)]
pub struct TypeAttrColumn(NonNull<TVMFFITypeAttrColumn>);

// Type-attribute columns and their cells are registry-owned process-lifetime
// data. Once registration is complete, reading a cell does not mutate the
// registry and is safe from any thread.
unsafe impl Send for TypeAttrColumn {}
unsafe impl Sync for TypeAttrColumn {}

impl TypeAttrColumn {
    /// Look up a registered type-attribute column by name.
    pub fn new(name: &str) -> Option<Self> {
        unsafe {
            let name = TVMFFIByteArray::from_str(name);
            NonNull::new(TVMFFIGetTypeAttrColumn(&name).cast_mut()).map(Self)
        }
    }

    /// Return an owning copy of this attribute for `type_index`.
    pub fn get(self, type_index: i32) -> Option<Any> {
        let raw = self.get_raw(type_index)?;
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return None;
        }
        Some(Any::from(unsafe { AnyView::from_raw_ffi_any(raw) }))
    }

    pub(crate) unsafe fn from_non_null(pointer: NonNull<TVMFFITypeAttrColumn>) -> Self {
        Self(pointer)
    }

    pub(crate) fn as_ptr(self) -> *mut TVMFFITypeAttrColumn {
        self.0.as_ptr()
    }

    /// Copy one borrowed cell without taking ownership.
    pub(crate) fn get_raw(self, type_index: i32) -> Option<TVMFFIAny> {
        unsafe {
            let column = self.0.as_ref();
            let index = type_index - column.begin_index;
            if index < 0 || index >= column.size || column.data.is_null() {
                None
            } else {
                Some(*column.data.offset(index as isize))
            }
        }
    }
}

/// Look up one type attribute and copy it into an owning value.
pub fn get_type_attr(type_index: i32, attr_name: &str) -> Option<Any> {
    TypeAttrColumn::new(attr_name)?.get(type_index)
}

/// Resolves a reflected field once, then uses its registered C ABI getter.
#[derive(Clone, Copy)]
pub struct FieldGetter {
    owner_type_index: i32,
    owner_type_depth: i32,
    field_offset: i64,
    getter: TVMFFIFieldGetter,
}

impl FieldGetter {
    /// Resolve a reflected field declared by `type_index` or one of its bases.
    pub fn new(type_index: i32, field_name: &str) -> Result<Self> {
        let type_info = unsafe { TVMFFIGetTypeInfo(type_index) };
        if type_info.is_null() {
            return Err(Error::new(
                TYPE_ERROR,
                &format!("Cannot find type info for type_index={type_index}"),
                "",
            ));
        }

        let field = unsafe { find_field(type_info, field_name) }.ok_or_else(|| {
            let type_key = unsafe { (*type_info).type_key.as_str() };
            Error::new(
                TYPE_ERROR,
                &format!("Cannot find reflected field `{field_name}` in type `{type_key}`"),
                "",
            )
        })?;
        let field = unsafe { field.as_ref() };
        let getter = field.getter.ok_or_else(|| {
            Error::new(
                TYPE_ERROR,
                &format!("Reflected field `{}` has no getter", field.name.as_str()),
                "",
            )
        })?;
        Ok(Self {
            owner_type_index: type_index,
            owner_type_depth: unsafe { (*type_info).type_depth },
            field_offset: field.offset,
            getter,
        })
    }

    /// Read the field as an owning [`Any`].
    ///
    /// `object` may have the declared owner type or any registered subtype.
    pub fn get_any<N: ObjectCore>(&self, object: &N) -> Result<Any> {
        let object_pointer = std::ptr::from_ref(object);
        let header = object_pointer.cast::<TVMFFIObject>();
        let dynamic_type_index = unsafe { (*header).type_index };
        if !unsafe {
            is_type_or_subtype(
                dynamic_type_index,
                self.owner_type_index,
                self.owner_type_depth,
            )
        } {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "Cannot read a field of type_index={} from object type_index={dynamic_type_index}",
                    self.owner_type_index
                ),
                "",
            ));
        }

        let field_address = unsafe {
            object_pointer
                .cast::<u8>()
                .offset(self.field_offset as isize)
                .cast_mut()
                .cast::<c_void>()
        };
        let mut result = Any::new();
        if unsafe { (self.getter)(field_address, Any::as_data_ptr(&mut result)) } != 0 {
            return Err(Error::from_raised());
        }
        Ok(result)
    }

    /// Read and convert the field to `T`.
    pub fn get<N, T>(&self, object: &N) -> Result<T>
    where
        N: ObjectCore,
        T: TryFrom<Any, Error = Error>,
    {
        T::try_from(self.get_any(object)?)
    }
}

unsafe fn find_field(
    type_info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    field_name: &str,
) -> Option<NonNull<TVMFFIFieldInfo>> {
    // Prefer the most-derived declaration, then search nearest ancestors.
    if let Some(field) = find_field_at_level(type_info, field_name) {
        return Some(field);
    }
    for depth in (0..(*type_info).type_depth).rev() {
        let ancestor = *(*type_info).type_acenstors.add(depth as usize);
        if let Some(field) = find_field_at_level(ancestor, field_name) {
            return Some(field);
        }
    }
    None
}

unsafe fn find_field_at_level(
    type_info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    field_name: &str,
) -> Option<NonNull<TVMFFIFieldInfo>> {
    if type_info.is_null() || (*type_info).fields.is_null() {
        return None;
    }
    for index in 0..(*type_info).num_fields as usize {
        let field = (*type_info).fields.add(index);
        if (*field).name.as_str() == field_name {
            return NonNull::new(field.cast_mut());
        }
    }
    None
}

unsafe fn is_type_or_subtype(
    dynamic_type_index: i32,
    target_type_index: i32,
    target_type_depth: i32,
) -> bool {
    if dynamic_type_index == target_type_index {
        return true;
    }
    let dynamic_info = TVMFFIGetTypeInfo(dynamic_type_index);
    if dynamic_info.is_null()
        || (*dynamic_info).type_depth <= target_type_depth
        || (*dynamic_info).type_acenstors.is_null()
    {
        return false;
    }
    let ancestor = *(*dynamic_info)
        .type_acenstors
        .add(target_type_depth as usize);
    !ancestor.is_null() && (*ancestor).type_index == target_type_index
}
