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

//! Move-aware object arguments compatible with C++ `ffi::RValueRef<T>`.

use std::cell::UnsafeCell;
use std::marker::PhantomData;

use tvm_ffi_sys::{TVMFFIAny, TVMFFIObject, TVMFFITypeIndex as TypeIndex};

use crate::any::ArgTryFromAnyView;
use crate::{AnyCompatible, AnyView, Error, ObjectRefCore, Result};

/// An object argument whose strong reference may be moved through the FFI ABI.
///
/// Passing this wrapper to [`crate::Function::call_tuple`] uses the same
/// `kTVMFFIObjectRValueRef` representation as C++ `ffi::RValueRef<T>`. If the
/// callee accepts an rvalue reference, it can take the stored reference without
/// incrementing its count; otherwise the slot remains owned by this wrapper and
/// is released normally.
pub struct RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    slot: UnsafeCell<*mut TVMFFIObject>,
    _marker: PhantomData<T>,
}

impl<T> RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    /// Transfer an owned object reference into an rvalue argument slot.
    pub fn new(value: T) -> Self {
        let mut raw = TVMFFIAny::new();
        unsafe { T::move_to_any(value, &mut raw) };
        debug_assert!(raw.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32);
        Self {
            slot: UnsafeCell::new(unsafe { raw.data_union.v_obj }),
            _marker: PhantomData,
        }
    }

    /// Take the object held by this wrapper.
    ///
    /// This is zero-copy: the owned strong reference is moved into `T`.
    pub fn into_inner(mut self) -> T {
        let object = *self.slot.get_mut();
        assert!(!object.is_null(), "RValueRef has already been moved");
        *self.slot.get_mut() = std::ptr::null_mut();
        unsafe {
            let mut raw = object_any(object);
            T::move_from_any_after_check(&mut raw)
        }
    }

    unsafe fn from_view(value: &AnyView<'_>, arg_index: Option<usize>) -> Result<Self> {
        let raw = value.as_raw_ffi_any();
        let converted = if raw.type_index == TypeIndex::kTVMFFIObjectRValueRef as i32 {
            let slot = raw.data_union.v_ptr.cast::<*mut TVMFFIObject>();
            if slot.is_null() || (*slot).is_null() {
                return Err(conversion_error::<T>(raw, arg_index, true));
            }
            let object = *slot;
            let object_view = object_any(object);
            if T::check_any_strict(&object_view) {
                *slot = std::ptr::null_mut();
                return Ok(Self {
                    slot: UnsafeCell::new(object),
                    _marker: PhantomData,
                });
            }
            T::try_cast_from_any_view(&object_view)
        } else if T::check_any_strict(raw) {
            Ok(T::copy_from_any_view_after_check(raw))
        } else {
            T::try_cast_from_any_view(raw)
        };

        converted
            .map(Self::new)
            .map_err(|()| conversion_error::<T>(raw, arg_index, false))
    }
}

impl<T> From<T> for RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<'a, T> From<&'a RValueRef<T>> for AnyView<'a>
where
    T: ObjectRefCore + AnyCompatible,
{
    fn from(value: &'a RValueRef<T>) -> Self {
        let mut raw = TVMFFIAny::new();
        raw.type_index = TypeIndex::kTVMFFIObjectRValueRef as i32;
        raw.data_union.v_ptr = value.slot.get().cast();
        unsafe { AnyView::from_raw_ffi_any(raw) }
    }
}

impl<T> TryFrom<AnyView<'_>> for RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    type Error = Error;

    fn try_from(value: AnyView<'_>) -> Result<Self> {
        unsafe { Self::from_view(&value, None) }
    }
}

impl<T> ArgTryFromAnyView for RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    fn try_from_any_view(value: &AnyView<'_>, arg_index: usize) -> Result<Self> {
        unsafe { Self::from_view(value, Some(arg_index)) }
    }
}

impl<T> Drop for RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    fn drop(&mut self) {
        let object = *self.slot.get_mut();
        if !object.is_null() {
            unsafe { crate::object::unsafe_::dec_ref(object) };
        }
    }
}

unsafe fn object_any(object: *mut TVMFFIObject) -> TVMFFIAny {
    let mut raw = TVMFFIAny::new();
    raw.type_index = (*object).type_index;
    raw.data_union.v_obj = object;
    raw
}

unsafe fn conversion_error<T>(
    raw: &TVMFFIAny,
    arg_index: Option<usize>,
    already_moved: bool,
) -> Error
where
    T: ObjectRefCore + AnyCompatible,
{
    let source = if already_moved {
        "an already-moved RValueRef".to_string()
    } else if raw.type_index == TypeIndex::kTVMFFIObjectRValueRef as i32 {
        "RValueRef with an incompatible object type".to_string()
    } else {
        T::get_mismatch_type_info(raw)
    };
    let prefix = arg_index
        .map(|index| format!("Argument #{index}: "))
        .unwrap_or_default();
    Error::new(
        crate::error::TYPE_ERROR,
        &format!(
            "{prefix}Cannot convert from `{source}` to `RValueRef<{}>`",
            T::type_str()
        ),
        "",
    )
}
