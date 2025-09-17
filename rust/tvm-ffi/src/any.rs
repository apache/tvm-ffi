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
use crate::error::Error;
use crate::object;
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIAnyViewToOwnedAny};

/// Unmanaged Any that can hold reference to values
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AnyView<'a> {
    data: TVMFFIAny,
    /// needs to explicit mark lifetime to avoid lifetime mismatch
    _phantom: std::marker::PhantomData<&'a ()>,
}

/// Managed Any that can hold reference to values
#[repr(C)]
pub struct Any {
    data: TVMFFIAny,
}

//---------------------
// AnyView
//---------------------
impl<'a> AnyView<'a> {
    pub fn new() -> Self {
        Self {
            data: TVMFFIAny::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn type_index(&self) -> i32 {
        self.data.type_index
    }

    /// More strict version than try_from/try_into
    ///
    /// This function will not try to cast the type
    /// and ensures invariance that the return value is only Some(T)
    /// Any::from(T) contains exactly the same value value
    ///
    /// will return Some(T) if the type is exactly compatible with T
    /// will return None if the type is not compatible with T
    #[inline]
    pub fn try_as<T>(&self) -> Option<T>
    where
        T: AnyCompatible,
    {
        unsafe {
            if T::check_any_strict(&self.data) {
                Some(T::copy_from_any_view_after_check(&self.data))
            } else {
                None
            }
        }
    }

    /// Get the strong count of the underlying object for testing/debugging purposes
    ///
    /// If the underlying object is not ref counted, return None
    pub fn debug_strong_count(&self) -> Option<usize> {
        unsafe {
            if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                Some(object::unsafe_::strong_count(self.data.data_union.v_obj))
            } else {
                None
            }
        }
    }
}

impl<'a, T: AnyCompatible> From<&'a T> for AnyView<'a> {
    #[inline]
    fn from(value: &'a T) -> Self {
        unsafe {
            let mut data = TVMFFIAny::new();
            T::copy_to_any_view(&value, &mut data);
            Self {
                data: data,
                _phantom: std::marker::PhantomData,
            }
        }
    }
}

impl Default for AnyView<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Holder for Any value
///
/// This is used to define try_from rule while conforming to orphan rule
/// Users should not use this directly
pub struct TryFromTemp<T> {
    value: T,
}

impl<T> TryFromTemp<T> {
    /// Create a new holder for the value
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Move the value out of the holder
    #[inline(always)]
    pub fn into_value(this: Self) -> T {
        this.value
    }
}

//---------------------
// Any
//---------------------
impl Any {
    pub fn new() -> Self {
        Self {
            data: TVMFFIAny::new(),
        }
    }
    #[inline]
    pub fn type_index(&self) -> i32 {
        self.data.type_index
    }
    /// Try to query if stored typed in Any exactly matches the type T
    ///
    /// This function is fast in the case of failure and can be used to check
    /// if the type is compatible with T
    ///
    /// This function will not try to cast the type
    /// and ensures invariance that the return value is only Some(T)
    /// `Any::from(T)` contains exactly the same value value
    /// `Any::try_as<T>()` contains exactly the same value value
    ///
    /// will return Some(T) if the type is exactly compatible with T
    /// will return None if the type is not compatible with T
    #[inline]
    pub fn try_as<T>(&self) -> Option<T>
    where
        T: AnyCompatible,
    {
        unsafe {
            if T::check_any_strict(&self.data) {
                Some(T::copy_from_any_view_after_check(&self.data))
            } else {
                None
            }
        }
    }

    #[inline]
    pub unsafe fn as_data_ptr(&mut self) -> *mut TVMFFIAny {
        &mut self.data
    }

    #[inline]
    pub unsafe fn into_raw_ffi_any(this: Self) -> TVMFFIAny {
        this.data
    }

    #[inline]
    pub unsafe fn from_raw_ffi_any(data: TVMFFIAny) -> Self {
        Self { data }
    }

    /// Get the strong count of the underlying object for testing/debugging purposes
    ///
    /// If the underlying object is not ref counted, return None
    pub fn debug_strong_count(&self) -> Option<usize> {
        unsafe {
            if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                Some(object::unsafe_::strong_count(self.data.data_union.v_obj))
            } else {
                None
            }
        }
    }
}

impl Default for Any {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Any {
    #[inline]
    fn clone(&self) -> Self {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { object::unsafe_::inc_ref(self.data.data_union.v_obj) }
        }
        Self { data: self.data }
    }
}

impl Drop for Any {
    #[inline]
    fn drop(&mut self) {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { object::unsafe_::dec_ref(self.data.data_union.v_obj) }
        }
    }
}

// convert Any ref to AnyView
impl<'a> From<&'a Any> for AnyView<'a> {
    #[inline]
    fn from(value: &'a Any) -> Self {
        Self {
            data: value.data,
            _phantom: std::marker::PhantomData,
        }
    }
}

// convert AnyView to Any
impl From<AnyView<'_>> for Any {
    #[inline]
    fn from(value: AnyView<'_>) -> Self {
        unsafe {
            let mut data = TVMFFIAny::new();
            crate::check_safe_call!(TVMFFIAnyViewToOwnedAny(&value.data, &mut data)).unwrap();
            Self { data }
        }
    }
}

impl<T: AnyCompatible> From<T> for Any {
    #[inline]
    fn from(value: T) -> Self {
        unsafe {
            let mut data = TVMFFIAny::new();
            T::move_to_any(value, &mut data);
            Self { data }
        }
    }
}

impl<'a, T: AnyCompatible> TryFrom<AnyView<'a>> for TryFromTemp<T> {
    type Error = crate::error::Error;
    #[inline]
    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        unsafe {
            if T::check_any_strict(&value.data) {
                Ok(TryFromTemp::new(T::copy_from_any_view_after_check(
                    &value.data,
                )))
            } else {
                T::try_cast_from_any_view(&value.data)
                    .map_err(|_| {
                        let msg = format!(
                            "Cannot convert from type `{}` to `{}`",
                            T::get_mismatch_type_info(&value.data),
                            T::type_str()
                        );
                        crate::error::Error::new(crate::error::TYPE_ERROR, &msg, "")
                    })
                    .map(TryFromTemp::new)
            }
        }
    }
}

impl<T: AnyCompatible> TryFrom<Any> for TryFromTemp<T> {
    type Error = crate::error::Error;
    #[inline]
    fn try_from(value: Any) -> Result<Self, Self::Error> {
        unsafe {
            if T::check_any_strict(&value.data) {
                let mut value = std::mem::ManuallyDrop::new(value);
                Ok(TryFromTemp::new(T::move_from_any_after_check(
                    &mut value.data,
                )))
            } else {
                T::try_cast_from_any_view(&value.data)
                    .map_err(|_| {
                        let msg = format!(
                            "Cannot convert from type `{}` to `{}`",
                            T::get_mismatch_type_info(&value.data),
                            T::type_str()
                        );
                        crate::error::Error::new(crate::error::TYPE_ERROR, &msg, "")
                    })
                    .map(TryFromTemp::new)
            }
        }
    }
}

crate::impl_try_from_any!(
    bool,
    i8,
    i16,
    i32,
    i64,
    isize,
    u8,
    u16,
    u32,
    u64,
    usize,
    f32,
    f64,
    (),
    *mut core::ffi::c_void,
    crate::string::String,
    crate::string::Bytes,
    crate::object::ObjectRef,
    tvm_ffi_sys::dlpack::DLDataType,
    tvm_ffi_sys::dlpack::DLDevice,
);

crate::impl_try_from_any_for_parametric!(Option<T>);

//------------------------------------------------------------
/// ArgTryFromAnyView: Helper for function argument passing
///-----------------------------------------------------------
pub(crate) trait ArgTryFromAnyView: Sized {
    fn try_from_any_view(value: &AnyView, arg_index: usize) -> Result<Self, Error>;
}

impl<T: AnyCompatible> ArgTryFromAnyView for T {
    fn try_from_any_view(value: &AnyView, arg_index: usize) -> Result<Self, Error> {
        unsafe {
            if T::check_any_strict(&value.data) {
                Ok(T::copy_from_any_view_after_check(&value.data))
            } else {
                T::try_cast_from_any_view(&value.data).map_err(|_| {
                    let msg = format!(
                        "Argument #{}: Cannot convert from type `{}` to `{}`",
                        arg_index,
                        T::get_mismatch_type_info(&value.data),
                        T::type_str()
                    );
                    crate::error::Error::new(crate::error::TYPE_ERROR, &msg, "")
                })
            }
        }
    }
}
