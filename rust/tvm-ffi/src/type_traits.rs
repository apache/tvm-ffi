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
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIGetTypeInfo};

//-----------------------------------------------------
// AnyCompatible
//-----------------------------------------------------
/// Trait to enable a value to be compatible with Any
/// Enables TryFrom/Into AnyView/Any
pub unsafe trait AnyCompatible: Sized {
    /// the value to copy to TVMFFIAny
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny);
    /// consume the value to move to Any
    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny);
    // check if the value is compatible with the type
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool;

    /// Copy value from TVMFFIAny after checking
    /// caller must ensure that the value is compatible with the type
    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self;
    /// the value to move from TVMFFIAny
    /// NOTE: pay very careful attention to avoid memory leak!
    /// - When calling from managed Any, remember to use std::mem::ManuallyDrop
    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self;
    /// try to cast the value from AnyView
    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()>;
    /// Get the type key of a type when TryCastFromAnyView fails.
    fn get_mismatch_type_info(data: &TVMFFIAny) -> String {
        unsafe {
            let info = TVMFFIGetTypeInfo(data.type_index);
            assert!(!info.is_null(), "TVMFFIGetTypeInfo returned null");
            (*info).type_key.as_str().to_string()
        }
    }
    /// the type string of the type
    fn type_str() -> String;
}

/// AnyCompatible for bool
unsafe impl AnyCompatible for bool {
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIBool as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = *src as i64;
    }
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFIBool as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        data.data_union.v_int64 != 0
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIBool as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = src as i64;
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        data.data_union.v_int64 != 0
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIBool as i32
            || data.type_index == TypeIndex::kTVMFFIInt as i32
        {
            unsafe { Ok(data.data_union.v_int64 != 0) }
        } else {
            Err(())
        }
    }

    fn type_str() -> String {
        "bool".to_string()
    }
}

/// Macro to implement AnyCompatible for integer types
macro_rules! impl_any_compatible_for_int {
    ($($int_type:ty),* $(,)?) => {
        $(
            unsafe impl AnyCompatible for $int_type {
                fn type_str() -> String {
                    "int".to_string()
                }

                unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIInt as i32;
                    data.small_str_len = 0;
                    data.data_union.v_int64 = *src as i64;
                }

                unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
                    data.type_index == TypeIndex::kTVMFFIInt as i32
                }

                unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
                    data.data_union.v_int64 as $int_type
                }

                unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIInt as i32;
                    data.small_str_len = 0;
                    data.data_union.v_int64 = src as i64;
                }

                unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
                    data.data_union.v_int64 as $int_type
                }

                unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
                    if data.type_index == TypeIndex::kTVMFFIInt as i32 ||
                       data.type_index == TypeIndex::kTVMFFIBool as i32 {
                        Ok(data.data_union.v_int64 as $int_type)
                    } else {
                        Err(())
                    }
                }
            }
        )*
    };
}

// Implement AnyCompatible for all integer types
impl_any_compatible_for_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

// Implement AnyCompatible for `Option<T>`
unsafe impl<T: AnyCompatible> AnyCompatible for Option<T> {
    fn type_str() -> String {
        // make it consistent with c++ representation
        "Optional<".to_string() + T::type_str().as_str() + ">"
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        if let Some(ref value) = src {
            T::copy_to_any_view(value, data);
        } else {
            data.type_index = TypeIndex::kTVMFFINone as i32;
            data.small_str_len = 0;
            data.data_union.v_int64 = 0;
        }
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        if let Some(value) = src {
            T::move_to_any(value, data);
        } else {
            data.type_index = TypeIndex::kTVMFFINone as i32;
            data.small_str_len = 0;
            data.data_union.v_int64 = 0;
        }
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        return T::check_any_strict(data) || data.type_index == TypeIndex::kTVMFFINone as i32;
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            None
        } else {
            Some(T::copy_from_any_view_after_check(data))
        }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            None
        } else {
            Some(T::move_from_any_after_check(data))
        }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            Ok(None)
        } else {
            T::try_cast_from_any_view(data).map(Some)
        }
    }
}

/// AnyCompatible for void*
unsafe impl AnyCompatible for *mut core::ffi::c_void {
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIOpaquePtr as i32;
        data.small_str_len = 0;
        data.data_union.v_ptr = *src;
    }
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFIOpaquePtr as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        data.data_union.v_ptr
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIOpaquePtr as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = src as i64;
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        data.data_union.v_ptr
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            Ok(std::ptr::null_mut())
        } else if data.type_index == TypeIndex::kTVMFFIOpaquePtr as i32 {
            unsafe { Ok(data.data_union.v_ptr) }
        } else {
            Err(())
        }
    }

    fn type_str() -> String {
        "void*".to_string()
    }
}

/// Macro to implement AnyCompatible for integer types
macro_rules! impl_any_compatible_for_float {
    ($($float_type:ty),* $(,)?) => {
        $(
            unsafe impl AnyCompatible for $float_type {
                fn type_str() -> String {
                    "float".to_string()
                }

                unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIFloat as i32;
                    data.small_str_len = 0;
                    data.data_union.v_float64 = *src as f64;
                }

                unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
                    data.type_index == TypeIndex::kTVMFFIFloat as i32
                }

                unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
                    data.data_union.v_float64 as $float_type
                }

                unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIFloat as i32;
                    data.small_str_len = 0;
                    data.data_union.v_float64 = src as f64;
                }

                unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
                    data.data_union.v_float64 as $float_type
                }

                unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
                    if data.type_index == TypeIndex::kTVMFFIFloat as i32 {
                        Ok(data.data_union.v_float64 as $float_type)
                    } else if data.type_index == TypeIndex::kTVMFFIInt as i32 ||
                       data.type_index == TypeIndex::kTVMFFIBool as i32 {
                        Ok(data.data_union.v_int64 as $float_type)
                    } else {
                        Err(())
                    }
                }
            }
        )*
    };
}

// Implement AnyCompatible for all float types
impl_any_compatible_for_float!(f32, f64);

// Special rule: we convert () to None
// This allows us to effectively pass () around as null value
// note that this is indeed a bit relaxation of the type
// but it is necessary for us to enable void/none interoperability
unsafe impl AnyCompatible for () {
    fn type_str() -> String {
        // make it consistent with c++ representation
        "None".to_string()
    }

    unsafe fn copy_to_any_view(_src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = 0;
    }

    unsafe fn move_to_any(_src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = 0;
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        return data.type_index == TypeIndex::kTVMFFINone as i32;
    }

    unsafe fn copy_from_any_view_after_check(_data: &TVMFFIAny) -> Self {
        ()
    }

    unsafe fn move_from_any_after_check(_data: &mut TVMFFIAny) -> Self {
        ()
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            Ok(())
        } else {
            Err(())
        }
    }
}
