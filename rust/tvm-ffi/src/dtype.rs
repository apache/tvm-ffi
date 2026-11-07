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
//! Data types: [`DLDataTypeExt`] and the Rust element types of tensors.

use crate::error::Result;
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIByteArray, TVMFFIDataTypeFromString, TVMFFIDataTypeToString};

/// Name conversions of a `DLDataType`.
pub trait DLDataTypeExt: Sized {
    /// Return the name of the data type, such as `"int32"`, `"float64"` or
    /// `"int8x4"` (bits and lanes), the name every language uses for it.
    ///
    /// ```
    /// use tvm_ffi::{DLDataType, DLDataTypeExt};
    /// let dtype = DLDataType::try_from_str("float16")?;
    /// assert_eq!(dtype.to_string(), "float16");
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    fn to_string(&self) -> std::string::String;

    /// Parse a data type from its name, the inverse of
    /// [`to_string`](Self::to_string).
    ///
    /// # Arguments
    /// * `dtype_str` - The name, such as `"int32"`.
    ///
    /// # Returns
    /// The data type.
    ///
    /// # Errors
    /// If the name is not a data type.
    fn try_from_str(dtype_str: &str) -> Result<Self>;
}

impl DLDataTypeExt for DLDataType {
    fn to_string(&self) -> std::string::String {
        unsafe {
            let mut ffi_any = TVMFFIAny::new();
            crate::check_safe_call!(TVMFFIDataTypeToString(self, &mut ffi_any)).unwrap();
            let name: crate::string::String = crate::any::Any::from_raw_ffi_any(ffi_any)
                .try_into()
                .unwrap();
            name.as_str().to_owned()
        }
    }

    fn try_from_str(dtype_str: &str) -> Result<Self> {
        let mut dtype = DLDataType {
            code: DLDataTypeCode::kDLOpaqueHandle as u8,
            bits: 0,
            lanes: 0,
        };
        unsafe {
            let dtype_byte_array = TVMFFIByteArray::from_str(dtype_str);
            crate::check_safe_call!(TVMFFIDataTypeFromString(&dtype_byte_array, &mut dtype))?;
        }
        Ok(dtype)
    }
}

unsafe impl AnyCompatible for DLDataType {
    fn type_str() -> String {
        // The C++ name.
        "DataType".to_string()
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIDataType as i32;
        data.small_str_len = 0;
        data.data_union.v_uint64 = 0;
        data.data_union.v_dtype = *src;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIDataType as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = 0;
        data.data_union.v_dtype = src;
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFIDataType as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        data.data_union.v_dtype
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        data.data_union.v_dtype
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIDataType as i32 {
            Ok(data.data_union.v_dtype)
        } else if let Ok(string) = crate::string::String::try_cast_from_any_view(data) {
            // A data type also converts from its name, such as "float32".
            DLDataType::try_from_str(string.as_str()).map_err(|_| ())
        } else {
            Err(())
        }
    }
}

/// A Rust element type of a tensor, such as `f32`, and its `DLDataType`.
///
/// # Safety
/// Tensor data of `DL_DATA_TYPE` is read as `Self` without checks, so `Self`
/// must have that data type's item size, be no more aligned than the item
/// size, and accept every bit pattern of it.
pub unsafe trait AsDLDataType: Copy {
    /// The data type of `Self`.
    const DL_DATA_TYPE: DLDataType;
}

macro_rules! impl_as_dl_data_type {
    ($type: ty, $code: expr, $bits: expr) => {
        unsafe impl AsDLDataType for $type {
            const DL_DATA_TYPE: DLDataType = DLDataType {
                code: $code as u8,
                bits: $bits as u8,
                lanes: 1,
            };
        }
    };
}

impl_as_dl_data_type!(i8, DLDataTypeCode::kDLInt, 8);
impl_as_dl_data_type!(i16, DLDataTypeCode::kDLInt, 16);
impl_as_dl_data_type!(i32, DLDataTypeCode::kDLInt, 32);
impl_as_dl_data_type!(i64, DLDataTypeCode::kDLInt, 64);
impl_as_dl_data_type!(u8, DLDataTypeCode::kDLUInt, 8);
impl_as_dl_data_type!(u16, DLDataTypeCode::kDLUInt, 16);
impl_as_dl_data_type!(u32, DLDataTypeCode::kDLUInt, 32);
impl_as_dl_data_type!(u64, DLDataTypeCode::kDLUInt, 64);
impl_as_dl_data_type!(f32, DLDataTypeCode::kDLFloat, 32);
impl_as_dl_data_type!(f64, DLDataTypeCode::kDLFloat, 64);
