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
use crate::error::Result;
use crate::type_traits::AnyCompatible;
/// Data type handling
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIByteArray, TVMFFIDataTypeFromString, TVMFFIDataTypeToString};

/// Extra methods for DLDataType
pub trait DLDataTypeExt: Sized {
    /// Convert the DLDataType to a string representation
    ///
    /// # Returns
    /// A string representation of the data type (e.g., "int32", "float64", "bool")
    fn to_string(&self) -> crate::string::String;

    /// Parse a string representation into a DLDataType
    ///
    /// # Arguments
    /// * `dtype_str` - The string representation of the data type to parse
    ///
    /// # Returns
    /// * `Ok(DLDataType)` - Successfully parsed data type
    /// * `Err(Error)` - Failed to parse the string
    ///
    /// # Examples
    /// ```
    /// use tvm_ffi::{DLDataType, DLDataTypeExt};
    ///
    /// let dtype = DLDataType::try_from_str("int32").unwrap();
    /// ```
    fn try_from_str(dtype_str: &str) -> Result<Self>;
}

impl DLDataTypeExt for DLDataType {
    fn to_string(&self) -> crate::string::String {
        unsafe {
            let mut ffi_any = TVMFFIAny::new();
            crate::check_safe_call!(TVMFFIDataTypeToString(&*self, &mut ffi_any)).unwrap();
            crate::any::Any::from_raw_ffi_any(ffi_any)
                .try_into()
                .unwrap()
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

/// AnyCompatible implementation for DLDataType
///
/// This implementation allows DLDataType to be used with the TVM FFI Any system,
/// enabling type-safe conversion between DLDataType and the generic Any type.
unsafe impl AnyCompatible for DLDataType {
    /// Get the type string identifier for DLDataType
    ///
    /// # Returns
    /// The string "DataType" to match the C++ representation
    fn type_str() -> String {
        // make it consistent with c++ representation
        "DataType".to_string()
    }

    /// Copy a DLDataType to an Any view
    ///
    /// # Arguments
    /// * `src` - The DLDataType to copy from
    /// * `data` - The Any view to copy to
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIDataType as i32;
        data.small_str_len = 0;
        data.data_union.v_uint64 = 0;
        data.data_union.v_dtype = *src;
    }

    /// Move a DLDataType into an Any
    ///
    /// # Arguments
    /// * `src` - The DLDataType to move from
    /// * `data` - The Any to move into
    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIDataType as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = 0;
        data.data_union.v_dtype = src;
    }

    /// Check if an Any contains a DLDataType
    ///
    /// # Arguments
    /// * `data` - The Any to check
    ///
    /// # Returns
    /// `true` if the Any contains a DLDataType, `false` otherwise
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        return data.type_index == TypeIndex::kTVMFFIDataType as i32;
    }

    /// Copy a DLDataType from an Any view (after type check)
    ///
    /// # Arguments
    /// * `data` - The Any view to copy from
    ///
    /// # Returns
    /// The copied DLDataType
    ///
    /// # Safety
    /// The caller must ensure that `data` contains a DLDataType
    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        data.data_union.v_dtype
    }

    /// Move a DLDataType from an Any (after type check)
    ///
    /// # Arguments
    /// * `data` - The Any to move from
    ///
    /// # Returns
    /// The moved DLDataType
    ///
    /// # Safety
    /// The caller must ensure that `data` contains a DLDataType
    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        data.data_union.v_dtype
    }

    /// Try to cast an Any view to a DLDataType
    ///
    /// This method supports both direct DLDataType conversion and string parsing.
    ///
    /// # Arguments
    /// * `data` - The Any view to cast from
    ///
    /// # Returns
    /// * `Ok(DLDataType)` - Successfully cast to DLDataType
    /// * `Err(())` - Failed to cast (wrong type or invalid string)
    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIDataType as i32 {
            Ok(data.data_union.v_dtype)
        } else if let Ok(string) = crate::string::String::try_cast_from_any_view(data) {
            DLDataType::try_from_str(string.as_str()).map_err(|_| ())
        } else {
            Err(())
        }
    }
}

/// Trait to convert standard data types to DLDataType
///
/// This trait provides a way to get the corresponding DLDataType for standard Rust types.
/// It's implemented for common integer, unsigned integer, and floating-point types.
pub trait AsDLDataType: Copy {
    /// The corresponding DLDataType for this type
    const DL_DATA_TYPE: DLDataType;
}

/// Macro to implement AsDLDataType for standard types
///
/// This macro generates implementations of the AsDLDataType trait for standard Rust types.
/// It takes the type, the DLPack data type code, and the number of bits.
macro_rules! impl_as_dl_data_type {
    ($type: ty, $code: expr, $bits: expr) => {
        impl AsDLDataType for $type {
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
