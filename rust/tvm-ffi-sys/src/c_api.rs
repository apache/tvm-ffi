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
// NOTE: we manually write the C ABI as they are reasonably minimal
// and we need to ensure clear control of the atomic access etc.
#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::sync::atomic::AtomicU64;

use crate::dlpack::DLDataType;
use crate::dlpack::DLDevice;

///  The index type of the FFI objects
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TVMFFITypeIndex {
    /// None/nullptr value
    kTVMFFINone = 0,
    /// POD int value
    kTVMFFIInt = 1,
    /// POD bool value
    kTVMFFIBool = 2,
    /// POD float value
    kTVMFFIFloat = 3,
    /// Opaque pointer object
    kTVMFFIOpaquePtr = 4,
    /// DLDataType
    kTVMFFIDataType = 5,
    /// DLDevice
    kTVMFFIDevice = 6,
    /// DLTensor*
    kTVMFFIDLTensorPtr = 7,
    /// const char*
    kTVMFFIRawStr = 8,
    /// TVMFFIByteArray*
    kTVMFFIByteArrayPtr = 9,
    /// R-value reference to ObjectRef
    kTVMFFIObjectRValueRef = 10,
    /// Small string on stack
    kTVMFFISmallStr = 11,
    /// Small bytes on stack
    kTVMFFISmallBytes = 12,
    /// Start of statically defined objects.
    kTVMFFIStaticObjectBegin = 64,
    /// String object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
    kTVMFFIStr = 65,
    /// Bytes object, layout = { TVMFFIObject, TVMFFIByteArray, ... }
    kTVMFFIBytes = 66,
    /// Error object.
    kTVMFFIError = 67,
    /// Function object.
    kTVMFFIFunction = 68,
    /// Shape object, layout = { TVMFFIObject, { const int64_t*, size_t }, ... }
    kTVMFFIShape = 69,
    /// Tensor object, layout = { TVMFFIObject, DLTensor, ... }
    kTVMFFITensor = 70,
    /// Array object.
    kTVMFFIArray = 71,
    //----------------------------------------------------------------
    // more complex objects
    //----------------------------------------------------------------
    /// Map object.
    kTVMFFIMap = 72,
    /// Runtime dynamic loaded module object.
    kTVMFFIModule = 73,
    /// Opaque python object.
    kTVMFFIOpaquePyObject = 74,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TVMFFIObjectDeleterFlagBitMask {
    kTVMFFIObjectDeleterFlagBitMaskStrong = 1 << 0,
    kTVMFFIObjectDeleterFlagBitMaskWeak = 1 << 1,
    kTVMFFIObjectDeleterFlagBitMaskBoth = (1 << 0) | (1 << 1),
}

/// Handle to Object from C API's pov
pub type TVMFFIObjectHandle = *mut c_void;
pub type TVMFFIObjectDeleter = unsafe extern "C" fn(self_ptr: *mut c_void, flags: i32);

// constants for working with combined reference count
pub const COMBINED_REF_COUNT_MASK_U32: u64 = (1u64 << 32) - 1;
pub const COMBINED_REF_COUNT_STRONG_ONE: u64 = 1;
pub const COMBINED_REF_COUNT_WEAK_ONE: u64 = 1u64 << 32;
pub const COMBINED_REF_COUNT_BOTH_ONE: u64 =
    COMBINED_REF_COUNT_STRONG_ONE | COMBINED_REF_COUNT_WEAK_ONE;

#[repr(C)]
pub struct TVMFFIObject {
    pub combined_ref_count: AtomicU64,
    pub type_index: i32,
    pub __padding: u32,
    pub deleter: Option<TVMFFIObjectDeleter>,
    // private padding to ensure 8 bytes alignment
    #[cfg(target_pointer_width = "32")]
    __padding: u32,
}

impl TVMFFIObject {
    pub fn new() -> Self {
        Self {
            combined_ref_count: AtomicU64::new(0),
            type_index: 0,
            __padding: 0,
            deleter: None,
        }
    }
}

/// Second union in TVMFFIAny - 8 bytes
#[repr(C)]
#[derive(Copy, Clone)]
pub union TVMFFIAnyDataUnion {
    /// Integers
    pub v_int64: i64,
    /// Floating-point numbers
    pub v_float64: f64,
    /// Typeless pointers
    pub v_ptr: *mut c_void,
    /// Raw C-string
    pub v_c_str: *const i8,
    /// Ref counted objects
    pub v_obj: *mut TVMFFIObject,
    /// Data type
    pub v_dtype: DLDataType,
    /// Device
    pub v_device: DLDevice,
    /// Small string
    pub v_bytes: [u8; 8],
    /// uint64 repr mainly used for hashing
    pub v_uint64: u64,
}

/// TVM FFI Any value - a union type that can hold various data types
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TVMFFIAny {
    /// Type index of the object.
    /// The type index of Object and Any are shared in FFI.
    pub type_index: i32,
    /// small string length or zero padding
    pub small_str_len: u32,
    /// data union - 8 bytes
    pub data_union: TVMFFIAnyDataUnion,
}

impl TVMFFIAny {
    /// create a new instance of TVMFFIAny that represents None
    pub fn new() -> Self {
        Self {
            type_index: TVMFFITypeIndex::kTVMFFINone as i32,
            small_str_len: 0,
            data_union: TVMFFIAnyDataUnion { v_int64: 0 },
        }
    }
}

/// Byte array data structure used by String and Bytes.
#[repr(C)]
pub struct TVMFFIByteArray {
    pub data: *const u8,
    pub size: usize,
}

impl TVMFFIByteArray {
    pub fn new(data: *const u8, size: usize) -> Self {
        Self { data, size }
    }
    /// Convert the TVMFFIByteArray to a str view
    ///
    /// # Arguments
    /// * `self` - The TVMFFIByteArray to convert.
    ///
    /// # Returns
    /// * `&str` - The converted str view.
    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.data, self.size)) }
    }
    /// Unsafe function to create a TVMFFIByteArray from a string
    /// This function is unsafe as it does not check lifetime of the string
    /// the caller must ensure that the string is valid for the lifetime of the TVMFFIByteArray
    ///
    /// # Arguments
    /// * `data` - The string to create the TVMFFIByteArray from.
    ///
    /// # Returns
    /// * `TVMFFIByteArray` - The created TVMFFIByteArray.
    pub unsafe fn from_str(data: &str) -> Self {
        Self {
            data: data.as_ptr(),
            size: data.len(),
        }
    }
}

/// Safe call type for function ABI
pub type TVMFFISafeCallType = unsafe extern "C" fn(
    handle: *mut c_void,
    args: *const TVMFFIAny,
    num_args: i32,
    result: *mut TVMFFIAny,
) -> i32;

/// Function cell
#[repr(C)]
pub struct TVMFFIFunctionCell {
    /// A C API compatible call with exception catching.
    pub safe_call: TVMFFISafeCallType,
    pub cxx_call: *mut c_void,
}

unsafe impl Send for TVMFFIFunctionCell {}
unsafe impl Sync for TVMFFIFunctionCell {}

#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TVMFFIBacktraceUpdateMode {
    kTVMFFIBacktraceUpdateModeReplace = 0,
    kTVMFFIBacktraceUpdateModeAppend = 1,
}

/// Error cell used in error object following header.
#[repr(C)]
pub struct TVMFFIErrorCell {
    pub kind: TVMFFIByteArray,
    pub message: TVMFFIByteArray,
    pub backtrace: TVMFFIByteArray,
    pub update_backtrace: unsafe extern "C" fn(
        self_ptr: *mut c_void,
        backtrace: *const TVMFFIByteArray,
        update_mode: i32,
    ),
}

/// Shape cell used in shape object following header.
#[repr(C)]
pub struct TVMFFIShapeCell {
    pub data: *const i64,
    pub size: usize,
}

/// Field getter function pointer type
pub type TVMFFIFieldGetter =
    unsafe extern "C" fn(field: *mut c_void, result: *mut TVMFFIAny) -> i32;

/// Field setter function pointer type
pub type TVMFFIFieldSetter =
    unsafe extern "C" fn(field: *mut c_void, value: *const TVMFFIAny) -> i32;

/// Information support for optional object reflection
#[repr(C)]
pub struct TVMFFIFieldInfo {
    /// The name of the field
    pub name: TVMFFIByteArray,
    /// The docstring about the field
    pub doc: TVMFFIByteArray,
    /// The metadata of the field in JSON string
    pub metadata: TVMFFIByteArray,
    /// bitmask flags of the field
    pub flags: i64,
    /// The size of the field
    pub size: i64,
    /// The alignment of the field
    pub alignment: i64,
    /// The offset of the field
    pub offset: i64,
    /// The getter to access the field
    pub getter: Option<TVMFFIFieldGetter>,
    /// The setter to access the field
    /// The setter is set even if the field is readonly for serialization
    pub setter: Option<TVMFFIFieldSetter>,
    /// The default value of the field, this field hold AnyView,
    /// valid when flags set kTVMFFIFieldFlagBitMaskHasDefault
    pub default_value: TVMFFIAny,
    /// Records the static type kind of the field
    pub field_static_type_index: i32,
}

/// Object creator function pointer type
pub type TVMFFIObjectCreator = unsafe extern "C" fn(result: *mut TVMFFIObjectHandle) -> i32;

/// Method information that can appear in reflection table
#[repr(C)]
pub struct TVMFFIMethodInfo {
    /// The name of the field
    pub name: TVMFFIByteArray,
    /// The docstring about the method
    pub doc: TVMFFIByteArray,
    /// Optional metadata of the method in JSON string
    pub metadata: TVMFFIByteArray,
    /// bitmask flags of the method
    pub flags: i64,
    /// The method wrapped as ffi::Function, stored as AnyView
    /// The first argument to the method is always the self for instance methods
    pub method: TVMFFIAny,
}

/// Extra information of object type that can be used for reflection
///
/// This information is optional and can be used to enable reflection based
/// creation of the object.
#[repr(C)]
pub struct TVMFFITypeMetadata {
    /// The docstring about the object
    pub doc: TVMFFIByteArray,
    /// An optional function that can create a new empty instance of the type
    pub creator: Option<TVMFFIObjectCreator>,
    /// Total size of the object struct, if it is fixed and known
    ///
    /// This field is set optional and set to 0 if not registered.
    pub total_size: i32,
    /// Optional meta-data for structural eq/hash
    pub structural_eq_hash_kind: i32,
}

/// Column array that stores extra attributes about types
///
/// The attributes stored in a column array that can be looked up by type index.
/// Note that the TypeAttr behaves like type_traits so column T so not contain
/// attributes from base classes.
#[repr(C)]
pub struct TVMFFITypeAttrColumn {
    /// The data of the column
    pub data: *const TVMFFIAny,
    /// The size of the column
    pub size: usize,
}

/// Runtime type information for object type checking
#[repr(C)]
pub struct TVMFFITypeInfo {
    /// The runtime type index
    /// It can be allocated during runtime if the type is dynamic
    pub type_index: i32,
    /// number of parent types in the type hierachy
    pub type_depth: i32,
    /// the unique type key to identify the type
    pub type_key: TVMFFIByteArray,
    /// `type_acenstors[depth]` stores the type_index of the acenstors at depth level
    /// To keep things simple, we do not allow multiple inheritance so the
    /// hieracy stays as a tree
    pub type_acenstors: *const *const TVMFFITypeInfo,
    /// Cached hash value of the type key, used for consistent structural hashing
    pub type_key_hash: u64,
    /// number of reflection accessible fields
    pub num_fields: i32,
    /// number of reflection acccesible methods
    pub num_methods: i32,
    /// The reflection field information
    pub fields: *const TVMFFIFieldInfo,
    /// The reflection method
    pub methods: *const TVMFFIMethodInfo,
    /// The extra information of the type
    pub metadata: *const TVMFFITypeMetadata,
}

unsafe extern "C" {
    pub fn TVMFFITypeKeyToIndex(type_key: *const TVMFFIByteArray, out_tindex: *mut i32) -> i32;
    pub fn TVMFFIFunctionGetGlobal(
        name: *const TVMFFIByteArray,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFIFunctionSetGlobal(
        name: *const TVMFFIByteArray,
        f: TVMFFIObjectHandle,
        can_override: i32,
    ) -> i32;
    pub fn TVMFFIFunctionCreate(
        self_ptr: *mut c_void,
        safe_call: TVMFFISafeCallType,
        deleter: Option<unsafe extern "C" fn(*mut c_void)>,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFIAnyViewToOwnedAny(any_view: *const TVMFFIAny, out: *mut TVMFFIAny) -> i32;
    pub fn TVMFFIFunctionCall(
        func: TVMFFIObjectHandle,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32;
    pub fn TVMFFIErrorMoveFromRaised(result: *mut TVMFFIObjectHandle);
    pub fn TVMFFIErrorSetRaised(error: TVMFFIObjectHandle);
    pub fn TVMFFIErrorSetRaisedFromCStr(kind: *const i8, message: *const i8);
    pub fn TVMFFIErrorCreate(
        kind: *const TVMFFIByteArray,
        message: *const TVMFFIByteArray,
        backtrace: *const TVMFFIByteArray,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFITensorFromDLPack(
        from: *mut c_void,
        require_alignment: i32,
        require_contiguous: i32,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFITensorToDLPack(from: TVMFFIObjectHandle, out: *mut *mut c_void) -> i32;
    pub fn TVMFFITensorFromDLPackVersioned(
        from: *mut c_void,
        require_alignment: i32,
        require_contiguous: i32,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;
    pub fn TVMFFITensorToDLPackVersioned(from: TVMFFIObjectHandle, out: *mut *mut c_void) -> i32;
    pub fn TVMFFIStringFromByteArray(input: *const TVMFFIByteArray, out: *mut TVMFFIAny) -> i32;
    pub fn TVMFFIBytesFromByteArray(input: *const TVMFFIByteArray, out: *mut TVMFFIAny) -> i32;
    pub fn TVMFFIDataTypeFromString(str: *const TVMFFIByteArray, out: *mut DLDataType) -> i32;
    pub fn TVMFFIDataTypeToString(dtype: *const DLDataType, out: *mut TVMFFIAny) -> i32;
    pub fn TVMFFITraceback(
        filename: *const i8,
        lineno: i32,
        func: *const i8,
        cross_ffi_boundary: i32,
    ) -> *const TVMFFIByteArray;
    pub fn TVMFFIGetTypeInfo(type_index: i32) -> *const TVMFFITypeInfo;
    pub fn TVMFFITestingDummyTarget() -> i32;
}
