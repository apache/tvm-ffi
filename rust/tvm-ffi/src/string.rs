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
//! [`String`] and [`Bytes`], immutable reference-counted strings.

use crate::derive::Object;
use crate::object::{unsafe_, Object, ObjectArc, ObjectCoreWithExtraItems};
use crate::type_traits::AnyCompatible;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIByteArray, TVMFFIObject};

//-----------------------------------------------------
// Bytes
//-----------------------------------------------------
/// An immutable byte string, passed across languages as `ffi.Bytes` (Python
/// `bytes`).
///
/// # Examples
///
/// ```
/// use tvm_ffi::Bytes;
///
/// let bytes = Bytes::from(&b"abc"[..]);
/// assert_eq!(bytes.as_slice(), b"abc");
/// ```
#[repr(C)]
pub struct Bytes {
    data: TVMFFIAny,
}

// SAFETY: `Bytes` is immutable: inline bytes or an immutable bytes object.
unsafe impl Send for Bytes {}
unsafe impl Sync for Bytes {}

// BytesObj for heap-allocated bytes
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.Bytes", unsafe type_index = TypeIndex::kTVMFFIBytes)]
pub(crate) struct BytesObj {
    object: Object,
    data: TVMFFIByteArray,
}

impl Bytes {
    /// An empty byte string.
    pub fn new() -> Self {
        Self {
            data: TVMFFIAny {
                type_index: TypeIndex::kTVMFFISmallBytes as i32,
                small_str_len: 0,
                data_union: TVMFFIAnyDataUnion { v_int64: 0 },
            },
        }
    }

    /// The number of bytes.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Whether there are no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes as a slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            if self.data.type_index == TypeIndex::kTVMFFISmallBytes as i32 {
                std::slice::from_raw_parts(
                    self.data.data_union.v_bytes.as_ptr(),
                    self.data.small_str_len as usize,
                )
            } else {
                let str_obj: &BytesObj = &*(self.data.data_union.v_obj as *const BytesObj);
                std::slice::from_raw_parts(str_obj.data.data, str_obj.data.size)
            }
        }
    }
}

impl Default for Bytes {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl ObjectCoreWithExtraItems for BytesObj {
    type ExtraItem = u8;
    /// The bytes plus a trailing NUL, so the data is also a C string.
    #[inline]
    fn extra_items_count(this: &Self) -> usize {
        this.data.size + 1
    }
}

impl From<&[u8]> for Bytes {
    /// Copy the bytes of `value`.
    fn from(value: &[u8]) -> Self {
        // to be compatible with normal c++
        const MAX_SMALL_BYTES_LEN: usize = 7;
        unsafe {
            if value.len() <= MAX_SMALL_BYTES_LEN {
                let mut data_union = TVMFFIAnyDataUnion { v_int64: 0 };
                data_union.v_bytes[..value.len()].copy_from_slice(value);
                // small bytes
                Self {
                    data: TVMFFIAny {
                        type_index: TypeIndex::kTVMFFISmallBytes as i32,
                        small_str_len: value.len() as u32,
                        data_union,
                    },
                }
            } else {
                // large bytes
                let mut obj_arc = ObjectArc::new_with_extra_items(BytesObj {
                    object: Object::new(),
                    data: TVMFFIByteArray {
                        data: std::ptr::null(),
                        size: value.len(),
                    },
                });
                // The fresh allocation is unique and has not been exposed.
                let object = ObjectArc::as_mut_ptr(&mut obj_arc);
                let items = BytesObj::extra_items_ptr(object);
                std::ptr::copy_nonoverlapping(value.as_ptr(), items, value.len());
                // write the trailing \0 for ffi compatibility
                items.add(value.len()).write(0);
                (*object).data.data = items;
                Self {
                    data: TVMFFIAny {
                        type_index: TypeIndex::kTVMFFIBytes as i32,
                        small_str_len: 0,
                        data_union: TVMFFIAnyDataUnion {
                            v_obj: ObjectArc::into_raw(obj_arc) as *mut BytesObj
                                as *mut TVMFFIObject,
                        },
                    },
                }
            }
        }
    }
}

impl Deref for Bytes {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Clone for Bytes {
    #[inline]
    fn clone(&self) -> Self {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { unsafe_::inc_ref(self.data.data_union.v_obj) }
        }
        Self { data: self.data }
    }
}

impl Drop for Bytes {
    #[inline]
    fn drop(&mut self) {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { unsafe_::dec_ref(self.data.data_union.v_obj) }
        }
    }
}

impl PartialEq for Bytes {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Bytes {}

impl PartialOrd for Bytes {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bytes {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Hash for Bytes {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

/// Formats as `[u8]` does, such as `[97, 98]`.
impl Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.as_slice(), f)
    }
}

macro_rules! impl_bytes_from {
    ($([$($g:tt)*] $t:ty),* $(,)?) => {
        $(impl<$($g)*> From<$t> for Bytes {
            #[inline]
            fn from(value: $t) -> Self {
                Self::from(&value[..])
            }
        })*
    };
}

impl_bytes_from!([] &mut [u8], [] Vec<u8>, [] &Vec<u8>, [] Box<[u8]>, [const N: usize] [u8; N], [const N: usize] &[u8; N]);

macro_rules! impl_bytes_eq {
    ($([$($g:tt)*] $t:ty),* $(,)?) => {
        $(impl<$($g)*> PartialEq<$t> for Bytes {
            #[inline]
            fn eq(&self, other: &$t) -> bool {
                self.as_slice() == &other[..]
            }
        }

        impl<$($g)*> PartialEq<Bytes> for $t {
            #[inline]
            fn eq(&self, other: &Bytes) -> bool {
                &self[..] == other.as_slice()
            }
        })*
    };
}

impl_bytes_eq!([] [u8], ['a] &'a [u8], [] Vec<u8>, [const N: usize] [u8; N], ['a, const N: usize] &'a [u8; N]);

impl AsRef<[u8]> for Bytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::borrow::Borrow<[u8]> for Bytes {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl From<Bytes> for Vec<u8> {
    #[inline]
    fn from(value: Bytes) -> Self {
        value.as_slice().to_vec()
    }
}

//-----------------------------------------------------
// String
//-----------------------------------------------------

/// An immutable UTF-8 string, passed across languages as `ffi.String`
/// (Python `str`).
///
/// An `ffi.String` holds UTF-8 by contract, which other languages keep, so
/// [`as_str`](Self::as_str) does not validate strings received through FFI.
///
/// Short strings are stored inline in an [`Any`](crate::Any) without an
/// allocation.
///
/// # Examples
///
/// ```
/// use tvm_ffi::String;
///
/// let text = String::from("hello");
/// assert_eq!(text.as_str(), "hello");
/// assert_eq!(text.len(), 5);
/// assert_eq!(text, "hello");
/// ```
#[repr(C)]
pub struct String {
    data: TVMFFIAny,
}

// SAFETY: a `String` is immutable: an inline small string or an immutable string object.
unsafe impl Send for String {}
unsafe impl Sync for String {}

// StringObj for heap-allocated string
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.String", unsafe type_index = TypeIndex::kTVMFFIStr)]
pub(crate) struct StringObj {
    object: Object,
    data: TVMFFIByteArray,
}

unsafe impl ObjectCoreWithExtraItems for StringObj {
    type ExtraItem = u8;
    /// The bytes plus a trailing NUL, so the data is also a C string.
    #[inline]
    fn extra_items_count(this: &Self) -> usize {
        this.data.size + 1
    }
}

impl String {
    /// An empty string.
    pub fn new() -> Self {
        Self {
            data: TVMFFIAny {
                type_index: TypeIndex::kTVMFFISmallStr as i32,
                small_str_len: 0,
                data_union: TVMFFIAnyDataUnion { v_int64: 0 },
            },
        }
    }

    /// The length in bytes.
    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Whether the string is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The string as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            if self.data.type_index == TypeIndex::kTVMFFISmallStr as i32 {
                std::slice::from_raw_parts(
                    self.data.data_union.v_bytes.as_ptr(),
                    self.data.small_str_len as usize,
                )
            } else {
                let str_obj: &StringObj = &*(self.data.data_union.v_obj as *const StringObj);
                std::slice::from_raw_parts(str_obj.data.data, str_obj.data.size)
            }
        }
    }

    /// The string as a `str`.
    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }
    }
}

impl From<&str> for String {
    /// Copy the string `value`.
    fn from(value: &str) -> Self {
        unsafe {
            let bytes = value.as_bytes();
            const MAX_SMALL_BYTES_LEN: usize = 7;
            if bytes.len() <= MAX_SMALL_BYTES_LEN {
                let mut data_union = TVMFFIAnyDataUnion { v_int64: 0 };
                data_union.v_bytes[..bytes.len()].copy_from_slice(bytes);
                Self {
                    data: TVMFFIAny {
                        type_index: TypeIndex::kTVMFFISmallStr as i32,
                        small_str_len: bytes.len() as u32,
                        data_union,
                    },
                }
            } else {
                let mut obj_arc = ObjectArc::new_with_extra_items(StringObj {
                    object: Object::new(),
                    data: TVMFFIByteArray {
                        data: std::ptr::null(),
                        size: bytes.len(),
                    },
                });
                // The fresh allocation is unique and has not been exposed.
                let object = ObjectArc::as_mut_ptr(&mut obj_arc);
                let items = StringObj::extra_items_ptr(object);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), items, bytes.len());
                // write the trailing \0 for ffi compatibility
                items.add(bytes.len()).write(0);
                (*object).data.data = items;
                Self {
                    data: TVMFFIAny {
                        type_index: TypeIndex::kTVMFFIStr as i32,
                        small_str_len: 0,
                        data_union: TVMFFIAnyDataUnion {
                            v_obj: ObjectArc::into_raw(obj_arc) as *mut StringObj
                                as *mut TVMFFIObject,
                        },
                    },
                }
            }
        }
    }
}

impl Default for String {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for String {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Clone for String {
    #[inline]
    fn clone(&self) -> Self {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { unsafe_::inc_ref(self.data.data_union.v_obj) }
        }
        Self { data: self.data }
    }
}

impl Drop for String {
    #[inline]
    fn drop(&mut self) {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { unsafe_::dec_ref(self.data.data_union.v_obj) }
        }
    }
}

impl PartialEq for String {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

macro_rules! impl_string_from {
    ($([$($g:tt)*] $t:ty),* $(,)?) => {
        $(impl<$($g)*> From<$t> for String {
            #[inline]
            fn from(value: $t) -> Self {
                Self::from(&value[..])
            }
        })*
    };
}

impl_string_from!([] &mut str, [] std::string::String, [] &std::string::String, [] Box<str>, ['a] std::borrow::Cow<'a, str>);

macro_rules! impl_string_eq {
    ($([$($g:tt)*] $t:ty),* $(,)?) => {
        $(impl<$($g)*> PartialEq<$t> for String {
            #[inline]
            fn eq(&self, other: &$t) -> bool {
                self.as_str() == &other[..]
            }
        }

        impl<$($g)*> PartialEq<String> for $t {
            #[inline]
            fn eq(&self, other: &String) -> bool {
                &self[..] == other.as_str()
            }
        })*
    };
}

impl_string_eq!([] str, ['a] &'a str, [] std::string::String, ['a] std::borrow::Cow<'a, str>);

impl AsRef<str> for String {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<[u8]> for String {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl std::borrow::Borrow<str> for String {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<String> for std::string::String {
    #[inline]
    fn from(value: String) -> Self {
        value.as_str().to_owned()
    }
}

impl Eq for String {}

impl PartialOrd for String {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for String {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for String {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl Display for String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.as_str(), f)
    }
}

/// Formats as `str` does, such as `"hi"`.
impl Debug for String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.as_str(), f)
    }
}

//-----------------------------------------------------
// AnyCompatible implementation for Bytes and String
//-----------------------------------------------------
unsafe impl AnyCompatible for Bytes {
    fn type_str() -> std::string::String {
        "ffi.Bytes".to_string()
    }

    // Bytes are small bytes or a bytes object, never None.
    #[inline]
    unsafe fn is_undefined(field: *const Self) -> bool {
        (*std::ptr::addr_of!((*field).data.type_index)) == TypeIndex::kTVMFFINone as i32
    }

    unsafe fn copy_to_any_view(this: &Self, data: &mut TVMFFIAny) {
        *data = this.data;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        *data = src.data;
        std::mem::forget(src);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFISmallBytes as i32
            || data.type_index == TypeIndex::kTVMFFIBytes as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        if data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { unsafe_::inc_ref(data.data_union.v_obj) }
        }
        Self { data: *data }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self { data: *data }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIByteArrayPtr as i32 {
            // deep copy from bytearray ptr
            let bytes = &*(data.data_union.v_ptr as *const TVMFFIByteArray);
            Ok(Self::from(crate::util::slice_from_nullable_raw_parts(
                bytes.data, bytes.size,
            )))
        } else if data.type_index == TypeIndex::kTVMFFISmallBytes as i32 {
            Ok(Self { data: *data })
        } else if data.type_index == TypeIndex::kTVMFFIBytes as i32 {
            unsafe { unsafe_::inc_ref(data.data_union.v_obj) }
            Ok(Self { data: *data })
        } else {
            Err(())
        }
    }
}

unsafe impl AnyCompatible for String {
    fn type_str() -> std::string::String {
        "ffi.String".to_string()
    }

    // A string is a small string or a string object, never None.
    #[inline]
    unsafe fn is_undefined(field: *const Self) -> bool {
        (*std::ptr::addr_of!((*field).data.type_index)) == TypeIndex::kTVMFFINone as i32
    }

    unsafe fn copy_to_any_view(this: &Self, data: &mut TVMFFIAny) {
        *data = this.data;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        *data = src.data;
        std::mem::forget(src);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFISmallStr as i32
            || data.type_index == TypeIndex::kTVMFFIStr as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        if data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { unsafe_::inc_ref(data.data_union.v_obj) }
        }
        Self { data: *data }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self { data: *data }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIRawStr as i32 {
            let c_str =
                std::ffi::CStr::from_ptr(data.data_union.v_c_str as *const std::os::raw::c_char);
            c_str.to_str().map(Self::from).map_err(|_| ())
        } else if data.type_index == TypeIndex::kTVMFFISmallStr as i32 {
            Ok(Self { data: *data })
        } else if data.type_index == TypeIndex::kTVMFFIStr as i32 {
            unsafe { unsafe_::inc_ref(data.data_union.v_obj) }
            Ok(Self { data: *data })
        } else {
            Err(())
        }
    }
}
