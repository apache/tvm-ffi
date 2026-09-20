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
use crate::object::unsafe_;
use crate::type_traits::AnyCompatible;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Write as _};
use std::hash::{Hash, Hasher};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIBigIntFromByteArray, TVMFFIBigIntGetContentByteArray,
    TVMFFIByteArray,
};

/// ABI stable arbitrary-precision signed integer for ffi.
///
/// Mirrors C++ `tvm::ffi::BigInt`: a value that fits `i64` is stored inline as
/// `kTVMFFIInt`, and any wider value owns a `kTVMFFIBigInt` object holding its
/// minimal two's-complement `i64` words, least-significant word first. The
/// runtime keeps that representation canonical, so equal values share one
/// representation and word-wise comparison decides equality.
///
/// The Rust side transports values across the ABI and converts them to and
/// from native integers; arithmetic stays with the C++ implementation.
#[repr(C)]
pub struct BigInt {
    data: TVMFFIAny,
}

impl BigInt {
    /// Create the integer zero.
    pub fn new() -> Self {
        Self::from_i64(0)
    }

    #[inline]
    fn from_i64(value: i64) -> Self {
        Self {
            data: TVMFFIAny {
                type_index: TypeIndex::kTVMFFIInt as i32,
                small_str_len: 0,
                data_union: TVMFFIAnyDataUnion { v_int64: value },
            },
        }
    }

    /// Create from two's-complement `i64` words, least-significant word first.
    ///
    /// Redundant sign-extension words are pruned and a value that fits `i64`
    /// is stored inline; an empty slice is zero.
    pub fn from_words(words: &[i64]) -> Self {
        let input = TVMFFIByteArray::new(words.as_ptr() as *const u8, std::mem::size_of_val(words));
        let mut data = TVMFFIAny::new();
        unsafe {
            crate::check_safe_call!(TVMFFIBigIntFromByteArray(&input, &mut data))
                .expect("TVMFFIBigIntFromByteArray failed");
        }
        Self { data }
    }

    /// Borrow the canonical two's-complement words, least-significant word first.
    ///
    /// The slice is never empty and its last word carries the sign.
    pub fn words(&self) -> &[i64] {
        unsafe {
            if self.data.type_index == TypeIndex::kTVMFFIInt as i32 {
                std::slice::from_ref(&self.data.data_union.v_int64)
            } else {
                let content = TVMFFIBigIntGetContentByteArray(&self.data);
                std::slice::from_raw_parts(
                    content.data as *const i64,
                    content.size / std::mem::size_of::<i64>(),
                )
            }
        }
    }

    /// Whether the value is negative.
    pub fn is_negative(&self) -> bool {
        let words = self.words();
        words[words.len() - 1] < 0
    }

    /// Return the value when it fits `i64`.
    #[inline]
    pub fn to_i64(&self) -> Option<i64> {
        if self.data.type_index == TypeIndex::kTVMFFIInt as i32 {
            Some(unsafe { self.data.data_union.v_int64 })
        } else {
            None
        }
    }

    /// Return the value when it fits `u64`.
    pub fn to_u64(&self) -> Option<u64> {
        match *self.words() {
            [w0] => u64::try_from(w0).ok(),
            [w0, 0] => Some(w0 as u64),
            _ => None,
        }
    }

    /// Return the value when it fits `i128`.
    pub fn to_i128(&self) -> Option<i128> {
        match *self.words() {
            [w0] => Some(i128::from(w0)),
            [w0, w1] => Some((i128::from(w1) << 64) | i128::from(w0 as u64)),
            _ => None,
        }
    }

    /// Return the value when it fits `u128`.
    pub fn to_u128(&self) -> Option<u128> {
        let combine = |w0: i64, w1: i64| (u128::from(w1 as u64) << 64) | u128::from(w0 as u64);
        match *self.words() {
            [w0] => u128::try_from(w0).ok(),
            [w0, w1] if w1 >= 0 => Some(combine(w0, w1)),
            [w0, w1, 0] => Some(combine(w0, w1)),
            _ => None,
        }
    }
}

impl Default for BigInt {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Macro to implement From for integer types that always fit i64
macro_rules! impl_big_int_from_int {
    ($($int_type:ty),* $(,)?) => {
        $(
            impl From<$int_type> for BigInt {
                #[inline]
                fn from(value: $int_type) -> Self {
                    Self::from_i64(value as i64)
                }
            }
        )*
    };
}

impl_big_int_from_int!(i8, i16, i32, i64, isize, u8, u16, u32);

impl From<u64> for BigInt {
    fn from(value: u64) -> Self {
        match i64::try_from(value) {
            Ok(value) => Self::from_i64(value),
            Err(_) => Self::from_words(&[value as i64, 0]),
        }
    }
}

impl From<usize> for BigInt {
    #[inline]
    fn from(value: usize) -> Self {
        Self::from(value as u64)
    }
}

impl From<i128> for BigInt {
    fn from(value: i128) -> Self {
        match i64::try_from(value) {
            Ok(value) => Self::from_i64(value),
            // The arithmetic shift keeps the sign in the high word.
            Err(_) => Self::from_words(&[value as i64, (value >> 64) as i64]),
        }
    }
}

impl From<u128> for BigInt {
    fn from(value: u128) -> Self {
        match i64::try_from(value) {
            Ok(value) => Self::from_i64(value),
            // The zero guard keeps the value positive; normalization prunes it when redundant.
            Err(_) => Self::from_words(&[value as i64, (value >> 64) as i64, 0]),
        }
    }
}

impl Clone for BigInt {
    #[inline]
    fn clone(&self) -> Self {
        if self.data.type_index == TypeIndex::kTVMFFIBigInt as i32 {
            unsafe { unsafe_::inc_ref(self.data.data_union.v_obj) }
        }
        Self { data: self.data }
    }
}

impl Drop for BigInt {
    #[inline]
    fn drop(&mut self) {
        if self.data.type_index == TypeIndex::kTVMFFIBigInt as i32 {
            unsafe { unsafe_::dec_ref(self.data.data_union.v_obj) }
        }
    }
}

impl PartialEq for BigInt {
    /// Canonical words make equal values word-for-word identical.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.words() == other.words()
    }
}

impl Eq for BigInt {}

impl PartialOrd for BigInt {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        let (a, b) = (self.words(), other.words());
        let (negative_a, negative_b) = (a[a.len() - 1] < 0, b[b.len() - 1] < 0);
        if negative_a != negative_b {
            return if negative_a {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        // With equal signs, sign-extended words compare as unsigned from the top.
        let extension = if negative_a { u64::MAX } else { 0 };
        let word = |words: &[i64], i: usize| words.get(i).map_or(extension, |&w| w as u64);
        (0..a.len().max(b.len()))
            .rev()
            .map(|i| word(a, i).cmp(&word(b, i)))
            .find(|ordering| ordering.is_ne())
            .unwrap_or(Ordering::Equal)
    }
}

impl Hash for BigInt {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words().hash(state);
    }
}

impl Display for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(value) = self.to_i64() {
            return Display::fmt(&value, f);
        }
        let words = self.words();
        let negative = words[words.len() - 1] < 0;
        // Magnitude words: negating two's complement is complement plus one.
        let mut magnitude: Vec<u64> = words
            .iter()
            .map(|&w| if negative { !(w as u64) } else { w as u64 })
            .collect();
        if negative {
            for word in magnitude.iter_mut() {
                let (sum, overflow) = word.overflowing_add(1);
                *word = sum;
                if !overflow {
                    break;
                }
            }
        }
        // Peel base-10^19 chunks, least significant first, by long division.
        const CHUNK: u128 = 10_000_000_000_000_000_000;
        let mut chunks = Vec::new();
        while magnitude.last() == Some(&0) {
            magnitude.pop();
        }
        while !magnitude.is_empty() {
            let mut remainder = 0u128;
            for word in magnitude.iter_mut().rev() {
                let current = (remainder << 64) | u128::from(*word);
                *word = (current / CHUNK) as u64;
                remainder = current % CHUNK;
            }
            chunks.push(remainder as u64);
            while magnitude.last() == Some(&0) {
                magnitude.pop();
            }
        }
        let mut digits = String::new();
        let mut chunks = chunks.iter().rev();
        // A heap value is never zero, so the leading chunk always exists.
        write!(digits, "{}", chunks.next().unwrap_or(&0))?;
        for chunk in chunks {
            write!(digits, "{chunk:019}")?;
        }
        f.pad_integral(!negative, "", &digits)
    }
}

impl Debug for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ffi.BigInt")
            .field("data", &format_args!("{self}"))
            .finish()
    }
}

//-----------------------------------------------------
// AnyCompatible implementation for BigInt
//-----------------------------------------------------
unsafe impl AnyCompatible for BigInt {
    fn type_str() -> std::string::String {
        "BigInt".to_string()
    }

    unsafe fn copy_to_any_view(this: &Self, data: &mut TVMFFIAny) {
        *data = this.data;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        *data = src.data;
        std::mem::forget(src);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFIInt as i32
            || data.type_index == TypeIndex::kTVMFFIBigInt as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        if data.type_index == TypeIndex::kTVMFFIBigInt as i32 {
            unsafe { unsafe_::inc_ref(data.data_union.v_obj) }
        }
        Self { data: *data }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self { data: *data }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if Self::check_any_strict(data) {
            Ok(Self::copy_from_any_view_after_check(data))
        } else if data.type_index == TypeIndex::kTVMFFIBool as i32 {
            Ok(Self::from_i64(data.data_union.v_int64))
        } else {
            Err(())
        }
    }
}
