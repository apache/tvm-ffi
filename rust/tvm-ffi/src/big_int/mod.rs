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
//! Integers with inline `i64` storage and arbitrary-precision arithmetic.
mod int_ops;

use crate::derive::Object;
use crate::error::{Error, Result, OVERFLOW_ERROR, VALUE_ERROR};
use crate::object::{unsafe_, Object, ObjectArc, ObjectCoreWithExtraItems};
use crate::type_traits::AnyCompatible;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Write as _};
use std::hash::{Hash, Hasher};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};
use std::str::FromStr;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIObject};

/// ABI stable arbitrary-precision signed integer for ffi.
///
/// Mirrors C++ `tvm::ffi::BigInt`: a value that fits `i64` is stored inline as
/// `kTVMFFIInt`, and any wider value owns a `kTVMFFIBigInt` object holding its
/// minimal two's-complement `i64` words, least-significant word first. Every
/// operation returns that canonical form, so word-wise comparison decides equality.
///
/// Arithmetic is unbounded and signed. `/` and `%` truncate toward zero;
/// [`floor_div`](Self::floor_div) and [`floor_mod`](Self::floor_mod) round down.
/// Bitwise operations sign-extend infinitely, and negative shift counts are errors.
/// Like the primitive operators, `/`, `%`, `<<` and `>>` panic on those errors;
/// the `try_*` methods return them instead, as ffi callbacks need since a panic
/// cannot unwind across them.
#[repr(C)]
pub struct BigInt {
    data: TVMFFIAny,
}

// BigIntObj for heap-allocated integers: the word count, then the canonical
// two's-complement words as trailing items. Mirrors C++ `details::BigIntObj`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.BigInt"]
#[type_index(TypeIndex::kTVMFFIBigInt)]
#[type_final]
struct BigIntObj {
    object: Object,
    size: usize,
}

unsafe impl ObjectCoreWithExtraItems for BigIntObj {
    type ExtraItem = i64;
    #[inline]
    /// Get the count of extra items (the logical word count)
    fn extra_items_count(this: &Self) -> usize {
        this.size
    }
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

    /// Take ownership of a canonical multiword object without an extra reference.
    #[inline]
    fn from_obj(obj: *mut BigIntObj) -> Self {
        Self {
            data: TVMFFIAny {
                type_index: TypeIndex::kTVMFFIBigInt as i32,
                small_str_len: 0,
                data_union: TVMFFIAnyDataUnion {
                    v_obj: obj.cast::<TVMFFIObject>(),
                },
            },
        }
    }

    /// Create from two's-complement `i64` words, least-significant word first.
    ///
    /// Redundant sign-extension words are pruned and a value that fits `i64`
    /// is stored inline; an empty slice is zero.
    pub fn from_words(words: &[i64]) -> Self {
        let size = int_ops::normalized_len(words);
        if size <= 1 {
            return Self::from_i64(words.first().copied().unwrap_or(0));
        }
        // Allocated at its final length, so the crate's generic extra-items deleter fits.
        unsafe {
            let mut obj = ObjectArc::new_with_extra_items(BigIntObj {
                object: Object::new(),
                size,
            });
            BigIntObj::extra_items_mut(&mut obj).copy_from_slice(&words[..size]);
            Self::from_obj(ObjectArc::into_raw(obj) as *mut BigIntObj)
        }
    }

    /// Borrow the canonical two's-complement words, least-significant word first.
    ///
    /// The slice is never empty and its last word carries the sign.
    pub fn words(&self) -> &[i64] {
        unsafe {
            if self.data.type_index == TypeIndex::kTVMFFIInt as i32 {
                std::slice::from_ref(&self.data.data_union.v_int64)
            } else {
                let obj: &BigIntObj = &*(self.data.data_union.v_obj as *const BigIntObj);
                BigIntObj::extra_items(obj)
            }
        }
    }

    /// Whether the value is negative.
    pub fn is_negative(&self) -> bool {
        int_ops::is_negative(self.words())
    }

    /// Whether the value is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.to_i64() == Some(0)
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

    /// Convert to the nearest finite double, rounding ties to even.
    ///
    /// Raises `OverflowError` when the value rounds beyond the finite range.
    pub fn to_f64(&self) -> Result<f64> {
        match self.to_i64() {
            Some(value) => Ok(value as f64),
            None => int_ops::to_f64(self.words()),
        }
    }

    /// Quotient rounded toward zero; raises `ZeroDivisionError`.
    pub fn try_div(&self, rhs: &BigInt) -> Result<BigInt> {
        div_impl(self, rhs)
    }

    /// Remainder of truncating division, zero or with the dividend's sign.
    pub fn try_rem(&self, rhs: &BigInt) -> Result<BigInt> {
        rem_impl(self, rhs)
    }

    /// Truncating quotient and remainder in one pass; raises `ZeroDivisionError`.
    pub fn div_rem(&self, rhs: &BigInt) -> Result<(BigInt, BigInt)> {
        if let (Some(x), Some(y)) = (self.to_i64(), rhs.to_i64()) {
            if let Some(q) = int_ops::trunc_div_i64(x, y)? {
                return Ok((Self::from_i64(q), Self::from_i64(x.wrapping_rem(y))));
            }
        }
        int_ops::div_rem(self.words(), rhs.words())
    }

    /// Quotient rounded toward negative infinity; raises `ZeroDivisionError`.
    pub fn floor_div(&self, rhs: &BigInt) -> Result<BigInt> {
        floor_div_impl(self, rhs)
    }

    /// Remainder of floor division, zero or with the divisor's sign.
    pub fn floor_mod(&self, rhs: &BigInt) -> Result<BigInt> {
        floor_mod_impl(self, rhs)
    }

    /// Left shift; negative counts raise `ValueError`, excessive growth `OverflowError`.
    pub fn try_shl<C: Into<BigInt>>(&self, count: C) -> Result<BigInt> {
        shl_impl(self, &count.into())
    }

    /// Sign-extending right shift; negative counts raise `ValueError`.
    pub fn try_shr<C: Into<BigInt>>(&self, count: C) -> Result<BigInt> {
        shr_impl(self, &count.into())
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

impl From<bool> for BigInt {
    #[inline]
    fn from(value: bool) -> Self {
        Self::from_i64(i64::from(value))
    }
}

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

impl TryFrom<f64> for BigInt {
    type Error = Error;

    /// Convert a finite double by truncating toward zero.
    ///
    /// NaN raises `ValueError` and infinity raises `OverflowError`.
    fn try_from(value: f64) -> Result<Self> {
        // The upper bound is exclusive: i64::MAX rounds up to 2^63 as a double.
        const LOWER: f64 = i64::MIN as f64;
        if (LOWER..-LOWER).contains(&value) {
            return Ok(Self::from_i64(value as i64));
        }
        if value.is_nan() {
            return Err(Error::new(VALUE_ERROR, "Cannot convert NaN to BigInt", ""));
        }
        if value.is_infinite() {
            return Err(Error::new(
                OVERFLOW_ERROR,
                "Cannot convert infinity to BigInt",
                "",
            ));
        }
        // |value| >= 2^63 is an integer: shift the 53-bit significand to its binary exponent.
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64 - 1075;
        let significand = ((bits & ((1u64 << 52) - 1)) | (1u64 << 52)) as i64;
        let magnitude = Self::from_i64(significand) << exponent;
        Ok(if value < 0.0 { -magnitude } else { magnitude })
    }
}

impl FromStr for BigInt {
    type Err = Error;

    /// Parse an optionally signed decimal integer; anything else raises `ValueError`.
    fn from_str(text: &str) -> Result<Self> {
        if let Ok(value) = text.parse::<i64>() {
            return Ok(Self::from_i64(value));
        }
        let (negative, digits) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(Error::new(
                VALUE_ERROR,
                "Invalid decimal string for BigInt",
                "",
            ));
        }
        // Accumulate 19-digit chunks: magnitude = magnitude * 10^len + chunk.
        let mut magnitude = Vec::new();
        for chunk in digits.as_bytes().chunks(19) {
            let value = chunk
                .iter()
                .fold(0u64, |acc, &digit| acc * 10 + u64::from(digit - b'0'));
            int_ops::mul_add_small(&mut magnitude, 10u64.pow(chunk.len() as u32), value);
        }
        let mut words: Vec<i64> = magnitude.into_iter().map(|word| word as i64).collect();
        words.push(0);
        if negative {
            int_ops::negate_in_place(&mut words);
        }
        Ok(Self::from_words(&words))
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
        let (negative_a, negative_b) = (int_ops::is_negative(a), int_ops::is_negative(b));
        if negative_a != negative_b {
            return if negative_a {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        // With equal signs, sign-extended words compare as unsigned from the top.
        let extension = int_ops::extension(a);
        (0..a.len().max(b.len()))
            .rev()
            .map(|i| int_ops::word_at(a, i, extension).cmp(&int_ops::word_at(b, i, extension)))
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
        let mut magnitude = words.to_vec();
        if negative {
            int_ops::negate_in_place(&mut magnitude);
        }
        // Peel base-10^19 chunks, least significant first, by long division.
        const CHUNK: u128 = 10_000_000_000_000_000_000;
        let mut chunks = Vec::new();
        while let Some(top) = magnitude.iter().rposition(|&word| word != 0) {
            magnitude.truncate(top + 1);
            let mut remainder = 0u128;
            for word in magnitude.iter_mut().rev() {
                let current = (remainder << 64) | u128::from(*word as u64);
                *word = (current / CHUNK) as i64;
                remainder = current % CHUNK;
            }
            chunks.push(remainder as u64);
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

// ---- Arithmetic: an inline fast path, then the word fallbacks.
#[inline]
fn add_impl(a: &BigInt, b: &BigInt) -> BigInt {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = x.checked_add(y) {
            return BigInt::from_i64(result);
        }
    }
    int_ops::add(a.words(), b.words())
}

#[inline]
fn sub_impl(a: &BigInt, b: &BigInt) -> BigInt {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = x.checked_sub(y) {
            return BigInt::from_i64(result);
        }
    }
    int_ops::sub(a.words(), b.words())
}

#[inline]
fn mul_impl(a: &BigInt, b: &BigInt) -> BigInt {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = x.checked_mul(y) {
            return BigInt::from_i64(result);
        }
    }
    int_ops::mul(a.words(), b.words())
}

#[inline]
fn div_impl(a: &BigInt, b: &BigInt) -> Result<BigInt> {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = int_ops::trunc_div_i64(x, y)? {
            return Ok(BigInt::from_i64(result));
        }
    }
    int_ops::trunc_div(a.words(), b.words())
}

#[inline]
fn rem_impl(a: &BigInt, b: &BigInt) -> Result<BigInt> {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = int_ops::trunc_mod_i64(x, y)? {
            return Ok(BigInt::from_i64(result));
        }
    }
    int_ops::trunc_mod(a.words(), b.words())
}

#[inline]
fn floor_div_impl(a: &BigInt, b: &BigInt) -> Result<BigInt> {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = int_ops::floor_div_i64(x, y)? {
            return Ok(BigInt::from_i64(result));
        }
    }
    int_ops::floor_div(a.words(), b.words())
}

#[inline]
fn floor_mod_impl(a: &BigInt, b: &BigInt) -> Result<BigInt> {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = int_ops::floor_mod_i64(x, y)? {
            return Ok(BigInt::from_i64(result));
        }
    }
    int_ops::floor_mod(a.words(), b.words())
}

#[inline]
fn shl_impl(a: &BigInt, b: &BigInt) -> Result<BigInt> {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if let Some(result) = int_ops::shl_i64(x, y)? {
            return Ok(BigInt::from_i64(result));
        }
    }
    int_ops::shl(a.words(), b.words())
}

#[inline]
fn shr_impl(a: &BigInt, b: &BigInt) -> Result<BigInt> {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        return Ok(BigInt::from_i64(int_ops::shr_i64(x, y)?));
    }
    int_ops::shr(a.words(), b.words())
}

#[inline]
fn and_impl(a: &BigInt, b: &BigInt) -> BigInt {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        return BigInt::from_i64(x & y);
    }
    int_ops::and(a.words(), b.words())
}

#[inline]
fn or_impl(a: &BigInt, b: &BigInt) -> BigInt {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        return BigInt::from_i64(x | y);
    }
    int_ops::or(a.words(), b.words())
}

#[inline]
fn xor_impl(a: &BigInt, b: &BigInt) -> BigInt {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        return BigInt::from_i64(x ^ y);
    }
    int_ops::xor(a.words(), b.words())
}

#[inline]
fn neg_impl(a: &BigInt) -> BigInt {
    if let Some(negated) = a.to_i64().and_then(i64::checked_neg) {
        return BigInt::from_i64(negated);
    }
    int_ops::negate(a.words())
}

#[inline]
fn not_impl(a: &BigInt) -> BigInt {
    if let Some(x) = a.to_i64() {
        return BigInt::from_i64(!x);
    }
    int_ops::not(a.words())
}

/// The operator form of a fallible operation panics like the primitive one would.
macro_rules! define_op_or_panic {
    ($($name:ident => $imp:ident),* $(,)?) => {
        $(
            #[inline]
            fn $name(a: &BigInt, b: &BigInt) -> BigInt {
                $imp(a, b).unwrap_or_else(|error| panic!("{}", error.message()))
            }
        )*
    };
}

define_op_or_panic!(
    div_or_panic => div_impl,
    rem_or_panic => rem_impl,
    shl_or_panic => shl_impl,
    shr_or_panic => shr_impl,
);

/// Forward an operator to its implementation for owned and borrowed operands,
/// with `i64` accepted on either side as C++ does.
macro_rules! forward_binop {
    ($Trait:ident::$method:ident, $Assign:ident::$assign:ident, $imp:ident) => {
        impl $Trait<&BigInt> for &BigInt {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: &BigInt) -> BigInt {
                $imp(self, rhs)
            }
        }
        impl $Trait<BigInt> for &BigInt {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: BigInt) -> BigInt {
                $imp(self, &rhs)
            }
        }
        impl $Trait<&BigInt> for BigInt {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: &BigInt) -> BigInt {
                $imp(&self, rhs)
            }
        }
        impl $Trait<BigInt> for BigInt {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: BigInt) -> BigInt {
                $imp(&self, &rhs)
            }
        }
        impl $Trait<i64> for &BigInt {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: i64) -> BigInt {
                $imp(self, &BigInt::from_i64(rhs))
            }
        }
        impl $Trait<i64> for BigInt {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: i64) -> BigInt {
                $imp(&self, &BigInt::from_i64(rhs))
            }
        }
        impl $Trait<&BigInt> for i64 {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: &BigInt) -> BigInt {
                $imp(&BigInt::from_i64(self), rhs)
            }
        }
        impl $Trait<BigInt> for i64 {
            type Output = BigInt;
            #[inline]
            fn $method(self, rhs: BigInt) -> BigInt {
                $imp(&BigInt::from_i64(self), &rhs)
            }
        }
        impl $Assign<&BigInt> for BigInt {
            #[inline]
            fn $assign(&mut self, rhs: &BigInt) {
                *self = $imp(self, rhs);
            }
        }
        impl $Assign<BigInt> for BigInt {
            #[inline]
            fn $assign(&mut self, rhs: BigInt) {
                *self = $imp(self, &rhs);
            }
        }
        impl $Assign<i64> for BigInt {
            #[inline]
            fn $assign(&mut self, rhs: i64) {
                *self = $imp(self, &BigInt::from_i64(rhs));
            }
        }
    };
}

forward_binop!(Add::add, AddAssign::add_assign, add_impl);
forward_binop!(Sub::sub, SubAssign::sub_assign, sub_impl);
forward_binop!(Mul::mul, MulAssign::mul_assign, mul_impl);
forward_binop!(Div::div, DivAssign::div_assign, div_or_panic);
forward_binop!(Rem::rem, RemAssign::rem_assign, rem_or_panic);
forward_binop!(BitAnd::bitand, BitAndAssign::bitand_assign, and_impl);
forward_binop!(BitOr::bitor, BitOrAssign::bitor_assign, or_impl);
forward_binop!(BitXor::bitxor, BitXorAssign::bitxor_assign, xor_impl);
forward_binop!(Shl::shl, ShlAssign::shl_assign, shl_or_panic);
forward_binop!(Shr::shr, ShrAssign::shr_assign, shr_or_panic);

impl Neg for &BigInt {
    type Output = BigInt;
    #[inline]
    fn neg(self) -> BigInt {
        neg_impl(self)
    }
}

impl Neg for BigInt {
    type Output = BigInt;
    #[inline]
    fn neg(self) -> BigInt {
        neg_impl(&self)
    }
}

impl Not for &BigInt {
    type Output = BigInt;
    #[inline]
    fn not(self) -> BigInt {
        not_impl(self)
    }
}

impl Not for BigInt {
    type Output = BigInt;
    #[inline]
    fn not(self) -> BigInt {
        not_impl(&self)
    }
}

// ---- AnyCompatible.
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

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        if Self::check_any_strict(data) {
            Ok(Self::copy_from_any_view_after_check(data))
        } else if data.type_index == TypeIndex::kTVMFFIBool as i32 {
            Ok(Self::from_i64(data.data_union.v_int64))
        } else {
            Err(())
        }
    }
}
