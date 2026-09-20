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
//! Word-level integer algorithms behind the `BigInt` operators.
//!
//! Every fallback takes canonical two's-complement words, least-significant
//! first, and returns a canonical `BigInt`. The functions mirror C++
//! `details::int_ops` in `include/tvm/ffi/big_int.h` one for one and follow its
//! index-based structure so the two can be reviewed side by side; keep them in
//! sync. Word arithmetic wraps modulo 2^64 on purpose, so every carry chain
//! uses the explicit `wrapping_*`/`overflowing_*` forms.
#![allow(clippy::needless_range_loop)]

use super::builder::WordsBuilder;
use super::BigInt;
use crate::error::{Error, Result, OVERFLOW_ERROR, VALUE_ERROR, ZERO_DIVISION_ERROR};

const BASE: u64 = 1 << 32;

pub(super) fn zero_division() -> Error {
    Error::new(ZERO_DIVISION_ERROR, "Division by zero", "")
}

fn negative_shift() -> Error {
    Error::new(VALUE_ERROR, "Negative BigInt shift count", "")
}

fn overflow(message: &str) -> Error {
    Error::new(OVERFLOW_ERROR, message, "")
}

/// Whether a nonempty signed-word view represents a negative integer.
#[inline]
pub(super) fn is_negative(x: &[i64]) -> bool {
    x[x.len() - 1] < 0
}

/// The word that sign-extends `x` beyond its length.
#[inline]
fn extension(x: &[i64]) -> u64 {
    if is_negative(x) {
        u64::MAX
    } else {
        0
    }
}

/// Word `i` of `x`, sign-extended past its length.
#[inline]
fn word_at(x: &[i64], i: usize, extension: u64) -> u64 {
    x.get(i).map_or(extension, |&w| w as u64)
}

/// Length after pruning redundant sign words; zero words stay zero.
pub(super) fn normalized_len(words: &[i64]) -> usize {
    let mut size = words.len();
    while size > 1 && words[size - 1] == (if words[size - 2] < 0 { -1 } else { 0 }) {
        size -= 1;
    }
    size
}

// ---- Inline (i64) forms. `Ok(None)` means the result needs the multiword fallback.

/// Quotient of division rounded toward zero.
#[inline]
pub(super) fn trunc_div_i64(a: i64, b: i64) -> Result<Option<i64>> {
    if b == 0 {
        return Err(zero_division());
    }
    // i64::MIN / -1 is the only quotient that needs promotion.
    Ok(a.checked_div(b))
}

/// Remainder of division rounded toward zero.
#[inline]
pub(super) fn trunc_mod_i64(a: i64, b: i64) -> Result<Option<i64>> {
    if b == 0 {
        return Err(zero_division());
    }
    // The remainder of i64::MIN / -1 is zero, which the wrapping form yields.
    Ok(Some(a.wrapping_rem(b)))
}

/// Quotient of division rounded down.
#[inline]
pub(super) fn floor_div_i64(a: i64, b: i64) -> Result<Option<i64>> {
    if b == 0 {
        return Err(zero_division());
    }
    Ok(a.checked_div(b).map(|q| {
        // A nonexact quotient of opposite signs rounds one lower.
        if a.wrapping_rem(b) != 0 && ((a < 0) != (b < 0)) {
            q - 1
        } else {
            q
        }
    }))
}

/// Remainder of division rounded down.
#[inline]
pub(super) fn floor_mod_i64(a: i64, b: i64) -> Result<Option<i64>> {
    if b == 0 {
        return Err(zero_division());
    }
    let r = a.wrapping_rem(b);
    // Compensate for the quotient rounding down, preserving a = q*b + r.
    Ok(Some(if r != 0 && ((a < 0) != (b < 0)) {
        r + b
    } else {
        r
    }))
}

/// Left shift, rejecting negative counts.
#[inline]
pub(super) fn shl_i64(a: i64, b: i64) -> Result<Option<i64>> {
    if b < 0 {
        return Err(negative_shift());
    }
    if b == 0 || a == 0 {
        return Ok(Some(a));
    }
    // For positive a require a <= i64::MAX >> b; for negative a require a >= -2^(63-b).
    if b >= 64 || a > (i64::MAX >> b) || a < -(1i64 << (63 - b)) {
        return Ok(None);
    }
    Ok(Some(a << b))
}

/// Right shift with arithmetic sign extension; this operation cannot overflow.
#[inline]
pub(super) fn shr_i64(a: i64, b: i64) -> Result<i64> {
    if b < 0 {
        return Err(negative_shift());
    }
    Ok(if b >= 64 {
        if a < 0 {
            -1
        } else {
            0
        }
    } else {
        a >> b
    })
}

// ---- Multiword fallbacks.

/// Add signed integers.
pub(super) fn add(a: &[i64], b: &[i64]) -> BigInt {
    // One sign-extension word retains the carry out until normalization.
    let size = a.len().max(b.len()) + 1;
    let mut builder = WordsBuilder::new(size);
    let (ext_a, ext_b) = (extension(a), extension(b));
    let mut carry = 0u64;
    for (i, out) in builder.words_mut()[..size].iter_mut().enumerate() {
        let (sum, c1) = word_at(a, i, ext_a).overflowing_add(word_at(b, i, ext_b));
        let (result, c2) = sum.overflowing_add(carry);
        carry = u64::from(c1 | c2);
        *out = result as i64;
    }
    builder.finish()
}

/// Subtract signed integers: a - b = a + !b + 1 after sign extension.
pub(super) fn sub(a: &[i64], b: &[i64]) -> BigInt {
    let size = a.len().max(b.len()) + 1;
    let mut builder = WordsBuilder::new(size);
    let (ext_a, ext_b) = (extension(a), extension(b));
    let mut carry = 1u64;
    for (i, out) in builder.words_mut()[..size].iter_mut().enumerate() {
        let (sum, c1) = word_at(a, i, ext_a).overflowing_add(!word_at(b, i, ext_b));
        let (result, c2) = sum.overflowing_add(carry);
        carry = u64::from(c1 | c2);
        *out = result as i64;
    }
    builder.finish()
}

/// Negate a signed integer: -x = !x + 1, with room for negating a minimum value.
pub(super) fn negate(x: &[i64]) -> BigInt {
    let size = x.len() + 1;
    let mut builder = WordsBuilder::new(size);
    let ext = extension(x);
    let mut carry = 1u64;
    for (i, out) in builder.words_mut()[..size].iter_mut().enumerate() {
        let word = !word_at(x, i, ext);
        let (result, overflow) = word.overflowing_add(carry);
        carry = u64::from(overflow);
        *out = result as i64;
    }
    builder.finish()
}

/// Negate two's-complement words in place, modulo their fixed width.
pub(super) fn negate_in_place(words: &mut [i64]) {
    let mut carry = true;
    for word in words.iter_mut() {
        let (sum, overflow) = (!(*word as u64)).overflowing_add(u64::from(carry));
        *word = sum as i64;
        carry = overflow;
    }
}

/// Multiply signed integers.
pub(super) fn mul(a: &[i64], b: &[i64]) -> BigInt {
    let (na, nb) = (a.len(), b.len());
    let mut builder = WordsBuilder::new(na + nb);
    let data = builder.words_mut();
    for i in 0..na {
        let x = u128::from(a[i] as u64);
        let mut carry = 0u128;
        for j in 0..nb {
            // x*y + word + carry < 2^128: each term is below its width's maximum.
            let product = x * u128::from(b[j] as u64) + u128::from(data[i + j] as u64) + carry;
            data[i + j] = product as u64 as i64;
            carry = product >> 64;
        }
        data[i + nb] = carry as u64 as i64;
    }
    // With stored unsigned A/B and 0/1 negative indicators neg_a/neg_b,
    // a = A - neg_a*2^(64*n), b = B - neg_b*2^(64*m). Thus ab is AB minus
    // neg_a*B*2^(64*n) and neg_b*A*2^(64*m); the remaining
    // neg_a*neg_b*2^(64*(n+m)) vanishes modulo the allocated n+m words.
    let subtract_shifted = |data: &mut [i64], x: &[i64], shift: usize| {
        let mut borrow = 0u64;
        for i in 0..x.len() {
            let word = x[i] as u64;
            let (rhs, o1) = word.overflowing_add(borrow);
            let (result, o2) = (data[shift + i] as u64).overflowing_sub(rhs);
            data[shift + i] = result as i64;
            borrow = u64::from(o1 | o2);
        }
    };
    if is_negative(a) {
        subtract_shifted(data, b, na);
    }
    if is_negative(b) {
        subtract_shifted(data, a, nb);
    }
    builder.finish()
}

/// Apply a sign-extended bitwise operation.
fn bitwise(a: &[i64], b: &[i64], op: impl Fn(u64, u64) -> u64) -> BigInt {
    let size = a.len().max(b.len());
    let mut builder = WordsBuilder::new(size);
    let (ext_a, ext_b) = (extension(a), extension(b));
    for (i, out) in builder.words_mut()[..size].iter_mut().enumerate() {
        *out = op(word_at(a, i, ext_a), word_at(b, i, ext_b)) as i64;
    }
    builder.finish()
}

pub(super) fn and(a: &[i64], b: &[i64]) -> BigInt {
    bitwise(a, b, |x, y| x & y)
}

pub(super) fn or(a: &[i64], b: &[i64]) -> BigInt {
    bitwise(a, b, |x, y| x | y)
}

pub(super) fn xor(a: &[i64], b: &[i64]) -> BigInt {
    bitwise(a, b, |x, y| x ^ y)
}

/// Complement all sign-extended bits.
pub(super) fn not(x: &[i64]) -> BigInt {
    let mut builder = WordsBuilder::new(x.len());
    for (out, &word) in builder.words_mut().iter_mut().zip(x) {
        *out = !word;
    }
    builder.finish()
}

/// Convert a nonnegative shift count; `None` when it does not fit `usize`.
fn shift_count(count: &[i64]) -> Result<Option<usize>> {
    if is_negative(count) {
        return Err(negative_shift());
    }
    if count[1..].iter().any(|&w| w != 0) {
        return Ok(None);
    }
    Ok(usize::try_from(count[0] as u64).ok())
}

/// Shift a signed integer left, growing as needed.
pub(super) fn shl(x: &[i64], count: &[i64]) -> Result<BigInt> {
    let shift = shift_count(count)?;
    if x.len() == 1 && x[0] == 0 {
        return Ok(BigInt::from_i64(0));
    }
    let too_large = || overflow("BigInt shift count is too large");
    let shift = shift.ok_or_else(too_large)?;
    let whole = shift / 64;
    let part = (shift % 64) as u32;
    let size = x
        .len()
        .checked_add(whole)
        .and_then(|size| size.checked_add(1))
        .ok_or_else(too_large)?;
    let mut builder = WordsBuilder::new(size);
    let data = builder.words_mut();
    let ext = extension(x);
    // The low `whole` words stay zero.
    for i in whole..size {
        let j = i - whole;
        let mut word = word_at(x, j, ext) << part;
        if part != 0 && j != 0 {
            word |= (x[j - 1] as u64) >> (64 - part);
        }
        data[i] = word as i64;
    }
    Ok(builder.finish())
}

/// Shift a signed integer right with sign extension; huge counts give 0 or -1.
pub(super) fn shr(x: &[i64], count: &[i64]) -> Result<BigInt> {
    let shift = match shift_count(count)? {
        Some(shift) if shift / 64 < x.len() => shift,
        _ => return Ok(BigInt::from_i64(if is_negative(x) { -1 } else { 0 })),
    };
    let whole = shift / 64;
    let size = x.len() - whole;
    let part = (shift % 64) as u32;
    let mut builder = WordsBuilder::new(size);
    let data = builder.words_mut();
    let ext = extension(x);
    for i in 0..size {
        let mut word = (x[i + whole] as u64) >> part;
        // The last partial word draws its high bits from the sign extension.
        if part != 0 {
            word |= word_at(x, i + whole + 1, ext) << (64 - part);
        }
        data[i] = word as i64;
    }
    Ok(builder.finish())
}

// Absolute-value access without copying operands: negation carries through the
// low zero words, negates the first nonzero word, then complements higher words.

/// The first nonzero index, or `x.len()` for zero.
fn first_nonzero(x: &[i64]) -> usize {
    x.iter().position(|&w| w != 0).unwrap_or(x.len())
}

/// Raw word `i` of the absolute value of `x`; indices beyond the view read as zero.
#[inline]
fn abs_word(x: &[i64], first_nonzero: usize, i: usize) -> i64 {
    if i >= x.len() {
        return 0;
    }
    let word = x[i];
    if !is_negative(x) {
        return word;
    }
    // Negation maps [0,0,w,v] to [0,0,-w,!v] as modulo-2^64 word patterns.
    if i < first_nonzero {
        0
    } else if i == first_nonzero {
        word.wrapping_neg()
    } else {
        !word
    }
}

/// Truncating quotient and remainder with an `i64` divisor.
///
/// The remainder is zero or has the dividend's sign.
pub(super) fn div_rem_single(a: &[i64], b: i64) -> Result<(BigInt, BigInt)> {
    if b == 0 {
        return Err(zero_division());
    }
    let divisor = b.unsigned_abs();
    let first_a = first_nonzero(a);
    let nq = a.len() + 1;
    let mut quotient = WordsBuilder::new(nq);
    let mut remainder = 0u64;
    {
        // The top word stays zero as the sign guard.
        let q = quotient.words_mut();
        let mut shift = 0u32;
        let mut normalized = divisor;
        if divisor > u64::from(u32::MAX) {
            while normalized >> 63 == 0 {
                normalized <<= 1;
                shift += 1;
            }
        }
        let divisor_high = normalized >> 32;
        let divisor_low = normalized as u32 as u64;
        for i in (0..a.len()).rev() {
            let word = abs_word(a, first_a, i) as u64;
            let quotient_word = if divisor <= u64::from(u32::MAX) {
                // remainder < divisor < 2^32, so each brought-down half fits u64
                // and yields a quotient half below 2^32 while restoring the bound.
                let high = (remainder << 32) | (word >> 32);
                let quotient_high = high / divisor;
                remainder = high % divisor;
                let low = (remainder << 32) | (word as u32 as u64);
                let quotient_low = low / divisor;
                remainder = low % divisor;
                (quotient_high << 32) | quotient_low
            } else {
                // Normalize the carried 128-bit numerator without ever shifting by 64.
                let high = if shift != 0 {
                    (remainder << shift) | (word >> (64 - shift))
                } else {
                    remainder
                };
                let low = word << shift;
                // Requires upper < normalized; returns a radix-2^32 quotient digit
                // and a remainder below normalized.
                let digit = |upper: u64, next: u64| -> (u64, u64) {
                    let mut estimate = upper / divisor_high;
                    let mut residual = upper % divisor_high;
                    // The normalized high digit bounds overestimation by two. Check
                    // the base before multiplying, and stop before a residual shift
                    // can overflow.
                    while estimate >= BASE || estimate * divisor_low > (residual << 32) + next {
                        estimate -= 1;
                        residual += divisor_high;
                        if residual >= BASE {
                            break;
                        }
                    }
                    // Wrapped subtraction is exact: the true remainder is below normalized.
                    let rest = (upper << 32)
                        .wrapping_add(next)
                        .wrapping_sub(estimate.wrapping_mul(normalized));
                    (estimate, rest)
                };
                let upper = digit(high, low >> 32);
                let lower = digit(upper.1, low as u32 as u64);
                remainder = lower.1 >> shift;
                (upper.0 << 32) | lower.0
            };
            q[i] = quotient_word as i64;
        }
        if is_negative(a) != (b < 0) {
            negate_in_place(&mut q[..nq]);
        }
    }
    // remainder < |b| <= 2^63 makes the conversion and signed negation safe.
    let mut signed_remainder = remainder as i64;
    if is_negative(a) {
        signed_remainder = -signed_remainder;
    }
    Ok((quotient.finish(), BigInt::from_i64(signed_remainder)))
}

/// Truncating quotient and remainder in one division pass.
///
/// The remainder is zero or has the dividend's sign.
pub(super) fn div_rem(a: &[i64], b: &[i64]) -> Result<(BigInt, BigInt)> {
    if b.len() == 1 {
        return div_rem_single(a, b[0]);
    }
    let first_a = first_nonzero(a);
    let first_b = first_nonzero(b);
    // Count magnitude digits, dropping the signed representation's high zero guards.
    let digit_count = |x: &[i64], first: usize| -> usize {
        let mut words = x.len();
        while words > 0 && abs_word(x, first, words - 1) == 0 {
            words -= 1;
        }
        if words == 0 {
            return 0;
        }
        let top = abs_word(x, first, words - 1) as u64;
        2 * words - usize::from(top >> 32 == 0)
    };
    let m = digit_count(a, first_a);
    let n = digit_count(b, first_b);
    if m < n {
        return Ok((BigInt::from_i64(0), BigInt::from_words(a)));
    }

    // Algorithm D uses radix 2^32, keeping products and estimates in u64.
    // Example: A[0] A[1] A[2] A[3] / B[0] B[1] B[2], with digits most-significant first.
    // Actual u/v digits are least-significant first, one radix digit per slot.
    // The scalar dispatch leaves at least two magnitude digits in the divisor.
    let top_word = abs_word(b, first_b, (n - 1) / 2) as u64;
    let mut top_digit = (top_word >> (((n - 1) % 2) * 32)) as u32;
    let mut shift = 0u32;
    // Normalize both inputs until B[0] >= base/2, bounding the quotient estimate error.
    while top_digit < (1u32 << 31) {
        top_digit <<= 1;
        shift += 1;
    }
    let nq = (m - n) / 2 + 2;
    let nr = n.div_ceil(2) + 1;
    // Scratch follows the logical quotient; normalization excludes it but retains its capacity.
    let capacity = nq
        .checked_add(m + 1)
        .and_then(|words| words.checked_add(n))
        .ok_or_else(|| overflow("BigInt division workspace is too large"))?;
    let mut quotient = WordsBuilder::with_capacity(nq, capacity);
    let mut remainder = WordsBuilder::new(nr);
    {
        let (q, scratch) = quotient.words_mut().split_at_mut(nq);
        let (u, v) = scratch.split_at_mut(m + 1);
        let r = remainder.words_mut();
        let normalize = |x: &[i64], first: usize, digits: usize, out: &mut [i64]| -> u32 {
            let mut carry = 0u64;
            for i in 0..digits {
                let word = abs_word(x, first, i / 2) as u64;
                let part = (word >> ((i % 2) * 32)) as u32 as u64;
                let value = (part << shift) | carry;
                out[i] = (value as u32) as i64;
                carry = value >> 32;
            }
            carry as u32
        };
        // The dividend also keeps its shifted high carry.
        u[m] = i64::from(normalize(a, first_a, m, &mut u[..m]));
        normalize(b, first_b, n, &mut v[..n]);

        let mut quotient_word = 0u64;
        for j in (0..=(m - n)).rev() {
            // - Step 1: Estimate q from the leading digits.
            // A[0]=u[j+n], A[1]=u[j+n-1], B[0]=v[n-1]; q=estimate and r=residual below.
            // A[0] is at most B[0]; clamp equality so q remains a radix digit.
            let (mut estimate, mut residual) = if u[j + n] == v[n - 1] {
                (BASE - 1, u[j + n - 1] as u64 + v[n - 1] as u64)
            } else {
                let numerator = ((u[j + n] as u64) << 32) | u[j + n - 1] as u64;
                (numerator / v[n - 1] as u64, numerator % v[n - 1] as u64)
            };
            // - Step 2: Refine q using B[1] until q*B[1] <= r*base + A[2].
            // Decrement q and add B[0] to r at most twice; r < base guards the shift.
            while residual < BASE
                && estimate * v[n - 2] as u64 > (residual << 32) + u[j + n - 2] as u64
            {
                estimate -= 1;
                residual += v[n - 1] as u64;
            }
            // - Step 3: Subtract the full q * B[...] from A[...].
            // The refined estimate is correct or one high; subtract the whole window to decide.
            let mut borrow = 0u64;
            for i in 0..n {
                // A radix-digit product plus the incoming borrow fits u64.
                let product = estimate * v[i] as u64 + borrow;
                let low = product as u32;
                let old = u[j + i] as u32;
                u[j + i] = i64::from(old.wrapping_sub(low));
                borrow = (product >> 32) + u64::from(old < low);
            }
            let old = u[j + n] as u64;
            // Only a borrow out of the full window means the result went negative.
            let negative = old < borrow;
            u[j + n] = i64::from(old.wrapping_sub(borrow) as u32);
            if negative {
                // - Step 4: Decrement q and add back all of B[...], restoring 0 <= R < divisor.
                estimate -= 1;
                let mut carry = 0u64;
                for i in 0..n {
                    let sum = u[j + i] as u64 + v[i] as u64 + carry;
                    u[j + i] = i64::from(sum as u32);
                    carry = sum >> 32;
                }
                u[j + n] = i64::from((u[j + n] as u64 + carry) as u32);
            }
            // Pack descending quotient digits into complete words.
            if j % 2 == 1 {
                quotient_word = estimate << 32;
            } else {
                q[j / 2] = (quotient_word | estimate) as i64;
            }
        }
        // Extract the remainder, undoing the normalization shift.
        for i in 0..nr - 1 {
            let digit = 2 * i;
            let mut word = u[digit] as u64;
            if digit + 1 < n {
                word |= (u[digit + 1] as u64) << 32;
            }
            word >>= shift;
            // Bring down bits from the next digit; shift zero never evaluates a shift by 64.
            if shift != 0 && digit + 2 < n {
                word |= (u[digit + 2] as u64) << (64 - shift);
            }
            r[i] = word as i64;
        }
        // Truncation gives the quotient the operand-sign XOR and a nonzero
        // remainder the dividend's sign.
        if is_negative(a) != is_negative(b) {
            negate_in_place(q);
        }
        if is_negative(a) {
            negate_in_place(r);
        }
    }
    Ok((quotient.finish(), remainder.finish()))
}

/// Quotient rounded toward zero.
pub(super) fn trunc_div(a: &[i64], b: &[i64]) -> Result<BigInt> {
    Ok(div_rem(a, b)?.0)
}

/// Remainder of truncating division: zero or with the dividend's sign.
pub(super) fn trunc_mod(a: &[i64], b: &[i64]) -> Result<BigInt> {
    if b.len() == 1 {
        if b[0] == 0 {
            return Err(zero_division());
        }
        let divisor = b[0].unsigned_abs();
        if divisor <= u64::from(u32::MAX) {
            // remainder < divisor < 2^32 lets each brought-down half fit u64.
            // The result fits inline, so a small divisor needs no quotient or allocation.
            let first = first_nonzero(a);
            let mut remainder = 0u64;
            for i in (0..a.len()).rev() {
                let word = abs_word(a, first, i) as u64;
                remainder = ((remainder << 32) | (word >> 32)) % divisor;
                remainder = ((remainder << 32) | (word as u32 as u64)) % divisor;
            }
            let result = remainder as i64;
            return Ok(BigInt::from_i64(if is_negative(a) {
                -result
            } else {
                result
            }));
        }
    }
    Ok(div_rem(a, b)?.1)
}

/// Quotient rounded toward negative infinity.
pub(super) fn floor_div(a: &[i64], b: &[i64]) -> Result<BigInt> {
    let (quotient, remainder) = div_rem(a, b)?;
    if !remainder.is_zero() && is_negative(a) != is_negative(b) {
        // With a nonzero remainder, (q-1)*b + (r+b) preserves a while rounding down.
        return Ok(sub(quotient.words(), &[1]));
    }
    Ok(quotient)
}

/// Remainder of floor division: zero or with the divisor's sign.
pub(super) fn floor_mod(a: &[i64], b: &[i64]) -> Result<BigInt> {
    let remainder = trunc_mod(a, b)?;
    if !remainder.is_zero() && is_negative(a) != is_negative(b) {
        // For a scalar divisor, |r| < |b| and opposite signs make r+b safe and inline.
        if let ([divisor], Some(r)) = (b, remainder.to_i64()) {
            return Ok(BigInt::from_i64(r + divisor));
        }
        // The remainder paired with the floor quotient q-1 is r+b.
        return Ok(add(remainder.words(), b));
    }
    Ok(remainder)
}

/// Convert to the nearest double, ties to even; `OverflowError` beyond the finite range.
pub(super) fn to_f64(x: &[i64]) -> Result<f64> {
    let first = first_nonzero(x);
    let mut size = x.len();
    while size > 0 && abs_word(x, first, size - 1) == 0 {
        size -= 1;
    }
    if size == 0 {
        return Ok(0.0);
    }
    let does_not_fit = || overflow("BigInt does not fit finite double");
    if size > 16 {
        return Err(does_not_fit());
    }
    let high = abs_word(x, first, size - 1) as u64;
    let bits = (size - 1) * 64 + (64 - high.leading_zeros() as usize);
    let shift = bits.saturating_sub(53);
    let bit = |i: usize| ((abs_word(x, first, i / 64) as u64) >> (i % 64)) & 1;
    let mut significand = 0u64;
    for i in (shift..bits).rev() {
        significand = (significand << 1) | bit(i);
    }
    if shift > 0 {
        let sticky = (0..shift - 1).any(|i| bit(i) != 0);
        if bit(shift - 1) == 1 && (sticky || significand & 1 == 1) {
            significand += 1;
        }
    }
    if bits == 1024 && significand == 1 << 53 {
        return Err(does_not_fit());
    }
    let result = significand as f64 * 2f64.powi(shift as i32);
    Ok(if is_negative(x) { -result } else { result })
}

/// Multiply magnitude words by `factor` and add `addend`, growing as needed.
pub(super) fn mul_add_small(magnitude: &mut Vec<u64>, factor: u64, addend: u64) {
    let mut carry = u128::from(addend);
    for word in magnitude.iter_mut() {
        let product = u128::from(*word) * u128::from(factor) + carry;
        *word = product as u64;
        carry = product >> 64;
    }
    if carry != 0 {
        magnitude.push(carry as u64);
    }
}
