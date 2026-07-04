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
//! In-place mirrors of C++ `ffi::Optional<T>` (`include/tvm/ffi/optional.h`).
//!
//! `ffi::Optional<T>` stores its value inline, in one of three ABI layouts
//! depending on `T`. The types here decode such a field's bytes directly — no
//! FFI call, no allocation, no reflection getter/setter. Pick the counterpart for
//! the field's `T`:
//!
//! - POD scalar (`i32`, `f64`, `bool`, …) → [`OptionPod<T>`](OptionPod)
//! - `String` → [`OptionStr`]
//! - `ObjectRef` subtype → [`OptionObjRef<T>`](OptionObjRef), an alias of plain
//!   `Option<T>` (a single nullable pointer, `nullptr` == `None`)
//!
//! # `OptionPod<T>` — POD scalars
//! Mirrors the `std::optional<T>` fallback as `#[repr(C)] { value: T, engaged:
//! bool }` (payload at offset 0, flag at `size_of::<T>()`), byte-verified against
//! libstdc++/libc++. `T` must implement [`OptionalCompatiblePod`] — the marker trait
//! carried by the fixed set of fixed-width scalars. Read with [`get`](OptionPod::get),
//! write with [`set`](OptionPod::set).
//!
//! # `OptionStr` — `String`
//! The C++ `String` specialization keeps the 16-byte string cell inline and marks
//! `nullopt` with the `type_index == kTVMFFINone` sentinel; [`OptionStr`] wraps
//! [`String`] the same way and reuses its refcounting `Clone`/`Drop`. Borrow with
//! [`as_str`](OptionStr::as_str), write with [`set`](OptionStr::set).
//! (`ffi::Optional<Bytes>` would follow the same pattern.)

use crate::String;
use std::fmt::{self, Debug};
use std::mem::MaybeUninit;

//-----------------------------------------------------
// OptionPod<T> — POD scalars
//-----------------------------------------------------

/// Marker for a POD scalar `T` that can back an [`OptionPod<T>`]; see the
/// [module docs](self).
///
/// Unsafe: an implementor guarantees `T` is trivially copyable and its Rust
/// representation is byte-identical to the C++ field type (`i32` ↔ `int32_t`,
/// `f64` ↔ `double`, …), so the mirror can overlay the C++ `std::optional<T>`.
///
/// Non-scalar payloads are rejected at compile time with a pointer to the
/// right counterpart:
///
/// ```compile_fail,E0277
/// // `Array` is an `ObjectRef` subtype → use `Option<Array<i64>>` (`OptionObjRef`).
/// let _ = tvm_ffi::option::OptionPod::<tvm_ffi::Array<i64>>::none();
/// ```
#[diagnostic::on_unimplemented(
    message = "`OptionPod<{Self}>` only mirrors `ffi::Optional` of fixed-width POD scalars",
    label = "`{Self}` is not a fixed-width POD scalar",
    note = "for an `ObjectRef` subtype use plain `Option<{Self}>` (alias `tvm_ffi::option::OptionObjRef`): the C++ side is a single nullable pointer",
    note = "for `String` use `tvm_ffi::option::OptionStr`"
)]
pub unsafe trait OptionalCompatiblePod: Copy {}

/// In-place mirror of C++ `ffi::Optional<T>` for POD `T`, laid out as
/// `std::optional<T>`: `{ T value @0; bool engaged @sizeof(T) }`.
///
/// Layout-compatible with the C++ type; see the [module docs](self).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OptionPod<T: OptionalCompatiblePod> {
    value: MaybeUninit<T>,
    engaged: bool,
}

impl<T: OptionalCompatiblePod> OptionPod<T> {
    /// Builds an engaged optional holding `value`.
    #[inline]
    pub fn some(value: T) -> Self {
        // Only payload+flag are written; padding isn't part of the ABI.
        Self {
            value: MaybeUninit::new(value),
            engaged: true,
        }
    }

    /// Builds a disengaged optional (`nullopt`).
    #[inline]
    pub fn none() -> Self {
        // Zeroed (not `uninit`) payload keeps the byte-image tests reading init bytes.
        Self {
            value: MaybeUninit::zeroed(),
            engaged: false,
        }
    }

    /// Decodes the value in place. No FFI call, no allocation.
    #[inline]
    pub fn get(&self) -> Option<T> {
        if self.engaged {
            // The payload is written whenever `engaged` is set.
            Some(unsafe { self.value.assume_init() })
        } else {
            None
        }
    }

    /// Returns whether a value is present.
    #[inline]
    pub fn has_value(&self) -> bool {
        self.engaged
    }

    /// Returns whether the optional is `nullopt`.
    #[inline]
    pub fn is_none(&self) -> bool {
        !self.has_value()
    }

    /// Overwrites the value in place.
    ///
    /// Mirrors C++ assignment: `Some(v)` engages and stores `v`; `None`
    /// disengages without touching the payload bytes, as `std::optional::reset`
    /// does for trivial `T`.
    #[inline]
    pub fn set(&mut self, value: Option<T>) {
        match value {
            Some(v) => {
                self.value = MaybeUninit::new(v);
                self.engaged = true;
            }
            None => self.engaged = false,
        }
    }
}

impl<T: OptionalCompatiblePod> Default for OptionPod<T> {
    /// `nullopt`, matching the C++ default constructor.
    #[inline]
    fn default() -> Self {
        Self::none()
    }
}

impl<T: OptionalCompatiblePod + PartialEq> PartialEq for OptionPod<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: OptionalCompatiblePod + Eq> Eq for OptionPod<T> {}

impl<T: OptionalCompatiblePod> From<Option<T>> for OptionPod<T> {
    #[inline]
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => Self::some(v),
            None => Self::none(),
        }
    }
}

impl<T: OptionalCompatiblePod> From<OptionPod<T>> for Option<T> {
    #[inline]
    fn from(value: OptionPod<T>) -> Self {
        value.get()
    }
}

impl<T: OptionalCompatiblePod + Debug> Debug for OptionPod<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "OptionPod::Some({v:?})"),
            None => f.write_str("OptionPod::None"),
        }
    }
}

// Registers each supported scalar from one list: the `OptionalCompatiblePod` impl and a
// compile-time guard that `OptionPod<T>` matches the `std::optional<T>` footprint
// (`size == round_up(size_of::<T>()+1, align)`). One list keeps the impl and its
// layout check from drifting.
macro_rules! impl_optional_compatible_pod {
    ($($t:ty),* $(,)?) => { $(
        // Fixed-width scalar; repr matches the C++ field's `std::optional`
        // fallback (layout proven by the `const` block below).
        unsafe impl OptionalCompatiblePod for $t {}
        const _: () = {
            let tsz = core::mem::size_of::<$t>();
            let tal = core::mem::align_of::<$t>();
            let expect = (tsz + 1).div_ceil(tal) * tal;
            assert!(core::mem::align_of::<OptionPod<$t>>() == tal);
            assert!(core::mem::size_of::<OptionPod<$t>>() == expect);
        };
    )* };
}
// Keep in sync with the static_assert guard at the end of include/tvm/ffi/optional.h.
impl_optional_compatible_pod!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

//-----------------------------------------------------
// OptionStr — String
//-----------------------------------------------------

/// In-place mirror of C++ `ffi::Optional<String>`: the 16-byte string cell
/// itself, with `type_index == kTVMFFINone` meaning `nullopt` (the C++
/// String/Bytes spec stores the sentinel in-cell, not as a separate flag).
/// Reuses [`String`]'s `Clone`/`Drop`, whose refcounting is a no-op on the
/// `nullopt` cell (`type_index` below `kTVMFFIStaticObjectBegin`).
#[repr(transparent)]
#[derive(Clone)]
pub struct OptionStr {
    // Never handed out or accessed while disengaged (a `nullopt` cell is not a
    // valid string).
    inner: String,
}

// Must stay 16 bytes to overlay C++ `ffi::Optional<String>` (parity with the POD
// guard in `impl_optional_compatible_pod!`).
const _: () = assert!(
    std::mem::size_of::<OptionStr>() == 16
        && std::mem::align_of::<OptionStr>() == std::mem::align_of::<crate::TVMFFIAny>()
);

impl OptionStr {
    /// An engaged optional holding `value`.
    #[inline]
    pub fn some(value: String) -> Self {
        Self { inner: value }
    }

    /// A disengaged optional (`nullopt`).
    #[inline]
    pub fn none() -> Self {
        Self {
            inner: String::none_cell(),
        }
    }

    /// Whether a value is present.
    #[inline]
    pub fn has_value(&self) -> bool {
        !self.inner.is_none_cell()
    }

    /// Whether the optional is `nullopt`.
    #[inline]
    pub fn is_none(&self) -> bool {
        self.inner.is_none_cell()
    }

    /// Borrows the engaged string as `&str`, or `None` when `nullopt`.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        if self.has_value() {
            Some(self.inner.as_str())
        } else {
            None
        }
    }

    /// Takes the value out, consuming self.
    #[inline]
    pub fn get(self) -> Option<String> {
        let OptionStr { inner } = self; // no `Drop` impl, so the move is allowed
        if inner.is_none_cell() {
            None
        } else {
            Some(inner)
        }
    }

    /// Overwrites the value in place, dropping the previous one first (dec-ref'd
    /// if it was a heap string).
    #[inline]
    pub fn set(&mut self, value: Option<String>) {
        // Assignment drops the old `String` (dec_ref if heap) before moving in the new.
        self.inner = match value {
            Some(s) => s,
            None => String::none_cell(),
        };
    }
}

impl Default for OptionStr {
    /// `nullopt`, matching the C++ default constructor.
    #[inline]
    fn default() -> Self {
        Self::none()
    }
}

impl From<Option<String>> for OptionStr {
    #[inline]
    fn from(value: Option<String>) -> Self {
        match value {
            Some(s) => Self::some(s),
            None => Self::none(),
        }
    }
}

impl From<OptionStr> for Option<String> {
    #[inline]
    fn from(value: OptionStr) -> Self {
        value.get()
    }
}

impl PartialEq for OptionStr {
    // NOT derivable: derived eq would run `String::eq` on a `nullopt` cell and
    // dereference its null object pointer; `as_str` checks the sentinel first.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for OptionStr {}

impl Debug for OptionStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "OptionStr::Some({s:?})"),
            None => f.write_str("OptionStr::None"),
        }
    }
}

//-----------------------------------------------------
// OptionObjRef — ObjectRef subtypes
//-----------------------------------------------------

/// Alias of [`Option`] for `ObjectRef`-subtype fields.
///
/// C++ `ffi::Optional<SomeRef>` is a single nullable pointer, so `Option<SomeRef>`
/// (niche-optimized over the ref's non-null [`ObjectArc`](crate::ObjectArc)
/// pointer) already mirrors it in place: `None` == `nullptr`. The alias only
/// names that contract for consistency.
pub type OptionObjRef<T> = Option<T>;
