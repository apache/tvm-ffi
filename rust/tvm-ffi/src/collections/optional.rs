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
//! - POD scalar (`i32`, `f64`, `bool`, …) → [`Optional<T>`](Optional)
//! - `String` → [`OptionalStr`]
//! - `ObjectRef` subtype → plain `Option<SomeRef>` (a single nullable pointer,
//!   `nullptr` == `None`; no dedicated type needed)
//!
//! # `Optional<T>` — POD scalars
//! Mirrors the `std::optional<T>` fallback as `#[repr(C)] { value: T, engaged:
//! bool }` (payload at offset 0, flag at `size_of::<T>()`), byte-verified against
//! libstdc++/libc++. `T` must implement [`OptionalPod`] — the marker trait
//! carried by the fixed set of fixed-width scalars. Read with [`get`](Optional::get),
//! write with [`set`](Optional::set); `set` takes `&self` via interior mutability,
//! so a shared `&Optional<T>` aliasing a C++ field stays writable (hence `!Sync`).
//!
//! # `OptionalStr` — `String`
//! The C++ `String` specialization keeps the 16-byte string cell inline and marks
//! `nullopt` with the `type_index == kTVMFFINone` sentinel; [`OptionalStr`] wraps
//! [`String`] the same way and reuses its refcounting `Clone`/`Drop`. Borrow with
//! [`as_str`](OptionalStr::as_str), write with [`set`](OptionalStr::set) — which
//! takes `&mut self`, since a shared-ref setter could drop the backing string
//! under a live `&str`. (`ffi::Optional<Bytes>` would follow the same pattern.)

use crate::String;
use std::cell::UnsafeCell;
use std::fmt::{self, Debug};
use std::mem::MaybeUninit;

//-----------------------------------------------------
// Optional<T> — POD scalars
//-----------------------------------------------------

/// Marker for a POD scalar `T` that can back an [`Optional<T>`]; see the
/// [module docs](self).
///
/// Unsafe: an implementor guarantees `T` is trivially copyable and its Rust
/// representation is byte-identical to the C++ field type (`i32` ↔ `int32_t`,
/// `f64` ↔ `double`, …), so the mirror can overlay the C++ `std::optional<T>`.
pub unsafe trait OptionalPod: Copy {}

/// Layout-mirror of `std::optional<T>`: `{ T value @0; bool engaged @sizeof(T) }`.
#[repr(C)]
struct OptionalCell<T: OptionalPod> {
    value: MaybeUninit<T>,
    engaged: bool,
}

/// In-place mirror of C++ `ffi::Optional<T>` for POD `T`.
///
/// Layout-compatible with the C++ type; see the [module docs](self).
#[repr(transparent)]
pub struct Optional<T: OptionalPod> {
    cell: UnsafeCell<OptionalCell<T>>,
}

impl<T: OptionalPod> Optional<T> {
    /// Builds an engaged optional holding `value`.
    #[inline]
    pub fn some(value: T) -> Self {
        // Only payload+flag are written; padding isn't part of the ABI.
        Self {
            cell: UnsafeCell::new(OptionalCell {
                value: MaybeUninit::new(value),
                engaged: true,
            }),
        }
    }

    /// Builds a disengaged optional (`nullopt`).
    #[inline]
    pub fn none() -> Self {
        // Zeroed (not `uninit`) payload keeps the byte-image tests reading init bytes.
        Self {
            cell: UnsafeCell::new(OptionalCell {
                value: MaybeUninit::zeroed(),
                engaged: false,
            }),
        }
    }

    /// Decodes the value in place. No FFI call, no allocation.
    #[inline]
    pub fn get(&self) -> Option<T> {
        // Read the payload only after confirming `engaged` (the cell is always initialized).
        let cell = unsafe { &*self.cell.get() };
        if cell.engaged {
            Some(unsafe { cell.value.assume_init() })
        } else {
            None
        }
    }

    /// Returns whether a value is present.
    #[inline]
    pub fn has_value(&self) -> bool {
        unsafe { (*self.cell.get()).engaged }
    }

    /// Returns whether the optional is `nullopt`.
    #[inline]
    pub fn is_none(&self) -> bool {
        !self.has_value()
    }

    /// Overwrites the value in place through a shared reference.
    ///
    /// Mirrors C++ assignment: `Some(v)` engages and stores `v`; `None`
    /// disengages without touching the payload bytes, as `std::optional::reset`
    /// does for trivial `T`.
    #[inline]
    pub fn set(&self, value: Option<T>) {
        // Interior mutation via `UnsafeCell`; caller must not race (`!Sync`).
        let cell = unsafe { &mut *self.cell.get() };
        match value {
            Some(v) => {
                cell.value = MaybeUninit::new(v);
                cell.engaged = true;
            }
            None => cell.engaged = false,
        }
    }
}

impl<T: OptionalPod> Default for Optional<T> {
    /// `nullopt`, matching the C++ default constructor.
    #[inline]
    fn default() -> Self {
        Self::none()
    }
}

impl<T: OptionalPod> Clone for Optional<T> {
    #[inline]
    fn clone(&self) -> Self {
        match self.get() {
            Some(v) => Self::some(v),
            None => Self::none(),
        }
    }
}

impl<T: OptionalPod> From<Option<T>> for Optional<T> {
    #[inline]
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => Self::some(v),
            None => Self::none(),
        }
    }
}

impl<T: OptionalPod> From<Optional<T>> for Option<T> {
    #[inline]
    fn from(value: Optional<T>) -> Self {
        value.get()
    }
}

impl<T: OptionalPod + Debug> Debug for Optional<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "Optional::Some({v:?})"),
            None => f.write_str("Optional::None"),
        }
    }
}

// Registers each supported scalar from one list: the `OptionalPod` impl and a
// compile-time guard that `Optional<T>` matches the `std::optional<T>` footprint
// (`size == round_up(size_of::<T>()+1, align)`). One list keeps the impl and its
// layout check from drifting.
macro_rules! impl_optional_pod {
    ($($t:ty),* $(,)?) => { $(
        // Fixed-width scalar; repr matches the C++ field's `std::optional`
        // fallback (layout proven by the `const` block below).
        unsafe impl OptionalPod for $t {}
        const _: () = {
            let tsz = core::mem::size_of::<$t>();
            let tal = core::mem::align_of::<$t>();
            let expect = (tsz + 1).div_ceil(tal) * tal;
            assert!(core::mem::align_of::<Optional<$t>>() == tal);
            assert!(core::mem::size_of::<Optional<$t>>() == expect);
        };
    )* };
}
impl_optional_pod!(bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

//-----------------------------------------------------
// OptionalStr — String
//-----------------------------------------------------

/// In-place mirror of C++ `ffi::Optional<String>`: the 16-byte string cell
/// itself, with `type_index == kTVMFFINone` meaning `nullopt` (the C++
/// String/Bytes spec stores the sentinel in-cell, not as a separate flag).
/// Reuses [`String`]'s `Clone`/`Drop`, whose refcounting is a no-op on the
/// `nullopt` cell (`type_index` below `kTVMFFIStaticObjectBegin`).
#[repr(transparent)]
#[derive(Clone)]
pub struct OptionalStr {
    // Never handed out or accessed while disengaged (a `nullopt` cell is not a
    // valid string).
    inner: String,
}

// Must stay 16 bytes to overlay C++ `ffi::Optional<String>` (parity with the POD
// guard in `impl_optional_pod!`).
const _: () = assert!(std::mem::size_of::<OptionalStr>() == 16);

impl OptionalStr {
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
        let OptionalStr { inner } = self; // no `Drop` impl, so the move is allowed
        if inner.is_none_cell() {
            None
        } else {
            Some(inner)
        }
    }

    /// Overwrites the value in place, dropping the previous one first (dec-ref'd
    /// if it was a heap string).
    ///
    /// `&mut self`, not `&self` like POD [`Optional::set`]: `as_str` hands out
    /// borrows into a refcounted cell, so a shared-ref setter could drop the
    /// backing string under a live `&str`.
    #[inline]
    pub fn set(&mut self, value: Option<String>) {
        // Assignment drops the old `String` (dec_ref if heap) before moving in the new.
        self.inner = match value {
            Some(s) => s,
            None => String::none_cell(),
        };
    }
}

impl Default for OptionalStr {
    /// `nullopt`, matching the C++ default constructor.
    #[inline]
    fn default() -> Self {
        Self::none()
    }
}

impl From<Option<String>> for OptionalStr {
    #[inline]
    fn from(value: Option<String>) -> Self {
        match value {
            Some(s) => Self::some(s),
            None => Self::none(),
        }
    }
}

impl From<OptionalStr> for Option<String> {
    #[inline]
    fn from(value: OptionalStr) -> Self {
        value.get()
    }
}

impl Debug for OptionalStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_str() {
            Some(s) => write!(f, "OptionalStr::Some({s:?})"),
            None => f.write_str("OptionalStr::None"),
        }
    }
}
