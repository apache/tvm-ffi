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
//! In-place mirror of C++ `ffi::Optional<T>` for non-object-pointer `T`
//! (`include/tvm/ffi/optional.h`).
//!
//! C++ `ffi::Optional<T>` is backed by one 16-byte `TVMFFIAny` for scalar,
//! string, and other non-object-pointer values, with
//! `type_index == kTVMFFINone` meaning `nullopt`. [`Optional<T>`] mirrors that
//! representation.
//!
//! ObjectRef, ObjectPtr, and Arc cases are intentionally excluded. Their C++
//! optional is one nullable object pointer, which Rust's niche-optimized
//! [`std::option::Option<X>`] already mirrors. Using `Optional<X>` for an object
//! class is rejected at compile time so the two ABI layouts cannot be confused.
//!
//! `Optional<T>` is `#[repr(transparent)]` over [`Any`] and decodes the cell in
//! place. It is named `Optional` (not `Option`) to match the C++ type while
//! keeping the pointer-backed object case visually distinct.

use crate::any::Any;
use crate::string::{Bytes, String};
use crate::type_traits::AnyCompatible;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

/// Marker for values whose C++ `ffi::Optional<T>` uses the 16-byte
/// `TVMFFIAny` representation.
///
/// Object classes deliberately do not implement this trait. Use `Option<X>`
/// for those values; it is the pointer-sized mirror of C++'s nullable
/// `ObjectPtr`-backed optional.
///
/// ```compile_fail,E0277
/// use tvm_ffi::{Array, Optional};
/// let _ = Optional::<Array<i64>>::none();
/// ```
#[diagnostic::on_unimplemented(
    message = "`Optional<{Self}>` only mirrors non-object-pointer `ffi::Optional` values",
    label = "`{Self}` uses the object-pointer optional representation",
    note = "use `Option<{Self}>` for object classes; it is the compatible nullable-pointer layout"
)]
pub unsafe trait OptionalCompatible: AnyCompatible {}

macro_rules! impl_optional_compatible {
    ($($t:ty),* $(,)?) => {
        $(unsafe impl OptionalCompatible for $t {})*
    };
}

impl_optional_compatible!(
    bool,
    i8,
    i16,
    i32,
    i64,
    isize,
    u8,
    u16,
    u32,
    u64,
    usize,
    f32,
    f64,
    *mut core::ffi::c_void,
    crate::DLDataType,
    crate::DLDevice,
    String,
    Bytes,
);

unsafe impl<T: OptionalCompatible> OptionalCompatible for Option<T> {}

/// In-place mirror of C++ `ffi::Optional<T>`: a single 16-byte `TVMFFIAny` cell
/// (wrapped as [`Any`]) whose `type_index == kTVMFFINone` encodes `nullopt`.
///
/// Layout-compatible with the C++ type (`size_of == 16`); see the [module
/// docs](self). Reuses [`Any`]'s reference-counting `Clone`/`Drop`, which are a
/// no-op on the `nullopt` cell (`type_index` below `kTVMFFIStaticObjectBegin`).
#[repr(transparent)]
pub struct Optional<T: OptionalCompatible> {
    // Holds either the value's `TVMFFIAny` representation or a `kTVMFFINone` cell.
    data: Any,
    _marker: PhantomData<T>,
}

// Must stay 16 bytes / `TVMFFIAny`-aligned to overlay a C++ `ffi::Optional<T>`
// field in place, independent of `T`.
const _: () = assert!(
    std::mem::size_of::<Optional<i64>>() == 16
        && std::mem::size_of::<Optional<String>>() == 16
        && std::mem::align_of::<Optional<i64>>() == std::mem::align_of::<crate::TVMFFIAny>()
);

impl<T: OptionalCompatible> Optional<T> {
    /// An engaged optional holding `value`.
    #[inline]
    pub fn some(value: T) -> Self {
        Self {
            data: Any::from(value),
            _marker: PhantomData,
        }
    }

    /// A disengaged optional (`nullopt`, a `kTVMFFINone` cell).
    #[inline]
    pub fn none() -> Self {
        Self {
            data: Any::new(),
            _marker: PhantomData,
        }
    }

    /// Whether a value is present.
    #[inline]
    pub fn has_value(&self) -> bool {
        self.data.type_index() != TypeIndex::kTVMFFINone as i32
    }

    /// Whether the optional is `nullopt`.
    #[inline]
    pub fn is_none(&self) -> bool {
        !self.has_value()
    }

    /// Decodes the value in place, cloning it out (ref-counted `inc_ref` for
    /// object payloads). Returns `None` when `nullopt`. No FFI call.
    #[inline]
    pub fn get(&self) -> Option<T> {
        self.data.try_as::<T>()
    }

    /// Takes the value out, consuming self (moves the payload, no `inc_ref`).
    #[inline]
    pub fn into_option(self) -> Option<T> {
        if self.has_value() {
            // Move the value out of the owning `Any` without dropping it, then
            // transfer ownership of the payload into `T` (no inc/dec ref).
            let mut raw = unsafe { Any::into_raw_ffi_any(self.data) };
            Some(unsafe { T::move_from_any_after_check(&mut raw) })
        } else {
            None
        }
    }

    /// Overwrites the value in place, dropping the previous one first (dec-ref'd
    /// if it was an object payload).
    #[inline]
    pub fn set(&mut self, value: Option<T>) {
        // Assignment drops the old `Any` (dec_ref if object) before storing the new.
        self.data = match value {
            Some(v) => Any::from(v),
            None => Any::new(),
        };
    }
}

impl Optional<String> {
    /// Borrows the engaged string as `&str`, or `None` when `nullopt`.
    ///
    /// Reinterprets the in-cell string without cloning: an engaged cell holds a
    /// `String`'s exact 16-byte representation, so `&Any` can be viewed as
    /// `&String`.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        if self.has_value() {
            // SAFETY: an engaged `Optional<String>` cell is byte-identical to a
            // `String` (both are `#[repr(transparent)]` over the 16-byte cell),
            // and the borrow is tied to `&self`.
            let s = unsafe { &*(&self.data as *const Any as *const String) };
            Some(s.as_str())
        } else {
            None
        }
    }
}

impl<T: OptionalCompatible> Default for Optional<T> {
    /// `nullopt`, matching the C++ default constructor.
    #[inline]
    fn default() -> Self {
        Self::none()
    }
}

impl<T: OptionalCompatible> Clone for Optional<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            // `Any::clone` inc_refs an object payload; a `nullopt` cell is a no-op.
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: OptionalCompatible + PartialEq> PartialEq for Optional<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl<T: OptionalCompatible + Eq> Eq for Optional<T> {}

impl<T: OptionalCompatible> From<Option<T>> for Optional<T> {
    #[inline]
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => Self::some(v),
            None => Self::none(),
        }
    }
}

impl<T: OptionalCompatible> From<Optional<T>> for Option<T> {
    #[inline]
    fn from(value: Optional<T>) -> Self {
        value.into_option()
    }
}

impl<T: OptionalCompatible + Debug> Debug for Optional<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            Some(v) => write!(f, "Optional::Some({v:?})"),
            None => f.write_str("Optional::None"),
        }
    }
}
