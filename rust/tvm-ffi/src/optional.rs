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
//! [`Optional<T>`], an optional value that is not an object, such as a scalar
//! or a string, stored in place.
//!
//! It is one 16-byte `TVMFFIAny` cell holding the value or `None`, the ABI
//! layout of an optional non-object value in every language. An optional
//! object is one nullable pointer, which [`std::option::Option<X>`] already
//! is, so `Optional<X>` of an object type is rejected at compile time.
//!
//! A reflected field holding an optional non-object value is an `Optional<T>`;
//! an `Option<T>` there is rejected at registration, since it is laid out as
//! neither form.

use crate::any::Any;
use crate::string::{Bytes, String};
use crate::type_traits::AnyCompatible;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use tvm_ffi_sys::TVMFFIAny;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

/// Marker for the types whose optional the ABI stores as one 16-byte
/// `TVMFFIAny` cell, the types [`Optional<T>`] accepts.
///
/// Object types deliberately do not implement it: use `Option<X>`, one
/// nullable pointer, for those.
///
/// # Safety
/// Implement only for types whose optional the ABI stores as a `TVMFFIAny`
/// cell; for an object type it would mismatch the ABI's nullable pointer.
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
    crate::big_int::BigInt,
);

/// An optional value that is not an object, stored in place as one 16-byte
/// cell holding the value or `None`. See the [module docs](self).
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
    /// An optional holding `value`.
    #[inline]
    pub fn some(value: T) -> Self {
        Self {
            data: Any::from(value),
            _marker: PhantomData,
        }
    }

    /// An empty optional, holding `None`.
    #[inline]
    pub fn none() -> Self {
        Self {
            data: Any::new(),
            _marker: PhantomData,
        }
    }

    /// Whether a value is present.
    #[inline]
    pub fn is_some(&self) -> bool {
        self.data.type_index() != TypeIndex::kTVMFFINone as i32
    }

    /// Whether the optional is empty.
    #[inline]
    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    /// Returns a clone of the value, or `None` when empty.
    #[inline]
    pub fn get(&self) -> Option<T> {
        // Strict: the value comes back exactly as stored, never converted.
        let cell = self.data.as_raw_ffi_any();
        // SAFETY: the cell is valid while `self` is borrowed, and the copy
        // runs only after the strict check.
        unsafe { T::check_any_strict(cell).then(|| T::copy_from_any_view_after_check(cell)) }
    }

    /// Takes the value out, without cloning it.
    #[inline]
    pub fn into_option(self) -> Option<T> {
        if self.is_some() {
            // Move the value out of the owning `Any` without dropping it, then
            // transfer ownership of the payload into `T` (no inc/dec ref).
            let mut raw = Any::into_raw_ffi_any(self.data);
            Some(unsafe { T::move_from_any_after_check(&mut raw) })
        } else {
            None
        }
    }

    /// Replaces the value, dropping the previous one.
    #[inline]
    pub fn set(&mut self, value: Option<T>) {
        // Assignment drops the old `Any` (dec_ref if object) before storing the new.
        self.data = match value {
            Some(v) => Any::from(v),
            None => Any::new(),
        };
    }
}

impl<T: OptionalCompatible + crate::type_traits::RefFromAnyUnchecked> Optional<T> {
    /// Borrow the value in place, like [`Option::as_ref`].
    ///
    /// Available for values the cell stores as themselves, such as `String`,
    /// `i64` and `f64`; use [`get`](Self::get) for the others.
    ///
    /// # Returns
    ///
    /// The borrowed value, or `None` when empty.
    ///
    /// ```
    /// use tvm_ffi::Optional;
    /// let value = Optional::some(7_i64);
    /// assert_eq!(value.as_ref(), Some(&7));
    /// assert_eq!(Optional::<i64>::none().as_ref(), None);
    /// ```
    #[inline]
    pub fn as_ref(&self) -> Option<&T> {
        if self.is_some() {
            // SAFETY: an engaged `Optional<T>` cell holds a `T`.
            Some(unsafe { T::ref_from_any_unchecked(self.data.as_raw_ffi_any()) })
        } else {
            None
        }
    }
}

impl Optional<String> {
    /// Borrows the string as `&str`, or `None` when empty.
    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        if self.is_some() {
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

// Converts as `Option<T>` does; the cell itself is the `Any` form, which makes
// `Optional<T>` a reflected field laid out as C++ `ffi::Optional<T>`.
unsafe impl<T: OptionalCompatible> AnyCompatible for Optional<T> {
    fn type_str() -> std::string::String {
        Option::<T>::type_str()
    }

    fn type_schema() -> std::string::String {
        Option::<T>::type_schema()
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        *data = *src.data.as_raw_ffi_any();
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        *data = Any::into_raw_ffi_any(src.data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        Option::<T>::check_any_strict(data)
    }

    unsafe fn match_constraints(data: &TVMFFIAny) -> bool {
        Option::<T>::match_constraints(data)
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        Option::<T>::copy_from_any_view_after_check(data).into()
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Option::<T>::move_from_any_after_check(data).into()
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        Option::<T>::try_cast_from_any_view(data).map(Self::from)
    }
}

impl<T: OptionalCompatible> Default for Optional<T> {
    /// The empty optional, holding `None`.
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
