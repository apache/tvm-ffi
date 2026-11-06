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

//! Typed unchanged-or-replacement results for structural mutation.

use std::marker::PhantomData;

use super::structural_common::{raw_ffi_any_same_as, structural_error_from_raw};

use crate::any::{Any, AnyView, TryFromTemp};
use crate::error::{Error, Result, TYPE_ERROR};
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use crate::type_traits::{AnyCompatible, ContainerElement};

/// A marker result that keeps the original value.
///
/// A map or mutation callback returns `Unchanged`, usually as
/// `Unchanged.into()`, to keep the value it was given without retaining or
/// cloning it. A pre-order map still descends into the original value's
/// children.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{Any, Unchanged, UnchangedOr};
///
/// let result: UnchangedOr<Any> = Unchanged.into();
/// assert!(result.is_unchanged());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unchanged;

/// Either a replacement value of type `T`, or [`Unchanged`] to keep the
/// original.
///
/// Structural map and mutation callbacks return `Result<UnchangedOr<Any>>`.
/// It is like `Option<T>` whose `None` means "keep what was there", stored in
/// one FFI value. `UnchangedOr<Any>` converts with `.into()` from
/// [`Unchanged`], from a managed reference or scalar, and from a typed
/// `UnchangedOr<T>`. `T` is an FFI value type or [`Any`]. Converting the
/// wrapper into `Any` keeps the marker; use [`value_or_unchanged`](Self::value_or_unchanged) or
/// [`into_option`](Self::into_option) when an actual value is needed.
///
/// # Examples
///
/// ```
/// use tvm_ffi::UnchangedOr;
///
/// fn double_even(value: i64) -> UnchangedOr<i64> {
///     if value % 2 == 0 {
///         UnchangedOr::changed(value * 2)
///     } else {
///         UnchangedOr::unchanged()
///     }
/// }
///
/// assert_eq!(double_even(4).value_or_unchanged(4), 8);
/// assert_eq!(double_even(3).value_or_unchanged(3), 3);
/// ```
#[repr(transparent)]
pub struct UnchangedOr<T: ContainerElement = Any> {
    data: Any,
    _marker: PhantomData<T>,
}

impl<T: ContainerElement> UnchangedOr<T> {
    /// Create a result that keeps the original value, without borrowing or cloning
    /// it.
    ///
    /// # Returns
    ///
    /// An unchanged result.
    #[inline]
    pub fn unchanged() -> Self {
        Self {
            data: Unchanged.into(),
            _marker: PhantomData,
        }
    }

    /// Create a result that replaces the original value.
    ///
    /// # Arguments
    ///
    /// * `value` - The replacement.
    ///
    /// # Returns
    ///
    /// A result holding `value`.
    #[inline]
    pub fn changed(value: T) -> Self {
        let mut raw = TVMFFIAny::new();
        // SAFETY: ContainerElement transfers exactly one owning value.
        unsafe {
            T::container_move_to_any(value, &mut raw);
            Self {
                data: Any::from_raw_ffi_any(raw),
                _marker: PhantomData,
            }
        }
    }

    /// Return whether this result keeps the original value.
    ///
    /// # Returns
    ///
    /// `true` if the result is unchanged.
    #[inline]
    pub fn is_unchanged(&self) -> bool {
        is_unchanged(&self.data)
    }

    /// Return whether this result is unchanged or holds `original` itself.
    ///
    /// An object counts as the same only when it is the same object, not an equal
    /// copy. Use it to tell whether anything was rewritten.
    ///
    /// # Arguments
    ///
    /// * `original` - The original value.
    ///
    /// # Returns
    ///
    /// `true` if the result is unchanged or holds `original`.
    ///
    /// ```
    /// use tvm_ffi::{Array, UnchangedOr};
    /// let original = Array::from_vec(vec![1_i64]);
    /// assert!(UnchangedOr::changed(original.clone()).unchanged_or_same_as(&original));
    /// ```
    ///
    /// See also [`UnchangedOr`].
    #[inline]
    pub fn unchanged_or_same_as(&self, original: &T) -> bool {
        if self.is_unchanged() {
            return true;
        }
        let mut raw = TVMFFIAny::new();
        // SAFETY: the borrowed cell is used only while original is alive.
        unsafe {
            T::container_copy_to_any_view(original, &mut raw);
        }
        raw_ffi_any_same_as(raw, *self.data.as_raw_ffi_any())
    }

    /// Move out the replacement.
    ///
    /// # Returns
    ///
    /// The replacement, or `None` when the result is unchanged.
    #[inline]
    pub fn into_option(self) -> Option<T> {
        if self.is_unchanged() {
            return None;
        }
        // SAFETY: constructors and conversions maintain the declared T.
        unsafe {
            let mut raw = Any::into_raw_ffi_any(self.data);
            Some(T::container_move_from_any_after_check(&mut raw))
        }
    }

    /// Move out the replacement, or `original` when the result is unchanged.
    ///
    /// `original` is the value the callback was given; to produce it only when
    /// needed, use `into_option().unwrap_or_else(..)`.
    ///
    /// # Arguments
    ///
    /// * `original` - The original value, kept when the result is unchanged.
    ///
    /// # Returns
    ///
    /// The replacement, or `original`.
    #[inline]
    pub fn value_or_unchanged(self, original: T) -> T {
        self.into_option().unwrap_or(original)
    }

    /// Convert only the replacement, preserving an unchanged result.
    ///
    /// # Arguments
    ///
    /// * `convert` - Converts the replacement; called only when there is one.
    ///
    /// # Returns
    ///
    /// The converted result.
    #[inline]
    pub fn map<U: ContainerElement>(self, convert: impl FnOnce(T) -> U) -> UnchangedOr<U> {
        match self.into_option() {
            Some(value) => UnchangedOr::changed(convert(value)),
            None => UnchangedOr::unchanged(),
        }
    }

    /// Fallibly convert only the replacement, preserving an unchanged result.
    ///
    /// # Arguments
    ///
    /// * `convert` - Converts the replacement; called only when there is one.
    ///
    /// # Returns
    ///
    /// The converted result.
    ///
    /// # Errors
    ///
    /// Returns the error of `convert` unchanged.
    ///
    /// ```
    /// use tvm_ffi::UnchangedOr;
    /// let result = UnchangedOr::changed(3_i64).try_map(u32::try_from).unwrap();
    /// assert_eq!(result.into_option(), Some(3_u32));
    /// ```
    ///
    /// See also [`UnchangedOr`].
    #[inline]
    pub fn try_map<U: ContainerElement, E>(
        self,
        convert: impl FnOnce(T) -> std::result::Result<U, E>,
    ) -> std::result::Result<UnchangedOr<U>, E> {
        match self.into_option() {
            Some(value) => convert(value).map(UnchangedOr::changed),
            None => Ok(UnchangedOr::unchanged()),
        }
    }

    /// Check a replacement's FFI type without cloning it.
    ///
    /// An unchanged result is compatible with every replacement type. Erased and
    /// narrowing conversions retain the strict type check of
    /// [`Any::downcast_ref`]. For a
    /// known typed conversion, use `map(Into::into)`.
    ///
    /// # Returns
    ///
    /// The result typed as `UnchangedOr<U>`.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the replacement is not stored as `U`.
    ///
    /// ```
    /// use tvm_ffi::{Any, UnchangedOr};
    /// let erased: UnchangedOr<Any> = 7_i64.into();
    /// assert_eq!(erased.try_cast::<i64>()?.into_option(), Some(7));
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`UnchangedOr`].
    #[inline]
    pub fn try_cast<U: ContainerElement>(self) -> Result<UnchangedOr<U>> {
        if unsafe { UnchangedOr::<U>::check_any_strict(self.data.as_raw_ffi_any()) } {
            Ok(UnchangedOr {
                data: self.data,
                _marker: PhantomData,
            })
        } else {
            Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "structural mutation result does not match {}",
                    U::container_type_str()
                ),
                "",
            ))
        }
    }
}

impl UnchangedOr<Any> {
    /// Wrap a successful erased carrier, which may hold the unchanged marker.
    #[inline(always)]
    pub(crate) fn from_any(data: Any) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn from_carrier(data: Any) -> Result<Self> {
        if data.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
            core::hint::cold_path();
            // Transfer ownership directly into the checked cold conversion.
            return unsafe {
                let raw = Any::into_raw_ffi_any(data);
                Err(structural_error_from_raw(raw))
            };
        }
        Ok(Self {
            data,
            _marker: PhantomData,
        })
    }
}

impl<T: ContainerElement> Clone for UnchangedOr<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

impl<'b, T: ContainerElement> TryFrom<&'b Any> for UnchangedOr<T> {
    type Error = Error;
    #[inline]
    fn try_from(value: &'b Any) -> Result<Self> {
        TryFromTemp::<Self>::try_from(value).map(TryFromTemp::into_value)
    }
}

impl<T: ContainerElement> TryFrom<Any> for UnchangedOr<T> {
    type Error = Error;
    #[inline]
    fn try_from(value: Any) -> Result<Self> {
        TryFromTemp::<Self>::try_from(value).map(TryFromTemp::into_value)
    }
}

impl<'a, T: ContainerElement> TryFrom<AnyView<'a>> for UnchangedOr<T> {
    type Error = Error;
    #[inline]
    fn try_from(value: AnyView<'a>) -> Result<Self> {
        TryFromTemp::<Self>::try_from(value).map(TryFromTemp::into_value)
    }
}

#[inline]
pub(crate) fn is_unchanged(value: &Any) -> bool {
    value.type_index() == TVMFFITypeIndex::kTVMFFIUnchanged as i32
}

// -----------------------------------------------------------------------------
// AnyCompatible: low-level FFI carrier conversions.
// -----------------------------------------------------------------------------

// SAFETY: the unchanged ABI tag has no payload or owned resources.
unsafe impl AnyCompatible for Unchanged {
    unsafe fn copy_to_any_view(_src: &Self, data: &mut TVMFFIAny) {
        *data = TVMFFIAny::new();
        data.type_index = TVMFFITypeIndex::kTVMFFIUnchanged as i32;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        Self::copy_to_any_view(&src, data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TVMFFITypeIndex::kTVMFFIUnchanged as i32
    }

    unsafe fn copy_from_any_view_after_check(_data: &TVMFFIAny) -> Self {
        Self
    }

    unsafe fn move_from_any_after_check(_data: &mut TVMFFIAny) -> Self {
        Self
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        Self::check_any_strict(data).then_some(Self).ok_or(())
    }

    fn type_str() -> std::string::String {
        "Unchanged".into()
    }

    fn type_schema() -> std::string::String {
        crate::type_traits::type_schema("ffi.Unchanged", &[])
    }
}

crate::impl_try_from_any!(Unchanged);

// SAFETY: the carrier contains either the resource-free marker or exactly
// T's owning representation. Borrowing never acquires ownership; moving
// transfers the carrier once. Non-strict conversions materialize a valid T.
unsafe impl<T: ContainerElement> AnyCompatible for UnchangedOr<T> {
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        *data = *src.data.as_raw_ffi_any();
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        *data = Any::into_raw_ffi_any(src.data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        if T::CONTAINER_IS_ANY {
            // An erased successful result excludes the ABI's error channel.
            data.type_index != TVMFFITypeIndex::kTVMFFIError as i32
        } else {
            Unchanged::check_any_strict(data) || T::container_check_any_strict(data)
        }
    }

    unsafe fn match_constraints(data: &TVMFFIAny) -> bool {
        Unchanged::check_any_strict(data) || T::container_match_constraints(data)
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        if Unchanged::check_any_strict(data) {
            return Self::unchanged();
        }
        // Materialize T, including numeric narrowing, before storing its value.
        Self::changed(T::container_copy_from_any_view_after_check(data))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self {
            data: Any::from_raw_ffi_any(std::mem::replace(data, TVMFFIAny::new())),
            _marker: PhantomData,
        }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        if T::CONTAINER_IS_ANY && data.type_index == TVMFFITypeIndex::kTVMFFIError as i32 {
            return Err(());
        }
        if Unchanged::check_any_strict(data) {
            return Ok(Self::unchanged());
        }
        T::container_try_cast_from_any_view(data).map(Self::changed)
    }

    fn type_str() -> std::string::String {
        format!("UnchangedOr<{}>", T::container_type_str())
    }

    fn type_schema() -> std::string::String {
        crate::type_traits::type_schema("UnchangedOr", &[T::container_type_schema()])
    }
}

// -----------------------------------------------------------------------------
// Conversions into an erased mutation result.
// -----------------------------------------------------------------------------

/// `Unchanged.into()` keeps the original value.
impl From<Unchanged> for UnchangedOr<Any> {
    #[inline]
    fn from(_: Unchanged) -> Self {
        Self::unchanged()
    }
}

/// A managed reference replaces the original value.
impl<T: crate::object::ObjectRefCore> From<T> for UnchangedOr<Any> {
    #[inline]
    fn from(value: T) -> Self {
        Self::changed(Any::from(value))
    }
}

/// A typed result keeps its unchanged marker or replacement.
impl<T: crate::object::ObjectRefCore> From<UnchangedOr<T>> for UnchangedOr<Any> {
    #[inline]
    fn from(value: UnchangedOr<T>) -> Self {
        Self::from_any(Any::from(value))
    }
}

macro_rules! impl_unchanged_or_any_from_scalar {
    ($($ty:ty),*) => {
        $(
            /// A scalar replaces the original value.
            impl From<$ty> for UnchangedOr<Any> {
                #[inline]
                fn from(value: $ty) -> Self {
                    Self::changed(Any::from(value))
                }
            }

            /// A typed scalar result keeps its unchanged marker or replacement.
            impl From<UnchangedOr<$ty>> for UnchangedOr<Any> {
                #[inline]
                fn from(value: UnchangedOr<$ty>) -> Self {
                    Self::from_any(Any::from(value))
                }
            }
        )*
    };
}
impl_unchanged_or_any_from_scalar!(
    bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64
);
