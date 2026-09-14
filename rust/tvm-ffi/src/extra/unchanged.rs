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

use crate::any::{Any, AnyView, TryFromTemp};
use crate::error::{Error, Result, TYPE_ERROR};
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use crate::type_traits::{AnyCompatible, ContainerElement};

/// A structural mutation that keeps its input without acquiring another owner.
///
/// This marker may be returned directly from a map or mutation callback. A
/// pre-order map still descends into the original value's children.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unchanged;

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
}

crate::impl_try_from_any!(Unchanged);

/// A typed replacement or an [`Unchanged`] marker, stored in one FFI Any cell.
///
/// `T` may be an FFI value type or [`Any`]. Use `Result<UnchangedOr<T>>` to
/// propagate errors. Callbacks can return this wrapper, [`Unchanged`], a
/// replacement value, or a `Result` containing any of these.
///
/// Converting this wrapper to `Any` preserves the marker. Use
/// [`Self::value_or`] or [`Self::value_or_else`] when an actual value is needed.
#[repr(transparent)]
pub struct UnchangedOr<T: ContainerElement = Any> {
    data: Any,
    _marker: PhantomData<T>,
}

impl<T: ContainerElement> UnchangedOr<T> {
    /// Keep the original value without borrowing or cloning it.
    #[inline]
    pub fn unchanged() -> Self {
        Self {
            data: Unchanged.into(),
            _marker: PhantomData,
        }
    }

    /// Supply an owning replacement value.
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

    /// Whether this result asks its caller to keep the original value.
    #[inline]
    pub fn is_unchanged(&self) -> bool {
        is_unchanged(&self.data)
    }

    /// Whether the result is unchanged or has the original shallow identity.
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
        super::structural_common::same_shallow(raw, *self.data.as_raw_ffi_any())
    }

    /// Move out the replacement, returning None when unchanged.
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

    /// Move the replacement or the supplied original value out of this result.
    #[inline]
    pub fn value_or(self, original: T) -> T {
        self.value_or_else(|| original)
    }

    /// Materialize the original only if this result is unchanged.
    #[inline]
    pub fn value_or_else(self, original: impl FnOnce() -> T) -> T {
        self.into_option().unwrap_or_else(original)
    }

    /// Convert only the replacement, preserving an unchanged marker.
    #[inline]
    pub fn map<U: ContainerElement>(self, convert: impl FnOnce(T) -> U) -> UnchangedOr<U> {
        match self.into_option() {
            Some(value) => UnchangedOr::changed(convert(value)),
            None => UnchangedOr::unchanged(),
        }
    }

    /// Fallibly convert only the replacement, propagating its error unchanged.
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
    /// An unchanged result is compatible with every replacement type. Erased
    /// and narrowing conversions retain the strict type check used by
    /// [`Any::try_as`]. For a known typed conversion, use `map(Into::into)`.
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
    #[inline]
    pub(crate) fn from_carrier(data: Any) -> Result<Self> {
        if data.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
            return match Error::try_from(data) {
                Ok(error) | Err(error) => Err(error),
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
