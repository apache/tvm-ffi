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
use crate::error::Error;
use crate::object;
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIAnyViewToOwnedAny};

/// A borrowed, type-erased FFI value.
///
/// `AnyView` holds any FFI value, such as an integer, a string or an object,
/// without owning it: it does not retain objects, and its lifetime `'a` keeps
/// it from outliving the value it borrows. It is `Copy` and is the argument
/// type of FFI calls. Convert it into an [`Any`] to own the value.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{Any, AnyView, Array};
///
/// let array = Array::from_vec(vec![1_i64, 2, 3]);
/// let view = AnyView::from(&array);
/// assert_eq!(view.try_as::<Array<i64>>().unwrap().len(), 3);
///
/// let owned = Any::from(view);
/// assert_eq!(owned.try_as::<Array<i64>>().unwrap().len(), 3);
/// ```
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AnyView<'a> {
    data: TVMFFIAny,
    /// needs to explicit mark lifetime to avoid lifetime mismatch
    _phantom: std::marker::PhantomData<&'a ()>,
}

/// An owned, type-erased FFI value.
///
/// `Any` holds any FFI value, such as an integer, a string or an object, and
/// owns it: cloning an `Any` that holds an object retains the object, and
/// dropping it releases the object. Any [`AnyCompatible`] value converts into
/// it with `From`; [`try_as`](Self::try_as) and [`match_any!`](crate::match_any)
/// read it back. Borrow it as an [`AnyView`] to pass it without retaining.
///
/// # Examples
///
/// ```
/// use tvm_ffi::Any;
///
/// let value = Any::from(42_i64);
/// assert_eq!(value.try_as::<i64>(), Some(42));
/// assert_eq!(value.try_as::<tvm_ffi::String>(), None);
///
/// let text = Any::from(tvm_ffi::String::from("hello"));
/// assert_eq!(text.try_as::<tvm_ffi::String>().unwrap().as_str(), "hello");
/// ```
#[repr(C)]
pub struct Any {
    data: TVMFFIAny,
}

//---------------------
// AnyView
//---------------------
impl<'a> AnyView<'a> {
    /// Create a view of `None`, the empty FFI value.
    ///
    /// # Returns
    ///
    /// A view holding `None`.
    pub fn new() -> Self {
        Self {
            data: TVMFFIAny::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Borrow a live managed object without acquiring ownership.
    #[doc(hidden)]
    #[inline(always)]
    pub fn from_object<T: object::ObjectCore>(value: &'a T) -> Self {
        let pointer = std::ptr::from_ref(value).cast::<tvm_ffi_sys::TVMFFIObject>();
        // ObjectCore guarantees the prefix and stable managed allocation. A
        // builder or a moved-out destructor value is not a live managed object.
        let header = unsafe { &*pointer };
        assert!(
            header.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32
                && header.deleter.is_some()
                && header
                    .combined_ref_count
                    .load(std::sync::atomic::Ordering::Relaxed)
                    & tvm_ffi_sys::COMBINED_REF_COUNT_MASK_U32
                    != 0,
            "cannot borrow an unmanaged or dead object as AnyView"
        );
        let mut data = TVMFFIAny::new();
        data.type_index = header.type_index;
        data.data_union.v_obj = pointer.cast_mut();
        // The source borrow bounds the view; the dynamic type comes from the
        // allocation even when value borrows a base-class prefix.
        unsafe { Self::from_raw_ffi_any(data) }
    }

    /// Return the runtime type index of the viewed value.
    ///
    /// Indices below `TypeIndex::kTVMFFIStaticObjectBegin` are the built-in
    /// non-object types; the others are object types.
    ///
    /// # Returns
    ///
    /// The type index.
    #[inline(always)]
    pub fn type_index(&self) -> i32 {
        self.data.type_index
    }

    /// Borrow the raw ABI representation.
    /// Copying it does not retain referenced data or extend its lifetime.
    #[inline(always)]
    pub fn as_raw_ffi_any(&self) -> &TVMFFIAny {
        &self.data
    }

    /// Construct a borrowed view from its ABI representation.
    ///
    /// # Safety
    ///
    /// `data.type_index` must describe the payload, and the caller must keep
    /// every resource it references alive for the view's complete lifetime.
    ///
    /// `kTVMFFIObjectRValueRef` carries a further obligation: `v_ptr` must
    /// point to a writable slot the caller uniquely owns, holding one strong
    /// reference. Owning the view takes that reference and writes null back
    /// through the pointer, so the view must be converted at most once and no
    /// other access to the slot may overlap the conversion.
    #[inline(always)]
    pub unsafe fn from_raw_ffi_any(data: TVMFFIAny) -> Self {
        Self {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Borrow the viewed value as `T` when it holds one, without copying it.
    ///
    /// `T` is an object type such as `ArrayObj`, a managed reference such as
    /// `Array<i64>`, or a value stored in place such as `String` or `i64`. The
    /// check is strict, as in [`try_as`](Self::try_as), and never converts.
    /// The borrow does not change any reference count and is tied to this view,
    /// which holds its own copy of the value's cell.
    ///
    /// # Returns
    ///
    /// The borrowed value, or `None`.
    ///
    /// See also [`Any::downcast_ref`].
    #[inline(always)]
    pub fn downcast_ref<T: crate::type_traits::DowncastRefFromAny>(&self) -> Option<&T> {
        // SAFETY: the view holds a valid cell for as long as it is borrowed.
        unsafe { T::downcast_ref_from_any(&self.data) }
    }

    /// Return the viewed value as a `T` when it is stored as exactly `T`.
    ///
    /// The check is strict and never converts: a stored integer is not returned as
    /// `f64`, and an object is returned only as a reference type it is an instance
    /// of. On success, `Any::from` of the result holds the same value.
    ///
    /// # Returns
    ///
    /// The value, or `None` when it is not stored as `T`.
    ///
    /// ```
    /// use tvm_ffi::AnyView;
    /// assert_eq!(AnyView::from(&7_i64).try_as::<i64>(), Some(7));
    /// assert_eq!(AnyView::from(&7_i64).try_as::<f64>(), None);
    /// ```
    ///
    /// See also [`Any::try_as`].
    #[inline(always)]
    pub fn try_as<T>(&self) -> Option<T>
    where
        T: AnyCompatible,
    {
        unsafe {
            if T::check_any_strict(&self.data) {
                Some(T::copy_from_any_view_after_check(&self.data))
            } else {
                None
            }
        }
    }

    /// Get the strong count of the underlying object for testing/debugging purposes
    ///
    /// If the underlying object is not ref counted, return None
    pub fn debug_strong_count(&self) -> Option<usize> {
        unsafe {
            if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                Some(object::unsafe_::strong_count(self.data.data_union.v_obj))
            } else {
                None
            }
        }
    }
}

impl<'a> From<&'a object::Object> for AnyView<'a> {
    #[inline(always)]
    fn from(value: &'a object::Object) -> Self {
        Self::from_object(value)
    }
}

impl<'a, T: AnyCompatible> From<&'a T> for AnyView<'a> {
    #[inline(always)]
    fn from(value: &'a T) -> Self {
        unsafe {
            let mut data = TVMFFIAny::new();
            T::copy_to_any_view(&value, &mut data);
            Self {
                data: data,
                _phantom: std::marker::PhantomData,
            }
        }
    }
}

impl<'a> From<&AnyView<'a>> for AnyView<'a> {
    #[inline(always)]
    fn from(value: &AnyView<'a>) -> Self {
        *value
    }
}

impl Default for AnyView<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Holder for Any value
///
/// This is used to define try_from rule while conforming to orphan rule
/// Users should not use this directly
pub struct TryFromTemp<T> {
    value: T,
}

impl<T> TryFromTemp<T> {
    /// Create a new holder for the value
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Move the value out of the holder
    #[inline(always)]
    pub fn into_value(this: Self) -> T {
        this.value
    }
}

//---------------------
// Any
//---------------------
impl Any {
    /// Create an `Any` holding `None`, the empty FFI value.
    ///
    /// # Returns
    ///
    /// An `Any` holding `None`.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            data: TVMFFIAny::new(),
        }
    }

    /// Return the runtime type index of the stored value.
    ///
    /// Indices below `TypeIndex::kTVMFFIStaticObjectBegin` are the built-in
    /// non-object types; the others are object types.
    ///
    /// # Returns
    ///
    /// The type index.
    #[inline(always)]
    pub fn type_index(&self) -> i32 {
        self.data.type_index
    }

    /// Borrow the raw ABI representation, which this value owns.
    /// Copying it does not retain referenced data or extend its lifetime.
    #[inline(always)]
    pub fn as_raw_ffi_any(&self) -> &TVMFFIAny {
        &self.data
    }
    /// Borrow the stored value as `T` when it holds one, without copying it.
    ///
    /// `T` is an object type such as `ArrayObj`, a managed reference such as
    /// `Array<i64>`, or a value stored in place such as `String` or `i64`. The
    /// check is strict, as in [`try_as`](Self::try_as), and never converts.
    /// This neither consumes `self` nor changes any reference count, and the
    /// returned reference cannot outlive `self`.
    ///
    /// # Returns
    ///
    /// The borrowed object, or `None`.
    ///
    /// ```
    /// use tvm_ffi::{collections::array::ArrayObj, Any, Array};
    /// let value = Any::from(Array::from_vec(vec![1_i64, 2]));
    /// assert_eq!(value.downcast_ref::<ArrayObj>().unwrap().size, 2);
    /// assert_eq!(value.downcast_ref::<Array<i64>>().unwrap()[1], 2);
    /// assert!(value.downcast_ref::<i64>().is_none());
    /// ```
    ///
    /// See also [`try_as`](Self::try_as).
    #[inline]
    pub fn downcast_ref<T: crate::type_traits::DowncastRefFromAny>(&self) -> Option<&T> {
        // SAFETY: the owned cell stays valid for as long as `self` is borrowed.
        unsafe { T::downcast_ref_from_any(&self.data) }
    }

    /// Return the stored value as a `T` when it is stored as exactly `T`.
    ///
    /// The check is strict and never converts: a stored integer is not returned as
    /// `f64`, and an object is returned only as a reference type it is an instance
    /// of. On success, `Any::from` of the result holds the same value. A failed
    /// check is cheap, so `try_as` also serves as a type test. A returned object
    /// reference is retained; use [`downcast_ref`](Self::downcast_ref) to borrow the
    /// object instead.
    ///
    /// # Returns
    ///
    /// The value, or `None` when it is not stored as `T`.
    ///
    /// ```
    /// use tvm_ffi::Any;
    /// let value = Any::from(7_i64);
    /// assert_eq!(value.try_as::<i64>(), Some(7));
    /// assert_eq!(value.try_as::<f64>(), None);
    /// ```
    ///
    /// See also [`match_any!`](crate::match_any).
    #[inline]
    pub fn try_as<T>(&self) -> Option<T>
    where
        T: AnyCompatible,
    {
        unsafe {
            if T::check_any_strict(&self.data) {
                Some(T::copy_from_any_view_after_check(&self.data))
            } else {
                None
            }
        }
    }

    /// Return a pointer to the raw ABI representation, for a runtime call that
    /// writes an owned result into it, as C++ `Any` out-parameters are filled.
    ///
    /// Writing through the pointer is unsafe: the cell must afterwards hold a
    /// value this `Any` may own and release, and the previous value must have
    /// been released or moved out.
    #[inline(always)]
    pub fn as_raw_ffi_any_mut(&mut self) -> *mut TVMFFIAny {
        &mut self.data
    }

    /// Give up ownership of the value as its raw ABI representation, as
    /// `Box::into_raw` does. Rebuild it with [`from_raw_ffi_any`](Self::from_raw_ffi_any)
    /// to release it.
    #[inline(always)]
    pub fn into_raw_ffi_any(this: Self) -> TVMFFIAny {
        let this = std::mem::ManuallyDrop::new(this);
        this.data
    }

    /// Take ownership of a raw ABI representation.
    ///
    /// # Safety
    ///
    /// `data.type_index` must describe the payload, and the caller must hand
    /// over the ownership `data` carries, such as one strong reference for an
    /// object, so it is released exactly once, by this `Any`.
    #[inline(always)]
    pub unsafe fn from_raw_ffi_any(data: TVMFFIAny) -> Self {
        Self { data }
    }

    /// Get the strong count of the underlying object for testing/debugging purposes
    ///
    /// If the underlying object is not ref counted, return None
    pub fn debug_strong_count(&self) -> Option<usize> {
        unsafe {
            if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                Some(object::unsafe_::strong_count(self.data.data_union.v_obj))
            } else {
                None
            }
        }
    }
}

impl Default for Any {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Any {
    #[inline]
    fn clone(&self) -> Self {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { object::unsafe_::inc_ref(self.data.data_union.v_obj) }
        }
        Self { data: self.data }
    }
}

impl Drop for Any {
    #[inline(always)]
    fn drop(&mut self) {
        if self.data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { object::unsafe_::dec_ref(self.data.data_union.v_obj) }
        }
    }
}

// convert Any ref to AnyView
impl<'a> From<&'a Any> for AnyView<'a> {
    #[inline(always)]
    fn from(value: &'a Any) -> Self {
        Self {
            data: value.data,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Whether a `TVMFFIAny` cell owns everything it holds, so its owning form is a
/// bitwise copy. Mirrors C++ `detail::InplaceConvertAnyViewToAny`.
#[inline]
pub(crate) fn is_plain_inline(type_index: i32) -> bool {
    type_index < TypeIndex::kTVMFFIRawStr as i32
        || type_index == TypeIndex::kTVMFFISmallStr as i32
        || type_index == TypeIndex::kTVMFFISmallBytes as i32
        || type_index == TypeIndex::kTVMFFIUnchanged as i32
}

// convert AnyView to Any
impl From<AnyView<'_>> for Any {
    #[inline]
    fn from(value: AnyView<'_>) -> Self {
        let data = value.data;
        // Owning a borrowed object is the same incref `Any::clone` does below.
        if data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            unsafe { object::unsafe_::inc_ref(data.data_union.v_obj) };
            return Self { data };
        }
        if is_plain_inline(data.type_index) {
            return Self { data };
        }
        // What is left borrows foreign storage and needs the runtime.
        any_view_to_owned_via_runtime(data)
    }
}

/// Out of line so the inlined conversions above stay leaf code, without the
/// stack frame and unwind path this call needs.
#[cold]
#[inline(never)]
fn any_view_to_owned_via_runtime(view: TVMFFIAny) -> Any {
    unsafe {
        let mut data = TVMFFIAny::new();
        crate::check_safe_call!(TVMFFIAnyViewToOwnedAny(&view, &mut data)).unwrap();
        Any { data }
    }
}

impl<T: AnyCompatible> From<T> for Any {
    #[inline(always)]
    fn from(value: T) -> Self {
        unsafe {
            let mut data = TVMFFIAny::new();
            T::move_to_any(value, &mut data);
            Self { data }
        }
    }
}

impl<'a, T: AnyCompatible> TryFrom<AnyView<'a>> for TryFromTemp<T> {
    type Error = crate::error::Error;
    #[inline]
    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        unsafe {
            if T::check_any_strict(&value.data) {
                Ok(TryFromTemp::new(T::copy_from_any_view_after_check(
                    &value.data,
                )))
            } else {
                T::try_cast_from_any_view(&value.data)
                    .map_err(|_| {
                        let msg = format!(
                            "Cannot convert from type `{}` to `{}`",
                            T::get_mismatch_type_info(&value.data),
                            T::type_str()
                        );
                        crate::error::Error::new(crate::error::TYPE_ERROR, &msg, "")
                    })
                    .map(TryFromTemp::new)
            }
        }
    }
}

impl<T: AnyCompatible> TryFrom<Any> for TryFromTemp<T> {
    type Error = crate::error::Error;
    #[inline]
    fn try_from(value: Any) -> Result<Self, Self::Error> {
        unsafe {
            if T::check_any_strict(&value.data) {
                let mut value = std::mem::ManuallyDrop::new(value);
                Ok(TryFromTemp::new(T::move_from_any_after_check(
                    &mut value.data,
                )))
            } else {
                T::try_cast_from_any_view(&value.data)
                    .map_err(|_| {
                        let msg = format!(
                            "Cannot convert from type `{}` to `{}`",
                            T::get_mismatch_type_info(&value.data),
                            T::type_str()
                        );
                        crate::error::Error::new(crate::error::TYPE_ERROR, &msg, "")
                    })
                    .map(TryFromTemp::new)
            }
        }
    }
}

crate::impl_try_from_any!(
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
    (),
    *mut core::ffi::c_void,
    crate::string::String,
    crate::string::Bytes,
    crate::big_int::BigInt,
    crate::object::ObjectRef,
    tvm_ffi_sys::dlpack::DLDataType,
    tvm_ffi_sys::dlpack::DLDevice,
);

crate::impl_try_from_any_for_parametric!(Option<T>);

//------------------------------------------------------------
/// ArgTryFromAnyView: Helper for function argument passing
///-----------------------------------------------------------
pub(crate) trait ArgTryFromAnyView: Sized + crate::type_traits::TypeSchema {
    fn try_from_any_view(value: &AnyView, arg_index: usize) -> Result<Self, Error>;
}

impl<T: AnyCompatible> ArgTryFromAnyView for T {
    fn try_from_any_view(value: &AnyView, arg_index: usize) -> Result<Self, Error> {
        unsafe {
            if T::check_any_strict(&value.data) {
                Ok(T::copy_from_any_view_after_check(&value.data))
            } else {
                T::try_cast_from_any_view(&value.data).map_err(|_| {
                    let msg = format!(
                        "Argument #{}: Cannot convert from type `{}` to `{}`",
                        arg_index,
                        T::get_mismatch_type_info(&value.data),
                        T::type_str()
                    );
                    crate::error::Error::new(crate::error::TYPE_ERROR, &msg, "")
                })
            }
        }
    }
}
