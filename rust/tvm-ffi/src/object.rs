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
use std::ops::{Deref, DerefMut};
use std::sync::atomic::AtomicU64;

use crate::derive::ObjectRef;
use crate::type_traits::AnyCompatible;
pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
/// Object related ABI handling
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIGetCustomAllocator, TVMFFIGetTypeInfo, TVMFFIObject,
    COMBINED_REF_COUNT_BOTH_ONE,
};

/// Object type is by default the TVMFFIObject
#[repr(C)]
pub struct Object {
    /// example implementation of the object
    header: TVMFFIObject,
}

/// Arc-like wrapper for Object that allows shared ownership
///
/// \tparam T The type of the object to be wrapped
#[repr(C)]
pub struct ObjectArc<T: ObjectCore> {
    ptr: std::ptr::NonNull<T>,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Send + Sync + ObjectCore> Send for ObjectArc<T> {}
unsafe impl<T: Send + Sync + ObjectCore> Sync for ObjectArc<T> {}

/// Traits that can be used to check if a type is an object
///
/// This trait is unsafe because it is used to access the object header
/// and the object header is unsafe to access
pub unsafe trait ObjectCore: Sized + 'static {
    /// the type key of the object
    const TYPE_KEY: &'static str;
    // return the type index of the object
    fn type_index() -> i32;
    /// Return the object header
    /// This function is implemented as a static function so
    ///
    /// # Arguments
    /// * `this` - The object to get the header
    ///
    /// # Returns
    /// * `&mut TVMFFIObject` - The object header
    /// \return The object header
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject;
}

/// Traits for objects with extra items that follows the object
///
/// This extra trait can be helpful to implement array types and string types
pub unsafe trait ObjectCoreWithExtraItems: ObjectCore {
    /// type of extra items storage that follows the object
    type ExtraItem;
    /// Return the number of extra items
    fn extra_items_count(this: &Self) -> usize;
    /// Return the extra items data pointer
    unsafe fn extra_items(this: &Self) -> &[Self::ExtraItem] {
        let extra_items_ptr = (this as *const Self as *const u8).add(std::mem::size_of::<Self>());
        std::slice::from_raw_parts(
            extra_items_ptr as *const Self::ExtraItem,
            Self::extra_items_count(this),
        )
    }
    /// Return the extra items data pointer
    unsafe fn extra_items_mut(this: &mut Self) -> &mut [Self::ExtraItem] {
        let extra_items_ptr = (this as *mut Self as *mut u8).add(std::mem::size_of::<Self>());
        std::slice::from_raw_parts_mut(
            extra_items_ptr as *mut Self::ExtraItem,
            Self::extra_items_count(this),
        )
    }
}

/// Traits to specify core operations of ObjectRef
///
/// used by the ffi Any system and not user facing
///
/// We mark as unsafe since it moves out the internal of the ObjectRef
///
/// # Safety
///
/// `data`, `into_data`, and `from_data` must preserve the same object
/// allocation. That allocation must start with a valid `TVMFFIObject` header
/// whose runtime type index is in the object range. When `Self` also implements
/// [`AnyCompatible`], its conversions to `TVMFFIAny` must preserve that object
/// pointer and dynamic type index.
pub unsafe trait ObjectRefCore: Sized + Clone {
    type ContainerType: ObjectCore;
    fn data(this: &Self) -> &ObjectArc<Self::ContainerType>;
    fn into_data(this: Self) -> ObjectArc<Self::ContainerType>;
    fn from_data(data: ObjectArc<Self::ContainerType>) -> Self;
}

/// Check whether a runtime type index refers to `target_type_index` or one of
/// its subtypes.
///
/// The subtype relation lives in the process-wide type table maintained by the
/// tvm-ffi library: every registered type records its depth in the single
/// inheritance tree together with the chain of its ancestors. The check is
/// O(1) — if `target` really is an ancestor, it must appear in the candidate's
/// ancestor array exactly at `target`'s depth.
#[inline(always)]
pub fn is_instance_of(object_type_index: i32, target_type_index: i32) -> bool {
    if object_type_index == target_type_index {
        return true;
    }
    // only object types participate in the type hierarchy
    if object_type_index < TypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return false;
    }
    unsafe {
        let object_info = TVMFFIGetTypeInfo(object_type_index);
        let target_info = TVMFFIGetTypeInfo(target_type_index);
        if object_info.is_null() || target_info.is_null() {
            return false;
        }
        let target_depth = (*target_info).type_depth;
        if (*object_info).type_depth <= target_depth {
            return false;
        }
        let ancestor = *(*object_info).type_acenstors.add(target_depth as usize);
        !ancestor.is_null() && (*ancestor).type_index == target_type_index
    }
}

/// Runtime-checked casting between arbitrary `ObjectRef` types.
///
/// The cast uses the target's [`AnyCompatible::check_any_strict`] implementation,
/// mirroring the semantics of `ObjectRef::as<T>` in C++. This supports both
/// object hierarchies and parameterized object containers.
///
/// This trait is blanket-implemented for every [`ObjectRefCore`] type that is
/// also [`AnyCompatible`].
pub trait ObjectRefCast: ObjectRefCore + AnyCompatible {
    /// Consume `self` and rewrap the underlying object as `B` without copying.
    #[inline(always)]
    fn try_cast<B>(self) -> crate::error::Result<B>
    where
        B: ObjectRefCore + AnyCompatible,
    {
        let mut any_data = TVMFFIAny::new();
        unsafe {
            Self::move_to_any(self, &mut any_data);
            debug_assert!(any_data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32);
            // SAFETY: ObjectRefCore's contract requires its AnyCompatible
            // representation to contain a valid object-range type index.
            std::hint::assert_unchecked(
                any_data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32,
            );

            if B::check_any_strict(&any_data) {
                Ok(B::move_from_any_after_check(&mut any_data))
            } else {
                let mismatch_data = any_data;
                let source = Self::move_from_any_after_check(&mut any_data);
                let msg = format!(
                    "Cannot convert from type `{}` to `{}`",
                    B::get_mismatch_type_info(&mismatch_data),
                    B::type_str()
                );
                drop(source);
                Err(crate::error::Error::new(crate::error::TYPE_ERROR, &msg, ""))
            }
        }
    }

    /// Same as [`try_cast`](Self::try_cast), but shares ownership with `self`
    /// instead of consuming it.
    fn cast_clone<B>(&self) -> crate::error::Result<B>
    where
        B: ObjectRefCore + AnyCompatible,
    {
        Self::from_data(Self::data(self).clone()).try_cast()
    }
}

impl<T: ObjectRefCore + AnyCompatible> ObjectRefCast for T {}

/// Base class for ObjectRef
///
/// This class is used to store the data of the ObjectRef
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ObjectRef {
    data: ObjectArc<Object>,
}

/// Unsafe operations on object
#[doc(hidden)]
pub mod unsafe_ {
    use tvm_ffi_sys::{
        COMBINED_REF_COUNT_BOTH_ONE, COMBINED_REF_COUNT_MASK_U32, COMBINED_REF_COUNT_STRONG_ONE,
        COMBINED_REF_COUNT_WEAK_ONE,
    };

    use std::ffi::c_void;
    use std::sync::atomic::{fence, Ordering};
    use tvm_ffi_sys::{TVMFFIObject, TVMFFIObjectAllocHeader};
    use tvm_ffi_sys::TVMFFIObjectDeleterFlagBitMask::{
        kTVMFFIObjectDeleterFlagBitMaskBoth, kTVMFFIObjectDeleterFlagBitMaskStrong,
        kTVMFFIObjectDeleterFlagBitMaskWeak,
    };

    /// Increase the strong reference count of the object
    ///
    /// This function is same as TVMFFIObjectIncRef but implemented natively in Rust
    ///
    /// # Arguments
    /// * `obj` - The object to increase the reference count
    #[inline]
    pub unsafe fn inc_ref(handle: *mut TVMFFIObject) {
        let obj = &mut *handle;
        obj.combined_ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrease the strong reference count of the object
    ///
    /// This function is same as TVMFFIObjectDecRef but implemented natively in Rust
    ///
    /// # Arguments
    /// * `obj` - The object to decrease the reference count
    #[inline]
    pub unsafe fn dec_ref(handle: *mut TVMFFIObject) {
        let obj = &mut *handle;
        let old_combined_count = obj
            .combined_ref_count
            .fetch_sub(COMBINED_REF_COUNT_STRONG_ONE, Ordering::Relaxed);
        if old_combined_count == COMBINED_REF_COUNT_BOTH_ONE {
            if let Some(deleter) = obj.deleter {
                fence(Ordering::Acquire);
                deleter(
                    obj as *mut TVMFFIObject as *mut c_void,
                    kTVMFFIObjectDeleterFlagBitMaskBoth as i32,
                );
            }
        } else if (old_combined_count & COMBINED_REF_COUNT_MASK_U32)
            == COMBINED_REF_COUNT_STRONG_ONE
        {
            // slow path, there is still a weak reference left
            // need to run two phase decrement
            fence(Ordering::Acquire);
            if let Some(deleter) = obj.deleter {
                deleter(
                    obj as *mut TVMFFIObject as *mut c_void,
                    kTVMFFIObjectDeleterFlagBitMaskStrong as i32,
                );
            }
            let old_weak_count = obj
                .combined_ref_count
                .fetch_sub(COMBINED_REF_COUNT_WEAK_ONE, Ordering::Release);
            if old_weak_count == COMBINED_REF_COUNT_WEAK_ONE {
                fence(Ordering::Acquire);
                if let Some(deleter) = obj.deleter {
                    deleter(
                        obj as *mut TVMFFIObject as *mut c_void,
                        kTVMFFIObjectDeleterFlagBitMaskWeak as i32,
                    );
                }
            }
        }
    }

    #[inline]
    pub unsafe fn strong_count(handle: *mut TVMFFIObject) -> usize {
        let obj = &mut *handle;
        (obj.combined_ref_count.load(Ordering::Relaxed) & COMBINED_REF_COUNT_MASK_U32) as usize
    }

    #[inline]
    pub unsafe fn weak_count(handle: *mut TVMFFIObject) -> usize {
        let obj = &mut *handle;
        (obj.combined_ref_count.load(Ordering::Relaxed) >> 32) as usize
    }

    /// Generic object deleter for objects allocated through the registered
    /// `TVMFFICustomAllocator`.
    pub unsafe extern "C" fn object_deleter_for_new<T>(ptr: *mut c_void, flags: i32)
    where
        T: super::ObjectCore,
    {
        let obj = ptr as *mut T;
        if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
            std::ptr::drop_in_place(obj);
        }
        if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
            let header_ptr = (ptr as *mut u8)
                .sub(std::mem::size_of::<TVMFFIObjectAllocHeader>())
                as *mut TVMFFIObjectAllocHeader;
            let delete_space = (*header_ptr)
                .delete_space
                .expect("TVMFFIObjectAllocHeader::delete_space must be set by the allocator");
            delete_space(ptr);
        }
    }
}

//---------------------
// Object
//---------------------

impl Object {
    pub fn new() -> Self {
        Self {
            header: TVMFFIObject::new(),
        }
    }
}

unsafe impl ObjectCore for Object {
    const TYPE_KEY: &'static str = "ffi.Object";
    #[inline]
    fn type_index() -> i32 {
        TypeIndex::kTVMFFIStaticObjectBegin as i32
    }
    #[inline]
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        &mut this.header
    }
}

//---------------------
// ObjectArc
//---------------------

/// Allocate via the process-wide `TVMFFICustomAllocator`.
unsafe fn allocate_via_registry(size: usize, alignment: usize, type_index: i32) -> *mut u8 {
    let alloc = TVMFFIGetCustomAllocator();
    let allocate_fn = (*alloc)
        .allocate
        .expect("TVMFFICustomAllocator::allocate must be set");
    allocate_fn(size, alignment, type_index, (*alloc).context) as *mut u8
}

impl<T: ObjectCore> ObjectArc<T> {
    pub fn new(data: T) -> Self {
        unsafe {
            let raw_data_ptr = allocate_via_registry(
                std::mem::size_of::<T>(),
                std::mem::align_of::<T>(),
                T::type_index(),
            );
            let ptr = raw_data_ptr as *mut T;
            std::ptr::write(ptr, data);
            // now override the header directly
            std::ptr::write(
                ptr as *mut TVMFFIObject,
                TVMFFIObject {
                    combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                    type_index: T::type_index(),
                    __padding: 0,
                    deleter: Some(unsafe_::object_deleter_for_new::<T>),
                },
            );
            // move into the object arc ptr
            Self {
                ptr: std::ptr::NonNull::new_unchecked(ptr as *mut T),
                _phantom: std::marker::PhantomData,
            }
        }
    }
    pub fn new_with_extra_items<U>(data: T) -> Self
    where
        T: ObjectCoreWithExtraItems<ExtraItem = U>,
    {
        unsafe {
            // ensure strict alignment requirements
            // so we can have { T, U*extra_items } layout
            assert_eq!(std::mem::align_of::<T>() % std::mem::align_of::<U>(), 0);
            assert_eq!(std::mem::size_of::<T>() % std::mem::align_of::<U>(), 0);
            let extra_items_count = T::extra_items_count(&data);
            let total_size = std::mem::size_of::<T>() + extra_items_count * std::mem::size_of::<U>();
            let raw_data_ptr =
                allocate_via_registry(total_size, std::mem::align_of::<T>(), T::type_index());
            let ptr = raw_data_ptr as *mut T;
            std::ptr::write(ptr, data);
            // now override the header directly
            std::ptr::write(
                ptr as *mut TVMFFIObject,
                TVMFFIObject {
                    combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                    type_index: T::type_index(),
                    __padding: 0,
                    deleter: Some(unsafe_::object_deleter_for_new::<T>),
                },
            );
            // move into the object arc ptr
            Self {
                ptr: std::ptr::NonNull::new_unchecked(ptr as *mut T),
                _phantom: std::marker::PhantomData,
            }
        }
    }

    /// Move a previously allocated object into the ObjectArc
    ///
    /// # Arguments
    /// * `ptr` - The raw pointer to move into the ObjectArc
    ///
    /// # Returns
    /// * `ObjectArc<T>` - The ObjectArc
    /// \return The ObjectArc
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            ptr: std::ptr::NonNull::new_unchecked(ptr as *mut T),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Move the ObjectArc into a raw pointer
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to move into a raw pointer
    ///
    /// # Returns
    /// * `*const T` - The raw pointer
    #[inline]
    pub unsafe fn into_raw(this: Self) -> *const T {
        let droped_this = std::mem::ManuallyDrop::new(this);
        droped_this.ptr.as_ptr() as *const T
    }

    /// Get the raw pointer from the ObjectArc
    ///
    /// Caller should view this as a non-owning reference
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the raw pointer
    ///
    /// # Returns
    /// * `*const T` - The raw pointer
    /// \return The raw pointer
    #[inline]
    pub unsafe fn as_raw(this: &Self) -> *const T {
        this.ptr.as_ptr() as *const T
    }

    /// Get the raw mutable pointer from the ObjectArc
    ///
    /// Caller should view this as a non-owning reference
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the raw pointer
    ///
    /// # Returns
    /// * `*mut T` - The raw pointer
    #[inline]
    pub unsafe fn as_raw_mut(this: &mut Self) -> *mut T {
        this.ptr.as_mut()
    }

    /// Get the strong reference count of the ObjectArc
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the strong reference count
    ///
    /// # Returns
    /// * `usize` - The strong reference count
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        unsafe {
            unsafe_::strong_count(this.ptr.as_ref() as *const T as *mut T as *mut TVMFFIObject)
        }
    }

    /// Get the weak reference count of the ObjectArc
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the weak reference count
    ///
    /// # Returns
    /// * `usize` - The weak reference count
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        unsafe { unsafe_::weak_count(this.ptr.as_ref() as *const T as *mut T as *mut TVMFFIObject) }
    }
}

// implement Deref for ObjectArc
impl<T: ObjectCore> Deref for ObjectArc<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

// implement DerefMut for ObjectArc
impl<T: ObjectCore> DerefMut for ObjectArc<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

// implement Drop for ObjectArc
impl<T: ObjectCore> Drop for ObjectArc<T> {
    fn drop(&mut self) {
        unsafe { unsafe_::dec_ref(self.ptr.as_mut() as *mut T as *mut TVMFFIObject) }
    }
}

// implement Clone for ObjectArc
impl<T: ObjectCore> Clone for ObjectArc<T> {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { unsafe_::inc_ref(self.ptr.as_ref() as *const T as *mut T as *mut TVMFFIObject) }
        Self {
            ptr: self.ptr,
            _phantom: std::marker::PhantomData,
        }
    }
}
