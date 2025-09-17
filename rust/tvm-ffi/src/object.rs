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
pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
/// Object related ABI handling
use tvm_ffi_sys::{TVMFFIObject, COMBINED_REF_COUNT_BOTH_ONE};

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
pub unsafe trait ObjectRefCore: Sized + Clone {
    type ContainerType: ObjectCore;
    fn data(this: &Self) -> &ObjectArc<Self::ContainerType>;
    fn into_data(this: Self) -> ObjectArc<Self::ContainerType>;
    fn from_data(data: ObjectArc<Self::ContainerType>) -> Self;
}

/// Base class for ObjectRef
///
/// This class is used to store the data of the ObjectRef
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ObjectRef {
    data: ObjectArc<Object>,
}

/// Unsafe operations on object
pub(crate) mod unsafe_ {
    use tvm_ffi_sys::{
        COMBINED_REF_COUNT_BOTH_ONE, COMBINED_REF_COUNT_MASK_U32, COMBINED_REF_COUNT_STRONG_ONE,
        COMBINED_REF_COUNT_WEAK_ONE,
    };

    use std::ffi::c_void;
    use std::sync::atomic::{fence, Ordering};
    use tvm_ffi_sys::TVMFFIObject;
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

    /// Generic object deleter that works for object allocated from Box then into_raw
    pub unsafe extern "C" fn object_deleter_for_new<T>(ptr: *mut c_void, flags: i32)
    where
        T: super::ObjectCore,
    {
        let obj = ptr as *mut T;
        if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
            // calling destructor of the object, does not free the memory
            std::ptr::drop_in_place(obj);
        }
        if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
            // free the memory
            std::alloc::dealloc(ptr as *mut u8, std::alloc::Layout::new::<T>());
        }
    }

    pub unsafe extern "C" fn object_deleter_for_new_with_extra_items<T, U>(
        ptr: *mut c_void,
        flags: i32,
    ) where
        T: super::ObjectCoreWithExtraItems<ExtraItem = U>,
    {
        let obj = ptr as *mut T;
        if flags == kTVMFFIObjectDeleterFlagBitMaskBoth as i32 {
            // must get extra items count before dropping the object
            let extra_items_count = T::extra_items_count(&(*obj));
            std::ptr::drop_in_place(obj);
            let layout = std::alloc::Layout::from_size_align(
                std::mem::size_of::<T>() + extra_items_count * std::mem::size_of::<U>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            // free the memory
            std::alloc::dealloc(ptr as *mut u8, layout);
        } else {
            assert_eq!(std::mem::size_of::<T>() % std::mem::size_of::<u64>(), 0);
            if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
                // must get extra items count before dropping the object
                let extra_items_count = T::extra_items_count(&(*obj));
                // calling destructor of the object, does not free the memory
                std::ptr::drop_in_place(obj);
                // record extra count in the original memory
                std::ptr::write(obj as *mut u64, extra_items_count as u64);
            }
            if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
                // read extra items count from the original memory
                // note we can no longer read it by calling T::extra_items_count(&(*obj))
                // because the object is already dropped
                let extra_items_count = std::ptr::read(obj as *mut u64) as usize;
                let layout = std::alloc::Layout::from_size_align(
                    std::mem::size_of::<T>() + extra_items_count * std::mem::size_of::<U>(),
                    std::mem::align_of::<T>(),
                )
                .unwrap();
                // free the memory
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
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
impl<T: ObjectCore> ObjectArc<T> {
    pub fn new(data: T) -> Self {
        unsafe {
            let layout = std::alloc::Layout::new::<T>();
            let raw_data_ptr = std::alloc::alloc(layout);
            if raw_data_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
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
            let layout = std::alloc::Layout::from_size_align(
                std::mem::size_of::<T>() + extra_items_count * std::mem::size_of::<U>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            let raw_data_ptr = std::alloc::alloc(layout);
            if raw_data_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let ptr = raw_data_ptr as *mut T;
            std::ptr::write(ptr, data);
            // now override the header directly
            std::ptr::write(
                ptr as *mut TVMFFIObject,
                TVMFFIObject {
                    combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                    type_index: T::type_index(),
                    __padding: 0,
                    deleter: Some(unsafe_::object_deleter_for_new_with_extra_items::<T, U>),
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
