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

//! Objects made with every field undefined, for reflection to set the fields,
//! as C++ makes an object with `UnsafeInit`.
//!
//! A C++ object made with `UnsafeInit` holds undefined (null) references, which
//! the field setters overwrite and the destructor skips. A Rust reference is
//! never null, so a Rust object cannot hold one; instead [`unsafe_init`]
//! allocates the object zeroed, and a field holding all-zero bytes that are no
//! value of its type ([`AnyCompatible::is_undefined`]) is undefined. The field
//! setter writes an undefined field without dropping it, the field getter reads
//! it as None, and the object's deleter drops only the defined fields, all
//! through raw pointers, so no undefined field is ever viewed as its type.

use std::alloc::Layout;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::AtomicU64;

use crate::object::{Object, ObjectArc, ObjectCore, ObjectRef, ObjectRefCore};
use crate::tvm_ffi_sys::TVMFFIObjectDeleterFlagBitMask::{
    kTVMFFIObjectDeleterFlagBitMaskStrong, kTVMFFIObjectDeleterFlagBitMaskWeak,
};
use crate::tvm_ffi_sys::{TVMFFIObject, TVMFFIObjectHandle, COMBINED_REF_COUNT_BOTH_ONE};
use crate::type_traits::AnyCompatible;
use crate::{Error, INTERNAL_ERROR};

/// An object type whose object can start with every field undefined, for
/// reflection to set the fields: C++ `UnsafeInit`.
///
/// The creator that `#[rs_class]` registers for its type makes the object with
/// [`unsafe_init`], which a JSON graph read back, `__ffi_init__` and
/// `__ffi_new__` then fill. `#[rs_class]` implements it for its object, so the
/// parent's object must implement it too: [`Object`] does, and so does the
/// object of a non-final C++ type bound by `tvm-ffi-stubgen`.
///
/// `__ffi_init__` leaves an `init_off` field as the creator made it, so such a
/// field's type must take all-zero bytes as a value, as a scalar, `Option<T>`
/// or `Any` does; the type's registration fails otherwise.
///
/// # Safety
///
/// `drop_defined_fields` drops each defined field of the object and no
/// undefined one (as [`drop_if_defined`] does), the parent's object's included.
pub unsafe trait UnsafeInit: ObjectCore {
    /// Drops the defined fields of the object at `this`, the parent's object's
    /// included, and leaves the undefined ones.
    ///
    /// # Safety
    ///
    /// `this` points to an object that [`unsafe_init`] made, which is not used
    /// afterwards.
    #[doc(hidden)]
    unsafe fn drop_defined_fields(this: *mut Self);
}

// The header holds nothing to drop.
unsafe impl UnsafeInit for Object {
    unsafe fn drop_defined_fields(_this: *mut Self) {}
}

/// Drops the field at `field` unless it is undefined.
///
/// # Safety
///
/// `field` points to a field of an object, defined or undefined, which is not
/// used afterwards.
#[doc(hidden)]
#[inline]
pub unsafe fn drop_if_defined<F: AnyCompatible>(field: *mut F) {
    unsafe {
        if !F::is_undefined(field) {
            std::ptr::drop_in_place(field);
        }
    }
}

/// An object of `T` with every field undefined, as C++
/// `make_object<T>(UnsafeInit{})` makes one, for reflection to set the fields
/// through their setters.
///
/// The [`ObjectRef`] views only the object's header, which is valid all along.
///
/// # Safety
///
/// Each field that all-zero bytes leave undefined
/// ([`AnyCompatible::is_undefined`]) is set before the object is viewed as a
/// `T` (by a cast of the reference) or the field is read as its type. Until
/// then, only the reflection's field setters and getters reach the fields.
pub unsafe fn unsafe_init<T: UnsafeInit>() -> ObjectRef {
    let layout = Layout::new::<T>();
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        std::ptr::write(
            ptr.cast::<TVMFFIObject>(),
            TVMFFIObject {
                combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                type_index: T::type_index(),
                __padding: 0,
                deleter: Some(unsafe_init_deleter::<T>),
            },
        );
        // SAFETY: `ObjectRef` is the view of any object.
        ObjectRef::from_data(ObjectArc::from_raw(ptr.cast::<Object>()))
    }
}

/// The deleter of an object that [`unsafe_init`] made, which may go before
/// reflection sets every field, as when a JSON graph lacks one.
unsafe extern "C" fn unsafe_init_deleter<T: UnsafeInit>(ptr: *mut c_void, flags: i32) {
    unsafe {
        if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
            T::drop_defined_fields(ptr.cast::<T>());
        }
        if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
            std::alloc::dealloc(ptr.cast::<u8>(), Layout::new::<T>());
        }
    }
}

/// The creator of `T` (`TVMFFITypeMetadata::creator`) that `#[rs_class]`
/// registers, as C++ `ObjectCreatorUnsafeInit<T>`: the owned handle of an
/// object that [`unsafe_init`] made.
///
/// # Safety
///
/// `result` is valid for writes.
#[doc(hidden)]
pub unsafe extern "C" fn unsafe_init_creator<T: UnsafeInit>(
    result: *mut TVMFFIObjectHandle,
) -> i32 {
    let made = catch_unwind(AssertUnwindSafe(|| unsafe {
        let object = unsafe_init::<T>();
        let handle = ObjectArc::into_raw(ObjectRef::into_data(object));
        std::ptr::write(result, handle as TVMFFIObjectHandle);
    }));
    match made {
        Ok(()) => 0,
        Err(_) => {
            let message = format!("panic while creating an object of `{}`", T::TYPE_KEY);
            Error::set_raised(&Error::new(INTERNAL_ERROR, &message, ""));
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use crate::tvm_ffi_sys::dlpack::{DLDevice, DLDeviceType};
    use crate::type_traits::AnyCompatible;
    use crate::{Array, BigInt, String};

    fn zero_is_undefined<F: AnyCompatible>() -> bool {
        let zero = MaybeUninit::<F>::zeroed();
        unsafe { F::is_undefined(zero.as_ptr()) }
    }

    fn is_undefined<F: AnyCompatible>(value: F) -> bool {
        unsafe { F::is_undefined(&value) }
    }

    #[test]
    fn all_zero_bytes_are_undefined_where_they_are_no_value() {
        assert!(zero_is_undefined::<Array<i64>>());
        assert!(zero_is_undefined::<String>());
        assert!(zero_is_undefined::<BigInt>());
        assert!(zero_is_undefined::<DLDevice>());

        assert!(!zero_is_undefined::<i64>());
        assert!(!zero_is_undefined::<bool>());
        // `None`, through the niche of a reference or a device, or through a tag.
        assert!(!zero_is_undefined::<Option<Array<i64>>>());
        assert!(!zero_is_undefined::<Option<DLDevice>>());
        assert!(!zero_is_undefined::<Option<String>>());
    }

    #[test]
    fn values_are_defined() {
        let cpu = DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: 0,
        };
        assert!(!is_undefined(Array::<i64>::new(vec![])));
        assert!(!is_undefined(String::from("")));
        assert!(!is_undefined(BigInt::from(0_i64)));
        assert!(!is_undefined(cpu));
        assert!(!is_undefined(Some(cpu)));
        assert!(!is_undefined(None::<DLDevice>));
        assert!(!is_undefined(None::<Array<i64>>));
    }
}
