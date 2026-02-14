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

//! Subtyping infrastructure for object hierarchy conversions.
//!
//! This module provides type-safe upcast and downcast operations for objects
//! that follow the TVM FFI object hierarchy.

use crate::object::{Object, ObjectArc, ObjectCore, ObjectRefCore};
use tvm_ffi_sys::TVMFFIGetTypeInfo;

/// Check if a type_index is an instance of target_index (including inheritance).
///
/// # Safety
/// This function accesses the type info table via FFI and follows ancestor pointers.
#[doc(hidden)]
pub unsafe fn is_instance_of(type_index: i32, target_index: i32) -> bool {
    if type_index == target_index {
        return true;
    }
    let info = TVMFFIGetTypeInfo(type_index);
    if info.is_null() {
        return false;
    }
    let info = &*info;
    let ancestors = info.type_acenstors;
    if ancestors.is_null() {
        return false;
    }
    for depth in 0..info.type_depth {
        let ancestor = *ancestors.add(depth as usize);
        if !ancestor.is_null() && (*ancestor).type_index == target_index {
            return true;
        }
    }
    false
}

/// Upcast an object reference from a subtype to a supertype.
///
/// This is a consuming operation that transfers ownership.
///
/// # Type Parameters
/// * `From` - The source type (subtype)
/// * `To` - The target type (supertype)
///
/// # Internal Implementation Detail
/// This function is public for macro expansion but should not be called directly.
/// Use `From::from()` or `.into()` for upcasting instead.
#[doc(hidden)]
pub fn upcast<From: ObjectRefCore, To: ObjectRefCore>(value: From) -> To {
    unsafe {
        let arc = <From as ObjectRefCore>::into_data(value);
        let raw = ObjectArc::into_raw(arc);
        let casted = ObjectArc::from_raw(raw as *const <To as ObjectRefCore>::ContainerType);
        <To as ObjectRefCore>::from_data(casted)
    }
}

/// Try to downcast an object reference from a supertype to a subtype.
///
/// This is a consuming operation that transfers ownership on success.
///
/// # Type Parameters
/// * `From` - The source type (supertype)
/// * `To` - The target type (subtype)
///
/// # Returns
/// * `Ok(To)` - If the runtime type check succeeds
/// * `Err(From)` - If the runtime type check fails, returns the original value
///
/// # Internal Implementation Detail
/// This function is public for macro expansion but should not be called directly.
/// Use `TryFrom::try_from()` or `.try_into()` for downcasting instead.
#[doc(hidden)]
pub fn try_downcast<From: ObjectRefCore, To: ObjectRefCore>(value: From) -> Result<To, From> {
    unsafe {
        let arc = <From as ObjectRefCore>::data(&value);
        let raw = ObjectArc::as_raw(arc) as *const Object as *const tvm_ffi_sys::TVMFFIObject;
        let type_index = (*raw).type_index;
        let target_index = <To as ObjectRefCore>::ContainerType::type_index();

        if is_instance_of(type_index, target_index) {
            // Type check passed, perform the downcast
            let arc = <From as ObjectRefCore>::into_data(value);
            let raw = ObjectArc::into_raw(arc);
            let casted = ObjectArc::from_raw(raw as *const <To as ObjectRefCore>::ContainerType);
            Ok(<To as ObjectRefCore>::from_data(casted))
        } else {
            // Type check failed, return the original value
            Err(value)
        }
    }
}
