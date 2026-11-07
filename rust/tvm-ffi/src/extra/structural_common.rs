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

use super::structural_visit::DefRegionKind;
use crate::any::Any;
use crate::error::{Error, Result, RUNTIME_ERROR};
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashDefPattern, kTVMFFIFieldFlagBitMaskSEqHashDefSimple,
    kTVMFFIFieldFlagBitMaskSEqHashIgnore, kTVMFFIFieldFlagBitSetterIsFunctionObj,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldInfo, TVMFFISEqHashKind, TVMFFITypeIndex,
    TVMFFITypeInfo, TVMFFITypeKeyToIndex,
};
use std::ops::ControlFlow;

// -----------------------------------------------------------------------------
// Shared outlined errors
// -----------------------------------------------------------------------------
// Perf note: Keep construction, formatting and checked-conversion scratch inside
// these cold outline boundaries, not in successful traversal callers' frames.

#[cold]
#[inline(never)]
pub(super) fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[cold]
#[inline(never)]
pub(super) fn wrong_dispatcher() -> ! {
    panic!("structural traversal view used with a different dispatcher instance")
}

#[cold]
#[inline(never)]
pub(super) fn missing_field_getter_error(operation: &str, name: &str) -> Error {
    runtime_error(&format!(
        "native {operation}: reflected field `{name}` has no getter"
    ))
}

#[cold]
#[inline(never)]
pub(super) fn with_structural_field_error_context(
    error: Error,
    operation: &str,
    name: &str,
) -> Error {
    with_structural_error_context(error, operation, &format!("field `{name}`"))
}

/// Consume an owning error carrier while keeping checked-conversion scratch
/// inside the cold path instead of successful recursive traversal frames.
///
/// # Safety
/// `raw` is one owning TVMFFIAny that has not been consumed.
// Perf note: Construct the addressable Any and perform checked Error conversion
// inside this outline boundary, not in the successful recursive caller's frame.
#[cold]
#[inline(never)]
pub(super) unsafe fn structural_error_from_raw(raw: TVMFFIAny) -> Error {
    let value = Any::from_raw_ffi_any(raw);
    match Error::try_from(value) {
        Ok(error) | Err(error) => error,
    }
}

// Perf note: cold marks these error-composition paths as uncommon; inlining
// within error paths is compiler-selected. Use explicit inline(never) at boundaries
// where scratch must stay outside recursive callers.
/// Add one structural traversal frame to an error's backtrace.
#[cold]
pub(crate) fn with_structural_error_context(error: Error, operation: &str, frame: &str) -> Error {
    Error::with_appended_backtrace(error, &format!("[native structural {operation}] {frame}\n"))
}

// -----------------------------------------------------------------------------
// Raw carrier utilities
// -----------------------------------------------------------------------------

/// Compare tags, inline metadata and payload bits without deep equality.
///
/// Object-backed values compare pointer identity. Both carriers must contain
/// initialized payload bits; no pointee is accessed or ownership transferred.
#[inline(always)]
pub(crate) fn raw_ffi_any_same_as(lhs: TVMFFIAny, rhs: TVMFFIAny) -> bool {
    lhs.type_index == rhs.type_index
        && lhs.small_str_len == rhs.small_str_len
        && unsafe { lhs.data_union.v_uint64 == rhs.data_union.v_uint64 }
}

/// Transfer an owning success or Error into exactly one raw result token.
///
/// This is the structural hook protocol: an Error tag represents failure, so
/// decoding `Ok(Any::from(error))` also yields `Err(error)`. Callers that carry
/// Error objects as ordinary values must keep a separate result channel.
/// No payload is cloned, normalized, or released by this transfer.
#[inline(always)]
pub(crate) fn result_into_raw_ffi_any(result: Result<Any>) -> TVMFFIAny {
    match result {
        Ok(value) => Any::into_raw_ffi_any(value),
        Err(error) => Any::into_raw_ffi_any(Any::from(error)),
    }
}

/// Take ownership of one value returned by a structural traversal ABI hook.
///
/// # Safety
///
/// `raw` must contain one owning TVMFFIAny result that has not already been
/// consumed. An Error object is converted into the Rust error channel, even
/// if it was encoded from `Ok`; all other tags become an owning success.
/// Borrowed views and rvalue slots must be normalized before this transfer.
#[inline(always)]
pub(crate) unsafe fn raw_ffi_any_into_result(raw: TVMFFIAny) -> Result<Any> {
    if raw.type_index != TVMFFITypeIndex::kTVMFFIError as i32 {
        return Ok(Any::from_raw_ffi_any(raw));
    }
    core::hint::cold_path();
    Err(structural_error_from_raw(raw))
}

pub(super) const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
pub(super) const FLAG_SEQ_HASH_DEF_PATTERN: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefPattern as i64;
pub(super) const FLAG_SEQ_HASH_DEF_SIMPLE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefSimple as i64;
pub(super) const FLAG_SETTER_IS_FUNCTION: i64 = kTVMFFIFieldFlagBitSetterIsFunctionObj as i64;

/// The runtime type index of the native type `key`; panics when it is unregistered.
#[cold]
pub(super) fn registered_type_index(key: &str) -> i32 {
    let mut type_index = 0;
    // SAFETY: `key` outlives the lookup.
    if unsafe { TVMFFITypeKeyToIndex(&TVMFFIByteArray::from_str(key), &mut type_index) } != 0 {
        panic!("{key} is not registered: {}", Error::from_raised());
    }
    type_index
}

/// The registered `structural_eq_hash_kind` of `info`, `Unsupported` without metadata.
///
/// # Safety
///
/// `info` is a registered type-info record.
#[inline(always)]
pub(super) unsafe fn seq_hash_kind(info: *const TVMFFITypeInfo) -> i32 {
    let metadata = (*info).metadata;
    if metadata.is_null() {
        TVMFFISEqHashKind::kTVMFFISEqHashKindUnsupported as i32
    } else {
        (*metadata).structural_eq_hash_kind
    }
}

#[inline(always)]
pub(super) fn field_def_region(field: &TVMFFIFieldInfo, inherited: DefRegionKind) -> DefRegionKind {
    // Precedence: a pattern region propagates; entering any kind inside it has no effect.
    if inherited == DefRegionKind::Pattern {
        DefRegionKind::Pattern
    } else if field.flags & FLAG_SEQ_HASH_DEF_SIMPLE != 0 {
        DefRegionKind::Simple
    } else if field.flags & FLAG_SEQ_HASH_DEF_PATTERN != 0 {
        DefRegionKind::Pattern
    } else {
        inherited
    }
}

/// A simple definition applies to a FreeVar value itself, but not to
/// the FreeVar's own reflected children: nested free vars there must resolve
/// against an outer binding instead of rebinding.
// Mirrors C++ `VisitReflectedFieldsExpected`.
#[inline(always)]
pub(super) fn free_var_child_region(
    inherited: DefRegionKind,
    structural_eq_hash_kind: i32,
) -> DefRegionKind {
    if inherited == DefRegionKind::Simple
        && structural_eq_hash_kind == TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar as i32
    {
        DefRegionKind::None
    } else {
        inherited
    }
}

/// Visit every reflected field described by `info` and its ancestors, from
/// the outermost parent's fields to the type's own.
///
/// # Safety
///
/// `info` must point to an immortal registered type-info record.
pub(super) unsafe fn for_each_field_info<B>(
    info: *const TVMFFITypeInfo,
    callback: &mut impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    // Ancestor slot 0 is the root Object. C++ starts at slot 1, walks toward
    // the immediate parent, then visits the concrete type's own fields.
    for depth in 1..(*info).type_depth {
        let ancestor = *(*info).type_acenstors.offset(depth as isize);
        if let Some(value) = visit_field_level(ancestor, callback) {
            return Some(value);
        }
    }
    visit_field_level(info, callback)
}

unsafe fn visit_field_level<B>(
    info: *const TVMFFITypeInfo,
    callback: &mut impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    if info.is_null() {
        return None;
    }
    let fields =
        crate::util::slice_from_nullable_raw_parts((*info).fields, (*info).num_fields as usize);
    for field in fields {
        // C reflection tables are immortal once registered.
        let field: &'static TVMFFIFieldInfo = &*(field as *const TVMFFIFieldInfo);
        if let ControlFlow::Break(value) = callback(field) {
            return Some(value);
        }
    }
    None
}
