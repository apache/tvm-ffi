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

use crate::any::{Any, AnyView};
use crate::collections::array::Array;
use crate::derive::Object;
use crate::error::{Error, Result, RUNTIME_ERROR};
use crate::object::{self, Object, ObjectArc, ObjectRef, ObjectRefCore};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeIndex,
};

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
/// The fields must describe one owning TVMFFIAny that has not been consumed.
// Perf note: Construct the addressable Any and perform checked Error conversion
// inside this outline boundary, not in the successful recursive caller's frame.
#[cold]
#[inline(never)]
pub(super) unsafe fn structural_error_from_raw(
    type_index: i32,
    small_str_len: u32,
    data_union: TVMFFIAnyDataUnion,
) -> Error {
    let value = Any::from_raw_ffi_any(TVMFFIAny {
        type_index,
        small_str_len,
        data_union,
    });
    match Error::try_from(value) {
        Ok(error) | Err(error) => error,
    }
}

/// Return the registered type key for `type_index`.
///
/// When the runtime returns no type information, return `<type_index N>` for diagnostics.
///
/// # Arguments
///
/// * `type_index` - Runtime type index whose registered type key is requested.
pub(crate) fn type_key_of(type_index: i32) -> String {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            format!("<type_index {type_index}>")
        } else {
            (*info).type_key.as_str().to_string()
        }
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

/// Rust binding of C++ `VisitErrorContextObj`, the payload that
/// `Error::extra_context` carries to name the nodes a failed visit unwound through.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.VisitErrorContext"]
#[type_final]
pub(crate) struct VisitErrorContextObj {
    base: Object,
    /// Visited nodes, innermost first; C++ `ObjectRef` is nullable.
    reverse_visit_pattern: Array<Option<ObjectRef>>,
    /// The `extra_context` the error carried before this context replaced it.
    prev_error_context: Option<ObjectRef>,
}

/// Name the visited object in `raw` in the error's visit context, like C++
/// `detail::UpdateVisitErrorContext`. Copy-on-write at each level: the Error, its
/// context and the context's pattern are each copied only when shared, so an error
/// owned by the unwinding frame is appended in place.
#[cold]
pub(crate) fn with_visit_error_context(mut error: Error, raw: TVMFFIAny) -> Error {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return error;
    }
    // SAFETY: the traversal input remains live during annotation. Conversion
    // acquires ownership before the temporary view ends; raw is an object tag.
    let Ok(node) = ObjectRef::try_from(unsafe { AnyView::from_raw_ffi_any(raw) }) else {
        return error;
    };
    let as_context = |context: &ObjectRef| {
        let object = ObjectArc::as_ptr(ObjectRef::data(context));
        // SAFETY: `context` keeps its object alive; the type is checked first.
        unsafe {
            object::is_instance_of::<VisitErrorContextObj>(
                (*object.cast::<TVMFFIObject>()).type_index,
            )
            .then(|| &*object.cast::<VisitErrorContextObj>())
        }
    };
    // Callback, policy and default descent can report the same frame.
    if let Some(context) = error.extra_context().and_then(as_context) {
        if context
            .reverse_visit_pattern
            .iter()
            .next_back()
            .and_then(Option::as_ref)
            .is_some_and(|last| last.same_as(&node))
        {
            return error;
        }
    }
    let slot = error.extra_context_mut();
    let previous = slot.take();
    let context = match previous {
        Some(previous) if as_context(&previous).is_some() => {
            // SAFETY: `previous` was checked to be a `VisitErrorContextObj`.
            let mut context = unsafe {
                ObjectArc::from_raw(
                    ObjectArc::into_raw(ObjectRef::into_data(previous))
                        .cast::<VisitErrorContextObj>(),
                )
            };
            if ObjectArc::strong_count(&context) != 1 {
                context = ObjectArc::new(VisitErrorContextObj {
                    base: Object::new(),
                    reverse_visit_pattern: context.reverse_visit_pattern.clone(),
                    prev_error_context: context.prev_error_context.clone(),
                });
            }
            // SAFETY: this handle is the context's only reference.
            unsafe {
                (*ObjectArc::as_mut_ptr(&mut context))
                    .reverse_visit_pattern
                    .push(Some(node))
            };
            context
        }
        previous => ObjectArc::new(VisitErrorContextObj {
            base: Object::new(),
            reverse_visit_pattern: Array::from_vec(vec![Some(node)]),
            prev_error_context: previous,
        }),
    };
    // SAFETY: every object starts with the root `Object` header.
    *slot = Some(unsafe {
        ObjectRef::from_data(ObjectArc::from_raw(
            ObjectArc::into_raw(context).cast::<Object>(),
        ))
    });
    error
}

// Generate the tuple arities supported by the standard library (1 through 12).
macro_rules! impl_callback_chain_tuple_arities {
    ($impl_chain:ident) => {
        impl_callback_chain_tuple_arities!(
            @prefixes $impl_chain;
            [];
            (F0, M0, 0),
            (F1, M1, 1),
            (F2, M2, 2),
            (F3, M3, 3),
            (F4, M4, 4),
            (F5, M5, 5),
            (F6, M6, 6),
            (F7, M7, 7),
            (F8, M8, 8),
            (F9, M9, 9),
            (F10, M10, 10),
            (F11, M11, 11)
        );
    };
    (@prefixes $impl_chain:ident; [$($prefix:tt)*];) => {};
    (
        @prefixes $impl_chain:ident;
        [$($prefix:tt)*];
        $next:tt $(, $rest:tt)*
    ) => {
        $impl_chain!($($prefix)* $next);
        impl_callback_chain_tuple_arities!(
            @prefixes $impl_chain;
            [$($prefix)* $next,];
            $($rest),*
        );
    };
}

pub(crate) use impl_callback_chain_tuple_arities;

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
    Err(structural_error_from_raw(
        raw.type_index,
        raw.small_str_len,
        raw.data_union,
    ))
}
