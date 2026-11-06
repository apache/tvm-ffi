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
use crate::error::{Error, Result, RUNTIME_ERROR};
use crate::function::Function;
use crate::object::{self, ObjectRefCast, ObjectRefCore};
use crate::reflection::FieldGetter;
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIByteArray, TVMFFIGetTypeInfo, TVMFFITypeIndex,
    TVMFFITypeKeyToIndex,
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

#[cold]
pub(crate) fn with_visit_error_context(error: Error, raw: TVMFFIAny) -> Error {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return error;
    }
    let context = (|| {
        let mut context_type = 0;
        unsafe {
            crate::check_safe_call!(TVMFFITypeKeyToIndex(
                &TVMFFIByteArray::from_str("ffi.VisitErrorContext"),
                &mut context_type,
            ))
            .ok()?;
        }
        // SAFETY: the traversal input remains live during annotation. Conversion
        // acquires ownership before the temporary view ends; raw is an object tag.
        let node = unsafe { AnyView::from_raw_ffi_any(raw) }.as_strict::<object::ObjectRef>()?;
        let mut previous = error.extra_context().cloned();
        let mut nodes = Vec::new();
        if let Some(prior) = previous.as_ref() {
            if AnyView::from(prior).type_index() == context_type {
                let object = &**object::ObjectRef::data(prior);
                let records = FieldGetter::new(context_type, "reverse_visit_pattern")
                    .ok()?
                    .get_any(object)
                    .ok()?;
                let size = Function::get_global("ffi.ListSize")
                    .ok()?
                    .call_tuple((records.clone(),))
                    .ok()?
                    .as_strict::<i64>()?;
                let get_item = Function::get_global("ffi.ListGetItem").ok()?;
                if size > 0 {
                    let last = get_item
                        .call_tuple((records.clone(), size - 1))
                        .ok()?
                        .as_strict::<object::ObjectRef>();
                    // Callback, policy and default descent can report the same frame.
                    if last.is_some_and(|last| last.same_as(&node)) {
                        return None;
                    }
                }
                for i in 0..size {
                    nodes.push(get_item.call_tuple((records.clone(), i)).ok()?);
                }
                previous = FieldGetter::new(context_type, "prev_error_context")
                    .ok()?
                    .get::<_, Option<object::ObjectRef>>(object)
                    .ok()?;
            }
        }
        nodes.push(Any::from(node));
        let records = Function::get_global("ffi.List")
            .ok()?
            .call_packed(&nodes.iter().map(AnyView::from).collect::<Vec<_>>())
            .ok()?;
        Function::get_global("ffi.MakeObjectFromPackedArgs")
            .ok()?
            .call_tuple((
                context_type,
                crate::String::from("reverse_visit_pattern"),
                records,
                crate::String::from("prev_error_context"),
                previous,
            ))
            .ok()?
            .as_strict::<object::ObjectRef>()
    })();
    // Diagnostic enrichment must not replace the original error on failure.
    match context {
        Some(context) => Error::new_with_cause_and_extra_context(
            error.kind(),
            error.message(),
            error.backtrace(),
            error.cause_chain(),
            Some(&context),
        ),
        None => error,
    }
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
