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
use crate::error::Error;
use crate::function::Function;
use crate::object::{self, ObjectRefCore};
use crate::reflection::FieldGetter;
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFIByteArray, TVMFFITypeIndex, TVMFFITypeKeyToIndex};

/// Initialize the borrowed ABI prefix of a scoped Rust-owned traversal engine.
///
/// # Safety
///
/// `header` points to the engine's stable Object prefix. The Box retains the
/// reserved owner reference and destroys the engine itself; low-level hooks may
/// only borrow it synchronously. All temporary strong and weak references must
/// end before the Box is dropped, so the foreign deleter is never used.
pub(crate) unsafe fn initialize_scoped_runtime(
    header: *mut crate::tvm_ffi_sys::TVMFFIObject,
    type_index: i32,
) {
    let mut value = crate::tvm_ffi_sys::TVMFFIObject::new();
    value.combined_ref_count =
        std::sync::atomic::AtomicU64::new(crate::tvm_ffi_sys::COMBINED_REF_COUNT_BOTH_ONE);
    value.type_index = type_index;
    std::ptr::write(header, value);
}

/// Add one structural traversal frame to an error's backtrace.
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
        let node = unsafe { borrowed_view(&raw) }.try_as::<object::ObjectRef>()?;
        let mut previous = error.extra_context();
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
                    .try_as::<i64>()?;
                let get_item = Function::get_global("ffi.ListGetItem").ok()?;
                if size > 0 {
                    let last = get_item
                        .call_tuple((records.clone(), size - 1))
                        .ok()?
                        .try_as::<object::ObjectRef>();
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
            .try_as::<object::ObjectRef>()
    })();
    // Diagnostic enrichment must not replace the original error on failure.
    match context {
        Some(context) => Error::new_with_cause_and_extra_context(
            error.kind(),
            error.message(),
            error.backtrace(),
            error.cause_chain().as_ref(),
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

/// Copy a borrowed FFI value when its owning representation is unchanged.
///
/// Raw string/byte views and `ObjectRValueRef` return `None` because they need
/// the runtime's normalization or move logic. A null object pointer also
/// returns `None` instead of constructing an invalid owning value.
#[inline]
pub(crate) fn try_to_owned_without_normalization(raw: TVMFFIAny) -> Option<Any> {
    if is_plain_inline(raw.type_index) {
        return Some(unsafe { Any::from_raw_ffi_any(raw) });
    }
    if raw.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        let object = unsafe { raw.data_union.v_obj };
        if object.is_null() {
            return None;
        }
        unsafe { object::unsafe_::inc_ref(object) };
        return Some(unsafe { Any::from_raw_ffi_any(raw) });
    }
    None
}

pub(crate) use crate::any::is_plain_inline;

#[inline]
pub(crate) fn same_shallow(lhs: TVMFFIAny, rhs: TVMFFIAny) -> bool {
    lhs.type_index == rhs.type_index
        && lhs.small_str_len == rhs.small_str_len
        && unsafe { lhs.data_union.v_uint64 == rhs.data_union.v_uint64 }
}

/// Borrow a traversal ABI cell for no longer than the cell's scope.
///
/// # Safety
/// The cell must satisfy `AnyView::from_raw_ffi_any`'s requirements, including
/// its unique, at-most-once conversion rule for rvalue slots. The traversal
/// driver must keep the owning root, field, or replacement alive for the borrow.
#[inline(always)]
pub(crate) unsafe fn borrowed_view(raw: &TVMFFIAny) -> AnyView<'_> {
    AnyView::from_raw_ffi_any(*raw)
}

#[inline(always)]
pub(crate) fn raw_of(value: AnyView<'_>) -> TVMFFIAny {
    *value.as_raw_ffi_any()
}
