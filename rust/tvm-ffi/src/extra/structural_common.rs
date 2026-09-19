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
use crate::object::{self, ObjectCore, ObjectRefCore};
use crate::reflection::FieldGetter;
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIGetTypeInfo, TVMFFITypeIndex, TVMFFITypeKeyToIndex,
};

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
        let node = StructuralView::from_raw(raw).cast::<object::ObjectRef>()?;
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
                        .try_as::<object::ObjectRef>()?;
                    // Callback, policy and default descent can report the same frame.
                    if last.same_as(&node) {
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

/// A borrowed value shared by structural visit, walk, map, and mutate callbacks.
///
/// Use `&StructuralView` for an erased callback argument. [`Self::as_node`]
/// borrows an object node, while [`Self::cast`] returns a typed value (acquiring
/// ownership for object handles). This view does not grant in-place permission;
/// consuming mutation callbacks use [`crate::MutateValue`] instead.
#[repr(transparent)]
pub struct StructuralView(TVMFFIAny);

impl<'a> From<&'a StructuralView> for AnyView<'a> {
    #[inline]
    fn from(value: &'a StructuralView) -> Self {
        // SAFETY: the view cannot outlive the borrowed structural value.
        unsafe { AnyView::from_raw_ffi_any(value.raw()) }
    }
}

impl StructuralView {
    #[inline]
    pub(crate) fn from_raw(raw: TVMFFIAny) -> Self {
        Self(raw)
    }

    #[inline]
    pub(crate) fn raw(&self) -> TVMFFIAny {
        self.0
    }

    /// Copy this borrowed value into an owning [`Any`].
    #[inline]
    pub fn to_owned(&self) -> Any {
        if let Some(owned) = try_to_owned_without_normalization(self.0) {
            return owned;
        }
        Any::from(unsafe { AnyView::from_raw_ffi_any(self.0) })
    }

    /// Convert the value into an owned typed handle.
    #[inline]
    pub fn cast<R: crate::type_traits::AnyCompatible>(&self) -> Option<R> {
        unsafe {
            if R::check_any_strict(&self.0) {
                Some(R::copy_from_any_view_after_check(&self.0))
            } else {
                None
            }
        }
    }

    /// Runtime type index stored in this value.
    #[inline]
    pub fn type_index(&self) -> i32 {
        self.0.type_index
    }

    /// Borrow the value as node type `N` if it is an instance of that type.
    #[inline]
    pub fn as_node<N: ObjectCore>(&self) -> Option<&N> {
        if self.0.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return None;
        }
        let base_type_index = N::type_index();
        if self.0.type_index != base_type_index {
            // A final type has no registered subtype, so a differing index can
            // never match: reject with the integer compare alone, mirroring the
            // `_type_final` fast path of C++ `IsObjectInstance`.
            if N::TYPE_FINAL {
                return None;
            }
            if !is_instance_at_depth(self.0.type_index, base_type_index, N::TYPE_DEPTH) {
                return None;
            }
        }
        Some(unsafe { &*(self.0.data_union.v_obj as *const N) })
    }
}

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

/// Subtype check with the base's inheritance depth supplied by the caller
/// (`ObjectCore::TYPE_DEPTH`), so only the object's type info is fetched.
#[inline]
fn is_instance_at_depth(object_type_index: i32, base_type_index: i32, base_depth: i32) -> bool {
    if object_type_index == base_type_index {
        return true;
    }
    // Parent type indices are registered before their descendants. An object
    // whose index precedes the target therefore cannot be its subtype.
    if object_type_index < base_type_index {
        return false;
    }
    unsafe {
        let info = TVMFFIGetTypeInfo(object_type_index);
        if info.is_null() {
            return false;
        }
        if (*info).type_depth <= base_depth {
            return false;
        }
        let ancestors = (*info).type_acenstors;
        if ancestors.is_null() {
            return false;
        }
        let ancestor = *ancestors.offset(base_depth as isize);
        !ancestor.is_null() && (*ancestor).type_index == base_type_index
    }
}
