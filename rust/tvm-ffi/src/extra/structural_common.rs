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
use crate::object::{self, ObjectCore};
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFIGetTypeInfo, TVMFFITypeIndex};

/// Add one structural traversal frame to an error's backtrace.
pub(crate) fn with_structural_error_context(error: Error, operation: &str, frame: &str) -> Error {
    Error::with_appended_backtrace(error, &format!("[native structural {operation}] {frame}\n"))
}

// Rust treats callback tuples of different lengths as different types, so
// each flat tuple arity needs its own impl. Generate arities 1 through 12
// here — the same width the standard library implements its tuple traits
// for — so structural visit and map accept the same flat tuple widths. This
// is a per-level width, not a cap on chain length: a tuple is itself a
// link, so nesting `(a, b, (c, d, ...))` chains any number of callbacks.
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

/// A borrowed value shared by structural visit and map callbacks.
///
/// This type centralizes the audited unsafe operations used to cast FFI
/// values and borrow object nodes. The public APIs expose it as `VisitValue`
/// or `MapValue` according to the callback context.
#[repr(transparent)]
pub struct StructuralValue(TVMFFIAny);

impl StructuralValue {
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
    if is_plain_inline_leaf(raw.type_index) {
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

#[inline]
pub(crate) fn is_plain_inline_leaf(type_index: i32) -> bool {
    type_index < TVMFFITypeIndex::kTVMFFIRawStr as i32
        || type_index == TVMFFITypeIndex::kTVMFFISmallStr as i32
        || type_index == TVMFFITypeIndex::kTVMFFISmallBytes as i32
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
