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

//! The context an error carries to name the nodes a failed visit unwound
//! through.

// The Rust side of C++ `tvm/ffi/extra/visit_error_context.h`.

use crate::any::AnyView;
use crate::collections::array::Array;
use crate::derive::Object;
use crate::error::Error;
use crate::object::{self, Object, ObjectArc, ObjectRef, ObjectRefCore};
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFIObject, TVMFFITypeIndex};

/// The payload that `Error::extra_context` carries to name the nodes a failed
/// visit unwound through, laid out as the C++ `VisitErrorContextObj`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.VisitErrorContext"]
#[type_final]
pub(crate) struct VisitErrorContextObj {
    base_: Object,
    /// Visited nodes, innermost first; an entry may be `None`.
    reverse_visit_pattern: Array<Option<ObjectRef>>,
    /// The `extra_context` the error carried before this context replaced it.
    prev_error_context: Option<ObjectRef>,
}

/// Name the visited object in `raw` in the error's visit context.
// As C++ `detail::UpdateVisitErrorContext`.
#[cold]
pub(crate) fn update_visit_error_context(mut error: Error, raw: TVMFFIAny) -> Error {
    // Copy-on-write at each level: the Error, its context and the context's pattern
    // are each copied only when shared, so an error owned by the unwinding frame is
    // appended in place.
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
            if !ObjectArc::is_unique(&context) {
                context = ObjectArc::new(VisitErrorContextObj {
                    base_: Object::new(),
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
            base_: Object::new(),
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
