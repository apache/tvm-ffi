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
//! Rust bindings for `tvm::ffi::StructuralVisitor`.
//!
//! The C++ side exposes the visitor as a layout-stable struct containing a
//! pair of function pointers (safe-call and optional C++ fast path), an opaque
//! self pointer, and a def-region tag. This module mirrors that struct with a
//! `#[repr(C)]` Rust type so visitors can be authored in either language and
//! interoperate over the FFI.

use crate::derive::{Object, ObjectRef};
use crate::error::{Error, Result};
use crate::object::{Object, ObjectArc, ObjectRef as BaseObjectRef, ObjectRefCore};

use std::ffi::{c_int, c_void};
use std::marker::PhantomPinned;
use std::pin::Pin;

use tvm_ffi_sys::{TVMFFIAny, TVMFFIObjectHandle};

//-----------------------------------------------------
// DefRegionKind
//-----------------------------------------------------

/// Mirrors C++ `TVMFFIDefRegionKind`.
///
/// Identifies whether the visitor is currently inside a def-region (a
/// binding scope that affects structural eq/hash semantics).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefRegionKind {
    None = 0,
    Recursive = 1,
    NonRecursive = 2,
}

impl DefRegionKind {
    /// Convert from the raw `i32` carried in `StructuralVisitor::def_region_mode`.
    pub fn from_i32(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Recursive),
            2 => Ok(Self::NonRecursive),
            _ => crate::bail!(crate::error::VALUE_ERROR, "Invalid DefRegionKind: {}", value),
        }
    }
}

//-----------------------------------------------------
// VisitInterrupt
//-----------------------------------------------------

/// Object node carrying the optional payload for an interrupted structural visit.
///
/// Mirrors C++ `tvm::ffi::VisitInterruptObj`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.VisitInterrupt"]
pub struct VisitInterruptObj {
    object: Object,
    /// Payload returned with the interrupt, or FFI None for no payload.
    pub value: TVMFFIAny,
}

/// ABI-stable owned `VisitInterrupt` ref class.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct VisitInterrupt {
    data: ObjectArc<VisitInterruptObj>,
}

//-----------------------------------------------------
// StructuralVisitor
//-----------------------------------------------------

/// C-ABI safe-call signature for structural visit.
///
/// Matches `tvm::ffi::FStructuralVisitSafe` on the C++ side. The output is
/// written as a raw `TVMFFIObjectHandle`: `NULL` means no interrupt, otherwise
/// the callee transfers ownership of a strong reference to a
/// `VisitInterruptObj` to the caller.
///
/// # Error propagation
///
/// Returns `0` on success and any non-zero value on error. The error is set
/// in thread-local storage via `TVMFFIErrorSetRaised` (or [`Error::set_raised`])
/// and retrieved by the caller via [`Error::from_raised`].
pub type StructuralVisitSafeCallType = unsafe extern "C" fn(
    self_ptr: *mut c_void,
    value: *const BaseObjectRef,
    out_interrupt: *mut TVMFFIObjectHandle,
) -> c_int;

/// Layout-compatible mirror of C++ `tvm::ffi::StructuralVisitor`.
///
/// Field order MUST match the C++ class for ABI compatibility:
///
/// 1. `safe_visit` — required C-ABI safe-call entry.
/// 2. `cpp_visit`  — optional C++ fast-path entry (always null on the Rust side).
/// 3. `self_ptr`   — opaque self pointer forwarded to the entries above.
/// 4. `def_region_mode` — current def-region context.
#[repr(C)]
pub struct StructuralVisitor {
    pub safe_visit: Option<StructuralVisitSafeCallType>,
    pub cpp_visit: *mut c_void,
    pub self_ptr: *mut c_void,
    pub def_region_mode: c_int,
}

impl StructuralVisitor {
    /// Visit a value, dispatching through this visitor's `safe_visit` entry.
    ///
    /// Returns `Ok(None)` to continue traversal, `Ok(Some(interrupt))` to halt
    /// with an interrupt payload, or `Err(_)` if the underlying visitor raised
    /// an error.
    pub fn visit(&mut self, value: &BaseObjectRef) -> Result<Option<VisitInterrupt>> {
        let safe_visit = self.safe_visit.expect("StructuralVisitor::safe_visit is null");
        unsafe {
            let mut out: TVMFFIObjectHandle = std::ptr::null_mut();
            let ret_code = safe_visit(self.self_ptr, value as *const BaseObjectRef, &mut out);
            if ret_code != 0 {
                return Err(Error::from_raised());
            }
            if out.is_null() {
                Ok(None)
            } else {
                let arc = ObjectArc::<VisitInterruptObj>::from_raw(out as *const VisitInterruptObj);
                Ok(Some(VisitInterrupt::from_data(arc)))
            }
        }
    }

    /// Get the current def-region context.
    pub fn def_region_kind(&self) -> DefRegionKind {
        DefRegionKind::from_i32(self.def_region_mode).unwrap_or(DefRegionKind::None)
    }

    /// Temporarily switch the def-region context while invoking `callback`.
    pub fn with_def_region_kind<R>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let saved = self.def_region_mode;
        self.def_region_mode = kind as c_int;
        let result = callback(self);
        self.def_region_mode = saved;
        result
    }
}

//-----------------------------------------------------
// RustStructuralVisitor: build a StructuralVisitor from a Rust closure
//-----------------------------------------------------

/// A Rust-defined structural visitor backed by a closure.
///
/// The closure receives the active visitor (so it can recurse via
/// `visitor.visit(...)`) and the value being visited. Returning:
///
/// * `Ok(None)` continues traversal,
/// * `Ok(Some(interrupt))` halts with an interrupt payload,
/// * `Err(_)` propagates an error across the FFI boundary via
///   `TVMFFIErrorSetRaised`, where the C++ caller will rethrow it.
///
/// Panics inside the closure are caught and converted into a `RuntimeError`
/// so that the C++ caller never observes an unwind across the FFI boundary.
///
/// The returned [`Pin<Box<_>>`] must outlive any consumer holding the
/// underlying [`StructuralVisitor`]; the pin keeps the embedded `self_ptr`
/// valid for the lifetime of the box.
pub struct RustStructuralVisitor<F>
where
    F: FnMut(&mut StructuralVisitor, &BaseObjectRef) -> Result<Option<VisitInterrupt>>,
{
    base: StructuralVisitor,
    callback: F,
    _pin: PhantomPinned,
}

impl<F> RustStructuralVisitor<F>
where
    F: FnMut(&mut StructuralVisitor, &BaseObjectRef) -> Result<Option<VisitInterrupt>>,
{
    /// Construct a new Rust-defined structural visitor.
    pub fn new(callback: F) -> Pin<Box<Self>> {
        let mut boxed = Box::pin(Self {
            base: StructuralVisitor {
                safe_visit: Some(Self::safe_visit_thunk),
                cpp_visit: std::ptr::null_mut(),
                self_ptr: std::ptr::null_mut(),
                def_region_mode: DefRegionKind::None as c_int,
            },
            callback,
            _pin: PhantomPinned,
        });
        unsafe {
            let this = Pin::into_inner_unchecked(boxed.as_mut());
            this.base.self_ptr = this as *mut Self as *mut c_void;
        }
        boxed
    }

    /// Get a mutable reference to the underlying [`StructuralVisitor`].
    ///
    /// Useful when the visitor must be passed by `&mut StructuralVisitor` to
    /// another API that operates on the C-ABI shape directly.
    pub fn as_visitor_mut(self: Pin<&mut Self>) -> &mut StructuralVisitor {
        unsafe { &mut Pin::into_inner_unchecked(self).base }
    }

    /// Convenience: invoke [`StructuralVisitor::visit`] on this visitor.
    pub fn visit(
        self: Pin<&mut Self>,
        value: &BaseObjectRef,
    ) -> Result<Option<VisitInterrupt>> {
        self.as_visitor_mut().visit(value)
    }

    /// Safe-call thunk plugged into `base.safe_visit`.
    ///
    /// Translates between Rust's panic/`Result`-based error reporting and the
    /// FFI's "set raised error then return non-zero" convention.
    unsafe extern "C" fn safe_visit_thunk(
        self_ptr: *mut c_void,
        value: *const BaseObjectRef,
        out_interrupt: *mut TVMFFIObjectHandle,
    ) -> c_int {
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let this = &mut *(self_ptr as *mut Self);
            let value_ref = &*value;
            (this.callback)(&mut this.base, value_ref)
        }));

        match panic_result {
            Ok(Ok(None)) => {
                *out_interrupt = std::ptr::null_mut();
                0
            }
            Ok(Ok(Some(interrupt))) => {
                let arc = VisitInterrupt::into_data(interrupt);
                *out_interrupt = ObjectArc::into_raw(arc) as TVMFFIObjectHandle;
                0
            }
            Ok(Err(err)) => {
                Error::set_raised(&err);
                -1
            }
            Err(_panic) => {
                let err = Error::new(
                    crate::error::RUNTIME_ERROR,
                    "panic in Rust structural visit callback",
                    "",
                );
                Error::set_raised(&err);
                -1
            }
        }
    }
}
