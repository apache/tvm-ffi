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

//! Native Rust structural visiting.
//!
//! Two public layers mirror the C++ API split:
//!
//! * [`Dispatcher`](crate::Dispatcher) + [`structural_visit`] — handlers drive
//!   recursion through an explicit [`StructuralVisitorView`] context. Pass state belongs on
//!   the dispatcher; `visitor.visit(self, &child)` reborrows it for recursive calls.
//!   Unmatched values use default child traversal.
//! * [`structural_walk`] — dispatcher or closure observers steer engine-owned
//!   recursion through [`WalkResult`] (advance, skip, interrupt).
//!
//! Both layers thread definition-region state explicitly. Walk handlers opt in
//! with a trailing [`DefRegionKind`]; visit handlers read it from their context
//! and may override it for a child through [`StructuralVisitorView::with_def_region_kind`].
//!
//! # Safety of the implementation
//!
//! The public interface is safe; its implementation deliberately uses raw pointers.
//! Each private runtime has a matching dispatcher vtable and a fixed dispatcher
//! identity. The dispatcher stays alive in the outer call, and the engine's dispatcher
//! pointer publishes the actual exclusive reborrow before native recursion, then
//! restores the enclosing borrow on return. Views and native runtime handles cannot
//! escape their synchronous calls. No reference to the whole runtime spans recursion,
//! and owning ABI results transfer exactly one reference at each boundary.
//!
//! Callbacks must return failures through [`Result`] and must not panic. A panic
//! reaching the non-unwinding `extern "C"` callback boundary aborts the process;
//! visitor callbacks do not catch or transport panic payloads across native frames.

// -----------------------------------------------------------------------------
// Implementation overview
// -----------------------------------------------------------------------------
//
// This file is organized into:
// - Data structures and visiting capabilities.
// - Outlined errors and private utility functions.
// - Engine implementation and the public entry point.
//
// We use narrowly scoped unsafe code for native interoperation and efficient
// traversal. Safety and ownership contracts are documented at each boundary.
//
// The main performance strategies are:
// - Outline cold work to keep its temporary storage out of recursive frames.
// - Use _raw helpers and private extern "C-unwind" boundaries for register
//   transport on AArch64 while preserving Rust unwinding.
// - Inline small conversions and forwarding code so ownership transfers and
//   redundant cleanup can optimize away.
//
// Local Perf note comments explain individual implementation choices.

use std::marker::PhantomData;

use super::marker::{Borrowed, Owned};
use std::ops::ControlFlow;
use std::os::raw::c_void;
use std::rc::Rc;

use super::cold_init_lazy_lock::ColdInitLazyLock;
pub use super::structural_walk::*;

use crate::any::{Any, AnyView};
use crate::derive::Object;
use crate::error::{Error, Result, TYPE_ERROR};
use crate::function::Function;
use crate::object::{Object, ObjectArc};
use crate::reflection::TypeAttrColumn;
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashDefPattern, kTVMFFIFieldFlagBitMaskSEqHashDefSimple,
    kTVMFFIFieldFlagBitMaskSEqHashIgnore,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIByteArray, TVMFFIDefRegionKind, TVMFFIFieldInfo,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFISEqHashKind, TVMFFITypeIndex, TVMFFITypeKeyToIndex,
};

use super::structural_common::{
    missing_field_getter_error, runtime_error, type_key_of, with_structural_error_context,
    with_structural_field_error_context, with_visit_error_context,
};

const STRUCTURAL_VISIT_ATTR: &str = "__s_visit__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SEQ_HASH_DEF_PATTERN: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefPattern as i64;
const FLAG_SEQ_HASH_DEF_SIMPLE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefSimple as i64;

// -----------------------------------------------------------------------------
// Data structures and FFI layout
// -----------------------------------------------------------------------------

type StructuralVisitorHandle = *mut StructuralVisitorObj;
type FStructuralVisit =
    unsafe extern "C" fn(StructuralVisitorHandle, AnyView<'static>) -> TVMFFIAny;

/// Exact C++ visitor vtable ABI.
#[repr(C)]
struct StructuralVisitorVTable {
    visit: FStructuralVisit,
}

/// Exact C++ StructuralVisitorObj ABI prefix.
#[repr(C)]
pub struct StructuralVisitorObj {
    base: Object,
    vtable: *const StructuralVisitorVTable,
    def_region_mode: i32,
}

/// Native `ffi.VisitInterrupt` object containing the interruption payload.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.VisitInterrupt"]
#[type_index(TVMFFITypeIndex::kTVMFFIVisitInterrupt)]
#[type_final]
pub struct VisitInterruptObj {
    base: Object,
    /// Payload returned with the interrupt, or FFI `None` for no payload.
    pub value: Any,
}

const _: () = {
    assert!(
        std::mem::offset_of!(StructuralVisitorObj, vtable) == std::mem::size_of::<TVMFFIObject>()
    );
    assert!(
        std::mem::offset_of!(StructuralVisitorObj, def_region_mode)
            == std::mem::size_of::<TVMFFIObject>() + std::mem::size_of::<*const c_void>()
    );
    assert!(std::mem::offset_of!(VisitInterruptObj, value) == std::mem::size_of::<TVMFFIObject>());
};

// Scoped runtime prefixes are not independently allocatable ObjectCore values.
impl StructuralVisitorObj {
    const TYPE_KEY: &'static str = "ffi.StructuralVisitor";

    fn type_index() -> i32 {
        static TYPE_INDEX: ColdInitLazyLock<i32> = ColdInitLazyLock::new(|| unsafe {
            let key = TVMFFIByteArray::from_str(StructuralVisitorObj::TYPE_KEY);
            let mut type_index = 0;
            let return_code = TVMFFITypeKeyToIndex(&key, &mut type_index);
            if return_code != 0 {
                panic!(
                    "ffi.StructuralVisitor is not registered: {}",
                    Error::from_raised()
                );
            }
            type_index
        });
        *TYPE_INDEX
    }
}

// The dispatcher is a raw reborrow handle; its storage belongs to the outer call.
// The runtime contains only the fixed ABI base and the dispatcher reborrow.
#[repr(C)]
pub(super) struct StructuralVisitEngine {
    base: StructuralVisitorObj,
    dispatcher: *mut c_void,
}

impl Default for StructuralVisitorObj {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base: Object::new(),
            vtable: std::ptr::null(),
            def_region_mode: DefRegionKind::None as i32,
        }
    }
}

/// Definition-region state active at the current value.
///
/// Reflected fields marked `SEqHashDefPattern` or
/// `SEqHashDefSimple` override the inherited state for that field's
/// complete recursive visit.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DefRegionKind {
    /// The value is outside a definition region.
    #[default]
    None = 0,
    /// Definitions apply recursively through the visited value.
    Pattern = 1,
    /// Definitions apply to the visited value alone; its type is walked as uses.
    Simple = 2,
}

const _: () = {
    assert!(DefRegionKind::None as i32 == TVMFFIDefRegionKind::kTVMFFIDefRegionKindNone as i32);
    assert!(
        DefRegionKind::Pattern as i32 == TVMFFIDefRegionKind::kTVMFFIDefRegionKindPattern as i32
    );
    assert!(DefRegionKind::Simple as i32 == TVMFFIDefRegionKind::kTVMFFIDefRegionKindSimple as i32);
};

crate::def_object_ref! {
    /// Interrupt state of a traversal, mirroring C++ `ffi.VisitInterrupt`.
    ///
    /// Entry points and visitor-layer calls return
    /// `Result<Option<VisitInterrupt>>`: `Ok(None)` means the (sub)graph was
    /// visited completely, `Ok(Some(..))` means a handler halted the traversal
    /// with this interrupt, and `Err` means it failed.
    ///
    /// This is an owning object reference, like [`Error`]. Payload reads through
    /// `interrupt.value` borrow the shared [`VisitInterruptObj`]; use [`Self::with`]
    /// instead of a payload-only struct literal. Moving or replacing the payload
    /// through this reference is not supported; clone it to retain a separate owner.
    pub VisitInterrupt, crate::object::ObjectRef, VisitInterruptObj
}

impl VisitInterrupt {
    /// Interrupt carrying an FFI-compatible payload.
    #[inline(always)]
    pub fn with<T: Into<Any>>(payload: T) -> Self {
        VisitInterruptObj {
            base: Object::new(),
            value: payload.into(),
        }
        .into()
    }
}

// -----------------------------------------------------------------------------
// Public visitor API
// -----------------------------------------------------------------------------

// Internal visit protocol matches the C++ owning TVMFFIAny result: None means
// continue; VisitInterrupt (including None/Error payloads) and Error stop descent.
// Each ABI ingress/import and egress/export transfers ownership exactly once.
// Recursive drivers move this carrier unchanged; typed public APIs decode it.
/// Recursive visiting and default child descent for a dispatcher.
///
/// Pass the handler's `self` to visit methods to recurse with the same state.
pub struct StructuralVisitorView<'a, D> {
    runtime: StructuralVisitorHandle,
    _scope: PhantomData<(&'a mut (), fn(&mut D), Rc<()>)>,
}

impl<D> StructuralVisitorView<'_, D> {
    #[inline(always)]
    pub fn def_region_kind(&self) -> DefRegionKind {
        // SAFETY: engine scopes and trusted ABI hooks store valid enum values.
        unsafe { std::mem::transmute((*self.runtime).def_region_mode) }
    }

    /// Run operations under a scoped definition region.
    /// An enclosing Pattern wins; the previous region is restored on all exits.
    #[inline(always)]
    pub fn with_def_region_kind<R>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut Self) -> R,
    ) -> R {
        StructuralVisitEngine::with_visitor_def_region(self.runtime, kind, || callback(self))
    }

    /// Visit a value through the dispatcher.
    #[inline(always)]
    pub fn visit<'input, T>(&mut self, dispatch: &mut D, value: T) -> Result<Option<VisitInterrupt>>
    where
        T: Into<AnyView<'input>>,
    {
        let value: AnyView<'input> = value.into();
        // SAFETY: the view ties this runtime to D; the check establishes the
        // instance, and publishing uses the caller's actual exclusive reborrow.
        unsafe {
            visit_result_from_any(Any::from_raw_ffi_any(StructuralVisitEngine::visit_raw(
                self.runtime,
                dispatch,
                *value.as_raw_ffi_any(),
            )))
        }
    }

    /// Run default descent through registered hooks or reflected fields.
    /// Children re-enter the dispatcher; the current value does not.
    #[inline(always)]
    pub fn default_visit<'input, T>(
        &self,
        dispatch: &mut D,
        value: T,
    ) -> Result<Option<VisitInterrupt>>
    where
        T: Into<AnyView<'input>>,
    {
        // SAFETY: type/layout and scope come from the private view constructor;
        // the supplied borrow must identify this runtime's dispatcher instance.
        unsafe {
            let context = StructuralVisitEngine::checked_dispatcher(self.runtime, dispatch);
            let value: AnyView<'input> = value.into();
            visit_result_from_any(Any::from_raw_ffi_any(
                StructuralVisitEngine::default_visit_raw(
                    self.runtime,
                    context,
                    *value.as_raw_ffi_any(),
                ),
            ))
        }
    }
}

/// Conversion into a dispatcher-driven visitor. Closures belong to structural_walk.
pub trait IntoVisitor<Marker> {
    #[doc(hidden)]
    fn visit_root(self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>>;
}

impl<D> IntoVisitor<Owned> for D
where
    for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
        (AnyView<'v>, &'c mut StructuralVisitorView<'r, D>),
        Option<Result<Option<VisitInterrupt>>>,
    >,
{
    #[inline(always)]
    fn visit_root(mut self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>> {
        visit_result_from_any(unsafe {
            Any::from_raw_ffi_any(StructuralVisitEngine::run_visitor_raw::<D>(
                *root.as_raw_ffi_any(),
                &mut self,
            ))
        })
    }
}

impl<D> IntoVisitor<Borrowed> for &mut D
where
    for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
        (AnyView<'v>, &'c mut StructuralVisitorView<'r, D>),
        Option<Result<Option<VisitInterrupt>>>,
    >,
{
    #[inline(always)]
    fn visit_root(self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>> {
        visit_result_from_any(unsafe {
            Any::from_raw_ffi_any(StructuralVisitEngine::run_visitor_raw::<D>(
                *root.as_raw_ffi_any(),
                self,
            ))
        })
    }
}

// -----------------------------------------------------------------------------
// Outlined errors
// -----------------------------------------------------------------------------
// Keep construction and formatting scratch outside successful traversal frames.

#[cold]
#[inline(never)]
fn unregistered_visit_type_error(type_index: i32) -> Error {
    runtime_error(&format!(
        "native visitor: unregistered type index {type_index}"
    ))
}

#[cold]
#[inline(never)]
fn invalid_visit_hook_result_error() -> Error {
    Error::new(
        TYPE_ERROR,
        "structural visit hook must return None or ffi.VisitInterrupt",
        "",
    )
}

/// Consume `error` and annotate it from borrowed input scalar fields. Internal
/// callers keep that input live; the Error remains owned if formatting unwinds.
// Perf note: Keep error formatting and context-allocation scratch out of visit frames.
#[cold]
#[inline(never)]
fn annotate_visit_error(
    error: Error,
    type_index: i32,
    small_str_len: u32,
    data_union: TVMFFIAnyDataUnion,
) -> Error {
    let value = TVMFFIAny {
        type_index,
        small_str_len,
        data_union,
    };
    let error = with_visit_error_context(error, value);
    with_structural_error_context(
        error,
        "visit",
        &format!("object `{}`", type_key_of(value.type_index)),
    )
}

// Perf note: Outline panic construction; the successful identity check stays small.
#[cold]
#[inline(never)]
fn wrong_visitor_dispatcher() -> ! {
    panic!("structural visitor called with a different dispatcher instance")
}

// -----------------------------------------------------------------------------
// Private utility functions
// -----------------------------------------------------------------------------

#[inline(always)]
fn dispatcher_context<D>(dispatch: &mut D) -> *mut c_void {
    std::ptr::from_mut(dispatch).cast()
}

#[inline(always)]
fn visit_result_raw(result: Result<Option<VisitInterrupt>>, raw: TVMFFIAny) -> TVMFFIAny {
    match result {
        Ok(None) => TVMFFIAny::new(),
        Ok(Some(interrupt)) => unsafe { Any::into_raw_ffi_any(interrupt.into()) },
        Err(error) => {
            core::hint::cold_path();
            StructuralVisitEngine::with_visit_context_raw(
                unsafe { Any::into_raw_ffi_any(error.into()) },
                raw,
            )
        }
    }
}

// A packed hook's owning Any is valid, but its result type is dynamic.
#[inline(always)]
fn checked_hook_result(value: Any) -> Any {
    let tag = value.type_index();
    if tag == TVMFFITypeIndex::kTVMFFINone as i32
        || tag == TVMFFITypeIndex::kTVMFFIError as i32
        || tag == TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32
    {
        value
    } else {
        invalid_visit_hook_result_error().into()
    }
}

/// Consume exactly one Error reference after its integer tag was checked.
#[inline(always)]
unsafe fn take_error(value: Any) -> Error {
    use crate::type_traits::AnyCompatible;
    let mut raw = Any::into_raw_ffi_any(value);
    Error::move_from_any_after_check(&mut raw)
}

#[inline(always)]
pub(crate) fn field_def_region(field: &TVMFFIFieldInfo, inherited: DefRegionKind) -> DefRegionKind {
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
/// against an outer binding instead of rebinding. Mirrors C++
/// `VisitReflectedFieldsExpected`.
#[inline(always)]
pub(crate) fn free_var_child_region(
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

#[inline(always)]
fn with_field_error_context(result: Any, name: &str) -> Any {
    if result.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
        core::hint::cold_path();
        // SAFETY: direct tag check establishes the Error representation.
        with_structural_field_error_context(unsafe { take_error(result) }, "visit", name).into()
    } else {
        result
    }
}

// Columns are process-lifetime registry entries; a missing per-type cell still
// falls back to reflected traversal. A missing required column is an error.
static STRUCTURAL_VISIT_COLUMN: ColdInitLazyLock<TypeAttrColumn> = ColdInitLazyLock::new(|| {
    TypeAttrColumn::new(STRUCTURAL_VISIT_ATTR).expect("missing __s_visit__ column")
});

/// Return the shared visit-hook column without caching optional per-type cells.
#[inline(always)]
fn structural_visit_column() -> TypeAttrColumn {
    *STRUCTURAL_VISIT_COLUMN
}

/// Visit every reflected field described by `info` and its ancestors in the
/// same parent-to-child order as C++ `ForEachFieldInfoWithEarlyStop`.
///
/// # Safety
///
/// `info` must point to an immortal registered type-info record.
pub(crate) unsafe fn for_each_field_info<B>(
    info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
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
    info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    callback: &mut impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    if info.is_null() || (*info).fields.is_null() {
        return None;
    }
    let fields = std::slice::from_raw_parts((*info).fields, (*info).num_fields as usize);
    for field in fields {
        // C reflection tables are immortal once registered.
        let field: &'static TVMFFIFieldInfo = &*(field as *const TVMFFIFieldInfo);
        if let ControlFlow::Break(value) = callback(field) {
            return Some(value);
        }
    }
    None
}

// -----------------------------------------------------------------------------
// StructuralVisitEngine implementation
// -----------------------------------------------------------------------------

// Perf note: Raw carriers avoid materializing an owning Any on the success path.
// Private C-unwind boundaries return this 16-byte carrier in x0/x1 on AArch64,
// avoiding the hidden result slot observed with the Rust ABI while allowing
// Rust cleanup to unwind. Native vtables remain extern "C".
// Raw bits provide no ownership guard: each `_raw` result carries one owned ABI
// value. Forward it exactly once or import it into Any before any operation
// that can unwind or discard it, including destruction of an observer.
impl StructuralVisitEngine {
    /// Consume owning `result` from descent, adding borrowed `value` context on Error.
    /// Internal callers supply valid tokens and keep the input live during annotation;
    /// nonerrors transfer unchanged, and Error ownership is guarded through formatting.
    #[inline(always)]
    fn with_visit_context_raw(result: TVMFFIAny, value: TVMFFIAny) -> TVMFFIAny {
        if result.type_index != TVMFFITypeIndex::kTVMFFIError as i32
            || value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32
        {
            result
        } else {
            core::hint::cold_path();
            // SAFETY: the Error tag establishes the owned object's representation.
            unsafe {
                let error = take_error(Any::from_raw_ffi_any(result));
                Any::into_raw_ffi_any(
                    annotate_visit_error(
                        error,
                        value.type_index,
                        value.small_str_len,
                        value.data_union,
                    )
                    .into(),
                )
            }
        }
    }

    /// Recover the Rust engine whose first field is this ABI base.
    ///
    /// # Safety
    /// The handle must belong to the live, synchronous Rust traversal that created
    /// this exact repr(C) engine. An arbitrary native visitor is not such an engine.
    #[inline(always)]
    unsafe fn from_handle(handle: StructuralVisitorHandle) -> *mut Self {
        handle.cast()
    }

    /// Descend through the native hook or reflected fields using this runtime.
    ///
    /// The dispatcher field must already hold the caller's actual reborrow.
    /// Descent itself never accesses D: every child reenters the same visitor
    /// callback, which restores its incoming pointer before the next sibling.
    /// Returns one owning raw None/interrupt/Error token.
    ///
    /// # Safety
    /// `runtime` must be this live engine and `value` a valid borrowed input whose
    /// owner survives descent. Hooks may not retain the runtime or input borrow.
    #[inline(always)]
    unsafe fn visit_children_raw(runtime: StructuralVisitorHandle, value: TVMFFIAny) -> TVMFFIAny {
        if let Some(attr) = structural_visit_column().get_raw(value.type_index) {
            if attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
                return Self::call_structural_visit_hook_raw(runtime, value, attr);
            }
        }

        if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            TVMFFIAny::new()
        } else {
            Self::visit_reflected_fields_raw(
                runtime,
                value.type_index,
                value.small_str_len,
                value.data_union,
            )
        }
    }

    /// Reconstruct borrowed input from its scalar fields and visit registered fields
    /// through live `runtime`. Return one owning None/interrupt/Error after local cleanup.
    /// The private C-unwind ABI preserves Rust unwinding and AArch64 register returns;
    /// public/native callback ABIs are unchanged.
    ///
    /// # Safety
    /// The input fields and runtime must be valid for synchronous descent; registered
    /// reflection metadata must satisfy its getter/offset contracts.
    // Perf note: Keep reflection scratch outside recursive frames. This fallback
    // can be frequent, so inline(never) is intentional without cold.
    // The AArch64 C ABI returns TVMFFIAny in registers; C-unwind preserves
    // Rust cleanup if reflected traversal panics inside this private helper.
    #[inline(never)]
    unsafe extern "C-unwind" fn visit_reflected_fields_raw(
        runtime: StructuralVisitorHandle,
        type_index: i32,
        small_str_len: u32,
        data_union: TVMFFIAnyDataUnion,
    ) -> TVMFFIAny {
        let value = TVMFFIAny {
            type_index,
            small_str_len,
            data_union,
        };
        let type_info = unsafe { TVMFFIGetTypeInfo(value.type_index) };
        if type_info.is_null() {
            return Any::into_raw_ffi_any(unregistered_visit_type_error(value.type_index).into());
        }
        let seq_hash_kind = unsafe {
            let metadata = (*type_info).metadata;
            if metadata.is_null() {
                TVMFFISEqHashKind::kTVMFFISEqHashKindUnsupported as i32
            } else {
                (*metadata).structural_eq_hash_kind
            }
        };
        let kind = std::mem::transmute::<i32, DefRegionKind>((*runtime).def_region_mode);
        let def_region_kind = free_var_child_region(kind, seq_hash_kind);
        let object = unsafe { value.data_union.v_obj } as *mut u8;
        let halted = unsafe {
            for_each_field_info(type_info, &mut |field| {
                let result = Self::visit_reflected_field(runtime, object, field, def_region_kind);
                if result.type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(result)
                }
            })
        };
        Any::into_raw_ffi_any(halted.unwrap_or_else(Any::new))
    }

    /// Visit one registered `field` in borrowed `object` using `inherited_region`.
    /// Own getter output through recursive traversal and return an owning result.
    ///
    /// # Safety
    /// `runtime` and `object` must be live; `field` must describe valid storage and a
    /// registered getter for that object. Neither raw input transfers ownership.
    unsafe fn visit_reflected_field(
        runtime: StructuralVisitorHandle,
        object: *mut u8,
        field: &TVMFFIFieldInfo,
        inherited_region: DefRegionKind,
    ) -> Any {
        if field.flags & FLAG_SEQ_HASH_IGNORE != 0 {
            return Any::new();
        }

        let Some(getter) = field.getter else {
            return Any::from(missing_field_getter_error("visitor", field.name.as_str()));
        };
        let address = object.offset(field.offset as isize) as *mut c_void;
        // Own the getter result so partial writes and recursive borrows drop safely.
        let mut child = Any::new();
        if getter(address, Any::as_data_ptr(&mut child)) != 0 {
            return with_field_error_context(Any::from(Error::from_raised()), field.name.as_str());
        }

        let borrowed = AnyView::from(&child);
        let child_region = field_def_region(field, inherited_region);
        with_field_error_context(
            Self::with_visitor_def_region(runtime, child_region, || {
                Any::from_raw_ffi_any(Self::call_visitor_raw(runtime, *borrowed.as_raw_ffi_any()))
            }),
            field.name.as_str(),
        )
    }

    /// Publish the current exclusive `dispatch` reborrow and visit borrowed `raw`.
    /// Return one owning raw None/interrupt/Error token.
    ///
    /// # Safety
    /// `runtime` must belong to this live `D` invocation and `raw` remain valid
    /// through recursion; the caller cannot use competing dispatcher borrows.
    #[inline(always)]
    unsafe fn visit_raw<D>(
        runtime: StructuralVisitorHandle,
        dispatch: &mut D,
        raw: TVMFFIAny,
    ) -> TVMFFIAny {
        let context = Self::checked_dispatcher(runtime, dispatch);
        (*Self::from_handle(runtime)).dispatcher = context;
        Self::call_visitor_raw(runtime, raw)
    }

    /// Check runtime identity and return the current `dispatch` reborrow pointer.
    ///
    /// # Safety
    /// `runtime` must be the live engine for `D`; address equality does not establish
    /// type or lifetime validity. Mismatch panics before a pointer is published.
    // The caller has already established matching runtime/D types and scope.
    // Read only the stored raw address: do not create a competing mutable borrow.
    #[inline(always)]
    unsafe fn checked_dispatcher<D>(
        runtime: StructuralVisitorHandle,
        dispatch: &mut D,
    ) -> *mut c_void {
        let current = dispatcher_context(dispatch);
        if current != (*Self::from_handle(runtime)).dispatcher {
            core::hint::cold_path();
            wrong_visitor_dispatcher();
        }
        current
    }

    /// Build the native visitor callback for concrete dispatcher `D`. Its higher-ranked
    /// view arguments restrict all borrows to the synchronous handler invocation.
    #[inline(always)]
    fn vtable<D>() -> &'static StructuralVisitorVTable
    where
        for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D>),
            Option<Result<Option<VisitInterrupt>>>,
        >,
    {
        &StructuralVisitorVTable {
            visit: Self::rust_vtable_visit_raw::<D>,
        }
    }

    /// Enter a native visitor callback with borrowed `value`; return one owning token.
    /// Restore the incoming dispatcher provenance before returning to the caller.
    ///
    /// # Safety
    /// `runtime` must have this `D` vtable and a live published reborrow. The erased
    /// ABI lifetime on `value` is valid only for this call. Callback failures use
    /// Result; Rust panics cannot cross this native C boundary.
    unsafe extern "C" fn rust_vtable_visit_raw<D>(
        runtime: StructuralVisitorHandle,
        value: AnyView<'static>,
    ) -> TVMFFIAny
    where
        for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D>),
            Option<Result<Option<VisitInterrupt>>>,
        >,
    {
        let context = (*Self::from_handle(runtime)).dispatcher;
        let raw = *value.as_raw_ffi_any();
        let result = if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            core::hint::cold_path();
            TVMFFIAny::new()
        } else {
            let dispatch = &mut *context.cast::<D>();
            let mut visitor = Self::visitor_context::<D>(runtime);
            // SAFETY: the native caller keeps the input alive for this synchronous
            // dispatch; the higher-ranked callback cannot retain its borrowed view.
            match crate::Dispatcher::dispatch_tuple(
                &mut *dispatch,
                (AnyView::from_raw_ffi_any(raw), &mut visitor),
            ) {
                Some(result) => visit_result_raw(result, raw),
                None => Self::default_visit_raw(runtime, dispatcher_context(dispatch), raw),
            }
        };
        // The enclosing callback may resume using its ancestor mutable borrow.
        // Native hooks cannot restore Rust-only borrow provenance for us.
        (*Self::from_handle(runtime)).dispatcher = context;
        result
    }

    /// Create the handler view for borrowed `runtime`, without retaining it.
    ///
    /// # Safety
    /// The live runtime's vtable must match D. Keep this view within the current
    /// higher-ranked handler call, never outliving the runtime or dispatcher.
    #[inline(always)]
    unsafe fn visitor_context<'a, D>(
        runtime: StructuralVisitorHandle,
    ) -> StructuralVisitorView<'a, D> {
        StructuralVisitorView {
            runtime,
            _scope: PhantomData,
        }
    }

    /// Publish `context` and descend through borrowed `raw`, then annotate errors.
    /// The raw return transfers one owning None/interrupt/Error token.
    ///
    /// # Safety
    /// `runtime` must be live and `context` its current exclusive dispatcher reborrow;
    /// `raw` remains borrowed and must survive the complete synchronous descent.
    #[inline(always)]
    unsafe fn default_visit_raw(
        runtime: StructuralVisitorHandle,
        context: *mut c_void,
        raw: TVMFFIAny,
    ) -> TVMFFIAny {
        // Context is the supplied exclusive reborrow. Publish it before descent;
        // the handle-only scanner never creates or uses a competing D reference.
        (*Self::from_handle(runtime)).dispatcher = context;
        Self::with_visit_context_raw(Self::visit_children_raw(runtime, raw), raw)
    }

    /// Run the public visit/walk adapter with borrowed `root` and exclusive `dispatch`.
    /// Internal callers keep the root owner live. Own the result across engine Drop,
    /// including unwind, then transfer one raw None/interrupt/Error token. Hooks must
    /// release temporary runtime owners and cannot retain the scoped dispatcher.
    pub(super) fn run_visitor_raw<D>(root: TVMFFIAny, dispatch: &mut D) -> TVMFFIAny
    where
        for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D>),
            Option<Result<Option<VisitInterrupt>>>,
        >,
    {
        let type_index = StructuralVisitorObj::type_index();
        // The managed runtime may borrow its dispatcher only for this call.
        // Trusted hooks release all temporary handles before the owner is dropped.
        let mut engine = unsafe {
            ObjectArc::<Object>::new_opaque_unchecked(
                StructuralVisitEngine {
                    base: StructuralVisitorObj::default(),
                    dispatcher: dispatcher_context(dispatch),
                },
                type_index,
            )
        };
        let handle = unsafe { ObjectArc::as_raw_mut(&mut engine).cast::<StructuralVisitorObj>() };
        // The engine and outer dispatcher storage stay stationary for the whole run.
        // Only raw field projections cross recursion; no whole-engine borrow survives.
        unsafe {
            (*handle).vtable = Self::vtable::<D>();
            (*handle).def_region_mode = DefRegionKind::None as i32;
        }
        // Keep the result owned while releasing the runtime, including on unwind.
        let result = unsafe { Any::from_raw_ffi_any(Self::call_visitor_raw(handle, root)) };
        // Trusted hooks have released every temporary strong/weak runtime reference.
        drop(engine);
        unsafe { Any::into_raw_ffi_any(result) }
    }

    /// Invoke the live visitor vtable on borrowed `raw`, returning one owning token.
    /// Internal callers keep runtime/input valid and publish the dispatcher reborrow.
    /// None bypasses dispatch; native callbacks report failures without unwinding.
    #[inline(always)]
    fn call_visitor_raw(visitor: StructuralVisitorHandle, raw: TVMFFIAny) -> TVMFFIAny {
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            core::hint::cold_path();
            return TVMFFIAny::new();
        }
        unsafe {
            let callback = (*(*visitor).vtable).visit;
            callback(visitor, AnyView::from_raw_ffi_any(raw))
        }
    }

    /// Invoke registered borrowed `attr` on live `visitor` and borrowed `raw`.
    /// Internal callers uphold hook registration and scoped lifetimes. Return one
    /// owning token, using packed validation when the attribute is not a native pointer.
    #[inline(always)]
    fn call_structural_visit_hook_raw(
        visitor: StructuralVisitorHandle,
        raw: TVMFFIAny,
        attr: TVMFFIAny,
    ) -> TVMFFIAny {
        unsafe {
            if attr.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
                let pointer = attr.data_union.v_ptr;
                // Native pointer registration is an unsafe contract: a non-null
                // callback with this ABI, borrowed runtime lifetime and owning result.
                let hook: FStructuralVisit = std::mem::transmute(pointer);
                hook(visitor, AnyView::from_raw_ffi_any(raw))
            } else {
                core::hint::cold_path();
                Self::call_structural_visit_hook_slow_raw(
                    visitor,
                    raw.type_index,
                    raw.small_str_len,
                    raw.data_union,
                    attr.type_index,
                    attr.small_str_len,
                    attr.data_union,
                )
            }
        }
    }

    /// Reconstruct borrowed value/attribute from their scalar fields for a packed call.
    /// Internal callers keep `visitor` and both inputs live; no input transfers ownership.
    /// Validate the owning packed result and release temporary owners before exporting
    /// one raw token. Private C-unwind preserves Rust cleanup and AArch64 register
    /// returns without changing the public or native callback ABI.
    // Keep packed calls and malformed registrations out of the native-hook hot path.
    // Outline uncommon packed-call scratch to keep native visit frames small.
    // Match the native raw-result ABI (registers on AArch64), without adding
    // an abort boundary around Rust packed-call cleanup.
    #[cold]
    #[inline(never)]
    extern "C-unwind" fn call_structural_visit_hook_slow_raw(
        visitor: StructuralVisitorHandle,
        value_type_index: i32,
        value_small_str_len: u32,
        value_data: TVMFFIAnyDataUnion,
        attr_type_index: i32,
        attr_small_str_len: u32,
        attr_data: TVMFFIAnyDataUnion,
    ) -> TVMFFIAny {
        // Construct addressable packed-call arguments only on this cold path.
        let raw = TVMFFIAny {
            type_index: value_type_index,
            small_str_len: value_small_str_len,
            data_union: value_data,
        };
        let attr = TVMFFIAny {
            type_index: attr_type_index,
            small_str_len: attr_small_str_len,
            data_union: attr_data,
        };
        // Packed hooks (including Python class methods) can return any well-formed
        // Any through safe APIs. Their result tag needs validation before decoding.
        let result = unsafe {
            Function::try_from(AnyView::from_raw_ffi_any(attr)).and_then(|function| {
                function.call_packed(&[
                    Self::borrowed_visitor_view(visitor),
                    AnyView::from_raw_ffi_any(raw),
                ])
            })
        };
        let result = match result {
            Ok(value) => checked_hook_result(value),
            Err(error) => error.into(),
        };
        unsafe { Any::into_raw_ffi_any(result) }
    }

    /// Borrow `visitor` as an object-valued packed-call argument without retaining it.
    ///
    /// # Safety
    /// The handle must point to a live visitor object and remain valid for the returned
    /// view lifetime; the view may not escape the synchronous invocation.
    #[inline(always)]
    unsafe fn borrowed_visitor_view<'a>(visitor: StructuralVisitorHandle) -> AnyView<'a> {
        let object = visitor.cast::<TVMFFIObject>();
        let mut raw = TVMFFIAny::new();
        raw.type_index = (*object).type_index;
        raw.small_str_len = 0;
        raw.data_union.v_obj = object;
        AnyView::from_raw_ffi_any(raw)
    }

    /// Run `callback` under requested `kind` on live borrowed `visitor`, preserving
    /// Pattern precedence. Restore prior region on return or Rust unwind. Internal
    /// callers keep the runtime live and retain no reference to its region field.
    #[inline(always)]
    fn with_visitor_def_region<T>(
        visitor: StructuralVisitorHandle,
        kind: DefRegionKind,
        callback: impl FnOnce() -> T,
    ) -> T {
        unsafe {
            let previous = (*visitor).def_region_mode;
            if previous == kind as i32 || previous == DefRegionKind::Pattern as i32 {
                // Inherited descent does not change state; Pattern keeps precedence.
                return callback();
            }
            (*visitor).def_region_mode = kind as i32;
            struct Restore {
                visitor: StructuralVisitorHandle,
                previous: i32,
            }
            impl Drop for Restore {
                #[inline(always)]
                fn drop(&mut self) {
                    unsafe { (*self.visitor).def_region_mode = self.previous };
                }
            }
            let _restore = Restore { visitor, previous };
            callback()
        }
    }
}

// -----------------------------------------------------------------------------
// Public entrypoint
// -----------------------------------------------------------------------------

/// Visit `root` with an owned or borrowed [`Dispatcher`](crate::Dispatcher).
///
/// Handlers receive [`StructuralVisitorView`] for explicit recursion and keep state on the
/// dispatcher. Unmatched values use default child traversal. Pass `&mut D` to
/// keep the dispatcher and inspect its state after the visit. An owned dispatcher
/// is held in the outer call and uses the same borrowed traversal engine.
///
/// Callbacks must report failures through [`Result`] and must not panic: a panic
/// reaching the non-unwinding native callback boundary aborts the process.
///
/// # Arguments
///
/// * `root` - Value borrowed for traversal, convertible into an [`AnyView`].
/// * `visitor` - Owned or mutably borrowed dispatcher implementing [`IntoVisitor`].
///   Matching handlers control recursion through [`StructuralVisitorView`];
///   unmatched values use default child traversal.
///
/// # Examples
///
/// Sum integer leaves while default traversal descends into the array:
///
/// ```
/// # pub use tvm_ffi::{extra, match_any, match_any_internal, Dispatcher};
/// # fn main() -> tvm_ffi::Result<()> {
/// use tvm_ffi::{
///     dispatch, structural_visit, AnyView, Array, Result, StructuralVisitorView,
///     VisitInterrupt,
/// };
///
/// struct Sum(i64);
/// #[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorView<'_, Self>), Result<Option<VisitInterrupt>>)]
/// impl Sum {
///     fn visit_integer(
///         &mut self,
///         value: i64,
///         _: &mut StructuralVisitorView<'_, Self>,
///     ) -> Result<Option<VisitInterrupt>> {
///         self.0 += value;
///         Ok(None)
///     }
/// }
///
/// let root = Array::new(vec![1_i64, 2, 3]);
/// let mut sum = Sum(0);
/// assert!(structural_visit(&root, &mut sum)?.is_none());
/// assert_eq!(sum.0, 6);
/// # Ok(())
/// # }
/// ```
pub fn structural_visit<'input, R, M>(
    root: R,
    visitor: impl IntoVisitor<M>,
) -> Result<Option<VisitInterrupt>>
where
    R: Into<AnyView<'input>>,
{
    visitor.visit_root(root.into())
}

#[inline(always)]
pub(super) fn visit_result_from_any(result: Any) -> Result<Option<VisitInterrupt>> {
    let tag = result.type_index();
    if tag == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(None);
    }
    core::hint::cold_path();
    if tag == TVMFFITypeIndex::kTVMFFIError as i32 {
        // SAFETY: internal results obey the documented owning carrier protocol.
        Err(unsafe { take_error(result) })
    } else {
        // SAFETY: the owning result protocol established a non-null interrupt.
        // Transfer that same reference, including carriers shared by native hooks.
        use crate::type_traits::AnyCompatible;
        let mut raw = unsafe { Any::into_raw_ffi_any(result) };
        Ok(Some(unsafe {
            VisitInterrupt::move_from_any_after_check(&mut raw)
        }))
    }
}
