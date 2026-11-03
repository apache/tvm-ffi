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

use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::os::raw::c_void;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::rc::Rc;

pub use super::structural_walk::*;
use std::sync::LazyLock;

use crate::any::{Any, AnyView};
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::{Object, ObjectArc, ObjectCore};
use crate::reflection::TypeAttrColumn;
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashDefPattern, kTVMFFIFieldFlagBitMaskSEqHashDefSimple,
    kTVMFFIFieldFlagBitMaskSEqHashIgnore,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIDefRegionKind, TVMFFIFieldInfo, TVMFFIGetTypeInfo,
    TVMFFIObject, TVMFFISEqHashKind, TVMFFITypeIndex, TVMFFITypeKeyToIndex,
};

use super::structural_common::{
    borrowed_view, raw_of, with_structural_error_context, with_visit_error_context,
};

const STRUCTURAL_VISIT_ATTR: &str = "__s_visit__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SEQ_HASH_DEF_PATTERN: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefPattern as i64;
const FLAG_SEQ_HASH_DEF_SIMPLE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefSimple as i64;

// -----------------------------------------------------------------------------
// ABI: C++ object prefixes, vtables, layouts and runtime type identity.
// -----------------------------------------------------------------------------

pub(super) type StructuralVisitorHandle = *mut StructuralVisitorObj;
type FStructuralVisit =
    unsafe extern "C" fn(StructuralVisitorHandle, AnyView<'static>) -> TVMFFIAny;

/// Rust vtable extension following the unchanged C++ ABI entries.
#[repr(C)]
pub(super) struct StructuralVisitorVTable {
    pub(super) visit: FStructuralVisit,
    pub(super) state_offset: usize,
}

/// Exact C++ StructuralVisitorObj ABI prefix.
#[repr(C)]
pub struct StructuralVisitorObj {
    base: Object,
    vtable: *const StructuralVisitorVTable,
    pub(super) def_region_mode: i32,
}

/// Rust layout used to create and read the ABI `ffi.VisitInterrupt` object.
#[repr(C)]
struct RuntimeVisitInterruptObj {
    base: Object,
    value: Any,
}

const _: () = {
    assert!(
        std::mem::offset_of!(StructuralVisitorObj, vtable) == std::mem::size_of::<TVMFFIObject>()
    );
    assert!(
        std::mem::offset_of!(StructuralVisitorObj, def_region_mode)
            == std::mem::size_of::<TVMFFIObject>() + std::mem::size_of::<*const c_void>()
    );
    assert!(
        std::mem::offset_of!(RuntimeVisitInterruptObj, value)
            == std::mem::size_of::<TVMFFIObject>()
    );
};

// Scoped runtime prefixes are not independently allocatable ObjectCore values.
impl StructuralVisitorObj {
    const TYPE_KEY: &'static str = "ffi.StructuralVisitor";

    fn type_index() -> i32 {
        static TYPE_INDEX: LazyLock<i32> = LazyLock::new(|| unsafe {
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

// SAFETY: `VisitInterruptObj` is final and consists of `Object` followed by
// one `Any`, exactly matching `RuntimeVisitInterruptObj`.
unsafe impl ObjectCore for RuntimeVisitInterruptObj {
    const TYPE_KEY: &'static str = "ffi.VisitInterrupt";
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH + 1;
    const TYPE_FINAL: bool = true;

    fn type_index() -> i32 {
        TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32
    }

    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.base)
    }
}

// -----------------------------------------------------------------------------
// Auxiliary internal data structures.
// -----------------------------------------------------------------------------

pub(super) struct VisitState {
    context: *mut c_void,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

#[repr(C)]
pub(super) struct StructuralVisitEngine<Base, Dispatcher> {
    base: Base,
    pub(super) state: VisitState,
    dispatcher: Dispatcher,
}

#[inline(always)]
unsafe fn visit_state(runtime: StructuralVisitorHandle) -> *mut VisitState {
    let offset = (*(*runtime).vtable).state_offset;
    runtime.cast::<u8>().add(offset).cast()
}

// -----------------------------------------------------------------------------
// Key data structures: Rust views, composition and callback outcomes.
// -----------------------------------------------------------------------------

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

/// Interrupt state of a traversal, mirroring C++ `ffi.VisitInterrupt`.
///
/// Entry points and visitor-layer calls return
/// `Result<Option<VisitInterrupt>>`: `Ok(None)` means the (sub)graph was
/// visited completely, `Ok(Some(..))` means a handler halted the traversal
/// with this interrupt, and `Err` means it failed.
pub struct VisitInterrupt {
    /// Payload returned with the interrupt, or FFI `None` for no payload.
    pub value: Any,
}

impl VisitInterrupt {
    /// Interrupt carrying an FFI-compatible payload.
    pub fn with<T: Into<Any>>(payload: T) -> Self {
        Self {
            value: payload.into(),
        }
    }
}

// Internal visit protocol matches the C++ owning TVMFFIAny result: None means
// continue; VisitInterrupt (including None/Error payloads) and Error stop descent.
// Each ABI ingress/import and egress/export transfers ownership exactly once.
// Recursive drivers move this carrier unchanged; typed public APIs decode it.
/// Recursive visiting and default child descent for a dispatcher.
///
/// Pass the handler's `self` to visit methods to recurse with the same state.
pub struct StructuralVisitorView<'a, D> {
    runtime: StructuralVisitorHandle,
    default_visit: DefaultVisitCallback<D>,
    builtin_default_visit: DefaultVisitCallback<D>,
    dispatcher_identity: *mut c_void,
    _scope: PhantomData<(&'a mut (), fn(&mut D), Rc<()>)>,
}

pub(super) fn dispatcher_context<D>(dispatch: &mut D) -> *mut c_void {
    std::ptr::from_mut(dispatch).cast()
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
        with_visitor_def_region(self.runtime, kind, || callback(self))
    }
    /// Visit a value through the dispatcher.
    #[inline(always)]
    pub fn visit<'input, T>(&mut self, dispatch: &mut D, value: T) -> Result<Option<VisitInterrupt>>
    where
        T: Into<AnyView<'input>>,
    {
        self.with_dispatch(dispatch, |runtime, dispatch| {
            publish_visitor_context(runtime, dispatcher_context(dispatch));
            finish(call_visitor(runtime, raw_of(value.into())))
        })
    }
    /// Run Base default descent, using registered hooks or reflected fields by default.
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
        self.with_dispatch(dispatch, |runtime, dispatch| unsafe {
            finish((self.default_visit)(
                runtime,
                dispatch,
                raw_of(value.into()),
            ))
        })
    }
    #[inline(always)]
    fn with_dispatch<T>(
        &self,
        dispatch: &mut D,
        callback: impl FnOnce(StructuralVisitorHandle, &mut D) -> T,
    ) -> T {
        let current = dispatcher_context(dispatch);
        // The supplied borrow must cover the dispatcher that runtime callbacks
        // access. Equal Rust types alone do not identify the same instance.
        if current != self.dispatcher_identity {
            wrong_visitor_dispatcher();
        }
        callback(self.runtime, dispatch)
    }
}

#[cold]
#[inline(never)]
fn wrong_visitor_dispatcher() -> ! {
    panic!("structural visitor called with a different dispatcher instance")
}

pub(crate) trait VisitDispatch: Sized {
    fn dispatch_visit(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Option<Result<Option<VisitInterrupt>>>;
}

impl<D> VisitDispatch for D
where
    for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
        (AnyView<'v>, &'c mut StructuralVisitorView<'r, D>),
        Option<Result<Option<VisitInterrupt>>>,
    >,
{
    #[inline(always)]
    fn dispatch_visit(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Option<Result<Option<VisitInterrupt>>> {
        crate::Dispatcher::dispatch_tuple(self, (value, visitor))
    }
}

/// Conversion into a dispatcher-driven visitor. Closures belong to structural_walk.
pub trait IntoVisitor<Marker> {
    #[doc(hidden)]
    fn visit_root(self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>>;
}

#[doc(hidden)]
pub enum ByOwnedVisitor {}
#[doc(hidden)]
pub enum ByBorrowedVisitor {}

/// Compose a concrete default-descent base with an owned or borrowed dispatcher.
pub struct StructuralVisitor<Base, D> {
    pub(super) base: Base,
    pub(super) dispatcher: D,
}

impl<Base, D> StructuralVisitor<Base, D> {
    pub fn new(base: Base, dispatcher: D) -> Self {
        Self { base, dispatcher }
    }
}

#[doc(hidden)]
pub enum ByOwnedBaseVisitor {}
#[doc(hidden)]
pub enum ByBorrowedBaseVisitor {}
impl<Base: StructuralVisitorBase, D: VisitDispatch> IntoVisitor<ByOwnedBaseVisitor>
    for StructuralVisitor<Base, D>
{
    fn visit_root(self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>> {
        finish(run_user_visitor(
            raw_of(root),
            self.base,
            OwnedVisitor(self.dispatcher),
        ))
    }
}

impl<Base: StructuralVisitorBase, D: VisitDispatch> IntoVisitor<ByBorrowedBaseVisitor>
    for StructuralVisitor<Base, &mut D>
{
    fn visit_root(self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>> {
        finish(run_user_visitor(
            raw_of(root),
            self.base,
            BorrowedVisitor(self.dispatcher),
        ))
    }
}

impl<D: VisitDispatch> IntoVisitor<ByOwnedVisitor> for D {
    fn visit_root(self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>> {
        finish(run_user_visitor(
            raw_of(root),
            StructuralVisitorObj::default(),
            OwnedVisitor(self),
        ))
    }
}

impl<D: VisitDispatch> IntoVisitor<ByBorrowedVisitor> for &mut D {
    fn visit_root(self, root: AnyView<'_>) -> Result<Option<VisitInterrupt>> {
        finish(run_user_visitor(
            raw_of(root),
            StructuralVisitorObj::default(),
            BorrowedVisitor(self),
        ))
    }
}

// -----------------------------------------------------------------------------
// Implementation of visit.
// -----------------------------------------------------------------------------

/// Action applied to each child found by the shared traversal.
pub(super) trait ChildVisit {
    fn context(&mut self) -> *mut c_void;
    fn visit_child(&mut self, child: AnyView<'_>, def_region_kind: DefRegionKind) -> Any;
}

struct UserChildren<'a, D> {
    dispatch: &'a mut D,
    runtime: StructuralVisitorHandle,
}

impl<D: VisitDispatch> ChildVisit for UserChildren<'_, D> {
    fn context(&mut self) -> *mut c_void {
        dispatcher_context(self.dispatch)
    }
    fn visit_child(&mut self, child: AnyView<'_>, kind: DefRegionKind) -> Any {
        publish_visitor_context(self.runtime, self.context());
        with_visitor_def_region(self.runtime, kind, || {
            call_visitor(self.runtime, raw_of(child))
        })
    }
}

#[inline(always)]
pub(super) fn visit_children_raw<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    def_region_kind: DefRegionKind,
    runtime: StructuralVisitorHandle,
) -> Any {
    if let Some(attr) = STRUCTURAL_VISIT_COLUMN.get_raw(value.type_index) {
        if attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
            publish_visitor_context(runtime, visitor.context());
            return call_structural_visit_hook(runtime, value, attr);
        }
    }

    if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        Any::new()
    } else {
        visit_reflected_fields(value, visitor, def_region_kind)
    }
}

#[inline(never)]
fn visit_reflected_fields<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    def_region_kind: DefRegionKind,
) -> Any {
    let type_info = unsafe { TVMFFIGetTypeInfo(value.type_index) };
    if type_info.is_null() {
        return runtime_error(&format!(
            "native visitor: unregistered type index {}",
            value.type_index
        ))
        .into();
    }
    let seq_hash_kind = unsafe {
        let metadata = (*type_info).metadata;
        if metadata.is_null() {
            TVMFFISEqHashKind::kTVMFFISEqHashKindUnsupported as i32
        } else {
            (*metadata).structural_eq_hash_kind
        }
    };
    let def_region_kind = free_var_child_region(def_region_kind, seq_hash_kind);
    let object = unsafe { value.data_union.v_obj } as *mut u8;
    let halted = unsafe {
        for_each_field_info(type_info, &mut |field| {
            let result = visit_reflected_field(object, field, visitor, def_region_kind);
            if result.type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(result)
            }
        })
    };
    halted.unwrap_or_else(Any::new)
}

unsafe fn visit_reflected_field<C: ChildVisit>(
    object: *mut u8,
    field: &TVMFFIFieldInfo,
    visitor: &mut C,
    inherited_region: DefRegionKind,
) -> Any {
    if field.flags & FLAG_SEQ_HASH_IGNORE != 0 {
        return Any::new();
    }

    let Some(getter) = field.getter else {
        return Any::from(runtime_error(&format!(
            "native visitor: reflected field `{}` has no getter",
            field.name.as_str()
        )));
    };
    let address = object.offset(field.offset as isize) as *mut c_void;
    // Own the getter result so partial writes and recursive borrows drop safely.
    let mut child = Any::new();
    if getter(address, Any::as_data_ptr(&mut child)) != 0 {
        return with_error_context(
            Any::from(Error::from_raised()),
            &format!("field `{}`", field.name.as_str()),
        );
    }

    let borrowed = AnyView::from(&child);
    let child_region = field_def_region(field, inherited_region);
    with_error_context(
        visitor.visit_child(borrowed, child_region),
        &format!("field `{}`", field.name.as_str()),
    )
}

impl Default for StructuralVisitorObj {
    fn default() -> Self {
        Self {
            base: Object::new(),
            vtable: std::ptr::null(),
            def_region_mode: DefRegionKind::None as i32,
        }
    }
}

impl StructuralVisitorObj {
    /// Run built-in hook or reflected-field descent, bypassing the Base override.
    #[inline(always)]
    pub fn default_visit<'input, D>(
        dispatch: &mut D,
        value: impl Into<AnyView<'input>>,
        visitor: &mut StructuralVisitorView<'_, D>,
    ) -> Result<Option<VisitInterrupt>> {
        visitor.with_dispatch(dispatch, |runtime, dispatch| unsafe {
            finish((visitor.builtin_default_visit)(
                runtime,
                dispatch,
                raw_of(value.into()),
            ))
        })
    }
}

/// A concrete base that customizes default descent.
///
/// # Safety
/// `Self` must have `#[repr(C)]` with a `StructuralVisitorObj` as its first field
/// at offset zero. Do not replace its runtime header or vtable. The callback is
/// synchronous and non-escaping; raw base pointers and runtime handles must not
/// be retained. Field references must end before recursion that may access them.
/// In particular, never form a reference to the whole base across recursive calls.
pub unsafe trait StructuralVisitorBase: Sized {
    /// Override default descent; delegate to `StructuralVisitorObj::default_visit`
    /// to visit children through the registered hook or reflected fields.
    ///
    /// # Safety
    /// `this` is the live base of `visitor`; use it only for this synchronous call.
    #[inline(always)]
    unsafe fn default_visit<D>(
        _this: *mut Self,
        dispatch: &mut D,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, D>,
    ) -> Result<Option<VisitInterrupt>> {
        StructuralVisitorObj::default_visit(dispatch, value, visitor)
    }

    /// Internal owning result bridge for default descent.
    ///
    /// # Safety
    /// The same scope rules as `default_visit` apply. Overrides must return an
    /// owning None, non-null ffi.VisitInterrupt, or non-null Error carrier.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn default_visit_result<D>(
        this: *mut Self,
        dispatch: &mut D,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, D>,
    ) -> Any {
        visit_result(
            Self::default_visit(this, dispatch, value, visitor),
            raw_of(value),
        )
    }
}

// SAFETY: the builtin base is the exact ABI prefix itself.
unsafe impl StructuralVisitorBase for StructuralVisitorObj {
    #[inline(always)]
    unsafe fn default_visit_result<D>(
        _this: *mut Self,
        dispatch: &mut D,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, D>,
    ) -> Any {
        // Builtin recursion retains the original owning ABI carrier. Only a
        // custom public typed override needs the default boundary adaptation.
        (visitor.builtin_default_visit)(visitor.runtime, dispatch, raw_of(value))
    }
}

fn user_runtime_vtable<Base: StructuralVisitorBase, S: VisitorStorage>(
) -> &'static StructuralVisitorVTable
where
    S::Dispatcher: VisitDispatch,
{
    &StructuralVisitorVTable {
        visit: rust_vtable_user::<Base, S::Dispatcher>,
        state_offset: std::mem::offset_of!(StructuralVisitEngine<Base, S>, state),
    }
}

unsafe extern "C" fn rust_vtable_user<Base: StructuralVisitorBase, V: VisitDispatch>(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    rust_vtable_visit_impl(visitor, value, |context, raw, kind| {
        runtime_user_visit::<Base, V>(visitor, context, raw, kind)
    })
}

#[inline(always)]
pub(super) unsafe fn rust_vtable_visit_impl(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
    callback: impl FnOnce(*mut c_void, TVMFFIAny, DefRegionKind) -> Any,
) -> TVMFFIAny {
    let context = (*visit_state(visitor)).context;
    let raw = *value.as_raw_ffi_any();
    let outcome = catch_unwind(AssertUnwindSafe(
        #[inline(always)]
        || {
            let result = match def_region_from_raw((*visitor).def_region_mode) {
                Ok(kind) => callback(context, raw, kind),
                Err(error) => Any::from(error),
            };
            Any::into_raw_ffi_any(result)
        },
    ));
    // Rust callbacks may resume using an ancestor mutable borrow after recursion.
    // Republish the entry pointer once all callback references have ended, also
    // on caught panic; native hooks cannot restore Rust-only borrow provenance.
    (*visit_state(visitor)).context = context;
    match outcome {
        Ok(result) => result,
        Err(payload) => visit_panic_result(visitor, payload),
    }
}

#[cold]
unsafe fn visit_panic_result(
    visitor: StructuralVisitorHandle,
    payload: Box<dyn std::any::Any + Send>,
) -> TVMFFIAny {
    (*visit_state(visitor)).panic = Some(payload);
    Any::into_raw_ffi_any(Any::from(runtime_error(
        "panic in structural visitor callback",
    )))
}

/// Publish the actual dispatcher reborrow before crossing the hook ABI.
#[inline(always)]
fn publish_visitor_context(visitor: StructuralVisitorHandle, context: *mut c_void) {
    unsafe {
        (*visit_state(visitor)).context = context;
    }
}

#[inline(always)]
unsafe fn runtime_user_visit<Base: StructuralVisitorBase, D: VisitDispatch>(
    runtime: StructuralVisitorHandle,
    context: *mut c_void,
    raw: TVMFFIAny,
    _kind: DefRegionKind,
) -> Any {
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Any::new();
    }
    let dispatch = &mut *context.cast::<D>();
    let mut visitor = visitor_context::<Base, D, UserDriver<Base>>(runtime, dispatch);
    match dispatch.dispatch_visit(borrowed_view(&raw), &mut visitor) {
        Some(result) => visit_result(result, raw),
        None => base_default_visit::<Base, D, UserDriver<Base>>(runtime, dispatch, raw),
    }
}

/// # Safety
/// The live runtime must have the matching Base, D and Driver. Keep the view
/// within the current handler or Base call: its lifetime must not outlast that
/// runtime or dispatcher. Only pass it through higher-ranked handler calls.
#[inline(always)]
pub(super) unsafe fn visitor_context<'a, Base: StructuralVisitorBase, D, Driver: VisitDriver<D>>(
    runtime: StructuralVisitorHandle,
    dispatch: &mut D,
) -> StructuralVisitorView<'a, D> {
    StructuralVisitorView {
        runtime,
        default_visit: base_default_visit::<Base, D, Driver>,
        builtin_default_visit: Driver::children,
        dispatcher_identity: dispatcher_context(dispatch),
        _scope: PhantomData,
    }
}

type DefaultVisitCallback<D> = unsafe fn(StructuralVisitorHandle, &mut D, TVMFFIAny) -> Any;

#[inline(always)]
pub(super) fn visit_result(result: Result<Option<VisitInterrupt>>, raw: TVMFFIAny) -> Any {
    match result {
        Ok(None) => Any::new(),
        Ok(Some(interrupt)) => interrupt_result(interrupt.value),
        Err(error) => with_value_context(error.into(), raw),
    }
}

#[inline(always)]
pub(super) unsafe fn base_default_visit<Base: StructuralVisitorBase, D, Driver: VisitDriver<D>>(
    runtime: StructuralVisitorHandle,
    dispatch: &mut D,
    raw: TVMFFIAny,
) -> Any {
    let mut visitor = visitor_context::<Base, D, Driver>(runtime, dispatch);
    Base::default_visit_result(runtime.cast(), dispatch, borrowed_view(&raw), &mut visitor)
}

pub(super) trait VisitDriver<D> {
    unsafe fn children(runtime: StructuralVisitorHandle, dispatch: &mut D, raw: TVMFFIAny) -> Any;
}
struct UserDriver<Base>(PhantomData<Base>);
impl<Base: StructuralVisitorBase, D: VisitDispatch> VisitDriver<D> for UserDriver<Base> {
    #[inline(always)]
    unsafe fn children(runtime: StructuralVisitorHandle, dispatch: &mut D, raw: TVMFFIAny) -> Any {
        let kind = std::mem::transmute((*runtime).def_region_mode);
        with_value_context(
            visit_children_raw(raw, &mut UserChildren { dispatch, runtime }, kind, runtime),
            raw,
        )
    }
}

pub(super) trait VisitorStorage {
    type Dispatcher;
    fn dispatch(&mut self) -> &mut Self::Dispatcher;
}
#[repr(transparent)]
pub(super) struct OwnedVisitor<D>(pub(super) D);
impl<D> VisitorStorage for OwnedVisitor<D> {
    type Dispatcher = D;
    fn dispatch(&mut self) -> &mut D {
        &mut self.0
    }
}
#[repr(transparent)]
struct BorrowedVisitor<'a, D>(&'a mut D);
impl<D> VisitorStorage for BorrowedVisitor<'_, D> {
    type Dispatcher = D;
    fn dispatch(&mut self) -> &mut D {
        self.0
    }
}

fn run_user_visitor<Base: StructuralVisitorBase, S: VisitorStorage>(
    root: TVMFFIAny,
    base: Base,
    storage: S,
) -> Any
where
    S::Dispatcher: VisitDispatch,
{
    run_visitor_engine(root, base, storage, user_runtime_vtable::<Base, S>())
}

pub(super) fn run_visitor_engine<Base: StructuralVisitorBase, S: VisitorStorage>(
    root: TVMFFIAny,
    base: Base,
    storage: S,
    vtable: &'static StructuralVisitorVTable,
) -> Any {
    let type_index = StructuralVisitorObj::type_index();
    // The managed payload may borrow its dispatcher/base only for this call.
    // Trusted hooks release all temporary handles before the owner is dropped.
    let mut engine = unsafe {
        ObjectArc::<Object>::new_opaque_unchecked(
            StructuralVisitEngine {
                base,
                state: VisitState {
                    context: std::ptr::null_mut(),
                    panic: None,
                },
                dispatcher: storage,
            },
            type_index,
        )
    };
    let allocation =
        unsafe { ObjectArc::as_raw_mut(&mut engine).cast::<StructuralVisitEngine<Base, S>>() };
    let (result, panic) = unsafe { run_visitor_engine_in_place(allocation, root, vtable) };
    drop(engine);
    if let Some(payload) = panic {
        drop(result);
        resume_unwind(payload);
    }
    result
}

// SAFETY: the caller supplies a live, aligned, fully constructed engine with
// the matching vtable. Storage stays stationary until this returns; no runtime
// pointer may escape. The owner is dropped before any captured panic resumes.
#[inline(always)]
unsafe fn run_visitor_engine_in_place<Base: StructuralVisitorBase, S: VisitorStorage>(
    engine_ptr: *mut StructuralVisitEngine<Base, S>,
    root: TVMFFIAny,
    vtable: &'static StructuralVisitorVTable,
) -> (Any, Option<Box<dyn std::any::Any + Send>>) {
    let handle = engine_ptr.cast::<StructuralVisitorObj>();
    (*handle).vtable = vtable;
    (*handle).def_region_mode = DefRegionKind::None as i32;
    let offset = std::mem::offset_of!(StructuralVisitEngine<Base, S>, dispatcher);
    let storage = &mut *engine_ptr.cast::<u8>().add(offset).cast::<S>();
    let current = dispatcher_context(storage.dispatch());
    (*visit_state(handle)).context = current;
    let result = call_visitor(handle, root);
    let panic = (*visit_state(handle)).panic.take();
    // Trusted hooks have released every temporary strong/weak runtime reference.
    (result, panic)
}

#[inline(always)]
fn call_visitor(visitor: StructuralVisitorHandle, raw: TVMFFIAny) -> Any {
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Any::new();
    }
    unsafe {
        let callback = (*(*visitor).vtable).visit;
        Any::from_raw_ffi_any(callback(visitor, AnyView::from_raw_ffi_any(raw)))
    }
}

#[inline(always)]
fn call_structural_visit_hook(
    visitor: StructuralVisitorHandle,
    raw: TVMFFIAny,
    attr: TVMFFIAny,
) -> Any {
    unsafe {
        if attr.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
            let pointer = attr.data_union.v_ptr;
            if pointer.is_null() {
                return runtime_error("structural visit hook is null").into();
            }
            // The registered opaque pointer has exactly the structural-visit ABI.
            let hook: FStructuralVisit = std::mem::transmute(pointer);
            checked_hook_result(Any::from_raw_ffi_any(hook(
                visitor,
                AnyView::from_raw_ffi_any(raw),
            )))
        } else {
            call_structural_visit_hook_slow(visitor, raw, attr)
        }
    }
}

// Keep packed calls and malformed registrations out of the native-hook hot path.
#[cold]
fn call_structural_visit_hook_slow(
    visitor: StructuralVisitorHandle,
    raw: TVMFFIAny,
    attr: TVMFFIAny,
) -> Any {
    unsafe {
        if attr.type_index == TVMFFITypeIndex::kTVMFFIFunction as i32 {
            let result = Function::try_from(AnyView::from_raw_ffi_any(attr)).and_then(|function| {
                function.call_packed(&[
                    borrowed_visitor_view(visitor),
                    AnyView::from_raw_ffi_any(raw),
                ])
            });
            match result {
                Ok(value) => checked_hook_result(value),
                Err(error) => error.into(),
            }
        } else {
            Error::new(
                TYPE_ERROR,
                "__s_visit__ must be an opaque function pointer or ffi.Function",
                "",
            )
            .into()
        }
    }
}

unsafe fn borrowed_visitor_view<'a>(visitor: StructuralVisitorHandle) -> AnyView<'a> {
    let object = visitor.cast::<TVMFFIObject>();
    let mut raw = TVMFFIAny::new();
    raw.type_index = (*object).type_index;
    raw.small_str_len = 0;
    raw.data_union.v_obj = object;
    AnyView::from_raw_ffi_any(raw)
}

/// Wrap a fresh public interrupt once. Payload tags never become outcome tags.
#[cold]
pub(super) fn interrupt_result(payload: Any) -> Any {
    let interrupt = ObjectArc::new(RuntimeVisitInterruptObj {
        base: Object::new(),
        value: payload,
    });
    let mut raw = TVMFFIAny::new();
    raw.type_index = TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32;
    unsafe {
        raw.data_union.v_obj = ObjectArc::into_raw(interrupt).cast_mut().cast();
        Any::from_raw_ffi_any(raw)
    }
}

// External registered hooks must honor the result protocol. Validate at ingress
// without decoding or cloning the owning carrier on each recursive return.
fn checked_hook_result(value: Any) -> Any {
    let tag = value.type_index();
    if tag == TVMFFITypeIndex::kTVMFFINone as i32 || tag == TVMFFITypeIndex::kTVMFFIError as i32 {
        value
    } else if tag == TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32 {
        if unsafe { value.as_raw_ffi_any().data_union.v_obj.is_null() } {
            // There is no object/reference to release for this malformed carrier.
            unsafe { Any::into_raw_ffi_any(value) };
            runtime_error("structural visit returned a null interrupt").into()
        } else {
            value
        }
    } else {
        Error::new(
            TYPE_ERROR,
            "structural visit hook must return None or ffi.VisitInterrupt",
            "",
        )
        .into()
    }
}

/// Consume exactly one Error reference after its integer tag was checked.
unsafe fn take_error(value: Any) -> Error {
    use crate::type_traits::AnyCompatible;
    let mut raw = Any::into_raw_ffi_any(value);
    Error::move_from_any_after_check(&mut raw)
}

#[inline(always)]
pub(super) fn with_visitor_def_region<T>(
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
            fn drop(&mut self) {
                unsafe { (*self.visitor).def_region_mode = self.previous };
            }
        }
        let _restore = Restore { visitor, previous };
        callback()
    }
}

#[inline(always)]
fn def_region_from_raw(kind: i32) -> Result<DefRegionKind> {
    match kind {
        x if x == DefRegionKind::None as i32 => Ok(DefRegionKind::None),
        x if x == DefRegionKind::Pattern as i32 => Ok(DefRegionKind::Pattern),
        x if x == DefRegionKind::Simple as i32 => Ok(DefRegionKind::Simple),
        _ => Err(runtime_error("invalid structural definition-region kind")),
    }
}

#[inline]
pub(super) fn with_value_context(result: Any, value: TVMFFIAny) -> Any {
    if result.type_index() != TVMFFITypeIndex::kTVMFFIError as i32
        || value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32
    {
        result
    } else {
        // SAFETY: the Error tag establishes the owned object's representation.
        let error = with_visit_error_context(unsafe { take_error(result) }, value);
        with_structural_error_context(
            error,
            "visit",
            &format!("object `{}`", type_key_of(value.type_index)),
        )
        .into()
    }
}

/// Visit `root` with an owned or borrowed [`Dispatcher`](crate::Dispatcher).
///
/// Handlers receive [`StructuralVisitorView`] for explicit recursion and keep state on the
/// dispatcher. Unmatched values use default child traversal.
pub fn structural_visit<'input, R, M>(
    root: R,
    visitor: impl IntoVisitor<M>,
) -> Result<Option<VisitInterrupt>>
where
    R: Into<AnyView<'input>>,
{
    visitor.visit_root(root.into())
}

pub(super) fn finish(result: Any) -> Result<Option<VisitInterrupt>> {
    let tag = result.type_index();
    if tag == TVMFFITypeIndex::kTVMFFINone as i32 {
        Ok(None)
    } else if tag == TVMFFITypeIndex::kTVMFFIError as i32 {
        // SAFETY: internal results obey the documented owning carrier protocol.
        Err(unsafe { take_error(result) })
    } else {
        // SAFETY: the only remaining protocol tag is VisitInterrupt. Clone only
        // at this typed public boundary: external hooks may share the carrier.
        let value = unsafe {
            (*result
                .as_raw_ffi_any()
                .data_union
                .v_obj
                .cast::<RuntimeVisitInterruptObj>())
            .value
            .clone()
        };
        Ok(Some(VisitInterrupt { value }))
    }
}

#[inline]
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
#[inline]
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

#[inline]
fn with_error_context(result: Any, frame: &str) -> Any {
    if result.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
        // SAFETY: direct tag check establishes the Error representation.
        with_structural_error_context(unsafe { take_error(result) }, "visit", frame).into()
    } else {
        result
    }
}

#[cold]
#[inline(never)]
fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

// Columns are process-lifetime registry entries; a missing per-type cell still
// falls back to reflected traversal. A missing required column is an error.
static STRUCTURAL_VISIT_COLUMN: LazyLock<TypeAttrColumn> = LazyLock::new(|| {
    TypeAttrColumn::new(STRUCTURAL_VISIT_ATTR).expect("missing __s_visit__ column")
});

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
