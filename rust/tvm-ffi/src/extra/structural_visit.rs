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
//! Two public layers:
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

// `_raw` functions move one owned `TVMFFIAny`: forward it once, or import it into `Any` before
// anything can unwind. Private `extern "C-unwind"` helpers return it in registers on AArch64.

use std::cell::UnsafeCell;
use std::marker::PhantomData;

use super::dispatch::IntoDispatcher;
use std::borrow::BorrowMut;
use std::ops::ControlFlow;
use std::os::raw::c_void;
use std::rc::Rc;

use super::cold_init_lazy_lock::ColdInitLazyLock;
pub use super::structural_walk::*;

use crate::any::{Any, AnyView};
use crate::derive::{Object, ObjectRef};
use crate::error::{Error, Result, TYPE_ERROR};
use crate::function::Function;
use crate::object::{Object, ObjectArc};
use crate::reflection::TypeAttrColumn;
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIDefRegionKind, TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFIObject,
    TVMFFITypeIndex,
};

use super::structural_common::{
    field_def_region, for_each_field_info, free_var_child_region, missing_field_getter_error,
    registered_type_index, runtime_error, seq_hash_kind, with_structural_error_context,
    with_structural_field_error_context, wrong_dispatcher, FLAG_SEQ_HASH_IGNORE,
};
use super::visit_error_context::update_visit_error_context;

const STRUCTURAL_VISIT_ATTR: &str = "__s_visit__";

// -----------------------------------------------------------------------------
// Data structures and FFI layout
// -----------------------------------------------------------------------------

type StructuralVisitorHandle = *mut StructuralVisitorObj;

type FStructuralVisit =
    unsafe extern "C" fn(StructuralVisitorHandle, AnyView<'static>) -> TVMFFIAny;

/// The visitor vtable, laid out as the C++ one.
#[repr(C)]
struct StructuralVisitorVTable {
    visit: FStructuralVisit,
}

/// The native visitor state, laid out as the prefix of the C++
/// `StructuralVisitorObj`.
#[repr(C)]
pub struct StructuralVisitorObj {
    base_: Object,
    vtable: *const StructuralVisitorVTable,
    def_region_mode: UnsafeCell<i32>,
}

// Scoped runtime prefixes are not independently allocatable ObjectCore values.
impl StructuralVisitorObj {
    const TYPE_KEY: &'static str = "ffi.StructuralVisitor";
}

impl Default for StructuralVisitorObj {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_: Object::new(),
            vtable: std::ptr::null(),
            def_region_mode: UnsafeCell::new(DefRegionKind::None as i32),
        }
    }
}

/// Native `ffi.VisitInterrupt` object containing the interruption payload.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.VisitInterrupt"]
#[type_index(unsafe TVMFFITypeIndex::kTVMFFIVisitInterrupt)]
#[type_final]
pub struct VisitInterruptObj {
    base_: Object,
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

/// Interrupt state of a traversal, an `ffi.VisitInterrupt` object.
///
/// Entry points and visitor-layer calls return
/// `Result<Option<VisitInterrupt>>`: `Ok(None)` means the (sub)graph was
/// visited completely, `Ok(Some(..))` means a handler halted the traversal
/// with this interrupt, and `Err` means it failed.
///
/// This is an owning object reference, like [`Error`]. Payload reads through
/// `interrupt.value` borrow the shared [`VisitInterruptObj`]; use [`Self::new`]
/// instead of a payload-only struct literal. Moving or replacing the payload
/// through this reference is not supported; clone it to retain a separate owner.
#[repr(C)]
#[derive(Clone, ObjectRef)]
pub struct VisitInterrupt {
    data: ObjectArc<VisitInterruptObj>,
}

impl VisitInterrupt {
    /// Create an interrupt carrying an FFI-compatible payload back to the caller of
    /// the visit.
    ///
    /// # Arguments
    ///
    /// * `payload` - The payload, such as the value that was found.
    ///
    /// # Returns
    ///
    /// The interrupt.
    #[inline(always)]
    pub fn new<T: Into<Any>>(payload: T) -> Self {
        Self::from_object(VisitInterruptObj {
            base_: Object::new(),
            value: payload.into(),
        })
    }
}

/// Native state and default traversal for [`StructuralVisitEngine`].
/// The engine owns allocation, scoped dispatch and results.
///
/// # Safety
///
/// `Self` must have a complete initialized C layout with [`StructuralVisitorObj`]
/// at offset zero, matching its registered native type and every field hooks access.
/// Native-mutated fields require interior mutability; shared references must not
/// freeze them during recursion. Hooks must preserve the vtable and valid region state.
///
/// The base must not implement [`crate::ObjectCore`] or otherwise allow callers
/// to retain the runtime. Hooks may access the runtime and borrowed traversal
/// state only synchronously, on the calling thread.
///
/// `default_visit_raw` returns one owning ABI None/Error/VisitInterrupt and recurses only
/// through the installed vtable, without a whole-base mutable reference spanning
/// recursion. Failures use that result carrier; panics must not cross native callbacks.
pub unsafe trait StructuralVisitorBase: Sized {
    /// Initialize the complete base before the engine installs its runtime header.
    fn new() -> Result<Self>;

    /// Native type registered for this complete base layout.
    fn type_index() -> i32;

    /// Descend from a borrowed value without dispatching that value again.
    /// Return one owning ABI None, Error, or VisitInterrupt value.
    /// Children reenter the installed vtable. The ordinary implementation accesses
    /// only the visitor prefix, so specialized bases may reuse it.
    ///
    /// # Safety
    /// `runtime` is this live initialized base with a matching installed vtable.
    /// The input and runtime remain borrowed for this synchronous call. The
    /// caller has published the current exclusive dispatcher reborrow.
    unsafe fn default_visit_raw(runtime: *mut Self, value: AnyView<'_>) -> TVMFFIAny;
}

// SAFETY: this prefix exactly matches the ordinary native visitor. Its mutable
// region is interior mutable, and it exposes no safe ownership conversion.
unsafe impl StructuralVisitorBase for StructuralVisitorObj {
    #[inline(always)]
    fn new() -> Result<Self> {
        Ok(Self::default())
    }

    fn type_index() -> i32 {
        static TYPE_INDEX: ColdInitLazyLock<i32> =
            ColdInitLazyLock::new(|| registered_type_index(StructuralVisitorObj::TYPE_KEY));
        *TYPE_INDEX
    }

    #[inline(always)]
    unsafe fn default_visit_raw(runtime: *mut Self, value: AnyView<'_>) -> TVMFFIAny {
        Self::visit_children_raw(runtime, *value.as_raw_ffi_any())
    }
}

/// Scoped Rust visitor runtime with a statically selected native base.
///
/// Enter through [`StructuralVisitEngine::visit_root`]. The engine owns the complete base
/// and the dispatcher reborrow handle, while the dispatcher stays in the outer
/// call. Engines and their owning handles are never exposed to callbacks.
///
/// Select a custom base with `StructuralVisitEngine::<Base>::visit_root(root, input)`.
/// Handlers use `StructuralVisitorView<'_, Target, Base>` for recursion and
/// base-state inspection; normalization preserves the actual target's identity.
#[repr(C)]
pub struct StructuralVisitEngine<Base = StructuralVisitorObj> {
    base: Base,
    dispatcher: *mut c_void,
}

/// The running structural visit, passed to each handler.
///
/// A handler that matches a node decides which children are visited: call
/// [`default_visit`](Self::default_visit) to visit every child,
/// [`visit`](Self::visit) to visit one value, or neither to skip the children.
/// Pass the handler's own `self` as `dispatch` to every method. A handler
/// returns `Ok(None)` to continue, or `Ok(Some(interrupt))` to stop the whole
/// visit with a [`VisitInterrupt`].
///
/// See also [`structural_visit`].
pub struct StructuralVisitorView<'a, D, Base = StructuralVisitorObj> {
    runtime: StructuralVisitorHandle,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<(&'a mut (), fn(&mut D) -> Base, Rc<()>)>,
}

// -----------------------------------------------------------------------------
// Public visitor API and entrypoint
// -----------------------------------------------------------------------------

impl<D, Base: StructuralVisitorBase> StructuralVisitorView<'_, D, Base> {
    /// Inspect the base state; the borrow cannot outlive this view.
    #[inline(always)]
    pub fn base(&self) -> &Base {
        // SAFETY: the unsafe base contract permits shared inspection, including
        // native mutation through interior-mutable fields during recursion.
        unsafe { &*self.runtime.cast::<Base>() }
    }

    /// Return the definition region the current value is in.
    ///
    /// Reflected fields marked as definitions set it while their values are
    /// visited; it is [`DefRegionKind::None`] elsewhere.
    ///
    /// # Returns
    ///
    /// The current definition region.
    #[inline(always)]
    pub fn def_region_kind(&self) -> DefRegionKind {
        // SAFETY: the view's runtime is live.
        unsafe { std::mem::transmute(*(*self.runtime).def_region_mode.get()) }
    }

    /// Run `callback` under a scoped definition region.
    ///
    /// An enclosing [`DefRegionKind::Pattern`] region keeps precedence; the
    /// previous region is restored on all exits.
    ///
    /// # Arguments
    ///
    /// * `kind` - The definition region to enter.
    /// * `callback` - Runs inside the region, with this view.
    ///
    /// # Returns
    ///
    /// The result of `callback`.
    #[inline(always)]
    pub fn with_def_region_kind<R>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut Self) -> R,
    ) -> R {
        // SAFETY: the view's runtime outlives the callback.
        StructuralVisitorObj::with_def_region_kind(self.runtime, kind, || callback(self))
    }

    /// Visit `value` through the dispatcher, as if the traversal had reached it.
    ///
    /// # Arguments
    ///
    /// * `dispatch` - The handler's own `self`.
    /// * `value` - The value to visit.
    ///
    /// # Returns
    ///
    /// `None` to continue, or the interrupt that stopped the visit.
    ///
    /// # Errors
    ///
    /// Returns the error of the handler that `value` reaches.
    #[inline(always)]
    pub fn visit<'input, T>(&mut self, dispatch: &mut D, value: T) -> Result<Option<VisitInterrupt>>
    where
        T: Into<AnyView<'input>>,
    {
        let value: AnyView<'input> = value.into();
        // SAFETY: the view ties this runtime to D; the check establishes the
        // instance, and publishing uses the caller's actual exclusive reborrow.
        unsafe {
            visit_result_from_any(Any::from_raw_ffi_any(
                StructuralVisitEngine::<Base>::visit_raw(
                    self.runtime,
                    dispatch,
                    *value.as_raw_ffi_any(),
                ),
            ))
        }
    }

    /// Visit the children of `value` through the dispatcher, without dispatching
    /// `value` itself again.
    ///
    /// Children are found by the type's registered visit hook or its reflected
    /// fields. The visit stops at the first child that returns an interrupt.
    ///
    /// # Arguments
    ///
    /// * `dispatch` - The handler's own `self`.
    /// * `value` - The value whose children to visit.
    ///
    /// # Returns
    ///
    /// `None` to continue, or the interrupt that stopped the visit.
    ///
    /// # Errors
    ///
    /// Returns the first error of a child's handler.
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
            let context = StructuralVisitEngine::<Base>::checked_dispatcher(self.runtime, dispatch);
            let value: AnyView<'input> = value.into();
            visit_result_from_any(Any::from_raw_ffi_any(
                StructuralVisitEngine::<Base>::default_visit_raw(
                    self.runtime,
                    context,
                    *value.as_raw_ffi_any(),
                ),
            ))
        }
    }
}

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
/// * `dispatcher` - Owned or mutably borrowed dispatcher implementing [`IntoDispatcher`].
///   Matching handlers control recursion through [`StructuralVisitorView`];
///   unmatched values use default child traversal.
///
/// # Examples
///
/// Sum integer leaves while default traversal descends into the array:
///
/// ```
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
/// let root = Array::from_vec(vec![1_i64, 2, 3]);
/// let mut sum = Sum(0);
/// assert!(structural_visit(&root, &mut sum)?.is_none());
/// assert_eq!(sum.0, 6);
/// # Ok(())
/// # }
/// ```
pub fn structural_visit<'input, R, H, M, D, O>(
    root: R,
    dispatcher: H,
) -> Result<Option<VisitInterrupt>>
where
    R: Into<AnyView<'input>>,
    O: BorrowMut<D>,
    for<'v, 'c, 'r> H: IntoDispatcher<
        (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, D, StructuralVisitorObj>,
        ),
        Option<Result<Option<VisitInterrupt>>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
        (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, D, StructuralVisitorObj>,
        ),
        Option<Result<Option<VisitInterrupt>>>,
    >,
{
    StructuralVisitEngine::visit_root(root, dispatcher)
}

impl super::dispatch::CallbackResult<Result<Option<VisitInterrupt>>>
    for Result<Option<VisitInterrupt>>
{
    #[inline(always)]
    fn callback_result(self) -> Result<Option<VisitInterrupt>> {
        self
    }
}

// -----------------------------------------------------------------------------
// StructuralVisitEngine implementation
// -----------------------------------------------------------------------------

// Generic runtime: only this implementation projects the dispatcher after Base.
impl<Base: StructuralVisitorBase> StructuralVisitEngine<Base> {
    /// Visit `root` with `dispatcher` on the native base `Base`. See also [`structural_visit`].
    #[inline(always)]
    pub fn visit_root<'input, R, H, M, D, O>(
        root: R,
        dispatcher: H,
    ) -> Result<Option<VisitInterrupt>>
    where
        R: Into<AnyView<'input>>,
        O: BorrowMut<D>,
        for<'v, 'c, 'r> H: IntoDispatcher<
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D, Base>),
            Option<Result<Option<VisitInterrupt>>>,
            M,
            Target = D,
            Output = O,
        >,
        for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D, Base>),
            Option<Result<Option<VisitInterrupt>>>,
        >,
    {
        // Normalize the input once. Named dispatchers keep their own recursive view
        // type; callback and tuple targets are their aggregate adapters.
        let mut storage = dispatcher.into_dispatcher();
        let root = root.into();
        Self::run_visitor::<D>(*root.as_raw_ffi_any(), storage.borrow_mut())
    }

    /// The engine whose first field is the native prefix `runtime`.
    ///
    /// # Safety
    /// `runtime` must belong to the live, synchronous Rust traversal that created
    /// this exact repr(C) engine. An arbitrary native visitor is not such an engine.
    #[inline(always)]
    unsafe fn from_handle(runtime: StructuralVisitorHandle) -> *mut Self {
        runtime.cast()
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
        StructuralVisitorObj::call_visitor_raw(runtime, raw)
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
        let current = dispatcher_ptr(dispatch);
        if current != (*Self::from_handle(runtime)).dispatcher {
            core::hint::cold_path();
            wrong_dispatcher();
        }
        current
    }

    /// Build the native visitor callback for concrete dispatcher `D`. Its higher-ranked
    /// view arguments restrict all borrows to the synchronous handler invocation.
    #[inline(always)]
    fn vtable<D>() -> &'static StructuralVisitorVTable
    where
        for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D, Base>),
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
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D, Base>),
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
                None => Self::default_visit_raw(runtime, dispatcher_ptr(dispatch), raw),
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
    ) -> StructuralVisitorView<'a, D, Base> {
        StructuralVisitorView {
            runtime,
            _marker: PhantomData,
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
        StructuralVisitorObj::with_visit_context_raw(
            Base::default_visit_raw(runtime.cast(), AnyView::from_raw_ffi_any(raw)),
            raw,
        )
    }

    /// Visit borrowed `root` with `dispatch` on a fresh engine; the root's owner outlives it.
    pub(super) fn run_visitor<D>(
        root: TVMFFIAny,
        dispatch: &mut D,
    ) -> Result<Option<VisitInterrupt>>
    where
        for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
            (AnyView<'v>, &'c mut StructuralVisitorView<'r, D, Base>),
            Option<Result<Option<VisitInterrupt>>>,
        >,
    {
        let base = Base::new()?;
        let type_index = Base::type_index();
        // The managed runtime may borrow its dispatcher only for this call.
        // Trusted hooks release all temporary handles before the owner is dropped.
        let mut engine = unsafe {
            ObjectArc::<Object>::new_opaque_unchecked(
                StructuralVisitEngine {
                    base,
                    dispatcher: dispatcher_ptr(dispatch),
                },
                type_index,
            )
        };
        let runtime = ObjectArc::as_mut_ptr(&mut engine).cast::<StructuralVisitorObj>();
        // The engine and outer dispatcher storage stay stationary for the whole run.
        // Only raw field projections cross recursion; no whole-engine borrow survives.
        unsafe {
            (*runtime).vtable = Self::vtable::<D>();
            *(*runtime).def_region_mode.get() = DefRegionKind::None as i32;
        }
        // Keep the result owned while releasing the runtime, including on unwind.
        let result =
            unsafe { Any::from_raw_ffi_any(StructuralVisitorObj::call_visitor_raw(runtime, root)) };
        // Trusted hooks have released every temporary strong/weak runtime reference.
        drop(engine);
        visit_result_from_any(result)
    }
}

// -----------------------------------------------------------------------------
// Prefix-only default traversal and native ABI operations
// -----------------------------------------------------------------------------

impl StructuralVisitorObj {
    /// Consume `result`, naming the borrowed `value` in it when it is an Error.
    #[inline(always)]
    fn with_visit_context_raw(result: TVMFFIAny, value: TVMFFIAny) -> TVMFFIAny {
        if result.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            result
        } else {
            Self::with_visit_stop_context_raw(result, value)
        }
    }

    // Continuing visits need no owning conversion or input inspection. Keep
    // interrupt/error handling out of their recursive frames.
    #[cold]
    #[inline(never)]
    extern "C-unwind" fn with_visit_stop_context_raw(
        result: TVMFFIAny,
        value: TVMFFIAny,
    ) -> TVMFFIAny {
        if result.type_index != TVMFFITypeIndex::kTVMFFIError as i32
            || value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32
        {
            result
        } else {
            // SAFETY: the Error tag establishes the owned object's representation.
            unsafe {
                let error = take_error(Any::from_raw_ffi_any(result));
                Any::into_raw_ffi_any(annotate_visit_error(error, value).into())
            }
        }
    }

    /// Visit the children of `value` through its hook or reflected fields, without
    /// dispatching `value` itself.
    ///
    /// # Safety
    ///
    /// `runtime` is live with its dispatcher published, and `value` outlives descent.
    #[inline(always)]
    unsafe fn visit_children_raw(runtime: StructuralVisitorHandle, value: TVMFFIAny) -> TVMFFIAny {
        if let Some(attr) = structural_visit_column().get_raw_ffi_any(value.type_index) {
            if attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
                return Self::call_structural_visit_hook_raw(runtime, value, attr);
            }
        }

        if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            TVMFFIAny::new()
        } else {
            Self::visit_reflected_fields_raw(runtime, value)
        }
    }

    /// Visit each reflected field of the object in `value`.
    ///
    /// # Safety
    ///
    /// As [`Self::visit_children_raw`]. Outlined (frequent, not cold) to keep reflection
    /// scratch out of recursive frames.
    #[inline(never)]
    unsafe extern "C-unwind" fn visit_reflected_fields_raw(
        runtime: StructuralVisitorHandle,
        value: TVMFFIAny,
    ) -> TVMFFIAny {
        let type_info = unsafe { TVMFFIGetTypeInfo(value.type_index) };
        if type_info.is_null() {
            return Any::into_raw_ffi_any(unregistered_visit_type_error(value.type_index).into());
        }
        let def_region_kind = free_var_child_region(
            std::mem::transmute::<i32, DefRegionKind>(*(*runtime).def_region_mode.get()),
            seq_hash_kind(type_info),
        );
        let object = unsafe { value.data_union.v_obj } as *mut u8;
        let halted = unsafe {
            for_each_field_info(type_info, &mut |field| {
                let result = Self::visit_reflected_field(runtime, object, field, def_region_kind);
                if result.is_none() {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(result)
                }
            })
        };
        Any::into_raw_ffi_any(halted.unwrap_or_else(Any::new))
    }

    /// Visit one reflected `field` of `object` under `inherited_region`.
    ///
    /// # Safety
    ///
    /// `runtime` and `object` are live, and `field` describes `object`'s storage.
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
        if getter(address, Any::as_raw_ffi_any_mut(&mut child)) != 0 {
            return with_field_error_context(Any::from(Error::from_raised()), field.name.as_str());
        }

        let borrowed = AnyView::from(&child);
        let child_region = field_def_region(field, inherited_region);
        with_field_error_context(
            Self::with_def_region_kind(runtime, child_region, || {
                Any::from_raw_ffi_any(Self::call_visitor_raw(runtime, *borrowed.as_raw_ffi_any()))
            }),
            field.name.as_str(),
        )
    }

    /// Dispatch borrowed `value` through the runtime's vtable.
    #[inline(always)]
    fn call_visitor_raw(runtime: StructuralVisitorHandle, value: TVMFFIAny) -> TVMFFIAny {
        if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            core::hint::cold_path();
            return TVMFFIAny::new();
        }
        unsafe {
            let callback = (*(*runtime).vtable).visit;
            callback(runtime, AnyView::from_raw_ffi_any(value))
        }
    }

    /// Call the registered visit `hook` (native pointer or packed function) on `value`.
    ///
    /// # Safety
    ///
    /// `runtime` is live with its dispatcher published, `value` and `hook` outlive the
    /// call, and a native hook has the visit ABI for this runtime's layout and obeys
    /// [`StructuralVisitorBase`]'s non-escaping contract.
    #[inline(always)]
    pub unsafe fn call_structural_visit_hook_raw(
        runtime: *mut Self,
        value: TVMFFIAny,
        hook: TVMFFIAny,
    ) -> TVMFFIAny {
        unsafe {
            if hook.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
                let pointer = hook.data_union.v_ptr;
                // Native pointer registration is an unsafe contract: a non-null
                // callback with this ABI, borrowed runtime lifetime and owning result.
                let hook: FStructuralVisit = std::mem::transmute(pointer);
                hook(runtime, AnyView::from_raw_ffi_any(value))
            } else {
                core::hint::cold_path();
                Self::call_structural_visit_hook_slow_raw(runtime, value, hook)
            }
        }
    }

    /// The packed-function arm of [`Self::call_structural_visit_hook_raw`], outlined as cold;
    /// its result tag is checked.
    #[cold]
    #[inline(never)]
    extern "C-unwind" fn call_structural_visit_hook_slow_raw(
        runtime: StructuralVisitorHandle,
        value: TVMFFIAny,
        hook: TVMFFIAny,
    ) -> TVMFFIAny {
        // Packed hooks (including Python class methods) can return any well-formed
        // Any through safe APIs. Their result tag needs validation before decoding.
        let result = unsafe {
            Function::try_from(AnyView::from_raw_ffi_any(hook)).and_then(|function| {
                // Borrow the live runtime for this synchronous hook; it may not retain it.
                let object = runtime.cast::<TVMFFIObject>();
                let mut runtime = TVMFFIAny::new();
                runtime.type_index = (*object).type_index;
                runtime.data_union.v_obj = object;
                function.call_packed(&[
                    AnyView::from_raw_ffi_any(runtime),
                    AnyView::from_raw_ffi_any(value),
                ])
            })
        };
        let result = match result {
            Ok(value) => checked_hook_result(value),
            Err(error) => error.into(),
        };
        Any::into_raw_ffi_any(result)
    }

    /// Run `callback` under requested `kind` on live borrowed `visitor`, preserving
    /// Pattern precedence. Restore prior region on return or Rust unwind. Internal
    /// callers keep the runtime live and retain no reference to its region field.
    #[inline(always)]
    fn with_def_region_kind<T>(
        visitor: StructuralVisitorHandle,
        kind: DefRegionKind,
        callback: impl FnOnce() -> T,
    ) -> T {
        unsafe {
            let previous = *(*visitor).def_region_mode.get();
            if previous == kind as i32 || previous == DefRegionKind::Pattern as i32 {
                // Inherited descent does not change state; Pattern keeps precedence.
                return callback();
            }
            *(*visitor).def_region_mode.get() = kind as i32;
            struct Restore {
                visitor: StructuralVisitorHandle,
                previous: i32,
            }
            impl Drop for Restore {
                #[inline(always)]
                fn drop(&mut self) {
                    unsafe { *(*self.visitor).def_region_mode.get() = self.previous };
                }
            }
            let _restore = Restore { visitor, previous };
            callback()
        }
    }
}

// -----------------------------------------------------------------------------
// Private utility functions
// -----------------------------------------------------------------------------

#[inline(always)]
fn dispatcher_ptr<D>(dispatch: &mut D) -> *mut c_void {
    std::ptr::from_mut(dispatch).cast()
}

// The visit result carrier, as C++: None continues; a VisitInterrupt or an Error stops.
#[inline(always)]
fn visit_result_raw(result: Result<Option<VisitInterrupt>>, raw: TVMFFIAny) -> TVMFFIAny {
    match result {
        Ok(None) => TVMFFIAny::new(),
        Ok(Some(interrupt)) => Any::into_raw_ffi_any(interrupt.into()),
        Err(error) => {
            core::hint::cold_path();
            StructuralVisitorObj::with_visit_context_raw(Any::into_raw_ffi_any(error.into()), raw)
        }
    }
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
        let mut raw = Any::into_raw_ffi_any(result);
        Ok(Some(unsafe {
            VisitInterrupt::move_from_any_after_check(&mut raw)
        }))
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
    TypeAttrColumn::lookup(STRUCTURAL_VISIT_ATTR).expect("missing __s_visit__ column")
});

/// Return the shared visit-hook column without caching optional per-type cells.
#[inline(always)]
fn structural_visit_column() -> TypeAttrColumn {
    *STRUCTURAL_VISIT_COLUMN
}

// -----------------------------------------------------------------------------
// Outlined errors
// -----------------------------------------------------------------------------

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
fn annotate_visit_error(error: Error, value: TVMFFIAny) -> Error {
    let error = update_visit_error_context(error, value);
    with_structural_error_context(
        error,
        "visit",
        &format!(
            "object `{}`",
            crate::object::type_key_of_index(value.type_index)
        ),
    )
}
