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
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    TVMFFIObject, TVMFFISEqHashKind, TVMFFITypeAttrColumn, TVMFFITypeIndex, TVMFFITypeKeyToIndex,
};

use super::structural_common::{
    borrowed_view, impl_callback_chain_tuple_arities, initialize_scoped_runtime, raw_of,
    with_structural_error_context, with_visit_error_context,
};

const STRUCTURAL_VISIT_ATTR: &str = "__s_visit__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SEQ_HASH_DEF_PATTERN: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefPattern as i64;
const FLAG_SEQ_HASH_DEF_SIMPLE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefSimple as i64;

/// What a callback asks the Rust walker to do with the current value.
pub enum WalkResult {
    /// Continue and visit this value's children.
    Advance,
    /// Continue without visiting this value's children or firing its exit hook.
    Skip,
    /// Halt the entire traversal.
    Interrupt,
    /// Halt the entire traversal and return a payload to the caller.
    InterruptWith(Any),
}

impl WalkResult {
    /// Halt traversal with an FFI-compatible payload.
    pub fn interrupt_with<T: Into<Any>>(payload: T) -> Self {
        Self::InterruptWith(payload.into())
    }
}

/// Convert either an infallible or fallible typed handler result.
///
/// This keeps simple handlers terse while allowing a handler to return
/// `tvm_ffi::Result<WalkResult>` and use `?`.
pub trait IntoWalkResult {
    fn into_walk_result(self) -> Result<WalkResult>;
}

impl IntoWalkResult for WalkResult {
    fn into_walk_result(self) -> Result<WalkResult> {
        Ok(self)
    }
}

impl IntoWalkResult for Result<WalkResult> {
    fn into_walk_result(self) -> Result<WalkResult> {
        self
    }
}

/// Callback order for [`structural_walk`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WalkOrder {
    /// Run the typed handler before the current value's children.
    #[default]
    PreOrder,
    /// Run the typed handler after the current value's children.
    PostOrder,
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

enum NativeHalt {
    Interrupt(Any),
    Error(Error),
}

impl From<Error> for NativeHalt {
    fn from(error: Error) -> Self {
        NativeHalt::Error(error)
    }
}

type NativeResult = std::result::Result<(), NativeHalt>;

/// Recursive visiting and default child descent for a dispatcher.
///
/// Pass the handler's `self` to visit methods to recurse with the same state.
#[repr(transparent)]
pub struct StructuralVisitorView<'a, D> {
    runtime: StructuralVisitorHandle,
    _scope: PhantomData<(&'a mut (), fn(&mut D), Rc<()>)>,
}

fn dispatcher_context<D>(dispatch: &mut D) -> *mut c_void {
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
        self.with_dispatch(dispatch, |runtime| {
            finish(call_visitor(
                runtime,
                raw_of(value.into()),
                self.def_region_kind(),
            ))
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
        self.with_dispatch(dispatch, |runtime| {
            let raw = raw_of(value.into());
            let children = unsafe { (*visit_state(runtime)).default_visit };
            finish(unsafe { children(runtime, raw, self.def_region_kind()) })
        })
    }
    #[inline(always)]
    fn with_dispatch<T>(
        &self,
        dispatch: &mut D,
        callback: impl FnOnce(StructuralVisitorHandle) -> T,
    ) -> T {
        let current = dispatcher_context(dispatch);
        // The supplied borrow must cover the dispatcher that runtime callbacks
        // access. Equal Rust types alone do not identify the same instance.
        if current != unsafe { (*visit_state(self.runtime)).context_identity } {
            wrong_visitor_dispatcher();
        }
        // Rebind from this actual exclusive reborrow, not the cached identity.
        with_current_visitor_context(self.runtime, current, || callback(self.runtime))
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
    base: Base,
    dispatcher: D,
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
#[doc(hidden)]
pub struct ByBaseWalker<M>(PhantomData<M>);

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

impl<Base: StructuralVisitorBase, D: IntoWalker<M>, M> IntoWalker<ByBaseWalker<M>>
    for StructuralVisitor<Base, D>
{
    type Walker = StructuralVisitor<Base, D::Walker>;

    fn into_walker(self) -> Self::Walker {
        StructuralVisitor::new(self.base, self.dispatcher.into_walker())
    }
}

impl<Base, D> NativeContext for StructuralVisitor<Base, D> {}

impl<Base: StructuralVisitorBase, D: NativeVisit> NativeVisit for StructuralVisitor<Base, D> {
    fn walk_root(self, root: AnyView<'_>, order: WalkOrder) -> Result<Option<VisitInterrupt>> {
        finish(run_walker(
            raw_of(root),
            self.base,
            OwnedVisitor(self.dispatcher),
            order,
        ))
    }

    fn visit(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<WalkResult> {
        self.dispatcher.visit(value, kind)
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

pub(crate) use super::dispatch::WalkDispatch;
pub use super::dispatch::{ByWalkDispatch, DispatchWalker};

/// Conversion into the walker argument of [`structural_walk`].
///
/// Accepts a mutable [`Dispatcher`](crate::Dispatcher), a typed callback or a nested callback
/// tuple.
/// `Marker` distinguishes the supported callback shapes.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a supported `structural_walk` walker",
    note = "accepted walkers: `&mut V` with the walk `Dispatcher` signature; a closure over `AnyView<'_>`, \
            an FFI value type `T`, or `&N` of an object node type (`N: ObjectCore`, e.g. \
            `&Object`), optionally with a trailing `DefRegionKind` argument; or a tuple of \
            up to 12 such links (tuples nest, so `(a, (b, c))` chains more)",
    note = "closure arguments need explicit type annotations; ObjectRef wrappers like `String` \
            or `Array<T>` are FFI value types — take them by value, not by reference"
)]
pub trait IntoWalker<Marker> {
    #[doc(hidden)]
    type Walker: NativeVisit;
    #[doc(hidden)]
    fn into_walker(self) -> Self::Walker;
}

impl<F, O> IntoWalker<ByCatchAllLink> for F
where
    F: for<'a> FnMut(AnyView<'a>) -> O,
    O: IntoWalkResult,
{
    type Walker = ChainWalker<F, ByCatchAllLink>;
    fn into_walker(self) -> Self::Walker {
        ChainWalker::new(self)
    }
}

impl<F, O> IntoWalker<ByCatchAllKindLink> for F
where
    F: for<'a> FnMut(AnyView<'a>, DefRegionKind) -> O,
    O: IntoWalkResult,
{
    type Walker = ChainWalker<F, ByCatchAllKindLink>;
    fn into_walker(self) -> Self::Walker {
        ChainWalker::new(self)
    }
}

/// One link in a first-match [`structural_walk`] callback chain.
///
/// Supported links are typed values, borrowed object nodes, `AnyView<'_>`,
/// and mutable [`Dispatcher`](crate::Dispatcher) implementations, optionally followed by
/// [`DefRegionKind`]. Tuples hold up to 12 links and may be nested.
pub trait WalkChainLink<Marker>: sealed::SealedLink<Marker> {
    /// Run this link if `value` matches its argument type; `None` hands the
    /// value to the next link.
    #[doc(hidden)]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>>;
}

pub(crate) use sealed::NativeContext;

mod sealed {
    // Only engine adapters may project the pointer used for ABI reentry.
    // Identity equality cannot establish that a pointer came from the current
    // exclusive reborrow, so this projection must not be user-overridable.
    pub trait NativeContext: Sized {
        fn context(&mut self) -> *mut std::ffi::c_void {
            super::dispatcher_context(self)
        }
    }

    use super::{AnyView, DefRegionKind, IntoWalkResult, ObjectCore, WalkDispatch};

    pub trait SealedLink<Marker> {}

    impl<F, T, O> SealedLink<super::ByOwnedLink<T>> for F
    where
        F: FnMut(T) -> O,
        O: IntoWalkResult,
    {
    }
    impl<F, T, O> SealedLink<super::ByOwnedKindLink<T>> for F
    where
        F: FnMut(T, DefRegionKind) -> O,
        O: IntoWalkResult,
    {
    }
    impl<F, N: ObjectCore, O> SealedLink<super::ByNodeLink<N>> for F
    where
        F: for<'a> FnMut(&'a N) -> O,
        O: IntoWalkResult,
    {
    }
    impl<F, N: ObjectCore, O> SealedLink<super::ByNodeKindLink<N>> for F
    where
        F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
        O: IntoWalkResult,
    {
    }
    impl<F, O> SealedLink<super::ByCatchAllLink> for F
    where
        F: for<'a> FnMut(AnyView<'a>) -> O,
        O: IntoWalkResult,
    {
    }
    impl<F, O> SealedLink<super::ByCatchAllKindLink> for F
    where
        F: for<'a> FnMut(AnyView<'a>, DefRegionKind) -> O,
        O: IntoWalkResult,
    {
    }
    impl<V: WalkDispatch> SealedLink<super::ByWalkDispatchLink> for &mut V {}
}

#[doc(hidden)]
pub struct ByOwnedLink<T>(PhantomData<T>);

impl<F, T, O> WalkChainLink<ByOwnedLink<T>> for F
where
    F: FnMut(T) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        value
            .try_as::<T>()
            .map(|typed| self(typed).into_walk_result())
    }
}

#[doc(hidden)]
pub struct ByOwnedKindLink<T>(PhantomData<T>);

impl<F, T, O> WalkChainLink<ByOwnedKindLink<T>> for F
where
    F: FnMut(T, DefRegionKind) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        value
            .try_as::<T>()
            .map(|typed| self(typed, def_region_kind).into_walk_result())
    }
}

#[doc(hidden)]
pub struct ByNodeLink<N>(PhantomData<N>);

impl<F, N, O> WalkChainLink<ByNodeLink<N>> for F
where
    F: for<'a> FnMut(&'a N) -> O,
    N: ObjectCore,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        value
            .downcast_ref::<N>()
            .map(|node| self(node).into_walk_result())
    }
}

#[doc(hidden)]
pub struct ByNodeKindLink<N>(PhantomData<N>);

impl<F, N, O> WalkChainLink<ByNodeKindLink<N>> for F
where
    F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
    N: ObjectCore,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        value
            .downcast_ref::<N>()
            .map(|node| self(node, def_region_kind).into_walk_result())
    }
}

#[doc(hidden)]
pub enum ByCatchAllLink {}

impl<F, O> WalkChainLink<ByCatchAllLink> for F
where
    F: for<'a> FnMut(AnyView<'a>) -> O,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        Some(self(value).into_walk_result())
    }
}

#[doc(hidden)]
pub enum ByCatchAllKindLink {}

impl<F, O> WalkChainLink<ByCatchAllKindLink> for F
where
    F: for<'a> FnMut(AnyView<'a>, DefRegionKind) -> O,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        Some(self(value, def_region_kind).into_walk_result())
    }
}

#[doc(hidden)]
pub struct ByChainLink<Markers>(PhantomData<fn(Markers)>);

#[doc(hidden)]
pub enum ByWalkDispatchLink {}

impl<V: WalkDispatch> WalkChainLink<ByWalkDispatchLink> for &mut V {
    #[inline]
    fn try_call(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        self.dispatch_walk(value, def_region_kind)
    }
}

/// Adapter from a [`WalkChainLink`] to the native traversal callback.
#[doc(hidden)]
pub struct ChainWalker<Link, Marker> {
    link: Link,
    marker: PhantomData<fn(Marker)>,
}

impl<Link, Marker> ChainWalker<Link, Marker> {
    #[inline]
    fn new(link: Link) -> Self {
        ChainWalker {
            link,
            marker: PhantomData,
        }
    }
}

impl<Link, Marker> NativeContext for ChainWalker<Link, Marker> {}

impl<Link, Marker> NativeVisit for ChainWalker<Link, Marker>
where
    Link: WalkChainLink<Marker>,
{
    #[inline]
    fn visit(&mut self, value: AnyView<'_>, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        self.link
            .try_call(value, def_region_kind)
            .unwrap_or_else(|| Ok(WalkResult::Advance))
    }
}

macro_rules! impl_chain_link {
    ($(($F:ident, $M:ident, $idx:tt)),+) => {
        impl<$($F, $M,)+> sealed::SealedLink<ByChainLink<($($M,)+)>> for ($($F,)+)
        where
            $($F: WalkChainLink<$M>,)+
        {
        }

        impl<$($F, $M,)+> WalkChainLink<ByChainLink<($($M,)+)>> for ($($F,)+)
        where
            $($F: WalkChainLink<$M>,)+
        {
            #[inline]
            fn try_call(
                &mut self,
                value: AnyView<'_>,
                def_region_kind: DefRegionKind,
            ) -> Option<Result<WalkResult>> {
                $(
                    if let Some(result) = self.$idx.try_call(value, def_region_kind) {
                        return Some(result);
                    }
                )+
                None
            }
        }

        impl<$($F, $M,)+> IntoWalker<($($M,)+)> for ($($F,)+)
        where
            $($F: WalkChainLink<$M>,)+
        {
            type Walker = ChainWalker<($($F,)+), ByChainLink<($($M,)+)>>;
            fn into_walker(self) -> Self::Walker {
                ChainWalker::new(self)
            }
        }
    };
}

impl_callback_chain_tuple_arities!(impl_chain_link);

macro_rules! impl_bare_link_walker {
    ($(($marker:ident, $($fn_args:ty),+)),+ $(,)?) => {
        $(
            impl<F, T, O> IntoWalker<$marker<T>> for F
            where
                F: FnMut($($fn_args),+) -> O,
                Self: WalkChainLink<$marker<T>>,
                O: IntoWalkResult,
            {
                type Walker = ChainWalker<F, $marker<T>>;
                fn into_walker(self) -> Self::Walker {
                    ChainWalker::new(self)
                }
            }
        )+
    };
}

impl_bare_link_walker!(
    (ByOwnedLink, T),
    (ByOwnedKindLink, T, DefRegionKind),
    (ByNodeLink, &T),
    (ByNodeKindLink, &T, DefRegionKind),
);

/// Internal callback protocol used by [`IntoWalker`].
#[doc(hidden)]
pub trait NativeVisit: Sized + sealed::NativeContext {
    fn walk_root(self, root: AnyView<'_>, order: WalkOrder) -> Result<Option<VisitInterrupt>> {
        finish(run_walker(
            raw_of(root),
            StructuralVisitorObj::default(),
            OwnedVisitor(self),
            order,
        ))
    }

    fn visit(&mut self, value: AnyView<'_>, def_region_kind: DefRegionKind) -> Result<WalkResult>;
}

impl<V: NativeVisit> NativeContext for &mut V {}

impl<V: NativeVisit> NativeVisit for &mut V {
    fn visit(&mut self, value: AnyView<'_>, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        (**self).visit(value, def_region_kind)
    }
}

/// Action applied to each child found by the shared traversal.
trait ChildVisit {
    fn visit_child(&mut self, child: AnyView<'_>, def_region_kind: DefRegionKind) -> NativeResult;
}

struct WalkChildren<'a, V, const PRE_ORDER: bool> {
    visitor: &'a mut V,
    runtime: StructuralVisitorHandle,
}

impl<V: NativeVisit, const PRE_ORDER: bool> ChildVisit for WalkChildren<'_, V, PRE_ORDER> {
    #[inline(always)]
    fn visit_child(&mut self, child: AnyView<'_>, def_region_kind: DefRegionKind) -> NativeResult {
        visit_raw::<V, PRE_ORDER>(raw_of(child), self.visitor, def_region_kind, self.runtime)
    }
}

struct UserChildren<'a, D> {
    dispatch: &'a mut D,
    runtime: StructuralVisitorHandle,
}

impl<D: VisitDispatch> ChildVisit for UserChildren<'_, D> {
    fn visit_child(&mut self, child: AnyView<'_>, kind: DefRegionKind) -> NativeResult {
        let context = std::ptr::from_mut(&mut *self.dispatch).cast::<c_void>();
        with_current_visitor_context(self.runtime, context, || {
            call_visitor(self.runtime, raw_of(child), kind)
        })
    }
}

#[inline(always)]
fn visit_raw<V: NativeVisit, const PRE_ORDER: bool>(
    value: TVMFFIAny,
    visitor: &mut V,
    def_region_kind: DefRegionKind,
    runtime: StructuralVisitorHandle,
) -> NativeResult {
    if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }
    let visit_value = unsafe { borrowed_view(&value) };
    if PRE_ORDER {
        let action = visitor
            .visit(visit_value, def_region_kind)
            .map_err(|error| with_value_context(error.into(), value))?;
        match action {
            WalkResult::Advance => {}
            WalkResult::Skip => return Ok(()),
            WalkResult::Interrupt => return Err(NativeHalt::Interrupt(Any::new())),
            WalkResult::InterruptWith(payload) => return Err(NativeHalt::Interrupt(payload)),
        }
    }

    let context = visitor.context();
    let default_visit = unsafe { (*visit_state(runtime)).default_visit };
    with_current_visitor_context(runtime, context, || {
        with_visitor_def_region(runtime, def_region_kind, || unsafe {
            default_visit(runtime, value, def_region_kind)
        })
    })?;

    if PRE_ORDER {
        Ok(())
    } else {
        match visitor.visit(visit_value, def_region_kind) {
            Ok(WalkResult::Advance | WalkResult::Skip) => Ok(()),
            Ok(WalkResult::Interrupt) => Err(NativeHalt::Interrupt(Any::new())),
            Ok(WalkResult::InterruptWith(payload)) => Err(NativeHalt::Interrupt(payload)),
            Err(error) => Err(with_value_context(error.into(), value)),
        }
    }
}

#[inline]
fn visit_children_raw<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    driver_context: *mut c_void,
    def_region_kind: DefRegionKind,
    runtime: StructuralVisitorHandle,
) -> NativeResult {
    if let Some(attr) =
        structural_visit_column().and_then(|column| column.get_raw(value.type_index))
    {
        if attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
            return with_current_visitor_context(runtime, driver_context, || {
                call_structural_visit_hook(runtime, value, def_region_kind, attr)
            });
        }
    }

    if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        Ok(())
    } else {
        visit_reflected_fields(value, visitor, def_region_kind)
    }
}

#[inline]
fn visit_reflected_fields<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    let type_info = unsafe { TVMFFIGetTypeInfo(value.type_index) };
    if type_info.is_null() {
        return Err(runtime_error(&format!(
            "native visitor: unregistered type index {}",
            value.type_index
        ))
        .into());
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
        for_each_field_info(type_info, &mut |field| match visit_reflected_field(
            object,
            field,
            visitor,
            def_region_kind,
        ) {
            Ok(()) => ControlFlow::Continue(()),
            Err(halt) => ControlFlow::Break(halt),
        })
    };
    halted.map_or(Ok(()), Err)
}

unsafe fn visit_reflected_field<C: ChildVisit>(
    object: *mut u8,
    field: &TVMFFIFieldInfo,
    visitor: &mut C,
    inherited_region: DefRegionKind,
) -> NativeResult {
    if field.flags & FLAG_SEQ_HASH_IGNORE != 0 {
        return Ok(());
    }

    let Some(getter) = field.getter else {
        return Err(NativeHalt::Error(runtime_error(&format!(
            "native visitor: reflected field `{}` has no getter",
            field.name.as_str()
        ))));
    };
    let address = object.offset(field.offset as isize) as *mut c_void;
    // Own the getter result so partial writes and recursive borrows drop safely.
    let mut child = Any::new();
    if getter(address, Any::as_data_ptr(&mut child)) != 0 {
        return Err(with_error_context(
            NativeHalt::Error(Error::from_raised()),
            &format!("field `{}`", field.name.as_str()),
        ));
    }

    let borrowed = AnyView::from(&child);
    let child_region = field_def_region(field, inherited_region);
    visitor
        .visit_child(borrowed, child_region)
        .map_err(|halt| with_error_context(halt, &format!("field `{}`", field.name.as_str())))
}

type StructuralVisitorHandle = *mut StructuralVisitorObj;
type FStructuralVisit =
    unsafe extern "C" fn(StructuralVisitorHandle, AnyView<'static>) -> TVMFFIAny;

/// Rust vtable extension following the unchanged C++ ABI entries.
#[repr(C)]
struct StructuralVisitorVTable {
    visit: FStructuralVisit,
    state_offset: usize,
}

/// Exact C++ StructuralVisitorObj ABI prefix.
#[repr(C)]
pub struct StructuralVisitorObj {
    base: Object,
    vtable: *const StructuralVisitorVTable,
    def_region_mode: i32,
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
    pub fn default_visit<'input, D>(
        dispatch: &mut D,
        value: impl Into<AnyView<'input>>,
        visitor: &mut StructuralVisitorView<'_, D>,
    ) -> Result<Option<VisitInterrupt>> {
        visitor.with_dispatch(dispatch, |runtime| {
            let raw = raw_of(value.into());
            let builtin = unsafe { (*visit_state(runtime)).builtin_default_visit };
            finish(unsafe { builtin(runtime, raw, visitor.def_region_kind()) })
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
    unsafe fn default_visit<D>(
        _this: *mut Self,
        dispatch: &mut D,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, D>,
    ) -> Result<Option<VisitInterrupt>> {
        StructuralVisitorObj::default_visit(dispatch, value, visitor)
    }
}

// SAFETY: the builtin base is the exact ABI prefix itself.
unsafe impl StructuralVisitorBase for StructuralVisitorObj {}

struct VisitState {
    context: *mut c_void,
    context_identity: *mut c_void,
    default_visit: DefaultVisitCallback,
    builtin_default_visit: DefaultVisitCallback,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

#[repr(C)]
struct StructuralVisitEngine<Base, Dispatcher> {
    base: Base,
    state: VisitState,
    dispatcher: Dispatcher,
}

#[inline(always)]
unsafe fn visit_state(runtime: StructuralVisitorHandle) -> *mut VisitState {
    let offset = (*(*runtime).vtable).state_offset;
    runtime.cast::<u8>().add(offset).cast()
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

// The Rust extension projects state after the concrete, potentially aligned Base.
fn walk_runtime_vtable<Base, S: VisitorStorage, const PRE_ORDER: bool>(
) -> &'static StructuralVisitorVTable
where
    S::Dispatcher: NativeVisit,
{
    &StructuralVisitorVTable {
        visit: rust_vtable_walk::<S::Dispatcher, PRE_ORDER>,
        state_offset: std::mem::offset_of!(StructuralVisitEngine<Base, S>, state),
    }
}

fn user_runtime_vtable<Base, S: VisitorStorage>() -> &'static StructuralVisitorVTable
where
    S::Dispatcher: VisitDispatch,
{
    &StructuralVisitorVTable {
        visit: rust_vtable_user::<S::Dispatcher>,
        state_offset: std::mem::offset_of!(StructuralVisitEngine<Base, S>, state),
    }
}

unsafe extern "C" fn rust_vtable_walk<V: NativeVisit, const PRE_ORDER: bool>(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    rust_vtable_visit_impl(visitor, value, |context, raw, kind| {
        runtime_walk::<V, PRE_ORDER>(visitor, context, raw, kind)
    })
}

unsafe extern "C" fn rust_vtable_user<V: VisitDispatch>(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    rust_vtable_visit_impl(visitor, value, |context, raw, kind| {
        runtime_user_visit::<V>(visitor, context, raw, kind)
    })
}

#[inline(always)]
unsafe fn rust_vtable_visit_impl(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
    callback: impl FnOnce(*mut c_void, TVMFFIAny, DefRegionKind) -> NativeResult,
) -> TVMFFIAny {
    let context = (*visit_state(visitor)).context;
    let raw = *value.as_raw_ffi_any();
    let outcome = catch_unwind(AssertUnwindSafe(
        #[inline(always)]
        || {
            let result = def_region_from_raw((*visitor).def_region_mode)
                .map_err(NativeHalt::Error)
                .and_then(|kind| callback(context, raw, kind));
            native_result_into_raw(result)
        },
    ));
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
    native_result_into_raw(Err(NativeHalt::Error(runtime_error(
        "panic in structural visitor callback",
    ))))
}

/// Bind a pointer derived from the current exclusive reborrow for a synchronous
/// trusted hook call. This internal path performs no liveness/identity discovery.
#[inline(always)]
fn with_current_visitor_context<T>(
    visitor: StructuralVisitorHandle,
    context: *mut c_void,
    callback: impl FnOnce() -> T,
) -> T {
    unsafe {
        let previous = (*visit_state(visitor)).context;
        (*visit_state(visitor)).context = context;
        struct Restore {
            visitor: StructuralVisitorHandle,
            previous: *mut c_void,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                unsafe {
                    (*visit_state(self.visitor)).context = self.previous;
                }
            }
        }
        let _restore = Restore { visitor, previous };
        callback()
    }
}

#[inline(always)]
unsafe fn runtime_walk<V: NativeVisit, const PRE_ORDER: bool>(
    runtime: StructuralVisitorHandle,
    context: *mut c_void,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    visit_raw::<V, PRE_ORDER>(raw, &mut *context.cast::<V>(), def_region_kind, runtime)
}

#[inline(always)]
unsafe fn runtime_user_visit<D: VisitDispatch>(
    runtime: StructuralVisitorHandle,
    context: *mut c_void,
    raw: TVMFFIAny,
    kind: DefRegionKind,
) -> NativeResult {
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }
    let dispatch = &mut *context.cast::<D>();
    let mut visitor = visitor_context(runtime);
    match dispatch.dispatch_visit(borrowed_view(&raw), &mut visitor) {
        Some(result) => visit_result(result, raw),
        None => {
            let current = dispatcher_context(dispatch);
            let default_visit = (*visit_state(runtime)).default_visit;
            with_current_visitor_context(runtime, current, || default_visit(runtime, raw, kind))
        }
    }
}

/// # Safety
/// The runtime and current scope must outlive the returned view, with callbacks
/// bound to D. The view is only passed through higher-ranked handler calls.
#[inline(always)]
unsafe fn visitor_context<'a, D>(runtime: StructuralVisitorHandle) -> StructuralVisitorView<'a, D> {
    StructuralVisitorView {
        runtime,
        _scope: PhantomData,
    }
}

type DefaultVisitCallback =
    unsafe fn(StructuralVisitorHandle, TVMFFIAny, DefRegionKind) -> NativeResult;

fn visit_result(result: Result<Option<VisitInterrupt>>, raw: TVMFFIAny) -> NativeResult {
    match result {
        Ok(None) => Ok(()),
        Ok(Some(interrupt)) => Err(NativeHalt::Interrupt(interrupt.value)),
        Err(error) => Err(with_value_context(error.into(), raw)),
    }
}

unsafe fn base_default_visit<Base: StructuralVisitorBase, D>(
    runtime: StructuralVisitorHandle,
    raw: TVMFFIAny,
    _kind: DefRegionKind,
) -> NativeResult {
    let dispatch = &mut *(*visit_state(runtime)).context.cast::<D>();
    let mut visitor = visitor_context(runtime);
    visit_result(
        Base::default_visit(runtime.cast(), dispatch, borrowed_view(&raw), &mut visitor),
        raw,
    )
}

unsafe fn walk_children<D: NativeVisit, const PRE_ORDER: bool>(
    runtime: StructuralVisitorHandle,
    raw: TVMFFIAny,
    kind: DefRegionKind,
) -> NativeResult {
    let visitor = &mut *(*visit_state(runtime)).context.cast::<D>();
    let current = dispatcher_context(visitor);
    visit_children_raw(
        raw,
        &mut WalkChildren::<D, PRE_ORDER> { visitor, runtime },
        current,
        kind,
        runtime,
    )
    .map_err(|halt| with_value_context(halt, raw))
}

unsafe fn user_children<D: VisitDispatch>(
    runtime: StructuralVisitorHandle,
    raw: TVMFFIAny,
    kind: DefRegionKind,
) -> NativeResult {
    let dispatch = &mut *(*visit_state(runtime)).context.cast::<D>();
    let current = dispatcher_context(dispatch);
    visit_children_raw(
        raw,
        &mut UserChildren { dispatch, runtime },
        current,
        kind,
        runtime,
    )
    .map_err(|halt| with_value_context(halt, raw))
}

trait VisitorStorage {
    type Dispatcher;
    fn dispatch(&mut self) -> &mut Self::Dispatcher;
}
#[repr(transparent)]
struct OwnedVisitor<D>(D);
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
) -> NativeResult
where
    S::Dispatcher: VisitDispatch,
{
    run_visitor_engine(
        root,
        base,
        storage,
        user_runtime_vtable::<Base, S>(),
        user_children::<S::Dispatcher>,
    )
}

fn run_walker<Base: StructuralVisitorBase, S: VisitorStorage>(
    root: TVMFFIAny,
    base: Base,
    storage: S,
    order: WalkOrder,
) -> NativeResult
where
    S::Dispatcher: NativeVisit,
{
    match order {
        WalkOrder::PreOrder => run_visitor_engine(
            root,
            base,
            storage,
            walk_runtime_vtable::<Base, S, true>(),
            walk_children::<S::Dispatcher, true>,
        ),
        WalkOrder::PostOrder => run_visitor_engine(
            root,
            base,
            storage,
            walk_runtime_vtable::<Base, S, false>(),
            walk_children::<S::Dispatcher, false>,
        ),
    }
}

fn run_visitor_engine<Base: StructuralVisitorBase, S: VisitorStorage>(
    root: TVMFFIAny,
    base: Base,
    storage: S,
    vtable: &'static StructuralVisitorVTable,
    builtin_default_visit: DefaultVisitCallback,
) -> NativeResult {
    let mut engine = Box::new(StructuralVisitEngine {
        base,
        state: VisitState {
            context: std::ptr::null_mut(),
            context_identity: std::ptr::null_mut(),
            default_visit: base_default_visit::<Base, S::Dispatcher>,
            builtin_default_visit,
            panic: None,
        },
        dispatcher: storage,
    });
    // Allocation is stable before projecting the aligned final dispatcher field.
    let engine_ptr = std::ptr::from_mut(&mut *engine);
    let handle = engine_ptr.cast::<StructuralVisitorObj>();
    unsafe {
        (*handle).vtable = vtable;
        (*handle).def_region_mode = DefRegionKind::None as i32;
        initialize_scoped_runtime(handle.cast(), StructuralVisitorObj::type_index());
        let offset = std::mem::offset_of!(StructuralVisitEngine<Base, S>, dispatcher);
        let storage = &mut *engine_ptr.cast::<u8>().add(offset).cast::<S>();
        let current = dispatcher_context(storage.dispatch());
        (*visit_state(handle)).context = current;
        (*visit_state(handle)).context_identity = current;
    }
    let result = call_visitor(handle, root, DefRegionKind::None);
    let panic = unsafe { (*visit_state(handle)).panic.take() };
    // Trusted hooks have released every temporary strong/weak runtime reference.
    drop(engine);
    if let Some(payload) = panic {
        drop(result);
        resume_unwind(payload);
    }
    result
}

fn call_visitor(
    visitor: StructuralVisitorHandle,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }
    let callback = unsafe { (*(*visitor).vtable).visit };
    with_visitor_def_region(visitor, def_region_kind, || unsafe {
        let value = AnyView::from_raw_ffi_any(raw);
        visit_result_from_raw(callback(visitor, value))
    })
}

fn call_structural_visit_hook(
    visitor: StructuralVisitorHandle,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    attr: TVMFFIAny,
) -> NativeResult {
    with_visitor_def_region(visitor, def_region_kind, || unsafe {
        match attr.type_index {
            x if x == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 => {
                let pointer = attr.data_union.v_ptr;
                if pointer.is_null() {
                    return Err(runtime_error("structural visit hook is null").into());
                }
                // The `__s_visit__` protocol stores exactly an
                // `FStructuralVisit` in an opaque-pointer attribute.
                let hook: FStructuralVisit = std::mem::transmute(pointer);
                let value = AnyView::from_raw_ffi_any(raw);
                visit_result_from_raw(hook(visitor, value))
            }
            x if x == TVMFFITypeIndex::kTVMFFIFunction as i32 => {
                let function = Function::try_from(AnyView::from_raw_ffi_any(attr))?;
                let visitor_value = borrowed_visitor_view(visitor);
                let value = AnyView::from_raw_ffi_any(raw);
                visit_result_from_any(function.call_packed(&[visitor_value, value])?)
            }
            _ => Err(Error::new(
                TYPE_ERROR,
                "__s_visit__ must be an opaque function pointer or ffi.Function",
                "",
            )
            .into()),
        }
    })
}

unsafe fn borrowed_visitor_view<'a>(visitor: StructuralVisitorHandle) -> AnyView<'a> {
    let object = visitor.cast::<TVMFFIObject>();
    let mut raw = TVMFFIAny::new();
    raw.type_index = (*object).type_index;
    raw.small_str_len = 0;
    raw.data_union.v_obj = object;
    AnyView::from_raw_ffi_any(raw)
}

#[inline(always)]
fn native_result_into_raw(result: NativeResult) -> TVMFFIAny {
    match result {
        Ok(()) => TVMFFIAny::new(),
        Err(NativeHalt::Error(error)) => unsafe { Any::into_raw_ffi_any(Any::from(error)) },
        Err(NativeHalt::Interrupt(payload)) => {
            let interrupt = ObjectArc::new(RuntimeVisitInterruptObj {
                base: Object::new(),
                value: payload,
            });
            let object = unsafe { ObjectArc::into_raw(interrupt) }.cast_mut();
            let mut raw = TVMFFIAny::new();
            raw.type_index = TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32;
            raw.data_union.v_obj = object.cast::<TVMFFIObject>();
            raw
        }
    }
}

unsafe fn visit_result_from_raw(raw: TVMFFIAny) -> NativeResult {
    // None is the overwhelmingly common success result. Avoid constructing
    // and dropping an owning Any unless the hook actually stopped or failed.
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        Ok(())
    } else {
        visit_result_from_any(Any::from_raw_ffi_any(raw))
    }
}

fn visit_result_from_any(value: Any) -> NativeResult {
    match value.type_index() {
        x if x == TVMFFITypeIndex::kTVMFFINone as i32 => Ok(()),
        x if x == TVMFFITypeIndex::kTVMFFIError as i32 => match Error::try_from(value) {
            Ok(error) | Err(error) => Err(NativeHalt::Error(error)),
        },
        x if x == TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32 => {
            let raw = *value.as_raw_ffi_any();
            let object = unsafe { raw.data_union.v_obj };
            if object.is_null() {
                return Err(runtime_error("structural visit returned a null interrupt").into());
            }
            let payload = unsafe { (*object.cast::<RuntimeVisitInterruptObj>()).value.clone() };
            Err(NativeHalt::Interrupt(payload))
        }
        _ => Err(Error::new(
            TYPE_ERROR,
            "structural visit hook must return None or ffi.VisitInterrupt",
            "",
        )
        .into()),
    }
}

#[inline(always)]
fn with_visitor_def_region<T>(
    visitor: StructuralVisitorHandle,
    kind: DefRegionKind,
    callback: impl FnOnce() -> T,
) -> T {
    unsafe {
        let previous = (*visitor).def_region_mode;
        // Precedence: a pattern region propagates; entering any kind inside it has no effect.
        if previous != DefRegionKind::Pattern as i32 {
            (*visitor).def_region_mode = kind as i32;
        }
        struct Restore {
            visitor: StructuralVisitorHandle,
            previous: i32,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                if self.previous != DefRegionKind::Pattern as i32 {
                    unsafe { (*self.visitor).def_region_mode = self.previous };
                }
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

#[cold]
fn with_value_context(halt: NativeHalt, value: TVMFFIAny) -> NativeHalt {
    if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        halt
    } else {
        let halt = match halt {
            NativeHalt::Error(error) => NativeHalt::Error(with_visit_error_context(error, value)),
            interrupt => interrupt,
        };
        with_error_context(halt, &format!("object `{}`", type_key_of(value.type_index)))
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

/// Walk `root` with an observer, the Rust analog of C++
/// `StructuralWalk<order>(root, callbacks...)`.
///
/// `walker` is anything implementing [`IntoWalker`]: a `&mut` reference to a
/// stateful [`Dispatcher`](crate::Dispatcher) walker (`#[dispatch(walk)]`), a bare closure
/// in any [`WalkChainLink`] shape (catch-all `AnyView<'_>`, typed, or node,
/// with an optional trailing [`DefRegionKind`]), or a tuple of such
/// callbacks tried in order — the C++ callback overloads and variadic
/// chain.
/// Callback views cannot escape their invocation; retain an [`Any`] explicitly
/// when a value must outlive the callback:
/// ```compile_fail,E0521
/// use tvm_ffi::{structural_walk, AnyView, WalkOrder, WalkResult};
/// let mut retained: Option<AnyView<'static>> = None;
/// structural_walk(&1_i64, |value: AnyView<'_>| {
///     retained = Some(value);
///     WalkResult::Advance
/// }, WalkOrder::PreOrder).unwrap();
/// ```
///
/// The walker owns recursion: the handler runs once per value,
/// before or after the value's children according to `order`, and steers
/// traversal through the returned [`WalkResult`].
pub fn structural_walk<'input, R, M, H>(
    root: R,
    walker: H,
    order: WalkOrder,
) -> Result<Option<VisitInterrupt>>
where
    H: IntoWalker<M>,
    R: Into<AnyView<'input>>,
{
    walker.into_walker().walk_root(root.into(), order)
}

fn finish(result: NativeResult) -> Result<Option<VisitInterrupt>> {
    match result {
        Ok(()) => Ok(None),
        Err(NativeHalt::Error(error)) => Err(error),
        Err(NativeHalt::Interrupt(payload)) => Ok(Some(VisitInterrupt { value: payload })),
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

#[cold]
fn with_error_context(halt: NativeHalt, frame: &str) -> NativeHalt {
    match halt {
        NativeHalt::Error(error) => {
            NativeHalt::Error(with_structural_error_context(error, "visit", frame))
        }
        interrupt => interrupt,
    }
}

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

pub(crate) fn type_attr_column(attr_name: &str) -> Option<TypeAttrColumn> {
    TypeAttrColumn::new(attr_name)
}

/// Cached `__s_visit__` column pointer (0 = not seen yet). A registry column
/// is stable once created — C++ `DefaultVisitExpected` caches the same
/// pointer in a function-local static — while an absent column is re-queried
/// because a later attr registration may create it.
static STRUCTURAL_VISIT_COLUMN: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn structural_visit_column() -> Option<TypeAttrColumn> {
    let cached = STRUCTURAL_VISIT_COLUMN.load(Ordering::Relaxed);
    if cached != 0 {
        let pointer = cached as *mut TVMFFITypeAttrColumn;
        return Some(unsafe { TypeAttrColumn::from_non_null(NonNull::new_unchecked(pointer)) });
    }
    initialize_structural_visit_column()
}

#[cold]
#[inline(never)]
fn initialize_structural_visit_column() -> Option<TypeAttrColumn> {
    let column = type_attr_column(STRUCTURAL_VISIT_ATTR)?;
    STRUCTURAL_VISIT_COLUMN.store(column.as_ptr() as usize, Ordering::Relaxed);
    Some(column)
}

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
