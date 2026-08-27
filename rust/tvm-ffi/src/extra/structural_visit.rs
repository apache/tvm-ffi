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
//! * [`StructuralVisitor`] + [`structural_visit`] — the visitor drives
//!   recursion itself, like a hand-written C++ `StructuralVisitorObj`:
//!   [`StructuralVisitor::visit`] runs once per reached value and descends
//!   only where it calls [`StructuralVisitor::default_visit_children`] or
//!   [`StructuralVisitor::visit_child`]. `#[dispatch(visit)]` generates this
//!   trait from typed `visit_*` methods.
//! * [`WalkDispatch`] + [`structural_walk`] — observer callbacks, like C++
//!   `StructuralWalk`: the walker recurses on its own and callbacks steer it
//!   through the returned [`WalkResult`] (advance, skip, interrupt).
//!
//! Both layers thread the definition-region state explicitly: walk handlers
//! opt in with a trailing [`DefRegionKind`] argument, and a visitor receives
//! and forwards it when descending.
//!
//! Underneath both, [`VisitValue`] provides borrowed matching for typed Rust
//! dispatch. Rust supplies a temporary `ffi.StructuralVisitor` ABI object so
//! every type's registered `__s_visit__` hook can enumerate its children and
//! call back into the active Rust visitor. Types without a hook fall back to
//! reflected structural fields, matching the C++ protocol.

use std::cell::Cell;
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
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive, kTVMFFIFieldFlagBitMaskSEqHashDefRecursive,
    kTVMFFIFieldFlagBitMaskSEqHashIgnore,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIDefRegionKind, TVMFFIFieldInfo, TVMFFIGetTypeAttrColumn,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFISEqHashKind, TVMFFITypeAttrColumn, TVMFFITypeIndex,
    TVMFFITypeKeyToIndex,
};

use super::structural_common::{impl_callback_chain_tuple_arities, with_structural_error_context};

const STRUCTURAL_VISIT_ATTR: &str = "__s_visit__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SEQ_HASH_DEF_RECURSIVE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64;
const FLAG_SEQ_HASH_DEF_NON_RECURSIVE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive as i64;

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
/// Reflected fields marked `SEqHashDefRecursive` or
/// `SEqHashDefNonRecursive` override the inherited state for that field's
/// complete recursive visit.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DefRegionKind {
    /// The value is outside a definition region.
    #[default]
    None = 0,
    /// Definitions apply recursively through the visited value.
    Recursive = 1,
    /// Definitions apply to the visited value using non-recursive semantics.
    NonRecursive = 2,
}

const _: () = {
    assert!(DefRegionKind::None as i32 == TVMFFIDefRegionKind::kTVMFFIDefRegionKindNone as i32);
    assert!(
        DefRegionKind::Recursive as i32
            == TVMFFIDefRegionKind::kTVMFFIDefRegionKindRecursive as i32
    );
    assert!(
        DefRegionKind::NonRecursive as i32
            == TVMFFIDefRegionKind::kTVMFFIDefRegionKindNonRecursive as i32
    );
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

/// Convert a callback result into structural-visit completion state.
#[doc(hidden)]
pub trait IntoVisitResult {
    fn into_visit_result(self) -> Result<Option<VisitInterrupt>>;
}

impl IntoVisitResult for () {
    #[inline]
    fn into_visit_result(self) -> Result<Option<VisitInterrupt>> {
        Ok(None)
    }
}

impl IntoVisitResult for Result<()> {
    #[inline]
    fn into_visit_result(self) -> Result<Option<VisitInterrupt>> {
        self.map(|()| None)
    }
}

impl IntoVisitResult for Option<VisitInterrupt> {
    #[inline]
    fn into_visit_result(self) -> Result<Option<VisitInterrupt>> {
        Ok(self)
    }
}

impl IntoVisitResult for Result<Option<VisitInterrupt>> {
    #[inline]
    fn into_visit_result(self) -> Result<Option<VisitInterrupt>> {
        self
    }
}

/// Fallible result returned by generated typed dispatch.
#[doc(hidden)]
pub type WalkCallbackResult = Result<WalkResult>;

/// A borrowed view of a raw tvm-ffi value passed to structural-visit callbacks.
///
/// Generated visitors match this value without taking ownership: borrowed
/// object-node handlers use [`VisitValue::as_node`], while POD or object-ref
/// value handlers use [`VisitValue::cast`].
pub use super::structural_common::StructuralValue as VisitValue;

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

/// State and recursive operations available to a visit callback.
///
/// A matched callback owns traversal of its value. Recursive operations
/// reborrow the visitor, so mutable state cannot remain borrowed across them.
pub struct VisitContext<'a, State> {
    driver: &'a mut dyn VisitContextDriver<State>,
    current: VisitValue,
    def_region_kind: DefRegionKind,
    _not_send_sync: PhantomData<Rc<()>>,
}

trait VisitContextDriver<State> {
    fn state(&self) -> &State;
    fn state_mut(&mut self) -> &mut State;
    fn visit_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>>;
    fn visit_children_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>>;
}

impl<State> VisitContext<'_, State> {
    /// User state shared by every callback in this traversal.
    pub fn state(&self) -> &State {
        self.driver.state()
    }

    /// Mutably borrow the user state.
    pub fn state_mut(&mut self) -> &mut State {
        self.driver.state_mut()
    }

    /// Complete borrowed value active at this callback.
    pub fn current(&self) -> &VisitValue {
        &self.current
    }

    /// Definition-region state active at the callback's current value.
    pub fn def_region_kind(&self) -> DefRegionKind {
        self.def_region_kind
    }

    /// Visit `child` using the current definition-region state.
    pub fn visit<T>(&mut self, child: &T) -> Result<Option<VisitInterrupt>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.visit_with(child, self.def_region_kind)
    }

    /// Visit `child` using an explicit definition-region state.
    pub fn visit_with<T>(
        &mut self,
        child: &T,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let raw = raw_of(AnyView::from(child));
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(None);
        }
        self.driver.visit_raw(raw, def_region_kind)
    }

    /// Visit the current value's children using registered hooks or reflected
    /// structural fields. The current value itself is not dispatched again.
    pub fn visit_children(&mut self) -> Result<Option<VisitInterrupt>> {
        self.driver
            .visit_children_raw(self.current.raw(), self.def_region_kind)
    }
}

/// Conversion into the visitor argument accepted by [`structural_visit`].
///
/// Accepts a mutable [`StructuralVisitor`] or a first-match callback chain.
/// Use [`VisitCallbacks`] when the chain needs mutable state.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a supported `structural_visit` visitor",
    note = "accepted visitors: `&mut V` where `V: StructuralVisitor`; an `Fn` callback over an FFI value type `T`, `&N` of an object node type, or `&VisitValue`, followed by `&mut VisitContext<'_, ()>`; or a tuple of up to 12 such callbacks (tuples may nest)",
    note = "callback arguments need explicit type annotations; use `VisitCallbacks::new(state, callbacks)` for ordinary mutable callback state"
)]
pub trait IntoVisitor<Marker> {
    #[doc(hidden)]
    fn visit_root(self, root: TVMFFIAny) -> Result<Option<VisitInterrupt>>;
}

impl<V: StructuralVisitor> IntoVisitor<V> for &mut V {
    fn visit_root(self, root: TVMFFIAny) -> Result<Option<VisitInterrupt>> {
        finish(run_structural_visitor(
            root,
            self,
            user_runtime_vtable::<V>(),
        ))
    }
}

/// One link in a first-match visitor callback chain.
pub trait VisitChainLink<State, Marker>: visit_sealed::SealedLink<State, Marker> {
    #[doc(hidden)]
    fn try_visit(
        &self,
        value: &VisitValue,
        visitor: &mut VisitContext<'_, State>,
    ) -> Option<Result<Option<VisitInterrupt>>>;
}

mod visit_sealed {
    use super::{IntoVisitResult, ObjectCore, VisitContext, VisitValue};

    pub trait SealedLink<State, Marker> {}

    impl<F, State, T, O> SealedLink<State, super::ByVisitOwnedLink<T>> for F
    where
        F: for<'visitor, 'driver> Fn(T, &'visitor mut VisitContext<'driver, State>) -> O,
        O: IntoVisitResult,
    {
    }

    impl<F, State, N: ObjectCore, O> SealedLink<State, super::ByVisitNodeLink<N>> for F
    where
        F: for<'value, 'visitor, 'driver> Fn(
            &'value N,
            &'visitor mut VisitContext<'driver, State>,
        ) -> O,
        O: IntoVisitResult,
    {
    }

    impl<F, State, O> SealedLink<State, super::ByVisitCatchAllLink> for F
    where
        F: for<'value, 'visitor, 'driver> Fn(
            &'value VisitValue,
            &'visitor mut VisitContext<'driver, State>,
        ) -> O,
        O: IntoVisitResult,
    {
    }
}

#[doc(hidden)]
pub struct ByVisitOwnedLink<T>(PhantomData<T>);

impl<F, State, T, O> VisitChainLink<State, ByVisitOwnedLink<T>> for F
where
    F: for<'visitor, 'driver> Fn(T, &'visitor mut VisitContext<'driver, State>) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoVisitResult,
{
    fn try_visit(
        &self,
        value: &VisitValue,
        visitor: &mut VisitContext<'_, State>,
    ) -> Option<Result<Option<VisitInterrupt>>> {
        value
            .cast::<T>()
            .map(|typed| self(typed, visitor).into_visit_result())
    }
}

#[doc(hidden)]
pub struct ByVisitNodeLink<N>(PhantomData<N>);

impl<F, State, N, O> VisitChainLink<State, ByVisitNodeLink<N>> for F
where
    F: for<'value, 'visitor, 'driver> Fn(
        &'value N,
        &'visitor mut VisitContext<'driver, State>,
    ) -> O,
    N: ObjectCore,
    O: IntoVisitResult,
{
    fn try_visit(
        &self,
        value: &VisitValue,
        visitor: &mut VisitContext<'_, State>,
    ) -> Option<Result<Option<VisitInterrupt>>> {
        value
            .as_node::<N>()
            .map(|node| self(node, visitor).into_visit_result())
    }
}

#[doc(hidden)]
pub enum ByVisitCatchAllLink {}

impl<F, State, O> VisitChainLink<State, ByVisitCatchAllLink> for F
where
    F: for<'value, 'visitor, 'driver> Fn(
        &'value VisitValue,
        &'visitor mut VisitContext<'driver, State>,
    ) -> O,
    O: IntoVisitResult,
{
    fn try_visit(
        &self,
        value: &VisitValue,
        visitor: &mut VisitContext<'_, State>,
    ) -> Option<Result<Option<VisitInterrupt>>> {
        Some(self(value, visitor).into_visit_result())
    }
}

#[doc(hidden)]
pub struct ByVisitChainLink<Markers>(PhantomData<fn(Markers)>);

macro_rules! impl_visit_chain_link {
    ($(($F:ident, $M:ident, $idx:tt)),+) => {
        impl<State, $($F, $M,)+>
            visit_sealed::SealedLink<State, ByVisitChainLink<($($M,)+)>> for ($($F,)+)
        where
            $($F: VisitChainLink<State, $M>,)+
        {
        }

        impl<State, $($F, $M,)+> VisitChainLink<State, ByVisitChainLink<($($M,)+)>>
            for ($($F,)+)
        where
            $($F: VisitChainLink<State, $M>,)+
        {
            fn try_visit(
                &self,
                value: &VisitValue,
                visitor: &mut VisitContext<'_, State>,
            ) -> Option<Result<Option<VisitInterrupt>>> {
                $(
                    if let Some(result) = self.$idx.try_visit(value, visitor) {
                        return Some(result);
                    }
                )+
                None
            }
        }
    };
}

impl_callback_chain_tuple_arities!(impl_visit_chain_link);

/// A reusable callback visitor with shared user state.
pub struct VisitCallbacks<State, Link, Marker> {
    state: State,
    callbacks: Rc<Link>,
    _marker: PhantomData<fn(Marker)>,
}

impl<State, Link, Marker> VisitCallbacks<State, Link, Marker>
where
    Link: VisitChainLink<State, Marker>,
{
    /// Construct a stateful callback visitor.
    pub fn new(state: State, callbacks: Link) -> Self {
        Self {
            state,
            callbacks: Rc::new(callbacks),
            _marker: PhantomData,
        }
    }
}

impl<State, Link, Marker> VisitCallbacks<State, Link, Marker> {
    /// Shared access to the callback state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Mutable access to the callback state outside an active recursive call.
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Consume the visitor and return its state.
    pub fn into_state(self) -> State {
        self.state
    }
}

struct DirectVisitCallbacks<'a, Link, Marker> {
    state: (),
    callbacks: &'a Link,
    _marker: PhantomData<fn(Marker)>,
}

trait VisitCallbackState<State> {
    fn callback_state(&self) -> &State;
    fn callback_state_mut(&mut self) -> &mut State;
}

impl<State, Link, Marker> VisitCallbackState<State> for VisitCallbacks<State, Link, Marker> {
    fn callback_state(&self) -> &State {
        &self.state
    }

    fn callback_state_mut(&mut self) -> &mut State {
        &mut self.state
    }
}

impl<Link, Marker> VisitCallbackState<()> for DirectVisitCallbacks<'_, Link, Marker> {
    fn callback_state(&self) -> &() {
        &self.state
    }

    fn callback_state_mut(&mut self) -> &mut () {
        &mut self.state
    }
}

#[doc(hidden)]
pub struct ByVisitCallbacks<Marker>(PhantomData<fn(Marker)>);

impl<Link, Marker> IntoVisitor<ByVisitCallbacks<Marker>> for Link
where
    Link: VisitChainLink<(), Marker>,
{
    fn visit_root(self, root: TVMFFIAny) -> Result<Option<VisitInterrupt>> {
        let callbacks = self;
        let mut visitor = DirectVisitCallbacks::<Link, Marker> {
            state: (),
            callbacks: &callbacks,
            _marker: PhantomData,
        };
        finish(run_structural_visitor(
            root,
            &mut visitor,
            user_runtime_vtable::<DirectVisitCallbacks<Link, Marker>>(),
        ))
    }
}

// Keep the generated walk-dispatch paths stable.
pub use super::dispatch::{ByWalkDispatch, DispatchWalker, WalkDispatch};

/// Conversion into the walker argument of [`structural_walk`].
///
/// Accepts a mutable [`WalkDispatch`], a typed callback, or a nested callback
/// tuple. `Marker` distinguishes the supported callback shapes.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a supported `structural_walk` walker",
    note = "accepted walkers: `&mut V` where `V: WalkDispatch`; a closure over `&VisitValue`, \
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

/// Adapter for a catch-all walk callback.
#[doc(hidden)]
pub struct ClosureWalker<F> {
    callback: F,
}

impl<F, O> NativeVisit for ClosureWalker<F>
where
    F: FnMut(&VisitValue) -> O,
    O: IntoWalkResult,
{
    fn visit(&mut self, value: &VisitValue, _def_region_kind: DefRegionKind) -> Result<WalkResult> {
        (self.callback)(value).into_walk_result()
    }
}

#[doc(hidden)]
pub enum ByValueClosure {}

impl<F, O> IntoWalker<ByValueClosure> for F
where
    F: FnMut(&VisitValue) -> O,
    O: IntoWalkResult,
{
    type Walker = ClosureWalker<F>;
    fn into_walker(self) -> Self::Walker {
        ClosureWalker { callback: self }
    }
}

/// Catch-all walk adapter that also supplies the definition-region state.
#[doc(hidden)]
pub struct ClosureKindWalker<F> {
    callback: F,
}

impl<F, O> NativeVisit for ClosureKindWalker<F>
where
    F: FnMut(&VisitValue, DefRegionKind) -> O,
    O: IntoWalkResult,
{
    fn visit(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        (self.callback)(value, def_region_kind).into_walk_result()
    }
}

#[doc(hidden)]
pub enum ByValueKindClosure {}

impl<F, O> IntoWalker<ByValueKindClosure> for F
where
    F: FnMut(&VisitValue, DefRegionKind) -> O,
    O: IntoWalkResult,
{
    type Walker = ClosureKindWalker<F>;
    fn into_walker(self) -> Self::Walker {
        ClosureKindWalker { callback: self }
    }
}

/// One link in a first-match [`structural_walk`] callback chain.
///
/// Supported links are typed values, borrowed object nodes, `&VisitValue`,
/// and mutable [`WalkDispatch`] implementations, optionally followed by
/// [`DefRegionKind`]. Tuples hold up to 12 links and may be nested.
pub trait WalkChainLink<Marker>: sealed::SealedLink<Marker> {
    /// Run this link if `value` matches its argument type; `None` hands the
    /// value to the next link.
    #[doc(hidden)]
    fn try_call(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult>;
}

mod sealed {
    use super::{DefRegionKind, IntoWalkResult, ObjectCore, VisitValue, WalkDispatch};

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
        F: for<'a> FnMut(&'a VisitValue) -> O,
        O: IntoWalkResult,
    {
    }
    impl<F, O> SealedLink<super::ByCatchAllKindLink> for F
    where
        F: for<'a> FnMut(&'a VisitValue, DefRegionKind) -> O,
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
        value: &VisitValue,
        _def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
        value
            .cast::<T>()
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
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
        value
            .cast::<T>()
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
        value: &VisitValue,
        _def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
        value
            .as_node::<N>()
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
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
        value
            .as_node::<N>()
            .map(|node| self(node, def_region_kind).into_walk_result())
    }
}

#[doc(hidden)]
pub enum ByCatchAllLink {}

impl<F, O> WalkChainLink<ByCatchAllLink> for F
where
    F: for<'a> FnMut(&'a VisitValue) -> O,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        _def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
        Some(self(value).into_walk_result())
    }
}

#[doc(hidden)]
pub enum ByCatchAllKindLink {}

impl<F, O> WalkChainLink<ByCatchAllKindLink> for F
where
    F: for<'a> FnMut(&'a VisitValue, DefRegionKind) -> O,
    O: IntoWalkResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
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
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
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

impl<Link, Marker> NativeVisit for ChainWalker<Link, Marker>
where
    Link: WalkChainLink<Marker>,
{
    #[inline]
    fn visit(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        self.link
            .try_call(value, def_region_kind)
            .unwrap_or(Ok(WalkResult::Advance))
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
                value: &VisitValue,
                def_region_kind: DefRegionKind,
            ) -> Option<WalkCallbackResult> {
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

/// A visitor that controls its own recursion.
///
/// Implementations descend with [`Self::visit_child`] or
/// [`Self::default_visit_children`]. `#[dispatch(visit)]` generates this trait
/// from typed `visit_*` methods.
pub trait StructuralVisitor: Sized {
    /// Visit one value under the definition-region state active at it.
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>>;

    /// Visit `child` under `def_region_kind`.
    #[inline]
    fn visit_child<T>(
        &mut self,
        child: &T,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let raw = raw_of(AnyView::from(child));
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(None);
        }
        let active = active_structural_visitor()?;
        let context = std::ptr::from_mut(self).cast::<c_void>();
        finish(with_current_visitor_context(active, context, || {
            call_visitor(active, raw, def_region_kind)
        }))
    }

    /// Visit `value`'s children with the default structural rules.
    #[inline]
    fn default_visit_children(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        let raw = value.raw();
        let context = std::ptr::from_mut(&mut *self).cast::<c_void>();
        let result = visit_children_raw(
            raw,
            &mut UserChildren { visitor: self },
            context,
            def_region_kind,
        )
        .map_err(|halt| with_value_context(halt, raw));
        finish(result)
    }
}

fn try_visit_callbacks<State, Link, Marker>(
    driver: &mut impl VisitContextDriver<State>,
    callback_ptr: *const Link,
    value: &VisitValue,
    def_region_kind: DefRegionKind,
) -> Result<Option<VisitInterrupt>>
where
    Link: VisitChainLink<State, Marker>,
{
    let mut visitor = VisitContext {
        driver,
        current: VisitValue::from_raw(value.raw()),
        def_region_kind,
        _not_send_sync: PhantomData,
    };
    // SAFETY: The owning `Rc` or the direct callback's stack slot remains live
    // and is never modified through the driver during recursive reentry.
    match unsafe { (&*callback_ptr).try_visit(value, &mut visitor) } {
        Some(outcome) => outcome,
        None => visitor.visit_children(),
    }
}

impl<State, Link, Marker> StructuralVisitor for VisitCallbacks<State, Link, Marker>
where
    Link: VisitChainLink<State, Marker>,
{
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        let callback_ptr = Rc::as_ptr(&self.callbacks);
        try_visit_callbacks::<State, Link, Marker>(self, callback_ptr, value, def_region_kind)
    }
}

impl<Link, Marker> StructuralVisitor for DirectVisitCallbacks<'_, Link, Marker>
where
    Link: VisitChainLink<(), Marker>,
{
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        let callback_ptr = std::ptr::from_ref(self.callbacks);
        try_visit_callbacks::<(), Link, Marker>(self, callback_ptr, value, def_region_kind)
    }
}

impl<State, Driver> VisitContextDriver<State> for Driver
where
    Driver: StructuralVisitor + VisitCallbackState<State>,
{
    fn state(&self) -> &State {
        self.callback_state()
    }

    fn state_mut(&mut self) -> &mut State {
        self.callback_state_mut()
    }

    fn visit_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(None);
        }
        let active = active_structural_visitor()?;
        let context = std::ptr::from_mut(self).cast::<c_void>();
        finish(with_current_visitor_context(active, context, || {
            call_visitor(active, raw, def_region_kind)
        }))
    }

    fn visit_children_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        <Self as StructuralVisitor>::default_visit_children(
            self,
            &VisitValue::from_raw(raw),
            def_region_kind,
        )
    }
}

/// Internal callback protocol used by [`IntoWalker`].
#[doc(hidden)]
pub trait NativeVisit {
    fn visit(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult>;
}

/// Action applied to each child found by the shared traversal.
trait ChildVisit {
    fn visit_child(&mut self, child: TVMFFIAny, def_region_kind: DefRegionKind) -> NativeResult;
}

struct WalkChildren<'a, V, const PRE_ORDER: bool> {
    visitor: &'a mut V,
}

impl<V: NativeVisit, const PRE_ORDER: bool> ChildVisit for WalkChildren<'_, V, PRE_ORDER> {
    fn visit_child(&mut self, child: TVMFFIAny, def_region_kind: DefRegionKind) -> NativeResult {
        visit_raw::<V, PRE_ORDER>(child, self.visitor, def_region_kind)
    }
}

struct UserChildren<'a, V> {
    visitor: &'a mut V,
}

impl<V: StructuralVisitor> ChildVisit for UserChildren<'_, V> {
    #[inline]
    fn visit_child(&mut self, child: TVMFFIAny, def_region_kind: DefRegionKind) -> NativeResult {
        if child.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(());
        }
        match self
            .visitor
            .visit(&VisitValue::from_raw(child), def_region_kind)
        {
            Ok(None) => Ok(()),
            Ok(Some(interrupt)) => Err(NativeHalt::Interrupt(interrupt.value)),
            Err(error) => Err(NativeHalt::Error(error)),
        }
    }
}

// Every registered container child re-enters this hot path.
#[inline(always)]
fn visit_raw<V: NativeVisit, const PRE_ORDER: bool>(
    value: TVMFFIAny,
    visitor: &mut V,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }

    let visit_value = VisitValue::from_raw(value);
    if PRE_ORDER {
        match visitor.visit(&visit_value, def_region_kind) {
            Ok(WalkResult::Advance) => {}
            Ok(WalkResult::Skip) => return Ok(()),
            Ok(WalkResult::Interrupt) => return Err(NativeHalt::Interrupt(Any::new())),
            Ok(WalkResult::InterruptWith(payload)) => return Err(NativeHalt::Interrupt(payload)),
            Err(error) => return Err(with_value_context(error.into(), value)),
        }
    }

    let context = std::ptr::from_mut(&mut *visitor).cast::<c_void>();
    let children = &mut WalkChildren::<V, PRE_ORDER> {
        visitor: &mut *visitor,
    };
    if let Err(halt) = visit_children_raw(value, children, context, def_region_kind) {
        return Err(with_value_context(halt, value));
    }

    if PRE_ORDER {
        Ok(())
    } else {
        match visitor.visit(&visit_value, def_region_kind) {
            Ok(WalkResult::Interrupt) => Err(NativeHalt::Interrupt(Any::new())),
            Ok(WalkResult::InterruptWith(payload)) => Err(NativeHalt::Interrupt(payload)),
            Ok(WalkResult::Advance | WalkResult::Skip) => Ok(()),
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
) -> NativeResult {
    if let Some(attr) = structural_visit_column().and_then(|column| column.get(value.type_index)) {
        if attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
            let active = active_structural_visitor()?;
            return with_current_visitor_context(active, driver_context, || {
                call_structural_visit_hook(active, value, def_region_kind, attr)
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

    let borrowed = raw_of_owned(&child);
    let child_region = field_def_region(field, inherited_region);
    visitor
        .visit_child(borrowed, child_region)
        .map_err(|halt| with_error_context(halt, &format!("field `{}`", field.name.as_str())))
}

type StructuralVisitorHandle = *mut RuntimeStructuralVisitorObj;
type FStructuralVisit =
    unsafe extern "C" fn(StructuralVisitorHandle, AnyView<'static>) -> TVMFFIAny;

/// Rust mirror of the C++ `StructuralVisitorVTable` ABI.
#[repr(C)]
struct StructuralVisitorVTable {
    visit: FStructuralVisit,
}

/// Rust visitor object with the C++ `StructuralVisitorObj` prefix.
#[repr(C)]
struct RuntimeStructuralVisitorObj {
    base: Object,
    vtable: *const StructuralVisitorVTable,
    def_region_mode: i32,
    // The live context remains in traversal-local TLS.
    context_identity: *mut c_void,
    owner_thread: std::thread::ThreadId,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

/// Rust layout used to create and read the ABI `ffi.VisitInterrupt` object.
#[repr(C)]
struct RuntimeVisitInterruptObj {
    base: Object,
    value: Any,
}

const _: () = {
    assert!(
        std::mem::offset_of!(RuntimeStructuralVisitorObj, vtable)
            == std::mem::size_of::<TVMFFIObject>()
    );
    assert!(
        std::mem::offset_of!(RuntimeStructuralVisitorObj, def_region_mode)
            == std::mem::size_of::<TVMFFIObject>() + std::mem::size_of::<*const c_void>()
    );
    assert!(
        std::mem::offset_of!(RuntimeVisitInterruptObj, value)
            == std::mem::size_of::<TVMFFIObject>()
    );
};

// SAFETY: the `repr(C)` prefix and assertions above match
// `StructuralVisitorObj`; the runtime type is registered by the C++ extra.
unsafe impl ObjectCore for RuntimeStructuralVisitorObj {
    const TYPE_KEY: &'static str = "ffi.StructuralVisitor";
    const TYPE_PARENT_KEY: Option<&'static str> = Some(Object::TYPE_KEY);
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH + 1;

    fn type_index() -> i32 {
        static TYPE_INDEX: LazyLock<i32> = LazyLock::new(|| unsafe {
            let key = TVMFFIByteArray::from_str(RuntimeStructuralVisitorObj::TYPE_KEY);
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

    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.base)
    }
}

// SAFETY: `VisitInterruptObj` is final and consists of `Object` followed by
// one `Any`, exactly matching `RuntimeVisitInterruptObj`.
unsafe impl ObjectCore for RuntimeVisitInterruptObj {
    const TYPE_KEY: &'static str = "ffi.VisitInterrupt";
    const TYPE_PARENT_KEY: Option<&'static str> = Some(Object::TYPE_KEY);
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH + 1;
    const TYPE_FINAL: bool = true;

    fn type_index() -> i32 {
        TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32
    }

    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.base)
    }
}

// Use a direct ABI entry for each concrete visitor type.
fn walk_runtime_vtable<V: NativeVisit, const PRE_ORDER: bool>() -> &'static StructuralVisitorVTable
{
    &StructuralVisitorVTable {
        visit: rust_vtable_walk::<V, PRE_ORDER>,
    }
}

fn user_runtime_vtable<V: StructuralVisitor>() -> &'static StructuralVisitorVTable {
    &StructuralVisitorVTable {
        visit: rust_vtable_user::<V>,
    }
}

struct RuntimeContextGuard {
    active: *mut ActiveStructuralVisitor,
    context: *mut c_void,
}

impl Drop for RuntimeContextGuard {
    fn drop(&mut self) {
        // SAFETY: `active` points to the traversal-local state installed in
        // TLS, which outlives every callback guard created during that run.
        unsafe { (*self.active).context = self.context };
    }
}

/// Traversal-local state exposed through TLS on the owner thread.
struct ActiveStructuralVisitor {
    visitor: StructuralVisitorHandle,
    context: *mut c_void,
    context_identity: *mut c_void,
}

/// Take the Rust callback context while one vtable call is active.
///
/// # Safety
///
/// `visitor` must be null or point to a live [`RuntimeStructuralVisitorObj`].
#[inline(always)]
unsafe fn take_runtime_context(visitor: StructuralVisitorHandle) -> Result<RuntimeContextGuard> {
    let active = active_structural_visitor_state(visitor)
        .ok_or_else(|| inactive_structural_visitor_error(visitor, "callback"))?;
    let context = (*active).context;
    if context.is_null() {
        return Err(runtime_error(
            "structural visitor may only be called by its active registered hook",
        ));
    }
    (*active).context = std::ptr::null_mut();
    Ok(RuntimeContextGuard { active, context })
}

unsafe extern "C" fn rust_vtable_walk<V: NativeVisit, const PRE_ORDER: bool>(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    rust_vtable_visit_impl(visitor, value, |context, raw, kind| {
        runtime_walk::<V, PRE_ORDER>(context, raw, kind)
    })
}

unsafe extern "C" fn rust_vtable_user<V: StructuralVisitor>(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    rust_vtable_visit_impl(visitor, value, |context, raw, kind| {
        runtime_user_visit::<V>(context, raw, kind)
    })
}

#[inline(always)]
unsafe fn rust_vtable_visit_impl(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
    callback: impl FnOnce(*mut c_void, TVMFFIAny, DefRegionKind) -> NativeResult,
) -> TVMFFIAny {
    let context_guard = match take_runtime_context(visitor) {
        Ok(guard) => guard,
        Err(error) => return native_result_into_raw(Err(NativeHalt::Error(error))),
    };
    let context = context_guard.context;
    let raw = *value.as_raw_ffi_any();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let kind = def_region_from_raw((*visitor).def_region_mode)?;
        callback(context, raw, kind)
    }));
    match outcome {
        Ok(result) => native_result_into_raw(result),
        Err(payload) => {
            (*visitor).panic = Some(payload);
            native_result_into_raw(Err(NativeHalt::Error(runtime_error(
                "panic in structural visitor callback",
            ))))
        }
    }
}

thread_local! {
    static ACTIVE_STRUCTURAL_VISITOR: Cell<*mut ActiveStructuralVisitor> = const {
        Cell::new(std::ptr::null_mut())
    };
}

fn with_active_structural_visitor<T>(
    active_state: &mut ActiveStructuralVisitor,
    callback: impl FnOnce() -> T,
) -> T {
    ACTIVE_STRUCTURAL_VISITOR.with(|active| {
        let previous = active.replace(std::ptr::from_mut(active_state));
        struct Restore<'a> {
            active: &'a Cell<*mut ActiveStructuralVisitor>,
            previous: *mut ActiveStructuralVisitor,
        }
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.active.set(self.previous);
            }
        }
        let _restore = Restore { active, previous };
        callback()
    })
}

fn active_structural_visitor() -> Result<StructuralVisitorHandle> {
    ACTIVE_STRUCTURAL_VISITOR.with(|active| {
        let state = active.get();
        if state.is_null() {
            Err(runtime_error(
                "structural visitor helper called outside structural_visit or structural_walk",
            ))
        } else {
            Ok(unsafe { (*state).visitor })
        }
    })
}

#[inline(always)]
fn active_structural_visitor_state(
    handle: StructuralVisitorHandle,
) -> Option<*mut ActiveStructuralVisitor> {
    ACTIVE_STRUCTURAL_VISITOR.with(|active| {
        let state = active.get();
        if state.is_null() || unsafe { (*state).visitor != handle } {
            None
        } else {
            Some(state)
        }
    })
}

#[cold]
fn inactive_structural_visitor_error(visitor: StructuralVisitorHandle, operation: &str) -> Error {
    if visitor.is_null() {
        return runtime_error("null active structural visitor");
    }
    // This branch is outside the hot path. The immutable owner id lets us
    // reject a foreign-thread call before reading context fields that the
    // owner thread may be updating.
    unsafe {
        if (*visitor).owner_thread != std::thread::current().id() {
            return runtime_error(&format!(
                "structural visitor {operation} invoked from a different thread"
            ));
        }
        if (*visitor).context_identity.is_null() {
            runtime_error("structural visitor was retained after its active call")
        } else {
            runtime_error(&format!(
                "structural visitor {operation} may only be used by its active registered hook"
            ))
        }
    }
}

/// Expose the current Rust visitor only while a registered hook may call its
/// vtable. The callback hides it again before returning to Rust user code.
fn with_current_visitor_context(
    visitor: StructuralVisitorHandle,
    context: *mut c_void,
    callback: impl FnOnce() -> NativeResult,
) -> NativeResult {
    let active = active_structural_visitor_state(visitor)
        .ok_or_else(|| inactive_structural_visitor_error(visitor, "helper"))?;
    unsafe {
        if (*active).context_identity != context {
            return Err(
                runtime_error("structural visitor helper called on a non-active visitor").into(),
            );
        }
        if !(*active).context.is_null() {
            return Err(runtime_error("structural visitor context is already exposed").into());
        }

        (*active).context = context;
        struct HideContext {
            active: *mut ActiveStructuralVisitor,
        }
        impl Drop for HideContext {
            fn drop(&mut self) {
                unsafe { (*self.active).context = std::ptr::null_mut() };
            }
        }
        let _hide = HideContext { active };
        callback()
    }
}

#[inline(always)]
unsafe fn runtime_walk<V: NativeVisit, const PRE_ORDER: bool>(
    context: *mut c_void,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        let visitor = &mut *context.cast::<V>();
        if PRE_ORDER {
            match visitor.visit(&VisitValue::from_raw(raw), def_region_kind) {
                Ok(WalkResult::Advance) => {}
                Ok(WalkResult::Skip) => return Ok(()),
                Ok(WalkResult::Interrupt) => return Err(NativeHalt::Interrupt(Any::new())),
                Ok(WalkResult::InterruptWith(payload)) => {
                    return Err(NativeHalt::Interrupt(payload));
                }
                Err(error) => return Err(with_value_context(error.into(), raw)),
            }
            if !has_registered_visit_hook(raw.type_index) {
                return Ok(());
            }
            let children = &mut WalkChildren::<V, PRE_ORDER> { visitor };
            return visit_children_raw(raw, children, context, def_region_kind)
                .map_err(|halt| with_value_context(halt, raw));
        }
        // Post-order inline values have no children unless their type
        // registered a visit hook. Handle the common case directly here.
        if !has_registered_visit_hook(raw.type_index) {
            return match visitor.visit(&VisitValue::from_raw(raw), def_region_kind) {
                Ok(WalkResult::Advance | WalkResult::Skip) => Ok(()),
                Ok(WalkResult::Interrupt) => Err(NativeHalt::Interrupt(Any::new())),
                Ok(WalkResult::InterruptWith(payload)) => Err(NativeHalt::Interrupt(payload)),
                Err(error) => Err(with_value_context(error.into(), raw)),
            };
        }
    }
    visit_raw::<V, PRE_ORDER>(raw, &mut *context.cast::<V>(), def_region_kind)
}

#[inline(always)]
unsafe fn runtime_user_visit<V: StructuralVisitor>(
    context: *mut c_void,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }
    match (&mut *context.cast::<V>()).visit(&VisitValue::from_raw(raw), def_region_kind) {
        Ok(None) => Ok(()),
        Ok(Some(interrupt)) => Err(NativeHalt::Interrupt(interrupt.value)),
        Err(error) => Err(NativeHalt::Error(error)),
    }
}

fn run_structural_visitor<D>(
    root: TVMFFIAny,
    driver: &mut D,
    vtable: &'static StructuralVisitorVTable,
) -> NativeResult {
    let context = std::ptr::from_mut(driver).cast::<c_void>();
    let mut active = ObjectArc::new(RuntimeStructuralVisitorObj {
        base: Object::new(),
        vtable,
        def_region_mode: DefRegionKind::None as i32,
        context_identity: context,
        owner_thread: std::thread::current().id(),
        panic: None,
    });
    let handle = unsafe { ObjectArc::as_raw_mut(&mut active) };
    let mut active_state = ActiveStructuralVisitor {
        visitor: handle,
        context,
        context_identity: context,
    };
    // Keep all borrow-sensitive Rust state in this traversal's stack frame.
    // Nested callbacks validate and temporarily take it through TLS without
    // repeatedly touching the heap-allocated FFI object.
    let result = with_active_structural_visitor(&mut active_state, || {
        call_visitor(handle, root, DefRegionKind::None)
    });
    unsafe {
        (*handle).context_identity = std::ptr::null_mut();
    }
    let panic = unsafe { (*handle).panic.take() };
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
    if visitor.is_null() {
        return Err(runtime_error("no active structural visitor").into());
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

fn with_visitor_def_region<T>(
    visitor: StructuralVisitorHandle,
    kind: DefRegionKind,
    callback: impl FnOnce() -> T,
) -> T {
    unsafe {
        let previous = (*visitor).def_region_mode;
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
        x if x == DefRegionKind::Recursive as i32 => Ok(DefRegionKind::Recursive),
        x if x == DefRegionKind::NonRecursive as i32 => Ok(DefRegionKind::NonRecursive),
        _ => Err(runtime_error("invalid structural definition-region kind")),
    }
}

fn with_value_context(halt: NativeHalt, value: TVMFFIAny) -> NativeHalt {
    if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        halt
    } else {
        with_error_context(halt, &format!("object `{}`", type_key_of(value.type_index)))
    }
}

/// Visit `root` with a [`StructuralVisitor`] or typed callback chain.
///
/// A matching callback owns recursion; unmatched values use default child
/// traversal. Use [`VisitCallbacks`] to attach mutable state to the chain.
pub fn structural_visit<R, M>(
    root: &R,
    visitor: impl IntoVisitor<M>,
) -> Result<Option<VisitInterrupt>>
where
    for<'x> AnyView<'x>: From<&'x R>,
{
    visitor.visit_root(raw_of(AnyView::from(root)))
}

/// Walk `root` with an observer, the Rust analog of C++
/// `StructuralWalk<order>(root, callbacks...)`.
///
/// `walker` is anything implementing [`IntoWalker`]: a `&mut` reference to a
/// stateful [`WalkDispatch`] walker (`#[dispatch(walk)]`), a bare closure
/// in any [`WalkChainLink`] shape (catch-all `&VisitValue`, typed, or node,
/// with an optional trailing [`DefRegionKind`]), or a tuple of such
/// callbacks tried in order — the C++ callback overloads and variadic
/// chain. The walker owns recursion: the handler runs once per value,
/// before or after the value's children according to `order`, and steers
/// traversal through the returned [`WalkResult`].
pub fn structural_walk<R, M, H>(
    root: &R,
    walker: H,
    order: WalkOrder,
) -> Result<Option<VisitInterrupt>>
where
    H: IntoWalker<M>,
    for<'x> AnyView<'x>: From<&'x R>,
{
    let mut dispatch = walker.into_walker();
    let root = raw_of(AnyView::from(root));
    let result = match order {
        WalkOrder::PreOrder => run_structural_visitor(
            root,
            &mut dispatch,
            walk_runtime_vtable::<H::Walker, true>(),
        ),
        WalkOrder::PostOrder => run_structural_visitor(
            root,
            &mut dispatch,
            walk_runtime_vtable::<H::Walker, false>(),
        ),
    };
    finish(result)
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
    if field.flags & FLAG_SEQ_HASH_DEF_NON_RECURSIVE != 0 {
        DefRegionKind::NonRecursive
    } else if field.flags & FLAG_SEQ_HASH_DEF_RECURSIVE != 0 {
        DefRegionKind::Recursive
    } else {
        inherited
    }
}

/// A non-recursive definition applies to a FreeVar value itself, but not to
/// the FreeVar's own reflected children: nested free vars there must resolve
/// against an outer binding instead of rebinding. Mirrors C++
/// `VisitReflectedFieldsExpected`.
#[inline]
pub(crate) fn free_var_child_region(
    inherited: DefRegionKind,
    structural_eq_hash_kind: i32,
) -> DefRegionKind {
    if inherited == DefRegionKind::NonRecursive
        && structural_eq_hash_kind == TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar as i32
    {
        DefRegionKind::None
    } else {
        inherited
    }
}

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

#[derive(Clone, Copy)]
pub(crate) struct TypeAttrColumn(NonNull<TVMFFITypeAttrColumn>);

impl TypeAttrColumn {
    pub(crate) unsafe fn from_non_null(pointer: NonNull<TVMFFITypeAttrColumn>) -> Self {
        Self(pointer)
    }

    pub(crate) fn as_ptr(self) -> *mut TVMFFITypeAttrColumn {
        self.0.as_ptr()
    }

    /// Copy one borrowed cell; ownership remains with the registry.
    pub(crate) fn get(self, type_index: i32) -> Option<TVMFFIAny> {
        unsafe {
            let column = self.0.as_ref();
            let index = type_index - column.begin_index;
            if index < 0 || index >= column.size || column.data.is_null() {
                None
            } else {
                Some(*column.data.offset(index as isize))
            }
        }
    }
}

pub(crate) fn type_attr_column(attr_name: &str) -> Option<TypeAttrColumn> {
    unsafe {
        let attr_name = TVMFFIByteArray::from_str(attr_name);
        NonNull::new(TVMFFIGetTypeAttrColumn(&attr_name).cast_mut()).map(TypeAttrColumn)
    }
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
        return Some(TypeAttrColumn(unsafe { NonNull::new_unchecked(pointer) }));
    }
    initialize_structural_visit_column()
}

#[inline]
fn has_registered_visit_hook(type_index: i32) -> bool {
    structural_visit_column()
        .and_then(|column| column.get(type_index))
        .is_some_and(|attr| attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32)
}

#[cold]
#[inline(never)]
fn initialize_structural_visit_column() -> Option<TypeAttrColumn> {
    let column = type_attr_column(STRUCTURAL_VISIT_ATTR)?;
    STRUCTURAL_VISIT_COLUMN.store(column.0.as_ptr() as usize, Ordering::Relaxed);
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

#[inline]
fn raw_of(view: AnyView<'_>) -> TVMFFIAny {
    *view.as_raw_ffi_any()
}

#[inline]
fn raw_of_owned(any: &Any) -> TVMFFIAny {
    *any.as_raw_ffi_any()
}
