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
//!   [`StructuralVisitor::visit_child`].
//! * [`VisitDispatch`] + [`structural_walk`] — observer callbacks, like C++
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
    TVMFFIAny, TVMFFIAnyViewToOwnedAny, TVMFFIByteArray, TVMFFIDefRegionKind, TVMFFIFieldInfo,
    TVMFFIGetTypeAttrColumn, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFISEqHashKind,
    TVMFFITypeAttrColumn, TVMFFITypeIndex, TVMFFITypeKeyToIndex,
};

use super::structural_common::impl_callback_chain_tuple_arities;

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
pub trait IntoVisitResult {
    fn into_visit_result(self) -> Result<WalkResult>;
}

impl IntoVisitResult for WalkResult {
    fn into_visit_result(self) -> Result<WalkResult> {
        Ok(self)
    }
}

impl IntoVisitResult for Result<WalkResult> {
    fn into_visit_result(self) -> Result<WalkResult> {
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

/// Fallible result returned by generated typed dispatch.
#[doc(hidden)]
pub type VisitResult = Result<WalkResult>;

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

// The typed-dispatch layer (`VisitDispatch`, its walker adapter, and the
// `&mut V` IntoWalker form) lives in `super::dispatch`; re-exported here so
// the module's public paths — which `#[dispatch(visit)]`-generated code
// names — stay stable.
pub use super::dispatch::{ByDispatch, DispatchVisitor, VisitDispatch};

/// Conversion into the walker argument of [`structural_walk`].
///
/// The `Marker` parameter lets one entry point accept several handler
/// shapes without overlapping implementations — the Rust equivalent of the
/// C++ `StructuralWalk` callback overload set:
///
/// * `&mut V` where `V: VisitDispatch` — a stateful typed visitor
///   (`#[dispatch(visit)]` or hand-written).
/// * A bare closure in any [`WalkChainLink`] shape — catch-all
///   `FnMut(&VisitValue)`, typed `FnMut(T)`, node `FnMut(&N)`, each with an
///   optional trailing [`DefRegionKind`] argument — the analog of a single
///   C++ callback. Values a typed closure does not match advance normally.
/// * A tuple of typed links `(link1, link2, ...)`, up to 8 — the analog of
///   the C++ variadic callback chain; see [`WalkChainLink`] for the
///   accepted link shapes. Larger handler sets belong in one
///   `#[dispatch(visit)]` visitor, which itself splices into a tuple as a
///   single link.
///
/// Closure arguments usually need explicit type annotations
/// (`|value: &VisitValue| ...`) for the marker to be inferred.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a supported `structural_walk` walker",
    note = "accepted walkers: `&mut V` where `V: VisitDispatch`; a closure over `&VisitValue`, \
            an FFI value type `T`, or `&N` of an object node type (`N: ObjectCore`, e.g. \
            `&Object`), optionally with a trailing `DefRegionKind` argument; or a tuple of \
            up to 8 such links",
    note = "closure arguments need explicit type annotations; ObjectRef wrappers like `String` \
            or `Array<T>` are FFI value types — take them by value, not by reference"
)]
pub trait IntoWalker<Marker> {
    #[doc(hidden)]
    type Walker: NativeVisit;
    #[doc(hidden)]
    fn into_walker(self, order: WalkOrder) -> Self::Walker;
}

/// Runs a catch-all closure at the phase selected by `order` — the closure
/// analog of `DispatchVisitor`, without the `Option<VisitResult>`
/// no-handler-matched layer a dispatch chain needs. (Routing closures
/// through `DispatchVisitor` instead measures ~10-20% slower on the bare
/// closure walk: the wrapped-and-unwrapped `Option<Result<..>>` does not
/// fold away.)
#[doc(hidden)]
pub struct ClosureWalker<F> {
    callback: F,
    order: WalkOrder,
}

impl<F, O> NativeVisit for ClosureWalker<F>
where
    F: FnMut(&VisitValue) -> O,
    O: IntoVisitResult,
{
    fn enter(&mut self, value: &VisitValue, _def_region_kind: DefRegionKind) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => (self.callback)(value).into_visit_result(),
            WalkOrder::PostOrder => Ok(WalkResult::Advance),
        }
    }

    fn exit(&mut self, value: &VisitValue, _def_region_kind: DefRegionKind) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => Ok(WalkResult::Advance),
            WalkOrder::PostOrder => (self.callback)(value).into_visit_result(),
        }
    }
}

#[doc(hidden)]
pub enum ByValueClosure {}

impl<F, O> IntoWalker<ByValueClosure> for F
where
    F: FnMut(&VisitValue) -> O,
    O: IntoVisitResult,
{
    type Walker = ClosureWalker<F>;
    fn into_walker(self, order: WalkOrder) -> Self::Walker {
        ClosureWalker {
            callback: self,
            order,
        }
    }
}

/// `ClosureWalker` variant whose callback also receives the definition-region
/// state.
#[doc(hidden)]
pub struct ClosureKindWalker<F> {
    callback: F,
    order: WalkOrder,
}

impl<F, O> NativeVisit for ClosureKindWalker<F>
where
    F: FnMut(&VisitValue, DefRegionKind) -> O,
    O: IntoVisitResult,
{
    fn enter(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => (self.callback)(value, def_region_kind).into_visit_result(),
            WalkOrder::PostOrder => Ok(WalkResult::Advance),
        }
    }

    fn exit(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => Ok(WalkResult::Advance),
            WalkOrder::PostOrder => (self.callback)(value, def_region_kind).into_visit_result(),
        }
    }
}

#[doc(hidden)]
pub enum ByValueKindClosure {}

impl<F, O> IntoWalker<ByValueKindClosure> for F
where
    F: FnMut(&VisitValue, DefRegionKind) -> O,
    O: IntoVisitResult,
{
    type Walker = ClosureKindWalker<F>;
    fn into_walker(self, order: WalkOrder) -> Self::Walker {
        ClosureKindWalker {
            callback: self,
            order,
        }
    }
}

/// One typed link of a tuple walker — a single callback of the C++ variadic
/// `StructuralWalk(root, callbacks...)` chain.
///
/// A tuple of up to 8 links passed to [`structural_walk`] is tried in order
/// and the first link whose argument type matches the value runs, exactly
/// like the C++ callback chain. (Python's `structural_walk` differs on one
/// point: it keeps `callbacks` and `with_def_region_kind` as two separately
/// ordered groups, trying every plain entry before any kind-taking entry,
/// so a mixed Rust tuple's single interleaved order has no exact Python
/// equivalent.) Accepted link shapes mirror `#[dispatch(visit)]` handlers:
///
/// * `FnMut(T) -> impl IntoVisitResult` for an FFI-convertible `T` — value
///   cast via [`VisitValue::cast`], which matches on the FFI type tag: a
///   numeric link claims every `Int` (or `Float`) regardless of width and
///   converts with `as` semantics, so prefer `i64`/`f64` links unless a
///   deliberate narrowing is wanted.
/// * `FnMut(&N) -> impl IntoVisitResult` for an object node `N` —
///   refcount-free subtype check via [`VisitValue::as_node`].
/// * `FnMut(&VisitValue) -> impl IntoVisitResult` — catch-all.
/// * `&mut V` where `V: VisitDispatch` — splice a typed visitor into the
///   chain; it claims every value one of its handlers matches.
///
/// Links after one that matches every value never run: place a catch-all
/// closure — or a spliced visitor whose own chain ends in a `&VisitValue`
/// handler — last. Unlike the in-visitor ordering check, misordering a
/// tuple is not a compile error.
///
/// Every closure shape may declare a trailing [`DefRegionKind`] argument,
/// and a single typed closure may also be passed to [`structural_walk`]
/// bare, without the tuple. Closure arguments need explicit type
/// annotations for the marker to be inferred. Borrow rules apply per link,
/// so state shared across links goes through a `Cell`/`RefCell` — or in a
/// single `#[dispatch(visit)]` visitor, which shares `&mut self` between
/// its handlers.
///
/// This trait is sealed: the link shapes above are the complete set, and
/// the dispatch method is an internal detail.
pub trait WalkChainLink<Marker>: sealed::SealedLink<Marker> {
    /// Run this link if `value` matches its argument type; `None` hands the
    /// value to the next link.
    #[doc(hidden)]
    fn try_call(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult>;
}

mod sealed {
    use super::{DefRegionKind, IntoVisitResult, ObjectCore, VisitDispatch, VisitValue};

    /// Seal for [`super::WalkChainLink`]: one impl per accepted link shape,
    /// mirroring the `WalkChainLink` impl set exactly.
    pub trait SealedLink<Marker> {}

    impl<F, T, O> SealedLink<super::ByOwnedLink<T>> for F
    where
        F: FnMut(T) -> O,
        O: IntoVisitResult,
    {
    }
    impl<F, T, O> SealedLink<super::ByOwnedKindLink<T>> for F
    where
        F: FnMut(T, DefRegionKind) -> O,
        O: IntoVisitResult,
    {
    }
    impl<F, N: ObjectCore, O> SealedLink<super::ByNodeLink<N>> for F
    where
        F: for<'a> FnMut(&'a N) -> O,
        O: IntoVisitResult,
    {
    }
    impl<F, N: ObjectCore, O> SealedLink<super::ByNodeKindLink<N>> for F
    where
        F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
        O: IntoVisitResult,
    {
    }
    impl<F, O> SealedLink<super::ByCatchAllLink> for F
    where
        F: for<'a> FnMut(&'a VisitValue) -> O,
        O: IntoVisitResult,
    {
    }
    impl<F, O> SealedLink<super::ByCatchAllKindLink> for F
    where
        F: for<'a> FnMut(&'a VisitValue, DefRegionKind) -> O,
        O: IntoVisitResult,
    {
    }
    impl<V: VisitDispatch> SealedLink<super::ByDispatchLink> for &mut V {}
}

#[doc(hidden)]
pub struct ByOwnedLink<T>(PhantomData<T>);

impl<F, T, O> WalkChainLink<ByOwnedLink<T>> for F
where
    F: FnMut(T) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoVisitResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        _def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        value
            .cast::<T>()
            .map(|typed| self(typed).into_visit_result())
    }
}

#[doc(hidden)]
pub struct ByOwnedKindLink<T>(PhantomData<T>);

impl<F, T, O> WalkChainLink<ByOwnedKindLink<T>> for F
where
    F: FnMut(T, DefRegionKind) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoVisitResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        value
            .cast::<T>()
            .map(|typed| self(typed, def_region_kind).into_visit_result())
    }
}

#[doc(hidden)]
pub struct ByNodeLink<N>(PhantomData<N>);

impl<F, N, O> WalkChainLink<ByNodeLink<N>> for F
where
    F: for<'a> FnMut(&'a N) -> O,
    N: ObjectCore,
    O: IntoVisitResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        _def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        value
            .as_node::<N>()
            .map(|node| self(node).into_visit_result())
    }
}

#[doc(hidden)]
pub struct ByNodeKindLink<N>(PhantomData<N>);

impl<F, N, O> WalkChainLink<ByNodeKindLink<N>> for F
where
    F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
    N: ObjectCore,
    O: IntoVisitResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        value
            .as_node::<N>()
            .map(|node| self(node, def_region_kind).into_visit_result())
    }
}

#[doc(hidden)]
pub enum ByCatchAllLink {}

impl<F, O> WalkChainLink<ByCatchAllLink> for F
where
    F: for<'a> FnMut(&'a VisitValue) -> O,
    O: IntoVisitResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        _def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        Some(self(value).into_visit_result())
    }
}

#[doc(hidden)]
pub enum ByCatchAllKindLink {}

impl<F, O> WalkChainLink<ByCatchAllKindLink> for F
where
    F: for<'a> FnMut(&'a VisitValue, DefRegionKind) -> O,
    O: IntoVisitResult,
{
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        Some(self(value, def_region_kind).into_visit_result())
    }
}

#[doc(hidden)]
pub enum ByDispatchLink {}

impl<V: VisitDispatch> WalkChainLink<ByDispatchLink> for &mut V {
    #[inline]
    fn try_call(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        self.dispatch_visit(value, def_region_kind)
    }
}

/// Runs a tuple of [`WalkChainLink`]s at the phase selected by `order`,
/// trying links in order and short-circuiting on the first whose type
/// matches — the Rust analog of C++ `StructuralWalkCallbackChain`. Static
/// dispatch throughout: each link's type test inlines to the same code the
/// `#[dispatch(visit)]` macro generates for a `visit_*` chain.
#[doc(hidden)]
pub struct ChainWalker<Links, Markers> {
    links: Links,
    order: WalkOrder,
    markers: PhantomData<fn(Markers)>,
}

macro_rules! impl_chain_walker {
    ($(($F:ident, $M:ident, $idx:tt)),+) => {
        impl<$($F, $M,)+> ChainWalker<($($F,)+), ($($M,)+)>
        where
            $($F: WalkChainLink<$M>,)+
        {
            #[inline]
            fn dispatch(
                &mut self,
                value: &VisitValue,
                def_region_kind: DefRegionKind,
            ) -> Result<WalkResult> {
                $(
                    if let Some(result) = self.links.$idx.try_call(value, def_region_kind) {
                        return result;
                    }
                )+
                Ok(WalkResult::Advance)
            }
        }

        impl<$($F, $M,)+> NativeVisit for ChainWalker<($($F,)+), ($($M,)+)>
        where
            $($F: WalkChainLink<$M>,)+
        {
            fn enter(
                &mut self,
                value: &VisitValue,
                def_region_kind: DefRegionKind,
            ) -> Result<WalkResult> {
                match self.order {
                    WalkOrder::PreOrder => self.dispatch(value, def_region_kind),
                    WalkOrder::PostOrder => Ok(WalkResult::Advance),
                }
            }

            fn exit(
                &mut self,
                value: &VisitValue,
                def_region_kind: DefRegionKind,
            ) -> Result<WalkResult> {
                match self.order {
                    WalkOrder::PreOrder => Ok(WalkResult::Advance),
                    WalkOrder::PostOrder => self.dispatch(value, def_region_kind),
                }
            }
        }

        impl<$($F, $M,)+> IntoWalker<($($M,)+)> for ($($F,)+)
        where
            $($F: WalkChainLink<$M>,)+
        {
            type Walker = ChainWalker<($($F,)+), ($($M,)+)>;
            fn into_walker(self, order: WalkOrder) -> Self::Walker {
                ChainWalker {
                    links: self,
                    order,
                    markers: PhantomData,
                }
            }
        }
    };
}

impl_callback_chain_tuple_arities!(impl_chain_walker);

// A bare typed closure — `FnMut(T)` or `FnMut(&N)`, optionally with a
// trailing `DefRegionKind` — walks as a single-link chain, so a lone typed
// handler needs no tuple wrapping; values that do not match its argument
// type advance normally. `&VisitValue` catch-all closures keep their
// dedicated `ClosureWalker`/`ClosureKindWalker` path above.
macro_rules! impl_bare_link_walker {
    ($(($marker:ident, $($fn_args:ty),+)),+ $(,)?) => {
        $(
            impl<F, T, O> IntoWalker<$marker<T>> for F
            where
                F: FnMut($($fn_args),+) -> O,
                Self: WalkChainLink<$marker<T>>,
                O: IntoVisitResult,
            {
                type Walker = ChainWalker<(F,), ($marker<T>,)>;
                fn into_walker(self, order: WalkOrder) -> Self::Walker {
                    ChainWalker {
                        links: (self,),
                        order,
                        markers: PhantomData,
                    }
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

/// A visitor that drives recursion itself, mirroring C++
/// `StructuralVisitorObj`.
///
/// [`structural_visit`] calls [`StructuralVisitor::visit`] for the root;
/// after that the visitor is in control, exactly like a C++ visitor whose
/// vtable `visit` runs per value. A `visit` implementation descends only
/// where it chooses:
///
/// * [`StructuralVisitor::default_visit_children`] delegates the default
///   child recursion — the analog of C++
///   `StructuralVisitorObj::DefaultVisitExpected`.
/// * [`StructuralVisitor::visit_child`] visits one selected child — the
///   analog of C++ `visitor->Visit(child)`, with the explicit
///   `def_region_kind` argument playing the role of `WithDefRegionKind`.
///
/// Returning without descending skips the value's children. There is no
/// [`WalkResult`] at this layer: control flow is what the implementation
/// visits, and `Ok(Some(interrupt))` halts the traversal — the analog of
/// returning a C++ `VisitInterrupt`. Nested `visit_child` and
/// `default_visit_children` calls report a nested interrupt through their
/// return value; propagate it (and errors, via `?`) upward instead of
/// dropping the result.
///
/// The definition-region state is threaded explicitly, exactly like walk
/// handlers that declare the trailing argument: `visit` receives the state
/// active at the value and forwards it — or an override — when descending.
/// Reflected-field annotations override the forwarded state automatically
/// inside `default_visit_children`.
pub trait StructuralVisitor: Sized {
    /// Visit one value under the definition-region state active at it.
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>>;

    /// Visit `child` now under `def_region_kind`, dispatching back into
    /// [`StructuralVisitor::visit`]. An FFI `None` child is skipped without
    /// a callback, matching the walk layer.
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

    /// Visit `value`'s children — not `value` itself — with the default
    /// rules, dispatching each child back into [`StructuralVisitor::visit`].
    ///
    /// Children are container contents for `Array`/`List`/`Map`/`Dict` and
    /// reflected structural fields otherwise. Field annotations override
    /// `def_region_kind` for that field's recursive visit exactly like the
    /// walk layer.
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

/// Internal per-value protocol driven by the recursion engine. Public only
/// as the bound of [`IntoWalker::Walker`]; not meant to be implemented
/// outside this crate.
#[doc(hidden)]
pub trait NativeVisit {
    fn enter(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult>;

    fn exit(&mut self, _value: &VisitValue, _def_region_kind: DefRegionKind) -> Result<WalkResult> {
        Ok(WalkResult::Advance)
    }
}

/// Per-child action invoked by the shared child-iteration engine.
///
/// The engine owns *finding* the children (container contents, reflected
/// fields) and computing each child's definition-region state; this trait
/// decides what happens at a child. The walk layer recurses
/// ([`WalkChildren`]); the visitor layer hands the child straight to user
/// code ([`UserChildren`]).
trait ChildVisit {
    fn visit_child(&mut self, child: TVMFFIAny, def_region_kind: DefRegionKind) -> NativeResult;
}

/// Walk-layer recursion: every child re-enters [`visit_raw`].
struct WalkChildren<'a, V> {
    visitor: &'a mut V,
}

impl<V: NativeVisit> ChildVisit for WalkChildren<'_, V> {
    fn visit_child(&mut self, child: TVMFFIAny, def_region_kind: DefRegionKind) -> NativeResult {
        visit_raw(child, self.visitor, def_region_kind)
    }
}

/// Visitor-layer dispatch: every child goes back into the user-driven
/// [`StructuralVisitor::visit`], which controls further descent itself.
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

/// Recurse into `value` on behalf of `visitor`: fire its enter hook, ask the
/// registered type hook to visit the children, then fire its exit hook.
fn visit_raw<V: NativeVisit>(
    value: TVMFFIAny,
    visitor: &mut V,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }

    let visit_value = VisitValue::from_raw(value);
    // Single by-value matches: splitting the Result match from the
    // WalkResult match leaves a partially-moved temporary whose drop glue
    // the compiler cannot fold away (measurably so on the container fast
    // path).
    match visitor.enter(&visit_value, def_region_kind) {
        Ok(WalkResult::Advance) => {}
        Ok(WalkResult::Skip) => return Ok(()),
        Ok(WalkResult::Interrupt) => return Err(NativeHalt::Interrupt(Any::new())),
        Ok(WalkResult::InterruptWith(payload)) => return Err(NativeHalt::Interrupt(payload)),
        Err(error) => return Err(with_value_context(error.into(), value)),
    }

    let context = std::ptr::from_mut(&mut *visitor).cast::<c_void>();
    let children = &mut WalkChildren {
        visitor: &mut *visitor,
    };
    if let Err(halt) = visit_children_raw(value, children, context, def_region_kind) {
        return Err(with_value_context(halt, value));
    }

    match visitor.exit(&visit_value, def_region_kind) {
        Ok(WalkResult::Interrupt) => Err(NativeHalt::Interrupt(Any::new())),
        Ok(WalkResult::InterruptWith(payload)) => Err(NativeHalt::Interrupt(payload)),
        Ok(WalkResult::Advance | WalkResult::Skip) => Ok(()),
        Err(error) => Err(with_value_context(error.into(), value)),
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
        for_each_field(value.type_index, |field| {
            match visit_reflected_field(object, field, visitor, def_region_kind) {
                Ok(()) => ControlFlow::Continue(()),
                Err(halt) => ControlFlow::Break(halt),
            }
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
    let mut child_raw = TVMFFIAny::new();
    if getter(address, &mut child_raw) != 0 {
        return Err(with_error_context(
            NativeHalt::Error(Error::from_raised()),
            &format!("field `{}`", field.name.as_str()),
        ));
    }

    // A reflection getter returns an owned Any. Keep it alive while the
    // recursive walk borrows its raw cell.
    let child = Any::from_raw_ffi_any(child_raw);
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

type RuntimeVisitCallback = unsafe fn(*mut c_void, TVMFFIAny, DefRegionKind) -> NativeResult;

/// Active Rust visitor with the exact C++ `StructuralVisitorObj` prefix.
///
/// Registered type hooks read `vtable` and `def_region_mode`; the erased Rust
/// callback state follows that shared prefix and is never inspected by C++.
#[repr(C)]
struct RuntimeStructuralVisitorObj {
    base: Object,
    vtable: *const StructuralVisitorVTable,
    def_region_mode: i32,
    context: *mut c_void,
    context_identity: *mut c_void,
    owner_thread: std::thread::ThreadId,
    callback: RuntimeVisitCallback,
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
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH + 1;
    const TYPE_FINAL: bool = true;

    fn type_index() -> i32 {
        TVMFFITypeIndex::kTVMFFIVisitInterrupt as i32
    }

    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.base)
    }
}

static RUST_STRUCTURAL_VISITOR_VTABLE: StructuralVisitorVTable = StructuralVisitorVTable {
    visit: rust_vtable_visit,
};

struct RuntimeContextGuard {
    visitor: StructuralVisitorHandle,
    context: *mut c_void,
}

impl Drop for RuntimeContextGuard {
    fn drop(&mut self) {
        // SAFETY: this guard exists only while the runtime visitor is live on
        // its owner thread.
        unsafe { (*self.visitor).context = self.context };
    }
}

/// Take the Rust callback context while one vtable call is active.
///
/// # Safety
///
/// `visitor` must be null or point to a live [`RuntimeStructuralVisitorObj`].
unsafe fn take_runtime_context(visitor: StructuralVisitorHandle) -> Result<RuntimeContextGuard> {
    if visitor.is_null() {
        return Err(runtime_error("null active structural visitor"));
    }
    if (*visitor).owner_thread != std::thread::current().id() {
        return Err(runtime_error(
            "structural visitor callback invoked from a different thread",
        ));
    }
    let context = (*visitor).context;
    if context.is_null() {
        let message = if (*visitor).context_identity.is_null() {
            "structural visitor was retained after its active call"
        } else {
            "structural visitor may only be called by its active registered hook"
        };
        return Err(runtime_error(message));
    }
    (*visitor).context = std::ptr::null_mut();
    Ok(RuntimeContextGuard { visitor, context })
}

unsafe extern "C" fn rust_vtable_visit(
    visitor: StructuralVisitorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    let context_guard = match take_runtime_context(visitor) {
        Ok(guard) => guard,
        Err(error) => return native_result_into_raw(Err(NativeHalt::Error(error))),
    };
    let callback = (*visitor).callback;
    let context = context_guard.context;
    let raw = *value.as_raw_ffi_any();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let kind = def_region_from_raw((*visitor).def_region_mode)?;
        with_active_structural_visitor(visitor, || callback(context, raw, kind))
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
    static ACTIVE_STRUCTURAL_VISITOR: Cell<StructuralVisitorHandle> = const {
        Cell::new(std::ptr::null_mut())
    };
}

fn with_active_structural_visitor<T>(
    handle: StructuralVisitorHandle,
    callback: impl FnOnce() -> T,
) -> T {
    ACTIVE_STRUCTURAL_VISITOR.with(|active| {
        let previous = active.replace(handle);
        struct Restore<'a> {
            active: &'a Cell<StructuralVisitorHandle>,
            previous: StructuralVisitorHandle,
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
        let handle = active.get();
        if handle.is_null() {
            Err(runtime_error(
                "structural visitor helper called outside structural_visit or structural_walk",
            ))
        } else {
            Ok(handle)
        }
    })
}

/// Expose the current Rust visitor only while a registered hook may call its
/// vtable. The callback hides it again before returning to Rust user code.
fn with_current_visitor_context(
    visitor: StructuralVisitorHandle,
    context: *mut c_void,
    callback: impl FnOnce() -> NativeResult,
) -> NativeResult {
    if visitor.is_null() {
        return Err(runtime_error("null active structural visitor").into());
    }
    unsafe {
        if (*visitor).owner_thread != std::thread::current().id() {
            return Err(
                runtime_error("structural visitor helper invoked from a different thread").into(),
            );
        }
        if (*visitor).context_identity != context {
            return Err(
                runtime_error("structural visitor helper called on a non-active visitor").into(),
            );
        }
        if !(*visitor).context.is_null() {
            return Err(runtime_error("structural visitor context is already exposed").into());
        }

        (*visitor).context = context;
        struct HideContext {
            visitor: StructuralVisitorHandle,
        }
        impl Drop for HideContext {
            fn drop(&mut self) {
                unsafe { (*self.visitor).context = std::ptr::null_mut() };
            }
        }
        let _hide = HideContext { visitor };
        callback()
    }
}

unsafe fn runtime_walk<V: NativeVisit>(
    context: *mut c_void,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    visit_raw(raw, &mut *context.cast::<V>(), def_region_kind)
}

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
    callback: RuntimeVisitCallback,
) -> NativeResult {
    let context = std::ptr::from_mut(driver).cast::<c_void>();
    let mut active = ObjectArc::new(RuntimeStructuralVisitorObj {
        base: Object::new(),
        vtable: &RUST_STRUCTURAL_VISITOR_VTABLE,
        def_region_mode: DefRegionKind::None as i32,
        context,
        context_identity: context,
        owner_thread: std::thread::current().id(),
        callback,
        panic: None,
    });
    let handle = unsafe { ObjectArc::as_raw_mut(&mut active) };
    let result = call_visitor(handle, root, DefRegionKind::None);
    unsafe {
        (*handle).context = std::ptr::null_mut();
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
                let function = Function::try_from(owned_from_raw(attr)?)?;
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
    visit_result_from_any(Any::from_raw_ffi_any(raw))
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

fn def_region_from_raw(kind: i32) -> Result<DefRegionKind> {
    match kind {
        x if x == DefRegionKind::None as i32 => Ok(DefRegionKind::None),
        x if x == DefRegionKind::Recursive as i32 => Ok(DefRegionKind::Recursive),
        x if x == DefRegionKind::NonRecursive as i32 => Ok(DefRegionKind::NonRecursive),
        _ => Err(runtime_error("invalid structural definition-region kind")),
    }
}

fn owned_from_raw(raw: TVMFFIAny) -> Result<Any> {
    let mut owned = Any::new();
    let return_code = unsafe { TVMFFIAnyViewToOwnedAny(&raw, Any::as_data_ptr(&mut owned)) };
    if return_code == 0 {
        Ok(owned)
    } else {
        Err(Error::from_raised())
    }
}

fn with_value_context(halt: NativeHalt, value: TVMFFIAny) -> NativeHalt {
    if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        halt
    } else {
        with_error_context(halt, &format!("object `{}`", type_key_of(value.type_index)))
    }
}

/// Visit `root` with a user-driven [`StructuralVisitor`].
///
/// The visitor's [`StructuralVisitor::visit`] runs for the root under
/// [`DefRegionKind::None`] and controls all further recursion itself. This is
/// the Rust analog of constructing a C++ `StructuralVisitorObj` and calling
/// `visitor->Visit(root)`. An FFI `None` root completes immediately.
pub fn structural_visit<R, V>(root: &R, visitor: &mut V) -> Result<Option<VisitInterrupt>>
where
    V: StructuralVisitor,
    for<'x> AnyView<'x>: From<&'x R>,
{
    finish(run_structural_visitor(
        raw_of(AnyView::from(root)),
        visitor,
        runtime_user_visit::<V>,
    ))
}

/// Walk `root` with an observer, the Rust analog of C++
/// `StructuralWalk<order>(root, callbacks...)`.
///
/// `walker` is anything implementing [`IntoWalker`]: a `&mut` reference to a
/// stateful [`VisitDispatch`] visitor (`#[dispatch(visit)]`), a bare closure
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
    let mut dispatch = walker.into_walker(order);
    finish(run_structural_visitor(
        raw_of(AnyView::from(root)),
        &mut dispatch,
        runtime_walk::<H::Walker>,
    ))
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
        NativeHalt::Error(error) => NativeHalt::Error(Error::with_appended_backtrace(
            error,
            &format!("[native structural visit] {frame}\n"),
        )),
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

/// Visit every reflected field of `type_index` and its ancestors in the same
/// parent-to-child order as C++ `ForEachFieldInfoWithEarlyStop`.
///
/// # Safety
///
/// `type_index` must be a registered type index.
pub(crate) unsafe fn for_each_field<B>(
    type_index: i32,
    mut callback: impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    let info = TVMFFIGetTypeInfo(type_index);
    if info.is_null() {
        return None;
    }

    // Ancestor slot 0 is the root Object. C++ starts at slot 1, walks toward
    // the immediate parent, then visits the concrete type's own fields.
    for depth in 1..(*info).type_depth {
        let ancestor = *(*info).type_acenstors.offset(depth as isize);
        if let Some(value) = visit_field_level(ancestor, &mut callback) {
            return Some(value);
        }
    }
    visit_field_level(info, &mut callback)
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
