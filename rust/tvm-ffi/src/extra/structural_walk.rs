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

//! Observer callbacks and adapters for structural walking.
//!
//! Callbacks report failures with [`Result`] and must not panic. A panic reaching
//! the non-unwinding native callback boundary aborts the process.

use super::structural_common::{impl_callback_chain_tuple_arities, raw_of};
use super::structural_visit::{
    visit_result_from_any, DefRegionKind, StructuralVisitEngine, StructuralVisitorView,
    VisitInterrupt,
};
use crate::{Any, AnyView, Dispatcher, ObjectCore, Result};
use std::marker::PhantomData;

// -----------------------------------------------------------------------------
// Key data structures and callback results.
// -----------------------------------------------------------------------------

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
    #[inline(always)]
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
    #[inline(always)]
    fn into_walk_result(self) -> Result<WalkResult> {
        Ok(self)
    }
}

impl IntoWalkResult for Result<WalkResult> {
    #[inline(always)]
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

// -----------------------------------------------------------------------------
// Walk adapter to Dispatcher.
// -----------------------------------------------------------------------------

// Borrowed traversal adapters forward this concrete tuple without making
// unknown-Args trait solving recurse through mutable reference layers.
impl<'receiver, 'inner, 'value, D, Ret> Dispatcher<(AnyView<'value>, DefRegionKind), Ret>
    for &'receiver mut &'inner mut D
where
    &'receiver mut D: Dispatcher<(AnyView<'value>, DefRegionKind), Ret>,
{
    #[inline(always)]
    fn dispatch_tuple(self, args: (AnyView<'value>, DefRegionKind)) -> Ret {
        <&'receiver mut D as Dispatcher<(AnyView<'value>, DefRegionKind), Ret>>::dispatch_tuple(
            &mut **self,
            args,
        )
    }
}

/// Dispatch for typed `structural_walk` observer callbacks.
///
/// This private engine bridge consumes the common normalized walk signature.
/// Matching is supplied by `Dispatcher`; its outer `None` means no match.
pub(crate) trait WalkDispatch: Sized {
    fn dispatch_walk(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>>;
}

impl<V> WalkDispatch for V
where
    for<'receiver, 'value> &'receiver mut V:
        Dispatcher<(AnyView<'value>, DefRegionKind), Option<Result<WalkResult>>>,
{
    #[inline(always)]
    fn dispatch_walk(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<WalkResult>> {
        self.dispatch_tuple((value, def_region_kind))
    }
}

#[doc(hidden)]
pub enum ByWalkDispatch {}

impl<'a, V: WalkDispatch> IntoWalker<ByWalkDispatch> for &'a mut V {
    type Walker = DispatchWalker<&'a mut V>;
    #[inline(always)]
    fn into_walker(self) -> Self::Walker {
        DispatchWalker { walker: self }
    }
}

/// Adapter from [`Dispatcher`] to the traversal's native callback.
#[doc(hidden)]
pub struct DispatchWalker<V> {
    walker: V,
}

impl<V> NativeContext for DispatchWalker<V> {}

impl<V: WalkDispatch> NativeVisit for DispatchWalker<&mut V> {
    #[inline(always)]
    fn visit(&mut self, value: AnyView<'_>, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        self.walker
            .dispatch_walk(value, def_region_kind)
            .unwrap_or_else(|| Ok(WalkResult::Advance))
    }
}

/// Conversion into the walker argument of [`structural_walk`].
///
/// Accepts a mutable [`Dispatcher`], a typed callback or a nested callback
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
    #[inline(always)]
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
    #[inline(always)]
    fn into_walker(self) -> Self::Walker {
        ChainWalker::new(self)
    }
}

/// One link in a first-match [`structural_walk`] callback chain.
///
/// Supported links are typed values, borrowed object nodes, `AnyView<'_>`,
/// and mutable [`Dispatcher`] implementations, optionally followed by
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
    // Keep native observer implementations within the supported adapters.
    pub trait NativeContext: Sized {}

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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
            #[inline(always)]
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
            #[inline(always)]
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
                #[inline(always)]
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
    #[inline(always)]
    fn walk_root(self, root: AnyView<'_>, order: WalkOrder) -> Result<Option<VisitInterrupt>> {
        visit_result_from_any(unsafe {
            Any::from_raw_ffi_any(run_walker_raw(raw_of(root), self, order))
        })
    }

    fn visit(&mut self, value: AnyView<'_>, def_region_kind: DefRegionKind) -> Result<WalkResult>;
}

impl<V: NativeVisit> NativeContext for &mut V {}

impl<V: NativeVisit> NativeVisit for &mut V {
    #[inline(always)]
    fn visit(&mut self, value: AnyView<'_>, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        (**self).visit(value, def_region_kind)
    }
}

// -----------------------------------------------------------------------------
// Implementation of walk.
// -----------------------------------------------------------------------------

// A walk is an ordinary visitor handler: the visitor core owns all child
// recursion, native hooks and region transitions. This adapter only chooses
// when the observer runs and translates its control result.
struct WalkVisitor<V, const PRE_ORDER: bool> {
    observer: V,
}

impl<'s, 'v, 'c, 'r, V: NativeVisit, const PRE_ORDER: bool>
    Dispatcher<
        (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, WalkVisitor<V, PRE_ORDER>>,
        ),
        Option<Result<Option<VisitInterrupt>>>,
    > for &'s mut WalkVisitor<V, PRE_ORDER>
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, visitor): (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, WalkVisitor<V, PRE_ORDER>>,
        ),
    ) -> Option<Result<Option<VisitInterrupt>>> {
        Some(self.visit(value, visitor))
    }
}

impl<V: NativeVisit, const PRE_ORDER: bool> WalkVisitor<V, PRE_ORDER> {
    #[inline(always)]
    fn visit(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let kind = visitor.def_region_kind();
        if !PRE_ORDER {
            if let Some(interrupt) = visitor.default_visit(self, value)? {
                core::hint::cold_path();
                return Ok(Some(interrupt));
            }
        }
        match self.observer.visit(value, kind)? {
            WalkResult::Advance if PRE_ORDER => visitor.default_visit(self, value),
            WalkResult::Advance => Ok(None),
            WalkResult::Skip => {
                core::hint::cold_path();
                Ok(None)
            }
            WalkResult::Interrupt => {
                core::hint::cold_path();
                Ok(Some(VisitInterrupt::with(Any::new())))
            }
            WalkResult::InterruptWith(payload) => {
                core::hint::cold_path();
                Ok(Some(VisitInterrupt::with(payload)))
            }
        }
    }
}

#[inline(always)]
fn run_walker_raw<D: NativeVisit>(
    root: crate::tvm_ffi_sys::TVMFFIAny,
    observer: D,
    order: WalkOrder,
) -> crate::tvm_ffi_sys::TVMFFIAny {
    // Own the result while dropping the observer: user capture destructors can
    // unwind after the native callback has returned.
    let result = match order {
        WalkOrder::PreOrder => {
            let mut visitor = WalkVisitor::<D, true> { observer };
            let result = unsafe {
                Any::from_raw_ffi_any(StructuralVisitEngine::run_visitor_raw(root, &mut visitor))
            };
            drop(visitor);
            result
        }
        WalkOrder::PostOrder => {
            let mut visitor = WalkVisitor::<D, false> { observer };
            let result = unsafe {
                Any::from_raw_ffi_any(StructuralVisitEngine::run_visitor_raw(root, &mut visitor))
            };
            drop(visitor);
            result
        }
    };
    unsafe { Any::into_raw_ffi_any(result) }
}

/// Walk `root` with an observer, the Rust analog of C++
/// `StructuralWalk<order>(root, callbacks...)`.
///
/// `walker` is anything implementing [`IntoWalker`]: a `&mut` reference to a
/// stateful [`Dispatcher`] walker (declared with `#[dispatch]`), a bare closure
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
/// traversal through the returned [`WalkResult`]. Walk accepts its existing
/// dispatcher borrows, closure observers and callback tuples; each adapter runs
/// through the same borrowed traversal engine.
///
/// Callbacks must report failures through [`Result`] and must not panic: a panic
/// reaching the non-unwinding native callback boundary aborts the process.
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
