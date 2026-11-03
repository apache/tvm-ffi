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

//! Ordered replacement callbacks and adapters for structural mapping.

use super::structural_common::{
    borrowed_view, impl_callback_chain_tuple_arities, raw_of, same_shallow,
};
use super::structural_mutate::{
    default_mutate_driver, run_mutator_engine, runtime_state, with_value_context,
    BorrowedDispatcher, ByBaseMutator, DispatcherStorage, IntoMutateResult, MutationDriver,
    OwnedDispatcher, Permit, StructuralMutator, StructuralMutatorBase, StructuralMutatorHandle,
    StructuralMutatorObj,
};
use super::structural_visit::DefRegionKind;
use super::structural_walk::WalkOrder;
use super::unchanged::is_unchanged;
use crate::tvm_ffi_sys::TVMFFIAny;
use crate::{Any, AnyView, Dispatcher, ObjectCore, Result};
use std::marker::PhantomData;
use std::os::raw::c_void;

// -----------------------------------------------------------------------------
// Map adapter to Dispatcher.
// -----------------------------------------------------------------------------

/// Ordered typed replacement dispatch for [`structural_map`].
///
/// This private engine bridge consumes the common normalized map signature.
/// `None` means no handler matched and preserves the current value; matching
/// and handler conversion are supplied by `Dispatcher`.
pub(crate) trait MapDispatch: Sized {
    fn dispatch_map(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>>;
}

/// Internal root-mapping protocol used by [`IntoMapper`].
#[doc(hidden)]
pub trait NativeMap: Sized {
    fn map_root(self, root: Any, order: WalkOrder) -> Result<Any>;
}

impl<D: MapDispatch> NativeMap for D {
    fn map_root(self, root: Any, order: WalkOrder) -> Result<Any> {
        run_native_mapper(root, OwnedDispatcher(self), order)
    }
}

impl<V> MapDispatch for V
where
    for<'receiver, 'value> &'receiver mut V:
        Dispatcher<(AnyView<'value>, DefRegionKind), Option<Result<Any>>>,
{
    #[inline(always)]
    fn dispatch_map(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        self.dispatch_tuple((value, def_region_kind))
    }
}

/// Conversion into the mapper consumed by [`structural_map`].
#[diagnostic::on_unimplemented(
    message = "unsupported structural-map callback shape",
    label = "this value cannot be used as a structural mapper",
    note = "pass `&mut` a matching `Dispatcher`, a supported closure or callback tuple"
)]
pub trait IntoMapper<Marker> {
    type Mapper: NativeMap;
    fn into_mapper(self) -> Self::Mapper;
}

#[doc(hidden)]
pub enum ByMapDispatch {}

impl<'a, V: MapDispatch> IntoMapper<ByMapDispatch> for &'a mut V {
    type Mapper = DispatchMapper<'a, V>;

    #[inline]
    fn into_mapper(self) -> Self::Mapper {
        DispatchMapper(self)
    }
}

#[doc(hidden)]
pub struct DispatchMapper<'a, V>(&'a mut V);

impl<V: MapDispatch> NativeMap for DispatchMapper<'_, V> {
    fn map_root(self, root: Any, order: WalkOrder) -> Result<Any> {
        run_native_mapper(root, BorrowedDispatcher(self.0), order)
    }
}

trait MapWithBase: Sized {
    fn map_with_base<B: StructuralMutatorBase>(
        self,
        root: Any,
        order: WalkOrder,
        base: B,
    ) -> Result<Any>;
}
impl<D: MapDispatch> MapWithBase for D {
    fn map_with_base<B: StructuralMutatorBase>(
        self,
        root: Any,
        order: WalkOrder,
        base: B,
    ) -> Result<Any> {
        run_mapper_with_base(root, OwnedDispatcher(self), order, base)
    }
}
impl<D: MapDispatch> MapWithBase for DispatchMapper<'_, D> {
    fn map_with_base<B: StructuralMutatorBase>(
        self,
        root: Any,
        order: WalkOrder,
        base: B,
    ) -> Result<Any> {
        run_mapper_with_base(root, BorrowedDispatcher(self.0), order, base)
    }
}
#[doc(hidden)]
pub struct BaseMapper<B, M>(B, M);
impl<B: StructuralMutatorBase, M: MapWithBase> NativeMap for BaseMapper<B, M> {
    fn map_root(self, root: Any, order: WalkOrder) -> Result<Any> {
        self.1.map_with_base(root, order, self.0)
    }
}
impl<B, H, M> IntoMapper<(ByBaseMutator, M)> for StructuralMutator<B, H>
where
    B: StructuralMutatorBase,
    H: IntoMapper<M>,
    H::Mapper: MapWithBase,
{
    type Mapper = BaseMapper<B, H::Mapper>;
    fn into_mapper(self) -> Self::Mapper {
        BaseMapper(self.base, self.dispatch.into_mapper())
    }
}
fn run_mapper_with_base<S: DispatcherStorage, B: StructuralMutatorBase>(
    root: Any,
    dispatch: S,
    order: WalkOrder,
    base: B,
) -> Result<Any>
where
    S::Dispatch: MapDispatch,
{
    match order {
        WalkOrder::PreOrder => {
            run_mutator_engine::<NativeMapper<'_, S::Dispatch, true, B>, _, _>(root, dispatch, base)
        }
        WalkOrder::PostOrder => {
            run_mutator_engine::<NativeMapper<'_, S::Dispatch, false, B>, _, _>(
                root, dispatch, base,
            )
        }
    }
}

/// One typed callback in a structural-map tuple.
///
/// Links use first-match order and may receive an owned FFI value, borrowed
/// object node, or `AnyView<'_>`, optionally followed by [`DefRegionKind`].
pub trait MapChainLink<Marker>: sealed_map::SealedMapLink<Marker> {
    #[doc(hidden)]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>>;
}

mod sealed_map {
    use super::{AnyView, DefRegionKind, IntoMutateResult, MapDispatch, ObjectCore};

    pub trait SealedMapLink<Marker> {}

    impl<F, T, O> SealedMapLink<super::ByMapOwned<T>> for F
    where
        F: FnMut(T) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, T, O> SealedMapLink<super::ByMapOwnedKind<T>> for F
    where
        F: FnMut(T, DefRegionKind) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, N: ObjectCore, O> SealedMapLink<super::ByMapNode<N>> for F
    where
        F: for<'a> FnMut(&'a N) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, N: ObjectCore, O> SealedMapLink<super::ByMapNodeKind<N>> for F
    where
        F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, O> SealedMapLink<super::ByMapCatchAll> for F
    where
        F: for<'a> FnMut(AnyView<'a>) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, O> SealedMapLink<super::ByMapCatchAllKind> for F
    where
        F: for<'a> FnMut(AnyView<'a>, DefRegionKind) -> O,
        O: IntoMutateResult,
    {
    }

    impl<V: MapDispatch> SealedMapLink<super::ByMapDispatchLink> for &mut V {}
}

#[doc(hidden)]
pub struct ByMapOwned<T>(PhantomData<T>);

impl<F, T, O> MapChainLink<ByMapOwned<T>> for F
where
    F: FnMut(T) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .try_as::<T>()
            .map(|typed| self(typed).into_mutate_result())
    }
}

#[doc(hidden)]
pub struct ByMapOwnedKind<T>(PhantomData<T>);

impl<F, T, O> MapChainLink<ByMapOwnedKind<T>> for F
where
    F: FnMut(T, DefRegionKind) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .try_as::<T>()
            .map(|typed| self(typed, def_region_kind).into_mutate_result())
    }
}

#[doc(hidden)]
pub struct ByMapNode<N>(PhantomData<N>);

impl<F, N, O> MapChainLink<ByMapNode<N>> for F
where
    F: for<'a> FnMut(&'a N) -> O,
    N: ObjectCore,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .downcast_ref::<N>()
            .map(|node| self(node).into_mutate_result())
    }
}

#[doc(hidden)]
pub struct ByMapNodeKind<N>(PhantomData<N>);

impl<F, N, O> MapChainLink<ByMapNodeKind<N>> for F
where
    F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
    N: ObjectCore,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .downcast_ref::<N>()
            .map(|node| self(node, def_region_kind).into_mutate_result())
    }
}

#[doc(hidden)]
pub enum ByMapCatchAll {}

impl<F, O> MapChainLink<ByMapCatchAll> for F
where
    F: for<'a> FnMut(AnyView<'a>) -> O,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        Some(self(value).into_mutate_result())
    }
}

#[doc(hidden)]
pub enum ByMapCatchAllKind {}

impl<F, O> MapChainLink<ByMapCatchAllKind> for F
where
    F: for<'a> FnMut(AnyView<'a>, DefRegionKind) -> O,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        Some(self(value, def_region_kind).into_mutate_result())
    }
}

#[doc(hidden)]
pub struct ByMapChainLink<Markers>(PhantomData<fn(Markers)>);

#[doc(hidden)]
pub enum ByMapDispatchLink {}

impl<V: MapDispatch> MapChainLink<ByMapDispatchLink> for &mut V {
    #[inline]
    fn try_map(
        &mut self,
        value: AnyView<'_>,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        self.dispatch_map(value, def_region_kind)
    }
}

/// Adapter from a [`MapChainLink`] to [`Dispatcher`].
#[doc(hidden)]
pub struct MapChain<Link, Marker> {
    link: Link,
    marker: PhantomData<fn(Marker)>,
}

impl<Link, Marker> MapChain<Link, Marker> {
    #[inline]
    fn new(link: Link) -> Self {
        MapChain {
            link,
            marker: PhantomData,
        }
    }
}

impl<'receiver, 'value, Link, Marker>
    Dispatcher<(AnyView<'value>, DefRegionKind), Option<Result<Any>>>
    for &'receiver mut MapChain<Link, Marker>
where
    Link: MapChainLink<Marker>,
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, def_region_kind): (AnyView<'value>, DefRegionKind),
    ) -> Option<Result<Any>> {
        self.link.try_map(value, def_region_kind)
    }
}

macro_rules! impl_map_chain_link {
    ($(($F:ident, $M:ident, $idx:tt)),+) => {
        impl<$($F, $M,)+> sealed_map::SealedMapLink<ByMapChainLink<($($M,)+)>> for ($($F,)+)
        where
            $($F: MapChainLink<$M>,)+
        {
        }

        impl<$($F, $M,)+> MapChainLink<ByMapChainLink<($($M,)+)>> for ($($F,)+)
        where
            $($F: MapChainLink<$M>,)+
        {
            #[inline]
            fn try_map(
                &mut self,
                value: AnyView<'_>,
                def_region_kind: DefRegionKind,
            ) -> Option<Result<Any>> {
                $(
                    if let Some(result) = self.$idx.try_map(value, def_region_kind) {
                        return Some(result);
                    }
                )+
                None
            }
        }

        impl<$($F, $M,)+> IntoMapper<($($M,)+)> for ($($F,)+)
        where
            $($F: MapChainLink<$M>,)+
        {
            type Mapper = MapChain<($($F,)+), ByMapChainLink<($($M,)+)>>;

            #[inline]
            fn into_mapper(self) -> Self::Mapper {
                MapChain::new(self)
            }
        }
    };
}

impl_callback_chain_tuple_arities!(impl_map_chain_link);

macro_rules! impl_bare_map_link {
    ($(($marker:ident, $($fn_args:ty),+)),+ $(,)?) => {
        $(
            impl<F, T, O> IntoMapper<$marker<T>> for F
            where
                F: FnMut($($fn_args),+) -> O,
                Self: MapChainLink<$marker<T>>,
                O: IntoMutateResult,
            {
                type Mapper = MapChain<F, $marker<T>>;

                #[inline]
                fn into_mapper(self) -> Self::Mapper {
                    MapChain::new(self)
                }
            }
        )+
    };
}

impl_bare_map_link!(
    (ByMapOwned, T),
    (ByMapOwnedKind, T, DefRegionKind),
    (ByMapNode, &T),
    (ByMapNodeKind, &T, DefRegionKind),
);

impl<F, O> IntoMapper<ByMapCatchAll> for F
where
    F: for<'a> FnMut(AnyView<'a>) -> O,
    O: IntoMutateResult,
{
    type Mapper = MapChain<F, ByMapCatchAll>;

    #[inline]
    fn into_mapper(self) -> Self::Mapper {
        MapChain::new(self)
    }
}

impl<F, O> IntoMapper<ByMapCatchAllKind> for F
where
    F: for<'a> FnMut(AnyView<'a>, DefRegionKind) -> O,
    O: IntoMutateResult,
{
    type Mapper = MapChain<F, ByMapCatchAllKind>;

    #[inline]
    fn into_mapper(self) -> Self::Mapper {
        MapChain::new(self)
    }
}

// -----------------------------------------------------------------------------
// Implementation of map.
// -----------------------------------------------------------------------------

struct NativeMapper<'a, D, const PRE_ORDER: bool, B> {
    _base: PhantomData<B>,
    dispatch: &'a mut D,
    runtime: StructuralMutatorHandle,
}

fn run_native_mapper<S: DispatcherStorage>(root: Any, dispatch: S, order: WalkOrder) -> Result<Any>
where
    S::Dispatch: MapDispatch,
{
    match order {
        WalkOrder::PreOrder => run_map_engine::<S, true>(root, dispatch),
        WalkOrder::PostOrder => run_map_engine::<S, false>(root, dispatch),
    }
}

fn run_map_engine<S: DispatcherStorage, const PRE: bool>(root: Any, dispatch: S) -> Result<Any>
where
    S::Dispatch: MapDispatch,
{
    run_mutator_engine::<NativeMapper<'_, S::Dispatch, PRE, StructuralMutatorObj>, S, _>(
        root,
        dispatch,
        StructuralMutatorObj::default(),
    )
}

impl<D: MapDispatch, const PRE_ORDER: bool, B: StructuralMutatorBase>
    NativeMapper<'_, D, PRE_ORDER, B>
{
    #[inline(always)]
    fn map_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        self.map_current_raw(raw, def_region_kind, permit)
    }

    fn map_current_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        match PRE_ORDER {
            true => {
                // A replacement inherits the input's permission, established
                // before the callback can acquire or release ownership.
                let permit = permit.inplace_mode(raw).permit();
                let value = unsafe { borrowed_view(&raw) };
                let Some(callback_result) = self.dispatch.dispatch_map(value, def_region_kind)
                else {
                    return self.default_map_current_raw(raw, def_region_kind, permit);
                };
                let mapped = callback_result.map_err(|error| with_value_context(error, raw))?;
                let mapped_raw = *mapped.as_raw_ffi_any();
                if is_unchanged(&mapped) || same_shallow(raw, mapped_raw) {
                    // Release the callback's temporary ownership before the
                    // runtime uniqueness check observes the original.
                    drop(mapped);
                    self.default_map_current_raw(raw, def_region_kind, permit)
                } else {
                    let descended = self.map_default_root(&mapped, def_region_kind, permit)?;
                    Ok(if is_unchanged(&descended) {
                        mapped
                    } else {
                        descended
                    })
                }
            }
            false => {
                let mapped = self.default_map_current_raw(raw, def_region_kind, permit)?;
                let original = unsafe { borrowed_view(&raw) };
                let value = if is_unchanged(&mapped) {
                    original
                } else {
                    AnyView::from(&mapped)
                };
                // Keep child rewrites when the callback leaves its input unchanged.
                match self.dispatch.dispatch_map(value, def_region_kind) {
                    Some(Ok(result)) if is_unchanged(&result) => Ok(mapped),
                    Some(result) => {
                        result.map_err(|error| with_value_context(error, raw_of(value)))
                    }
                    None => Ok(mapped),
                }
            }
        }
    }

    /// Descend into a pre-order replacement without dispatching its root again.
    fn map_default_root(
        &mut self,
        mapped: &Any,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        let raw = *mapped.as_raw_ffi_any();
        self.default_map_current_raw(raw, def_region_kind, permit)
    }
}

impl<D: MapDispatch, const PRE_ORDER: bool, B: StructuralMutatorBase> MutationDriver
    for NativeMapper<'_, D, PRE_ORDER, B>
{
    type Dispatch = D;
    type Base = B;
    fn dispatcher(&mut self) -> &mut D {
        self.dispatch
    }
    fn runtime(&self) -> StructuralMutatorHandle {
        self.runtime
    }
    fn context(&mut self) -> *mut c_void {
        std::ptr::from_mut(&mut *self.dispatch).cast()
    }
    #[inline(always)]
    unsafe fn runtime_dispatch<const INPLACE: bool>(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> TVMFFIAny {
        NativeMapper::<D, PRE_ORDER, B> {
            dispatch: &mut *(*runtime_state(runtime)).context.cast::<D>(),
            runtime,
            _base: PhantomData,
        }
        .dispatch_abi_raw::<INPLACE>(raw, kind)
    }
    #[inline(always)]
    unsafe fn runtime_recurse(
        runtime: StructuralMutatorHandle,
        dispatch: &mut D,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        NativeMapper::<D, PRE_ORDER, B> {
            dispatch,
            runtime,
            _base: PhantomData,
        }
        .dispatch_raw(raw, kind, permit)
    }
    #[inline(always)]
    unsafe fn runtime_default(
        runtime: StructuralMutatorHandle,
        dispatch: &mut D,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        default_mutate_driver(
            &mut NativeMapper::<D, PRE_ORDER, B> {
                dispatch,
                runtime,
                _base: PhantomData,
            },
            raw,
            kind,
            permit,
        )
    }
    #[inline(always)]
    fn dispatch_raw(&mut self, raw: TVMFFIAny, kind: DefRegionKind, permit: Permit) -> Result<Any> {
        self.map_raw(raw, kind, permit)
    }
}

/// Transform a structured value graph with ordered replacement callbacks.
///
/// The root is consumed. A uniquely owned built-in container may therefore be
/// reused in place, while passing `root.clone()` keeps the original shared and
/// selects copy-on-write behavior. Map and Dict keys are anchors and are not
/// mapped. Their registered structural hooks own container traversal.
///
/// Callbacks run at every occurrence. Default reflected descent handles FreeVar
/// and DAG-node remapping; registered hooks own their type's remapping policy.
/// Callback replacements are not automatically recorded as substitutions.
///
/// In-place changes completed before an error are not rolled back. Because
/// this function consumes `root`, an error does not return the partly mapped
/// root to the caller.
pub fn structural_map<R, M, H>(root: R, mapper: H, order: WalkOrder) -> Result<Any>
where
    R: Into<Any>,
    H: IntoMapper<M>,
{
    mapper.into_mapper().map_root(root.into(), order)
}
