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
    with_value_context, IntoMutateResult, MutationView, StructuralMutateEngine,
    StructuralMutatorView,
};
use super::structural_visit::DefRegionKind;
use super::structural_walk::WalkOrder;
use super::unchanged::is_unchanged;
use crate::{Any, AnyView, Dispatcher, ObjectCore, Result};
use std::marker::PhantomData;

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
    fn map_root(mut self, root: Any, order: WalkOrder) -> Result<Any> {
        run_native_mapper(root, &mut self, order)
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

    #[inline(always)]
    fn into_mapper(self) -> Self::Mapper {
        DispatchMapper(self)
    }
}

#[doc(hidden)]
pub struct DispatchMapper<'a, V>(&'a mut V);

impl<V: MapDispatch> NativeMap for DispatchMapper<'_, V> {
    fn map_root(self, root: Any, order: WalkOrder) -> Result<Any> {
        run_native_mapper(root, self.0, order)
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
    #[inline(always)]
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
            #[inline(always)]
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

            #[inline(always)]
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

                #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
    fn into_mapper(self) -> Self::Mapper {
        MapChain::new(self)
    }
}

// -----------------------------------------------------------------------------
// Implementation of map.
// -----------------------------------------------------------------------------

struct NativeMapper<'a, D, const PRE_ORDER: bool> {
    dispatch: &'a mut D,
}

fn run_native_mapper<D: MapDispatch>(root: Any, dispatch: &mut D, order: WalkOrder) -> Result<Any> {
    match order {
        WalkOrder::PreOrder => StructuralMutateEngine::run_mutator::<_, false>(
            root,
            &mut NativeMapper::<D, true> { dispatch },
        ),
        WalkOrder::PostOrder => StructuralMutateEngine::run_mutator::<_, false>(
            root,
            &mut NativeMapper::<D, false> { dispatch },
        ),
    }
}

impl<'v, 'c, 'r, 'a, D: MapDispatch, const PRE_ORDER: bool>
    Dispatcher<
        (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, NativeMapper<'a, D, PRE_ORDER>>,
        ),
        Option<Result<Any>>,
    > for &mut NativeMapper<'a, D, PRE_ORDER>
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, mutator): (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, NativeMapper<'a, D, PRE_ORDER>>,
        ),
    ) -> Option<Result<Any>> {
        Some(self.map_current(value, mutator))
    }
}

impl<D: MapDispatch, const PRE_ORDER: bool> NativeMapper<'_, D, PRE_ORDER> {
    #[inline(always)]
    fn map_current(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        let mode = value.inplace_mode();
        let raw = raw_of(value.as_any_view());
        let kind = mutator.def_region_kind();
        if PRE_ORDER {
            // Snapshot permission before a callback can retain or release ownership.
            let Some(callback_result) = self
                .dispatch
                .dispatch_map(unsafe { borrowed_view(&raw) }, kind)
            else {
                return mutator.default_mutate(self, value).map(Into::into);
            };
            let mapped = callback_result.map_err(|error| with_value_context(error, raw))?;
            if is_unchanged(&mapped) || same_shallow(raw, *mapped.as_raw_ffi_any()) {
                // Drop temporary ownership before default descent checks uniqueness.
                drop(mapped);
                mutator.default_mutate(self, value).map(Into::into)
            } else {
                let replacement = MutationView::new(AnyView::from(&mapped), mode);
                let descended: Any = mutator.default_mutate(self, replacement)?.into();
                Ok(if is_unchanged(&descended) {
                    mapped
                } else {
                    descended
                })
            }
        } else {
            let mapped: Any = mutator.default_mutate(self, value)?.into();
            let current = if is_unchanged(&mapped) {
                unsafe { borrowed_view(&raw) }
            } else {
                AnyView::from(&mapped)
            };
            match self.dispatch.dispatch_map(current, kind) {
                Some(Ok(result)) if is_unchanged(&result) => Ok(mapped),
                Some(result) => result.map_err(|error| with_value_context(error, raw_of(current))),
                None => Ok(mapped),
            }
        }
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
///
/// # Arguments
///
/// * `root` - Owned value to transform, convertible into an [`Any`]. Clone it
///   first to preserve the original through copy-on-write behavior.
/// * `mapper` - Mutable dispatcher, closure, or first-match callback tuple
///   implementing [`IntoMapper`]. Callbacks return replacements; unmatched
///   values retain their value while default traversal maps their children.
/// * `order` - [`WalkOrder::PreOrder`] applies the callback before traversing the
///   replacement's children; [`WalkOrder::PostOrder`] maps children before
///   applying the callback to their parent.
///
/// # Examples
///
/// Increment integer leaves while keeping a shared original unchanged:
///
/// ```
/// use tvm_ffi::{structural_map, Array, WalkOrder};
///
/// let root = Array::new(vec![1_i64, 2, 3]);
/// let mapped = structural_map(root.clone(), |value: i64| value + 1, WalkOrder::PostOrder)?;
/// let mapped = Array::<i64>::try_from(mapped)?;
/// assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![2, 3, 4]);
/// assert_eq!(root.iter().collect::<Vec<_>>(), vec![1, 2, 3]);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub fn structural_map<R, M, H>(root: R, mapper: H, order: WalkOrder) -> Result<Any>
where
    R: Into<Any>,
    H: IntoMapper<M>,
{
    mapper.into_mapper().map_root(root.into(), order)
}
