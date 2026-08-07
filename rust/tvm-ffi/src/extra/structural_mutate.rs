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

//! Native Rust structural mapping.
//!
//! [`structural_map`] mirrors the callback-controlled C++ `StructuralMap`,
//! but keeps recursion, typed dispatch, definition-region propagation, and
//! identity remapping in Rust.  The root is consumed so Rust ownership and
//! the runtime strong count jointly define the boundary for optional
//! in-place container mutation.  Passing a clone naturally selects
//! copy-on-write behavior.
//!
//! Container hooks are implemented natively.  A non-container object with a
//! foreign `__s_mutate__` or `__s_maybe_inplace_mutate__` hook is rejected
//! rather than silently replacing its custom semantics with reflection.

use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

use crate::any::{Any, AnyView};
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::{self, ObjectCore};
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashIgnore, kTVMFFIFieldFlagBitSetterIsFunctionObj,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIFieldInfo, TVMFFIFieldSetter, TVMFFIFunctionCall, TVMFFIGetTypeInfo,
    TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex,
};
use crate::tvm_ffi_sys::{TVMFFIObjectHandle, TVMFFISEqHashKind};

use super::structural_visit::{
    field_def_region, for_each_field, free_var_child_region, map_layout_usable, type_attr_column,
    type_key_of, DefRegionKind, MapCursor, MapPrefix, SeqPrefix, TypeAttrColumn, VisitValue,
    WalkOrder,
};

const STRUCTURAL_MUTATE_ATTR: &str = "__s_mutate__";
const STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR: &str = "__s_maybe_inplace_mutate__";
const SHALLOW_COPY_ATTR: &str = "__ffi_shallow_copy__";
const MAP_SHALLOW_COPY_GLOBAL: &str = "ffi.MapShallowCopy";
const DICT_SHALLOW_COPY_GLOBAL: &str = "ffi.DictShallowCopy";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SETTER_IS_FUNCTION: i64 = kTVMFFIFieldFlagBitSetterIsFunctionObj as i64;
const MAP_SMALL_TAG: u64 = 1 << 63;
const MAP_INVALID_INDEX: u64 = u64::MAX;
const MAP_BLOCK_CAP: u64 = 16;
const MAP_ITEM_SIZE: usize = 48;
const MAP_BLOCK_SIZE: usize = 16 + 16 * MAP_ITEM_SIZE;
const MAP_ITEM_VALUE_OFFSET: usize = 16;
const MAP_ITEM_NEXT_OFFSET: usize = 40;
const MAP_EMPTY_SLOT: u8 = 0xff;
const MAP_PROTECTED_SLOT: u8 = 0xfe;

/// Borrowed value passed to structural-map callbacks.
///
/// This is the same representation used by structural-visit callbacks; the
/// alias keeps typed casts and borrowed node checks on one audited unsafe
/// implementation.
pub type MapValue = VisitValue;

/// Result type produced by a structural-map callback.
#[doc(hidden)]
pub type MapResult = Result<Any>;

/// Convert an infallible or fallible callback result into [`MapResult`].
pub trait IntoMapResult {
    fn into_map_result(self) -> MapResult;
}

impl IntoMapResult for Any {
    #[inline]
    fn into_map_result(self) -> MapResult {
        Ok(self)
    }
}

impl IntoMapResult for Result<Any> {
    #[inline]
    fn into_map_result(self) -> MapResult {
        self
    }
}

/// Ordered typed replacement dispatch for [`structural_map`].
///
/// `None` means no handler matched and preserves the current value.  A
/// generated `#[dispatch(map)]` implementation tests `map_*` methods in
/// source order and returns the first match.
pub trait MapDispatch: Sized {
    fn dispatch_map(
        &mut self,
        value: &MapValue,
        def_region_kind: DefRegionKind,
    ) -> Option<MapResult>;
}

impl<V: MapDispatch> MapDispatch for &mut V {
    #[inline]
    fn dispatch_map(
        &mut self,
        value: &MapValue,
        def_region_kind: DefRegionKind,
    ) -> Option<MapResult> {
        (**self).dispatch_map(value, def_region_kind)
    }
}

/// Conversion into the mapper consumed by [`structural_map`].
#[diagnostic::on_unimplemented(
    message = "unsupported structural-map callback shape",
    label = "this value cannot be used as a structural mapper",
    note = "pass `&mut` a type implementing `MapDispatch`, a supported closure, or a tuple of callbacks"
)]
pub trait IntoMapper<Marker> {
    type Mapper: MapDispatch;
    fn into_mapper(self) -> Self::Mapper;
}

#[doc(hidden)]
pub enum ByMapDispatch {}

impl<'a, V: MapDispatch> IntoMapper<ByMapDispatch> for &'a mut V {
    type Mapper = &'a mut V;

    #[inline]
    fn into_mapper(self) -> Self::Mapper {
        self
    }
}

/// One typed callback in a structural-map tuple.
///
/// Links are tried in tuple order and the first matching link supplies the
/// replacement.  Supported shapes are owned FFI values, borrowed object
/// nodes, and `&MapValue`, each optionally followed by [`DefRegionKind`].
/// Numeric links match the complete FFI `Int` or `Float` type tag and then use
/// Rust `as` conversion semantics, so prefer `i64` and `f64` unless narrowing
/// is intentional.
pub trait MapChainLink<Marker>: sealed_map::SealedMapLink<Marker> {
    #[doc(hidden)]
    fn try_map(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Option<MapResult>;
}

mod sealed_map {
    use super::{DefRegionKind, IntoMapResult, MapDispatch, MapValue, ObjectCore};

    pub trait SealedMapLink<Marker> {}

    impl<F, T, O> SealedMapLink<super::ByMapOwned<T>> for F
    where
        F: FnMut(T) -> O,
        O: IntoMapResult,
    {
    }

    impl<F, T, O> SealedMapLink<super::ByMapOwnedKind<T>> for F
    where
        F: FnMut(T, DefRegionKind) -> O,
        O: IntoMapResult,
    {
    }

    impl<F, N: ObjectCore, O> SealedMapLink<super::ByMapNode<N>> for F
    where
        F: for<'a> FnMut(&'a N) -> O,
        O: IntoMapResult,
    {
    }

    impl<F, N: ObjectCore, O> SealedMapLink<super::ByMapNodeKind<N>> for F
    where
        F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
        O: IntoMapResult,
    {
    }

    impl<F, O> SealedMapLink<super::ByMapCatchAll> for F
    where
        F: for<'a> FnMut(&'a MapValue) -> O,
        O: IntoMapResult,
    {
    }

    impl<F, O> SealedMapLink<super::ByMapCatchAllKind> for F
    where
        F: for<'a> FnMut(&'a MapValue, DefRegionKind) -> O,
        O: IntoMapResult,
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
    O: IntoMapResult,
{
    #[inline]
    fn try_map(&mut self, value: &MapValue, _def_region_kind: DefRegionKind) -> Option<MapResult> {
        value.cast::<T>().map(|typed| self(typed).into_map_result())
    }
}

#[doc(hidden)]
pub struct ByMapOwnedKind<T>(PhantomData<T>);

impl<F, T, O> MapChainLink<ByMapOwnedKind<T>> for F
where
    F: FnMut(T, DefRegionKind) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoMapResult,
{
    #[inline]
    fn try_map(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Option<MapResult> {
        value
            .cast::<T>()
            .map(|typed| self(typed, def_region_kind).into_map_result())
    }
}

#[doc(hidden)]
pub struct ByMapNode<N>(PhantomData<N>);

impl<F, N, O> MapChainLink<ByMapNode<N>> for F
where
    F: for<'a> FnMut(&'a N) -> O,
    N: ObjectCore,
    O: IntoMapResult,
{
    #[inline]
    fn try_map(&mut self, value: &MapValue, _def_region_kind: DefRegionKind) -> Option<MapResult> {
        value
            .as_node::<N>()
            .map(|node| self(node).into_map_result())
    }
}

#[doc(hidden)]
pub struct ByMapNodeKind<N>(PhantomData<N>);

impl<F, N, O> MapChainLink<ByMapNodeKind<N>> for F
where
    F: for<'a> FnMut(&'a N, DefRegionKind) -> O,
    N: ObjectCore,
    O: IntoMapResult,
{
    #[inline]
    fn try_map(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Option<MapResult> {
        value
            .as_node::<N>()
            .map(|node| self(node, def_region_kind).into_map_result())
    }
}

#[doc(hidden)]
pub enum ByMapCatchAll {}

impl<F, O> MapChainLink<ByMapCatchAll> for F
where
    F: for<'a> FnMut(&'a MapValue) -> O,
    O: IntoMapResult,
{
    #[inline]
    fn try_map(&mut self, value: &MapValue, _def_region_kind: DefRegionKind) -> Option<MapResult> {
        Some(self(value).into_map_result())
    }
}

#[doc(hidden)]
pub enum ByMapCatchAllKind {}

impl<F, O> MapChainLink<ByMapCatchAllKind> for F
where
    F: for<'a> FnMut(&'a MapValue, DefRegionKind) -> O,
    O: IntoMapResult,
{
    #[inline]
    fn try_map(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Option<MapResult> {
        Some(self(value, def_region_kind).into_map_result())
    }
}

#[doc(hidden)]
pub enum ByMapDispatchLink {}

impl<V: MapDispatch> MapChainLink<ByMapDispatchLink> for &mut V {
    #[inline]
    fn try_map(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Option<MapResult> {
        self.dispatch_map(value, def_region_kind)
    }
}

/// Statically dispatched tuple mapper, public only as an [`IntoMapper`]
/// projection.
#[doc(hidden)]
pub struct MapChain<Links, Markers> {
    links: Links,
    markers: PhantomData<fn(Markers)>,
}

macro_rules! impl_map_chain {
    ($(($F:ident, $M:ident, $idx:tt)),+) => {
        impl<$($F, $M,)+> MapDispatch for MapChain<($($F,)+), ($($M,)+)>
        where
            $($F: MapChainLink<$M>,)+
        {
            #[inline]
            fn dispatch_map(
                &mut self,
                value: &MapValue,
                def_region_kind: DefRegionKind,
            ) -> Option<MapResult> {
                $(
                    if let Some(result) = self.links.$idx.try_map(value, def_region_kind) {
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
            type Mapper = MapChain<($($F,)+), ($($M,)+)>;

            #[inline]
            fn into_mapper(self) -> Self::Mapper {
                MapChain {
                    links: self,
                    markers: PhantomData,
                }
            }
        }
    };
}

impl_map_chain!((F0, M0, 0));
impl_map_chain!((F0, M0, 0), (F1, M1, 1));
impl_map_chain!((F0, M0, 0), (F1, M1, 1), (F2, M2, 2));
impl_map_chain!((F0, M0, 0), (F1, M1, 1), (F2, M2, 2), (F3, M3, 3));
impl_map_chain!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4)
);
impl_map_chain!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4),
    (F5, M5, 5)
);
impl_map_chain!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4),
    (F5, M5, 5),
    (F6, M6, 6)
);
impl_map_chain!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4),
    (F5, M5, 5),
    (F6, M6, 6),
    (F7, M7, 7)
);

macro_rules! impl_bare_map_link {
    ($(($marker:ident, $($fn_args:ty),+)),+ $(,)?) => {
        $(
            impl<F, T, O> IntoMapper<$marker<T>> for F
            where
                F: FnMut($($fn_args),+) -> O,
                Self: MapChainLink<$marker<T>>,
                O: IntoMapResult,
            {
                type Mapper = MapChain<(F,), ($marker<T>,)>;

                #[inline]
                fn into_mapper(self) -> Self::Mapper {
                    MapChain {
                        links: (self,),
                        markers: PhantomData,
                    }
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
    F: for<'a> FnMut(&'a MapValue) -> O,
    O: IntoMapResult,
{
    type Mapper = MapChain<(F,), (ByMapCatchAll,)>;

    #[inline]
    fn into_mapper(self) -> Self::Mapper {
        MapChain {
            links: (self,),
            markers: PhantomData,
        }
    }
}

impl<F, O> IntoMapper<ByMapCatchAllKind> for F
where
    F: for<'a> FnMut(&'a MapValue, DefRegionKind) -> O,
    O: IntoMapResult,
{
    type Mapper = MapChain<(F,), (ByMapCatchAllKind,)>;

    #[inline]
    fn into_mapper(self) -> Self::Mapper {
        MapChain {
            links: (self,),
            markers: PhantomData,
        }
    }
}

/// Engine-issued permission to attempt in-place mutation of one value.
///
/// This capability cannot be constructed by callers. The native recursion
/// engine issues it only when the parent path permits mutation and the object
/// is uniquely owned at dispatch time. It is deliberately distinct from
/// [`MapValue`], which can also be obtained from a read-only structural walk.
/// Implementations should normally inspect it and then call
/// [`StructuralMutator::default_maybe_inplace_mutate`].
pub struct InplaceValue<'a> {
    value: MapValue,
    _scope: PhantomData<&'a mut TVMFFIAny>,
}

impl<'a> InplaceValue<'a> {
    #[inline]
    fn from_raw(raw: &'a mut TVMFFIAny) -> Self {
        Self {
            value: MapValue::from_raw(*raw),
            _scope: PhantomData,
        }
    }

    /// Borrow the value without its in-place capability.
    #[inline]
    pub fn as_value(&self) -> &MapValue {
        &self.value
    }

    /// Retain an owning copy of the value.
    ///
    /// Retaining an object creates an alias. The default in-place helper
    /// rechecks uniqueness and automatically falls back to copying.
    #[inline]
    pub fn to_owned(&self) -> Any {
        self.value.to_owned()
    }
}

impl Deref for InplaceValue<'_> {
    type Target = MapValue;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_value()
    }
}

/// Stateful identity-substitution environment for a hand-written
/// [`StructuralMutator`].
///
/// The map owns both its object keys and mapped values, preventing an object
/// address from being recycled while a mutation is in progress. A mutator
/// can delegate its required `var_remap_get` and `var_remap_set` methods to
/// this type.
#[derive(Default)]
pub struct StructuralVarRemap {
    entries: HashMap<NonNull<TVMFFIObject>, MemoEntry>,
}

impl StructuralVarRemap {
    /// Look up an identity replacement previously stored for `var`.
    pub fn get(&self, var: &MapValue) -> Result<Option<Any>> {
        let key = object_identity_key(var.raw())?;
        Ok(self.entries.get(&key).map(|entry| entry.mapped.clone()))
    }

    /// Store the final mapped value for `var`.
    pub fn set(&mut self, var: &MapValue, mapped_value: &Any) -> Result<()> {
        let key = object_identity_key(var.raw())?;
        self.entries.insert(
            key,
            MemoEntry {
                _original: var.to_owned(),
                mapped: mapped_value.clone(),
            },
        );
        Ok(())
    }

    /// Remove every recorded identity substitution.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// User-driven structural mutation, analogous to the low-level C++
/// `StructuralMutatorObj` API.
///
/// [`structural_mutate`] dispatches the root to [`Self::mutate`] or, when
/// ownership permits, [`Self::maybe_inplace_mutate`]. An implementation
/// chooses where to recurse by calling `default_*` for the current value or a
/// child helper for a selected value. This also provides the Rust takeover
/// path for a type whose foreign `__s_mutate__` hook cannot be called by the
/// native engine.
pub trait StructuralMutator: Sized {
    /// Mutate one borrowed value without modifying its source storage.
    fn mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any>;

    /// Mutate one value for which the engine permits an in-place attempt.
    ///
    /// The default delegates to [`Self::mutate`] and therefore remains
    /// non-in-place. Override this method to opt into the default container
    /// reuse path.
    fn maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.mutate(value.as_value(), def_region_kind)
    }

    /// Re-enter this mutator for a borrowed child. The child and all of its
    /// descendants use the non-in-place path.
    fn mutate_child<T>(&mut self, child: &T, def_region_kind: DefRegionKind) -> Result<Any>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(child);
        user_dispatch_raw(self, *view.as_raw_ffi_any(), def_region_kind, Permit::Copy)
    }

    /// Re-enter this mutator for an owned child, permitting reuse only when
    /// the converted value remains uniquely owned.
    fn maybe_inplace_mutate_child<T>(
        &mut self,
        child: T,
        def_region_kind: DefRegionKind,
    ) -> Result<Any>
    where
        T: Into<Any>,
    {
        let child = child.into();
        user_dispatch_raw(
            self,
            *child.as_raw_ffi_any(),
            def_region_kind,
            Permit::MaybeInPlace,
        )
    }

    /// Apply default non-in-place mutation to `value`'s children.
    fn default_mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        user_default_mutate(self, value.raw(), def_region_kind, Permit::Copy)
    }

    /// Apply the default mutation under an engine-issued in-place capability.
    ///
    /// Uniqueness is checked again here because user code may have retained
    /// an owning alias after the capability was issued.
    fn default_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        let raw = value.raw();
        let permit = if object_is_unique(raw) {
            Permit::MaybeInPlace
        } else {
            Permit::Copy
        };
        user_default_mutate(self, raw, def_region_kind, permit)
    }

    /// Look up a previously completed FreeVar identity substitution.
    fn var_remap_get(&mut self, var: &MapValue) -> Result<Option<Any>>;

    /// Store the final mapped result for a FreeVar identity.
    fn var_remap_set(&mut self, var: &MapValue, mapped_value: &Any) -> Result<()>;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Permit {
    Copy,
    MaybeInPlace,
}

struct MemoEntry {
    // Keeps the pointer-valued key alive so its address cannot be reused
    // during the same mapping invocation.
    _original: Any,
    mapped: Any,
}

struct NativeMapper<D> {
    dispatch: D,
    order: WalkOrder,
    memo: HashMap<NonNull<TVMFFIObject>, MemoEntry>,
}

impl<D: MapDispatch> NativeMapper<D> {
    fn map_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return owned_from_raw(raw);
        }

        let identity = identity_key(raw)?;
        if let Some(key) = identity {
            if let Some(entry) = self.memo.get(&key) {
                return Ok(entry.mapped.clone());
            }
        }

        // Identity nodes need an owning key for the complete invocation.  The
        // extra owner intentionally disables mutation of the original
        // identity node; a distinct callback replacement may still be unique.
        let original = identity.map(|_| owned_from_raw(raw)).transpose()?;
        let effective_permit = if identity.is_some() {
            Permit::Copy
        } else {
            permit
        };
        let result = self
            .map_uncached_raw(raw, def_region_kind, effective_permit)
            .map_err(|error| with_value_context(error, raw))?;

        if let (Some(key), Some(original)) = (identity, original) {
            self.memo.insert(
                key,
                MemoEntry {
                    _original: original,
                    mapped: result.clone(),
                },
            );
        }
        Ok(result)
    }

    fn map_uncached_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        match self.order {
            WalkOrder::PreOrder => {
                let value = MapValue::from_raw(raw);
                let Some(callback_result) = self.dispatch.dispatch_map(&value, def_region_kind)
                else {
                    return self.default_map_current_raw(raw, def_region_kind, permit);
                };
                let mapped = callback_result?;
                let mapped_raw = *mapped.as_raw_ffi_any();
                if same_shallow(raw, mapped_raw) {
                    // Release the callback's temporary ownership before the
                    // runtime uniqueness check observes the original.
                    drop(mapped);
                    self.default_map_current_raw(raw, def_region_kind, permit)
                } else {
                    self.map_default_root_raw(
                        mapped_raw,
                        def_region_kind,
                        Permit::MaybeInPlace,
                        &mapped,
                    )
                }
            }
            WalkOrder::PostOrder => {
                let mapped = self.default_map_current_raw(raw, def_region_kind, permit)?;
                let mapped_raw = *mapped.as_raw_ffi_any();
                let value = MapValue::from_raw(mapped_raw);
                match self.dispatch.dispatch_map(&value, def_region_kind) {
                    Some(result) => result,
                    None => Ok(mapped),
                }
            }
        }
    }

    /// Map a pre-order callback replacement without invoking a callback for
    /// the replacement root.  Its children still enter the full map engine,
    /// and an identity replacement is memoized with its final default result.
    fn map_default_root_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
        _owner: &Any,
    ) -> Result<Any> {
        let identity = identity_key(raw)?;
        if let Some(key) = identity {
            if let Some(entry) = self.memo.get(&key) {
                return Ok(entry.mapped.clone());
            }
        }
        let original = identity.map(|_| owned_from_raw(raw)).transpose()?;
        let effective_permit = if identity.is_some() {
            Permit::Copy
        } else {
            permit
        };
        let result = self
            .default_map_current_raw(raw, def_region_kind, effective_permit)
            .map_err(|error| with_value_context(error, raw))?;
        if let (Some(key), Some(original)) = (identity, original) {
            self.memo.insert(
                key,
                MemoEntry {
                    _original: original,
                    mapped: result.clone(),
                },
            );
        }
        Ok(result)
    }
}

/// Shared default-recursion implementation. Its only customization point is
/// how a child re-enters the high-level map or a user-driven mutator.
trait DefaultMutationDriver: Sized {
    fn recurse_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any>;

    fn default_map_current_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        match raw.type_index {
            x if x == TVMFFITypeIndex::kTVMFFIArray as i32
                || x == TVMFFITypeIndex::kTVMFFIList as i32 =>
            {
                return self.map_sequence(raw, def_region_kind, permit);
            }
            x if x == TVMFFITypeIndex::kTVMFFIMap as i32
                || x == TVMFFITypeIndex::kTVMFFIDict as i32 =>
            {
                return self.map_values(raw, def_region_kind, permit);
            }
            _ => {}
        }

        reject_foreign_structural_mutation(raw.type_index)?;
        if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            owned_from_raw(raw)
        } else {
            self.map_reflected(raw, def_region_kind)
        }
    }

    fn map_sequence(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        let seq = checked_sequence(raw)?;
        let size = seq.size;
        if size == 0 {
            return owned_from_raw(raw);
        }

        if permit == Permit::MaybeInPlace && object_is_unique(raw) {
            // The consumed root or a unique parent exclusively owns the
            // Array/List storage. Re-checking after the parent callback
            // prevents a callback-retained alias from observing raw writes.
            let cells = seq.data;
            for index in 0..size {
                let cell = unsafe { cells.add(index) };
                let old_raw = unsafe { *cell };
                let mapped = self
                    .recurse_raw(old_raw, def_region_kind, Permit::MaybeInPlace)
                    .map_err(|error| {
                        with_error_context(error, &format!("sequence item [{index}]"))
                    })?;
                let mapped_raw = *mapped.as_raw_ffi_any();
                if !same_shallow(old_raw, mapped_raw) {
                    unsafe { replace_owned_cell(cell, mapped) };
                }
            }
            return owned_from_raw(raw);
        }

        // List is mutable, so snapshot every cell before the first callback.
        // Array cells are stable, but using the same owned vector keeps the
        // copy path simple and prevents source children from being mutated.
        let mut output = Vec::with_capacity(size);
        for child_raw in unsafe { std::slice::from_raw_parts(seq.data, size) } {
            output.push(owned_from_raw(*child_raw)?);
        }
        let mut changed = false;
        for (index, child) in output.iter_mut().enumerate() {
            let child_raw = *child.as_raw_ffi_any();
            let mapped = self
                .recurse_raw(child_raw, def_region_kind, Permit::Copy)
                .map_err(|error| with_error_context(error, &format!("sequence item [{index}]")))?;
            if !same_shallow(child_raw, *mapped.as_raw_ffi_any()) {
                *child = mapped;
                changed = true;
            }
        }
        if !changed {
            return owned_from_raw(raw);
        }
        construct_sequence(raw.type_index, &output)
    }

    fn map_values(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        if permit == Permit::MaybeInPlace && object_is_unique(raw) && map_layout_usable(raw) {
            if let Some((mut cursor, size)) = checked_map_value_cursor(raw) {
                let kind = if raw.type_index == TVMFFITypeIndex::kTVMFFIDict as i32 {
                    "dict"
                } else {
                    "map"
                };
                for index in 0..size {
                    // The complete cursor was validated before the first
                    // callback, so no partial write can precede discovering
                    // an unusable private layout.
                    let Some(cell) = (unsafe { cursor.next_value_slot() }) else {
                        return Err(runtime_error(
                            "native structural map: validated map iteration ended early",
                        ));
                    };
                    let old_raw = unsafe { *cell };
                    let mapped = self
                        .recurse_raw(old_raw, def_region_kind, Permit::MaybeInPlace)
                        .map_err(|error| {
                            with_error_context(error, &format!("{kind} value [{index}]"))
                        })?;
                    if !same_shallow(old_raw, *mapped.as_raw_ffi_any()) {
                        unsafe { replace_owned_cell(cell, mapped) };
                    }
                }
                return owned_from_raw(raw);
            }
        }

        // A runtime shallow-copy helper owns the non-trivial Map/Dict object
        // allocation and deleter contract. Rust still owns recursion and
        // updates only value cells through the fully validated cursor above.
        // The helper is optional so a newer crate remains usable with an
        // older runtime, falling back to the canonical constructors below.
        if map_layout_usable(raw) {
            if let (Some(helper), Some((cursor, size))) = (
                map_shallow_copy_helper(raw.type_index),
                checked_map_value_cursor(raw),
            ) {
                if size == 0 {
                    return owned_from_raw(raw);
                }
                if raw.type_index == TVMFFITypeIndex::kTVMFFIDict as i32 {
                    return self.map_shared_dict_with_shallow_copy(
                        raw,
                        def_region_kind,
                        size,
                        helper,
                    );
                }
                return self.map_shared_map_with_shallow_copy(
                    raw,
                    def_region_kind,
                    cursor,
                    size,
                    helper,
                );
            }
        }

        let mut entries = snapshot_map_entries(raw)?;
        let mut changed = false;
        let kind = if raw.type_index == TVMFFITypeIndex::kTVMFFIDict as i32 {
            "dict"
        } else {
            "map"
        };
        for (index, (_, old_value)) in entries.iter_mut().enumerate() {
            let old_raw = *old_value.as_raw_ffi_any();
            let mapped = self
                .recurse_raw(old_raw, def_region_kind, Permit::Copy)
                .map_err(|error| with_error_context(error, &format!("{kind} value [{index}]")))?;
            if !same_shallow(old_raw, *mapped.as_raw_ffi_any()) {
                *old_value = mapped;
                changed = true;
            }
        }
        if !changed {
            return owned_from_raw(raw);
        }
        construct_map(raw.type_index, &entries)
    }

    /// Map an immutable shared Map and allocate its output lazily at the first
    /// changed value. Map storage cannot be invalidated by a callback, so the
    /// source cursor remains stable for the complete traversal.
    fn map_shared_map_with_shallow_copy(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        mut source_cursor: MapValueCursor,
        size: usize,
        helper: &Function,
    ) -> Result<Any> {
        for index in 0..size {
            let Some(source_cell) = (unsafe { source_cursor.next_value_slot() }) else {
                return Err(runtime_error(
                    "native structural map: validated map iteration ended early",
                ));
            };
            let old_raw = unsafe { *source_cell };
            let mapped = self
                .recurse_raw(old_raw, def_region_kind, Permit::Copy)
                .map_err(|error| with_error_context(error, &format!("map value [{index}]")))?;
            if same_shallow(old_raw, *mapped.as_raw_ffi_any()) {
                continue;
            }

            let output = call_map_shallow_copy(raw, helper, "map")?;
            let output_raw = *output.as_raw_ffi_any();
            let Some((mut output_cursor, copied_size)) = runtime_copy_map_value_cursor(output_raw)
            else {
                return Err(runtime_error(
                    "native structural map: shallow-copied map has an unusable layout",
                ));
            };
            if copied_size != size {
                return Err(runtime_error(
                    "native structural map: shallow-copied map changed size",
                ));
            }
            // Values before the first change remain untouched, but the
            // output cursor must be aligned with the current source item.
            for _ in 0..index {
                if unsafe { output_cursor.next_value_slot() }.is_none() {
                    return Err(runtime_error(
                        "native structural map: shallow-copied map iteration ended early",
                    ));
                }
            }
            let Some(output_cell) = (unsafe { output_cursor.next_value_slot() }) else {
                return Err(runtime_error(
                    "native structural map: shallow-copied map iteration ended early",
                ));
            };
            unsafe { replace_owned_cell(output_cell, mapped) };

            // Once the output exists, advance source and output cursors in
            // lockstep without checking an Option on every remaining item.
            for remaining_index in index + 1..size {
                let Some(source_cell) = (unsafe { source_cursor.next_value_slot() }) else {
                    return Err(runtime_error(
                        "native structural map: validated map iteration ended early",
                    ));
                };
                let old_raw = unsafe { *source_cell };
                let mapped = self
                    .recurse_raw(old_raw, def_region_kind, Permit::Copy)
                    .map_err(|error| {
                        with_error_context(error, &format!("map value [{remaining_index}]"))
                    })?;
                let Some(output_cell) = (unsafe { output_cursor.next_value_slot() }) else {
                    return Err(runtime_error(
                        "native structural map: shallow-copied map iteration ended early",
                    ));
                };
                if !same_shallow(old_raw, *mapped.as_raw_ffi_any()) {
                    unsafe { replace_owned_cell(output_cell, mapped) };
                }
            }
            if !output_cursor.is_complete() {
                return Err(runtime_error(
                    "native structural map: shallow-copied map iteration did not terminate",
                ));
            }
            return Ok(output);
        }
        owned_from_raw(raw)
    }

    /// Map a mutable shared Dict through a private shallow copy made before
    /// the first callback. The copy is the stable snapshot: a callback may
    /// re-enter and mutate an alias of the source Dict without invalidating
    /// this cursor or changing the output's initial key set.
    fn map_shared_dict_with_shallow_copy(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        size: usize,
        helper: &Function,
    ) -> Result<Any> {
        let output = call_map_shallow_copy(raw, helper, "dict")?;
        let output_raw = *output.as_raw_ffi_any();
        let Some((mut cursor, copied_size)) = runtime_copy_map_value_cursor(output_raw) else {
            return Err(runtime_error(
                "native structural map: shallow-copied dict has an unusable layout",
            ));
        };
        if copied_size != size {
            return Err(runtime_error(
                "native structural map: shallow-copied dict changed size",
            ));
        }

        let mut changed = false;
        for index in 0..size {
            let Some(cell) = (unsafe { cursor.next_value_slot() }) else {
                return Err(runtime_error(
                    "native structural map: shallow-copied dict iteration ended early",
                ));
            };
            let old_raw = unsafe { *cell };
            let mapped = self
                .recurse_raw(old_raw, def_region_kind, Permit::Copy)
                .map_err(|error| with_error_context(error, &format!("dict value [{index}]")))?;
            if !same_shallow(old_raw, *mapped.as_raw_ffi_any()) {
                unsafe { replace_owned_cell(cell, mapped) };
                changed = true;
            }
        }
        if !cursor.is_complete() {
            return Err(runtime_error(
                "native structural map: shallow-copied dict iteration did not terminate",
            ));
        }
        if changed {
            Ok(output)
        } else {
            owned_from_raw(raw)
        }
    }

    fn map_reflected(&mut self, raw: TVMFFIAny, def_region_kind: DefRegionKind) -> Result<Any> {
        let type_info = checked_type_info(raw.type_index)?;
        let seq_hash_kind = unsafe {
            if (*type_info).metadata.is_null() {
                TVMFFISEqHashKind::kTVMFFISEqHashKindUnsupported as i32
            } else {
                (*(*type_info).metadata).structural_eq_hash_kind
            }
        };
        let inherited_region = free_var_child_region(def_region_kind, seq_hash_kind);
        // Match the C++ reflected-mutation contract: resolve and invoke the
        // shallow-copy hook before inspecting any fields. Besides providing
        // isolated setter storage, this means a missing or failing hook is an
        // error even when no field eventually changes.
        let output = shallow_copy(raw)?;
        let output_raw = *output.as_raw_ffi_any();
        let output_object = unsafe { output_raw.data_union.v_obj.cast::<u8>() };
        if output_object.is_null() {
            return Err(runtime_error(
                "native structural map: shallow copy has a null object pointer",
            ));
        }

        let mut field_changed = false;
        let mut failure: Option<Error> = None;
        unsafe {
            for_each_field(raw.type_index, |field| {
                if field.flags & FLAG_SEQ_HASH_IGNORE != 0 {
                    return ControlFlow::Continue(());
                }
                match self.map_reflected_field(
                    output_object,
                    field,
                    inherited_region,
                    &mut field_changed,
                ) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(error) => {
                        failure = Some(error);
                        ControlFlow::Break(())
                    }
                }
            });
        }
        if let Some(error) = failure {
            return Err(error);
        }
        if field_changed {
            Ok(output)
        } else {
            owned_from_raw(raw)
        }
    }

    unsafe fn map_reflected_field(
        &mut self,
        output_object: *mut u8,
        field: &TVMFFIFieldInfo,
        inherited_region: DefRegionKind,
        field_changed: &mut bool,
    ) -> Result<()> {
        let Some(getter) = field.getter else {
            return Err(runtime_error(&format!(
                "native structural map: reflected field `{}` has no getter",
                field.name.as_str()
            )));
        };
        // Read every field from the copy so earlier setters' side effects are
        // visible to later field mappings, exactly as in the C++ fallback.
        let source_address = output_object.offset(field.offset as isize).cast::<c_void>();
        // Keep the output slot owned before entering foreign code.  A getter
        // may populate an owning result and still report an error; `Any`
        // then releases that partial result on every return path.
        let mut child = Any::new();
        if getter(source_address, Any::as_data_ptr(&mut child)) != 0 {
            return Err(with_error_context(
                Error::from_raised(),
                &format!("field `{}`", field.name.as_str()),
            ));
        }
        // Reflection getters return owning values. Keep the child alive for
        // the complete recursive call, then let normal Drop release it.
        let child_raw = *child.as_raw_ffi_any();
        let child_region = field_def_region(field, inherited_region);
        let mapped = self
            .recurse_raw(child_raw, child_region, Permit::Copy)
            .map_err(|error| {
                with_error_context(error, &format!("field `{}`", field.name.as_str()))
            })?;
        if same_shallow(child_raw, *mapped.as_raw_ffi_any()) {
            return Ok(());
        }

        call_field_setter(field, source_address, mapped.as_raw_ffi_any()).map_err(|error| {
            with_error_context(error, &format!("field `{}`", field.name.as_str()))
        })?;
        *field_changed = true;
        Ok(())
    }
}

impl<D: MapDispatch> DefaultMutationDriver for NativeMapper<D> {
    #[inline]
    fn recurse_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        self.map_raw(raw, def_region_kind, permit)
    }
}

struct UserDriver<'a, U> {
    mutator: &'a mut U,
}

impl<U: StructuralMutator> DefaultMutationDriver for UserDriver<'_, U> {
    #[inline]
    fn recurse_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        user_dispatch_raw(self.mutator, raw, def_region_kind, permit)
    }
}

fn user_dispatch_raw<U: StructuralMutator>(
    mutator: &mut U,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return owned_from_raw(raw);
    }
    let result = if permit == Permit::MaybeInPlace && object_is_unique(raw) {
        // Tie the capability lifetime to this dispatch frame, so safe user
        // code cannot retain a borrowed raw view past its owning storage.
        let mut scoped_raw = raw;
        mutator.maybe_inplace_mutate(InplaceValue::from_raw(&mut scoped_raw), def_region_kind)
    } else {
        mutator.mutate(&MapValue::from_raw(raw), def_region_kind)
    };
    result.map_err(|error| with_value_context(error, raw))
}

fn user_default_mutate<U: StructuralMutator>(
    mutator: &mut U,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    let is_free_var =
        structural_hash_kind(raw)? == Some(TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar as i32);
    if is_free_var {
        let var = MapValue::from_raw(raw);
        if let Some(mapped) = mutator.var_remap_get(&var)? {
            return Ok(mapped);
        }
    }

    let result = {
        let mut driver = UserDriver { mutator };
        driver.default_map_current_raw(raw, def_region_kind, permit)?
    };
    if is_free_var {
        let var = MapValue::from_raw(raw);
        mutator.var_remap_set(&var, &result)?;
    }
    Ok(result)
}

/// Mutate a structured value with a user-driven [`StructuralMutator`].
///
/// The root is consumed to establish the ownership boundary for optional
/// in-place mutation. Completed in-place changes are not rolled back on an
/// error, and an error does not return the consumed root.
pub fn structural_mutate<R, U>(root: R, mutator: &mut U) -> Result<Any>
where
    R: Into<Any>,
    U: StructuralMutator,
{
    let root = root.into();
    user_dispatch_raw(
        mutator,
        *root.as_raw_ffi_any(),
        DefRegionKind::None,
        Permit::MaybeInPlace,
    )
}

/// Transform a structured value graph with ordered replacement callbacks.
///
/// The root is consumed. A uniquely owned Array, List, Map, or Dict may
/// therefore be reused in place, while passing `root.clone()` keeps the
/// original shared and selects copy-on-write behavior. Map and Dict keys are
/// anchors and are not mapped.
///
/// In-place changes completed before an error are not rolled back. Because
/// this function consumes `root`, an error does not return the partly mapped
/// root to the caller.
pub fn structural_map<R, M, H>(root: R, mapper: H, order: WalkOrder) -> Result<Any>
where
    R: Into<Any>,
    H: IntoMapper<M>,
{
    let root = root.into();
    let raw = *root.as_raw_ffi_any();
    let mut native = NativeMapper {
        dispatch: mapper.into_mapper(),
        order,
        memo: HashMap::new(),
    };
    native.map_raw(raw, DefRegionKind::None, Permit::MaybeInPlace)
}

#[derive(Clone, Copy)]
struct CheckedSequence {
    data: *mut TVMFFIAny,
    size: usize,
}

fn checked_sequence(raw: TVMFFIAny) -> Result<CheckedSequence> {
    let pointer = unsafe { raw.data_union.v_obj };
    if pointer.is_null() {
        return Err(runtime_error(
            "native structural map: sequence has a null object pointer",
        ));
    }
    let seq = pointer.cast::<SeqPrefix>();
    let size = unsafe { (*seq).size };
    let capacity = unsafe { (*seq).capacity };
    let data = unsafe { (*seq).data.cast_mut() };
    if size < 0 || capacity < 0 {
        return Err(runtime_error(
            "native structural map: sequence reports a negative size or capacity",
        ));
    }
    if size > capacity {
        return Err(runtime_error(
            "native structural map: sequence size exceeds capacity",
        ));
    }
    if data.is_null() && size != 0 {
        return Err(runtime_error(
            "native structural map: non-empty sequence has a null data pointer",
        ));
    }
    if !data.is_null() && data.align_offset(std::mem::align_of::<TVMFFIAny>()) != 0 {
        return Err(runtime_error(
            "native structural map: sequence data pointer is misaligned",
        ));
    }
    let size = usize::try_from(size)
        .map_err(|_| runtime_error("native structural map: sequence size does not fit usize"))?;
    Ok(CheckedSequence { data, size })
}

#[repr(C)]
struct DenseMapPrefixForMutation {
    _base: MapPrefix,
    _fib_shift: u32,
    iter_list_head: u64,
    _iter_list_tail: u64,
}

const _: () = {
    assert!(std::mem::offset_of!(DenseMapPrefixForMutation, iter_list_head) == 64);
};

enum MapValueCursor {
    Small {
        data: *mut TVMFFIAny,
        index: usize,
        size: usize,
    },
    Dense {
        data: *mut u8,
        index: u64,
        remaining: usize,
        slots: u64,
    },
}

impl MapValueCursor {
    unsafe fn new(map: &MapPrefix) -> Option<Self> {
        let size = usize::try_from(map.size).ok()?;
        if map.slots & MAP_SMALL_TAG != 0 {
            let slots = usize::try_from(map.slots & !MAP_SMALL_TAG).ok()?;
            if size > slots || (size != 0 && map.data.is_null()) {
                return None;
            }
            let data = map.data.cast::<TVMFFIAny>();
            if !data.is_null() && data.align_offset(std::mem::align_of::<TVMFFIAny>()) != 0 {
                return None;
            }
            size.checked_mul(2)?;
            Some(Self::Small {
                data,
                index: 0,
                size,
            })
        } else {
            if size == 0 {
                return Some(Self::Dense {
                    data: map.data,
                    index: MAP_INVALID_INDEX,
                    remaining: 0,
                    slots: map.slots,
                });
            }
            if map.slots == 0
                || !map.slots.is_power_of_two()
                || map.size > map.slots
                || map.data.is_null()
                || map.data.align_offset(std::mem::align_of::<TVMFFIAny>()) != 0
            {
                return None;
            }
            let dense = &*(map as *const MapPrefix as *const DenseMapPrefixForMutation);
            Some(Self::Dense {
                data: map.data,
                index: dense.iter_list_head,
                remaining: size,
                slots: map.slots,
            })
        }
    }

    unsafe fn next_value_slot(&mut self) -> Option<*mut TVMFFIAny> {
        match self {
            Self::Small { data, index, size } => {
                if *index >= *size {
                    return None;
                }
                let slot = data.add(index.checked_mul(2)?.checked_add(1)?);
                *index += 1;
                Some(slot)
            }
            Self::Dense {
                data,
                index,
                remaining,
                slots,
            } => {
                if *remaining == 0 || *index == MAP_INVALID_INDEX || *index >= *slots {
                    return None;
                }
                let block = usize::try_from(*index / MAP_BLOCK_CAP)
                    .ok()?
                    .checked_mul(MAP_BLOCK_SIZE)?;
                let in_block = usize::try_from(*index % MAP_BLOCK_CAP).ok()?;
                let metadata = *data.add(block.checked_add(in_block)?);
                if metadata == MAP_EMPTY_SLOT || metadata == MAP_PROTECTED_SLOT {
                    return None;
                }
                let item = block
                    .checked_add(MAP_BLOCK_CAP as usize)?
                    .checked_add(in_block.checked_mul(MAP_ITEM_SIZE)?)?;
                let value_offset = item.checked_add(MAP_ITEM_VALUE_OFFSET)?;
                let value = data.add(value_offset).cast::<TVMFFIAny>();
                if value.align_offset(std::mem::align_of::<TVMFFIAny>()) != 0 {
                    return None;
                }
                let next = data
                    .add(item.checked_add(MAP_ITEM_NEXT_OFFSET)?)
                    .cast::<u64>();
                if next.align_offset(std::mem::align_of::<u64>()) != 0 {
                    return None;
                }
                *index = *next;
                *remaining -= 1;
                Some(value)
            }
        }
    }

    fn is_complete(&self) -> bool {
        match self {
            Self::Small { index, size, .. } => index == size,
            Self::Dense {
                index, remaining, ..
            } => *remaining == 0 && *index == MAP_INVALID_INDEX,
        }
    }
}

fn checked_map_value_cursor(raw: TVMFFIAny) -> Option<(MapValueCursor, usize)> {
    let pointer = unsafe { raw.data_union.v_obj };
    let map = unsafe { pointer.cast::<MapPrefix>().as_ref()? };
    let size = usize::try_from(map.size).ok()?;
    let mut validation = unsafe { MapValueCursor::new(map)? };
    for _ in 0..size {
        unsafe { validation.next_value_slot()? };
    }
    if !validation.is_complete() {
        return None;
    }
    Some((unsafe { MapValueCursor::new(map)? }, size))
}

/// Create a cursor for a private object returned by the runtime shallow-copy
/// helper. The source layout was fully validated before the helper call, and
/// the helper contract preserves the concrete type, size, and iteration order.
/// The cursor still checks every slot and its final state while traversing;
/// unlike `checked_map_value_cursor`, it avoids a redundant full preflight of
/// an output that has not yet been published.
fn runtime_copy_map_value_cursor(raw: TVMFFIAny) -> Option<(MapValueCursor, usize)> {
    let pointer = unsafe { raw.data_union.v_obj };
    let map = unsafe { pointer.cast::<MapPrefix>().as_ref()? };
    let size = usize::try_from(map.size).ok()?;
    Some((unsafe { MapValueCursor::new(map)? }, size))
}

fn snapshot_map_entries(raw: TVMFFIAny) -> Result<Vec<(Any, Any)>> {
    if map_layout_usable(raw) {
        let pointer = unsafe { raw.data_union.v_obj };
        if pointer.is_null() {
            return Err(runtime_error(
                "native structural map: map has a null object pointer",
            ));
        }
        let map = unsafe { &*(pointer as *const MapPrefix) };
        let size = usize::try_from(map.size)
            .map_err(|_| runtime_error("native structural map: map size does not fit usize"))?;
        let mut cursor = unsafe { MapCursor::new(map) };
        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            let Some((key, value)) = (unsafe { cursor.next() }) else {
                return Err(runtime_error(
                    "native structural map: map iteration ended early",
                ));
            };
            entries.push((owned_from_raw(key)?, owned_from_raw(value)?));
        }
        return Ok(entries);
    }
    snapshot_map_entries_fallback(raw)
}

fn snapshot_map_entries_fallback(raw: TVMFFIAny) -> Result<Vec<(Any, Any)>> {
    let is_dict = raw.type_index == TVMFFITypeIndex::kTVMFFIDict as i32;
    let (size_name, iter_name) = if is_dict {
        ("ffi.DictSize", "ffi.DictForwardIterFunctor")
    } else {
        ("ffi.MapSize", "ffi.MapForwardIterFunctor")
    };
    let owner = owned_from_raw(raw)?;
    let size = Function::get_global(size_name)?
        .call_packed(&[AnyView::from(&owner)])
        .and_then(i64::try_from)?;
    if size < 0 {
        return Err(runtime_error(
            "native structural map: map reports a negative size",
        ));
    }
    let size = usize::try_from(size)
        .map_err(|_| runtime_error("native structural map: map size does not fit usize"))?;
    let iter_any = Function::get_global(iter_name)?.call_packed(&[AnyView::from(&owner)])?;
    let iter = Function::try_from(iter_any)?;
    let mut entries = Vec::with_capacity(size);
    for index in 0..size {
        let key = iter.call_packed(&[AnyView::from(&0i64)])?;
        let value = iter.call_packed(&[AnyView::from(&1i64)])?;
        entries.push((key, value));
        if index + 1 != size {
            iter.call_packed(&[AnyView::from(&2i64)])?;
        }
    }
    Ok(entries)
}

fn construct_sequence(type_index: i32, items: &[Any]) -> Result<Any> {
    let name = if type_index == TVMFFITypeIndex::kTVMFFIList as i32 {
        "ffi.List"
    } else {
        "ffi.Array"
    };
    let args: Vec<AnyView<'_>> = items.iter().map(AnyView::from).collect();
    Function::get_global(name)?.call_packed(&args)
}

fn construct_map(type_index: i32, entries: &[(Any, Any)]) -> Result<Any> {
    let name = if type_index == TVMFFITypeIndex::kTVMFFIDict as i32 {
        "ffi.Dict"
    } else {
        "ffi.Map"
    };
    let mut args = Vec::with_capacity(entries.len() * 2);
    for (key, value) in entries {
        args.push(AnyView::from(key));
        args.push(AnyView::from(value));
    }
    Function::get_global(name)?.call_packed(&args)
}

fn map_shallow_copy_helper(type_index: i32) -> Option<&'static Function> {
    static MAP_HELPER: LazyLock<Option<Function>> =
        LazyLock::new(|| Function::get_global(MAP_SHALLOW_COPY_GLOBAL).ok());
    static DICT_HELPER: LazyLock<Option<Function>> =
        LazyLock::new(|| Function::get_global(DICT_SHALLOW_COPY_GLOBAL).ok());

    if type_index == TVMFFITypeIndex::kTVMFFIMap as i32 {
        MAP_HELPER.as_ref()
    } else if type_index == TVMFFITypeIndex::kTVMFFIDict as i32 {
        DICT_HELPER.as_ref()
    } else {
        None
    }
}

fn call_map_shallow_copy(raw: TVMFFIAny, helper: &Function, kind: &str) -> Result<Any> {
    let source = owned_from_raw(raw)?;
    let output = helper
        .call_packed(&[AnyView::from(&source)])
        .map_err(|error| with_error_context(error, &format!("{kind} shallow copy")))?;
    let output_raw = *output.as_raw_ffi_any();
    if output_raw.type_index != raw.type_index {
        return Err(Error::new(
            TYPE_ERROR,
            &format!("{kind} shallow-copy helper returned a different type"),
            "",
        ));
    }
    let source_pointer = unsafe { raw.data_union.v_obj };
    let output_pointer = unsafe { output_raw.data_union.v_obj };
    if output_pointer.is_null() || output_pointer == source_pointer {
        return Err(Error::new(
            TYPE_ERROR,
            &format!("{kind} shallow-copy helper must return a distinct non-null object"),
            "",
        ));
    }
    if !object_is_unique(output_raw) {
        return Err(runtime_error(&format!(
            "native structural map: {kind} shallow-copy helper returned a shared object"
        )));
    }
    Ok(output)
}

fn shallow_copy(raw: TVMFFIAny) -> Result<Any> {
    let Some(attr) = shallow_copy_column().and_then(|column| column.get(raw.type_index)) else {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "type `{}` cannot use reflected structural mutation because it does not define `{SHALLOW_COPY_ATTR}`",
                type_key_of(raw.type_index)
            ),
            "",
        ));
    };
    if attr.type_index != TVMFFITypeIndex::kTVMFFIFunction as i32 {
        return Err(Error::new(
            TYPE_ERROR,
            &format!("{SHALLOW_COPY_ATTR} must be an ffi.Function"),
            "",
        ));
    }
    let function = Function::try_from(owned_from_raw(attr)?)?;
    let source = owned_from_raw(raw)?;
    let result = function.call_packed(&[AnyView::from(&source)])?;
    let result_raw = *result.as_raw_ffi_any();
    let result_pointer = unsafe { result_raw.data_union.v_obj };
    let source_pointer = unsafe { raw.data_union.v_obj };
    if result_raw.type_index != raw.type_index
        || result_pointer.is_null()
        || result_pointer == source_pointer
    {
        return Err(Error::new(
            TYPE_ERROR,
            "shallow copy callback must return a distinct object with the same type as its input",
            "",
        ));
    }
    Ok(result)
}

fn call_field_setter(
    field: &TVMFFIFieldInfo,
    field_address: *mut c_void,
    value: &TVMFFIAny,
) -> Result<()> {
    if field.setter.is_null() {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "cannot structurally mutate field `{}` because it does not define a setter",
                field.name.as_str()
            ),
            "",
        ));
    }
    let return_code = unsafe {
        if field.flags & FLAG_SETTER_IS_FUNCTION == 0 {
            let setter: TVMFFIFieldSetter = std::mem::transmute(field.setter);
            setter(field_address, value)
        } else {
            let mut args = [TVMFFIAny::new(), *value];
            args[0].type_index = TVMFFITypeIndex::kTVMFFIOpaquePtr as i32;
            args[0].data_union.v_ptr = field_address;
            // The safe-call contract permits a callee to populate `result`
            // before returning an error. Own the slot up front so either a
            // successful return value or a partial failure is released.
            let mut result = Any::new();
            TVMFFIFunctionCall(
                field.setter as TVMFFIObjectHandle,
                args.as_ptr(),
                2,
                Any::as_data_ptr(&mut result),
            )
        }
    };
    if return_code == 0 {
        Ok(())
    } else {
        Err(Error::from_raised())
    }
}

fn identity_key(raw: TVMFFIAny) -> Result<Option<NonNull<TVMFFIObject>>> {
    let kind = structural_hash_kind(raw)?;
    if kind != Some(TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar as i32)
        && kind != Some(TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode as i32)
    {
        return Ok(None);
    }
    object_identity_key(raw).map(Some)
}

fn structural_hash_kind(raw: TVMFFIAny) -> Result<Option<i32>> {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return Ok(None);
    }
    let type_info = checked_type_info(raw.type_index)?;
    unsafe {
        if (*type_info).metadata.is_null() {
            Ok(None)
        } else {
            Ok(Some((*(*type_info).metadata).structural_eq_hash_kind))
        }
    }
}

fn object_identity_key(raw: TVMFFIAny) -> Result<NonNull<TVMFFIObject>> {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return Err(Error::new(
            TYPE_ERROR,
            "variable-remap keys must be object-backed values",
            "",
        ));
    }
    let pointer = unsafe { raw.data_union.v_obj };
    NonNull::new(pointer)
        .ok_or_else(|| runtime_error("native structural map: identity object has a null pointer"))
}

fn checked_type_info(type_index: i32) -> Result<*const crate::tvm_ffi_sys::TVMFFITypeInfo> {
    let info = unsafe { TVMFFIGetTypeInfo(type_index) };
    if info.is_null() {
        Err(runtime_error(&format!(
            "native structural map: unregistered type index {type_index}"
        )))
    } else {
        Ok(info)
    }
}

#[inline]
fn object_is_unique(raw: TVMFFIAny) -> bool {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return false;
    }
    let pointer = unsafe { raw.data_union.v_obj };
    !pointer.is_null() && unsafe { object::unsafe_::strong_count(pointer) == 1 }
}

#[inline]
fn same_shallow(lhs: TVMFFIAny, rhs: TVMFFIAny) -> bool {
    lhs.type_index == rhs.type_index
        && lhs.small_str_len == rhs.small_str_len
        && unsafe { lhs.data_union.v_uint64 == rhs.data_union.v_uint64 }
}

fn owned_from_raw(raw: TVMFFIAny) -> Result<Any> {
    let view = unsafe { AnyView::from_raw_ffi_any(raw) };
    Ok(Any::from(view))
}

/// Replace one owning container cell after its replacement mapped
/// successfully.
///
/// # Safety
///
/// `cell` must be an initialized owning `TVMFFIAny` in uniquely and
/// exclusively owned parent storage. `value` must not borrow `cell`, and no
/// concurrent or re-entrant access to that storage may occur during the
/// replacement.
unsafe fn replace_owned_cell(cell: *mut TVMFFIAny, value: Any) {
    let replacement = Any::into_raw_ffi_any(value);
    let old = std::ptr::replace(cell, replacement);
    drop(Any::from_raw_ffi_any(old));
}

fn reject_foreign_structural_mutation(type_index: i32) -> Result<()> {
    for (name, column) in [
        (STRUCTURAL_MUTATE_ATTR, structural_mutate_column()),
        (
            STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR,
            structural_maybe_inplace_mutate_column(),
        ),
    ] {
        let Some(attr) = column.and_then(|column| column.get(type_index)) else {
            continue;
        };
        if attr.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            continue;
        }
        if attr.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32
            || attr.type_index == TVMFFITypeIndex::kTVMFFIFunction as i32
        {
            return Err(runtime_error(&format!(
                "native structural map: type `{}` registers foreign `{name}`; implement its mutation explicitly in Rust",
                type_key_of(type_index)
            )));
        }
        return Err(Error::new(
            TYPE_ERROR,
            &format!("{name} must be an opaque function pointer or ffi.Function"),
            "",
        ));
    }
    Ok(())
}

fn with_value_context(error: Error, raw: TVMFFIAny) -> Error {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        error
    } else {
        with_error_context(error, &format!("object `{}`", type_key_of(raw.type_index)))
    }
}

fn with_error_context(error: Error, frame: &str) -> Error {
    Error::with_appended_backtrace(error, &format!("[native structural map] {frame}\n"))
}

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[derive(Clone, Copy)]
struct CachedColumn {
    cache: &'static AtomicUsize,
    name: &'static str,
}

impl CachedColumn {
    fn get(self) -> Option<TypeAttrColumn> {
        let cached = self.cache.load(Ordering::Relaxed);
        if cached != 0 {
            let pointer = cached as *mut TVMFFITypeAttrColumn;
            return Some(unsafe { TypeAttrColumn::from_non_null(NonNull::new_unchecked(pointer)) });
        }
        let column = type_attr_column(self.name)?;
        // TypeAttrColumn is a transparent NonNull wrapper shared with the
        // structural-visit module.  Registry column addresses are immortal.
        self.cache
            .store(column.as_ptr() as usize, Ordering::Relaxed);
        Some(column)
    }
}

static STRUCTURAL_MUTATE_COLUMN: AtomicUsize = AtomicUsize::new(0);
static STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN: AtomicUsize = AtomicUsize::new(0);
static SHALLOW_COPY_COLUMN: AtomicUsize = AtomicUsize::new(0);

fn structural_mutate_column() -> Option<TypeAttrColumn> {
    CachedColumn {
        cache: &STRUCTURAL_MUTATE_COLUMN,
        name: STRUCTURAL_MUTATE_ATTR,
    }
    .get()
}

fn structural_maybe_inplace_mutate_column() -> Option<TypeAttrColumn> {
    CachedColumn {
        cache: &STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN,
        name: STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR,
    }
    .get()
}

fn shallow_copy_column() -> Option<TypeAttrColumn> {
    CachedColumn {
        cache: &SHALLOW_COPY_COLUMN,
        name: SHALLOW_COPY_ATTR,
    }
    .get()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tvm_ffi_sys::{TVMFFIAnyViewToOwnedAny, TVMFFIByteArray};
    use crate::String;

    unsafe extern "C" fn clone_any_then_fail(source: *mut c_void, result: *mut TVMFFIAny) -> i32 {
        let code = TVMFFIAnyViewToOwnedAny(source.cast(), result);
        if code != 0 {
            return code;
        }
        Error::set_raised(&Error::new(
            RUNTIME_ERROR,
            "callback failed after writing an owning result",
            "",
        ));
        -1
    }

    unsafe extern "C" fn setter_safe_call(
        _handle: *mut c_void,
        args: *const TVMFFIAny,
        _num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32 {
        clone_any_then_fail(args.add(1).cast_mut().cast(), result)
    }

    struct NoopDispatch;

    impl MapDispatch for NoopDispatch {
        fn dispatch_map(
            &mut self,
            _value: &MapValue,
            _def_region_kind: DefRegionKind,
        ) -> Option<MapResult> {
            None
        }
    }

    #[test]
    fn reflected_getter_releases_partial_result_on_error() {
        let tracked = String::from("a reference-counted reflected field value");
        let mut value = Any::from(tracked.clone());
        let count_before = AnyView::from(&tracked).debug_strong_count();
        let mut field: TVMFFIFieldInfo = unsafe { std::mem::zeroed() };
        field.name = unsafe { TVMFFIByteArray::from_str("value") };
        field.getter = Some(clone_any_then_fail);
        let mut mapper = NativeMapper {
            dispatch: NoopDispatch,
            order: WalkOrder::PreOrder,
            memo: HashMap::new(),
        };
        let mut field_changed = false;

        let error = unsafe {
            mapper.map_reflected_field(
                (&mut value as *mut Any).cast(),
                &field,
                DefRegionKind::None,
                &mut field_changed,
            )
        }
        .expect_err("failing getter unexpectedly succeeded");

        assert_eq!(
            error.message(),
            "callback failed after writing an owning result"
        );
        assert!(!field_changed);
        assert_eq!(AnyView::from(&tracked).debug_strong_count(), count_before);
    }

    #[test]
    fn function_setter_releases_partial_result_on_error() {
        let tracked = String::from("a reference-counted setter result");
        let value = Any::from(tracked.clone());
        let count_before = AnyView::from(&tracked).debug_strong_count();
        let setter =
            unsafe { Function::from_extern_c(std::ptr::null_mut(), setter_safe_call, None) };
        let setter_owner = Any::from(setter);
        let mut field: TVMFFIFieldInfo = unsafe { std::mem::zeroed() };
        field.name = unsafe { TVMFFIByteArray::from_str("value") };
        field.flags = FLAG_SETTER_IS_FUNCTION;
        field.setter = unsafe { setter_owner.as_raw_ffi_any().data_union.v_obj.cast() };
        let mut storage = 0u8;

        let error = call_field_setter(
            &field,
            (&mut storage as *mut u8).cast(),
            value.as_raw_ffi_any(),
        )
        .expect_err("failing Function setter unexpectedly succeeded");

        assert_eq!(
            error.message(),
            "callback failed after writing an owning result"
        );
        assert_eq!(AnyView::from(&tracked).debug_strong_count(), count_before);
    }
}
