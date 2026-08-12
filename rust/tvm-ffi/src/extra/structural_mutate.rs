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
//! Rust owns callback dispatch, memoization, identity remapping, and most
//! container traversal. Map/Dict storage is traversed through a narrow C ABI
//! that calls directly back into Rust for each value, allowing unique maps to
//! be updated in place without exposing the runtime's private hash layout.
//! A non-container object with a foreign `__s_mutate__` or
//! `__s_maybe_inplace_mutate__` hook is rejected rather than silently
//! replacing its custom semantics with reflection.

use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::any::{Any, AnyView};
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::{self, ObjectCore};
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashIgnore, kTVMFFIFieldFlagBitSetterIsFunctionObj,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIFieldInfo, TVMFFIFieldSetter, TVMFFIFunctionCall, TVMFFIGetTypeInfo,
    TVMFFIMapMutateValues, TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex,
};
use crate::tvm_ffi_sys::{TVMFFIObjectHandle, TVMFFISEqHashKind};

use super::structural_visit::{
    field_def_region, for_each_field, free_var_child_region, type_attr_column, type_key_of,
    DefRegionKind, SeqPrefix, TypeAttrColumn, WalkOrder,
};

const STRUCTURAL_MUTATE_ATTR: &str = "__s_mutate__";
const STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR: &str = "__s_maybe_inplace_mutate__";
const SHALLOW_COPY_ATTR: &str = "__ffi_shallow_copy__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SETTER_IS_FUNCTION: i64 = kTVMFFIFieldFlagBitSetterIsFunctionObj as i64;

/// Borrowed value passed to structural-map callbacks.
///
/// Structural visit and map callbacks share the same audited implementation
/// for typed casts and borrowed node checks.
pub use super::common::StructuralValue as MapValue;

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

// Rust has no variadic generics, so implement each supported callback-tuple
// arity explicitly. Keep this arity limit in sync with the structural-visit
// callback chain.
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

        // Plain inline leaves have no children or structural identity.  Map
        // them directly instead of routing through identity lookup and the
        // default-mutation path, whose owning conversion crosses the C ABI.
        // Raw strings, byte-array views, and ObjectRValueRef are deliberately
        // excluded because converting those borrowed special values into an
        // Any performs normalization rather than a bitwise copy.
        if is_plain_inline_leaf(raw.type_index) {
            let value = MapValue::from_raw(raw);
            return match self.dispatch.dispatch_map(&value, def_region_kind) {
                Some(result) => result,
                // SAFETY: `is_plain_inline_leaf` excludes every borrowed
                // representation that needs normalization.  These values own
                // no external resource, so their owning form is the same
                // bitwise TVMFFIAny value.
                None => Ok(unsafe { Any::from_raw_ffi_any(raw) }),
            };
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
                return self.mutate_mapping_values(raw, def_region_kind, permit);
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
        // Array/List values come from the runtime and satisfy the container
        // layout invariants by contract.
        let seq = unsafe { &*raw.data_union.v_obj.cast::<SeqPrefix>() };
        let data = seq.data.cast_mut();
        let size = seq.size as usize;
        if size == 0 {
            return owned_from_raw(raw);
        }

        if permit == Permit::MaybeInPlace && object_is_unique(raw) {
            // The consumed root or a unique parent exclusively owns the
            // Array/List storage. Re-checking after the parent callback
            // prevents a callback-retained alias from observing raw writes.
            for index in 0..size {
                let cell = unsafe { data.add(index) };
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
        for child_raw in unsafe { std::slice::from_raw_parts(data, size) } {
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

    fn mutate_mapping_values(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        runtime_mutate_mapping_values(self, raw, def_region_kind, permit)
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
        // Own the output slot before entering foreign code. A getter may
        // populate an owning result and still report an error.
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
/// The root is consumed. A uniquely owned built-in container may therefore be
/// reused in place, while passing `root.clone()` keeps the original shared and
/// selects copy-on-write behavior. Map and Dict keys are anchors and are not
/// mapped; their values re-enter the Rust mapper through the runtime C ABI.
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

struct RuntimeMapMutationContext<D> {
    driver: *mut D,
    def_region_kind: DefRegionKind,
    kind: &'static str,
}

unsafe extern "C" fn runtime_map_value_mutator<D: DefaultMutationDriver>(
    context: *mut c_void,
    value: *const TVMFFIAny,
    index: i64,
    allow_inplace: i32,
    result: *mut TVMFFIAny,
) -> i32 {
    let context = &mut *context.cast::<RuntimeMapMutationContext<D>>();
    let permit = if allow_inplace != 0 {
        Permit::MaybeInPlace
    } else {
        Permit::Copy
    };
    let mapped = (&mut *context.driver)
        .recurse_raw(*value, context.def_region_kind, permit)
        .map_err(|error| with_error_context(error, &format!("{} value [{index}]", context.kind)));
    match mapped {
        Ok(mapped) => {
            *result = Any::into_raw_ffi_any(mapped);
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

fn runtime_mutate_mapping_values<D: DefaultMutationDriver>(
    driver: &mut D,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    let kind = if raw.type_index == TVMFFITypeIndex::kTVMFFIDict as i32 {
        "dict"
    } else {
        "map"
    };
    let mut context = RuntimeMapMutationContext {
        driver,
        def_region_kind,
        kind,
    };
    let mut result = Any::new();
    let return_code = unsafe {
        TVMFFIMapMutateValues(
            &raw,
            i32::from(permit == Permit::MaybeInPlace),
            std::ptr::from_mut(&mut context).cast(),
            runtime_map_value_mutator::<D>,
            Any::as_data_ptr(&mut result),
        )
    };
    if return_code == 0 {
        Ok(result)
    } else {
        Err(Error::from_raised())
    }
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
            // Own the result slot before entering foreign code so a partial
            // owning result is released on both success and failure.
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
    // Built-in containers always use container-specific structural mutation
    // and can never be FreeVar or DAG identities.  Avoid a runtime type-info
    // lookup for every Array/List/Map/Dict encountered during recursion.
    if is_builtin_container(raw.type_index) {
        return Ok(None);
    }
    let kind = structural_hash_kind(raw)?;
    if kind != Some(TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar as i32)
        && kind != Some(TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode as i32)
    {
        return Ok(None);
    }
    object_identity_key(raw).map(Some)
}

#[inline]
fn is_plain_inline_leaf(type_index: i32) -> bool {
    type_index < TVMFFITypeIndex::kTVMFFIRawStr as i32
        || type_index == TVMFFITypeIndex::kTVMFFISmallStr as i32
        || type_index == TVMFFITypeIndex::kTVMFFISmallBytes as i32
}

#[inline]
fn is_builtin_container(type_index: i32) -> bool {
    type_index == TVMFFITypeIndex::kTVMFFIArray as i32
        || type_index == TVMFFITypeIndex::kTVMFFIList as i32
        || type_index == TVMFFITypeIndex::kTVMFFIMap as i32
        || type_index == TVMFFITypeIndex::kTVMFFIDict as i32
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
