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
//! dispatch and the stateless recursion engine (`visit_raw` and the
//! `visit_*` helpers below) owns iteration over containers and reflected
//! fields.
//!
//! The runtime object registry is open, so the walker uses the stable tvm-ffi
//! reflection ABI for arbitrary registered object types. That ABI is only the
//! object-description boundary: traversal, control flow, typed dispatch,
//! visitor state, and definition-region propagation remain in Rust.
//!
//! Mutable `List`/`Dict` contents are snapshotted before callbacks run, so a
//! callback mutating the container it was reached through cannot invalidate
//! the traversal; the walk sees the pre-mutation contents.
//!
//! No C++ `ffi.StructuralVisitor` is constructed and no C++ default-visit
//! function is called. A non-container type with a foreign `__s_visit__` hook
//! is rejected instead of silently substituting reflection with potentially
//! different semantics; visit such a type's children explicitly from a
//! [`StructuralVisitor`], or skip the value in a walk.

use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crate::any::{Any, AnyView};
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::ObjectCore;
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive, kTVMFFIFieldFlagBitMaskSEqHashDefRecursive,
    kTVMFFIFieldFlagBitMaskSEqHashIgnore,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIDefRegionKind, TVMFFIFieldInfo, TVMFFIGetTypeAttrColumn,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFISEqHashKind, TVMFFITypeAttrColumn, TVMFFITypeIndex,
};

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

/// A borrowed view of a raw tvm-ffi value.
///
/// Generated visitors match this value without taking ownership: borrowed
/// object-node handlers use [`VisitValue::as_node`], while POD or object-ref
/// value handlers use [`VisitValue::cast`].
#[repr(transparent)]
pub struct VisitValue(TVMFFIAny);

impl VisitValue {
    #[inline]
    fn from_raw(raw: TVMFFIAny) -> Self {
        VisitValue(raw)
    }

    /// Convert the value into an owned typed handle.
    #[inline]
    pub fn cast<R: crate::type_traits::AnyCompatible>(&self) -> Option<R> {
        unsafe {
            if R::check_any_strict(&self.0) {
                Some(R::copy_from_any_view_after_check(&self.0))
            } else {
                None
            }
        }
    }

    /// Runtime type index stored in this value.
    #[inline]
    pub fn type_index(&self) -> i32 {
        self.0.type_index
    }

    /// Borrow the value as node type `N` if it is an instance of that type.
    #[inline]
    pub fn as_node<N: ObjectCore>(&self) -> Option<&N> {
        if self.0.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return None;
        }
        let base_type_index = N::type_index();
        if self.0.type_index != base_type_index {
            // A final type has no registered subtype, so a differing index can
            // never match: reject with the integer compare alone, mirroring the
            // `_type_final` fast path of C++ `IsObjectInstance`.
            if N::TYPE_FINAL {
                return None;
            }
            if !is_instance_at_depth(self.0.type_index, base_type_index, N::TYPE_DEPTH) {
                return None;
            }
        }
        Some(unsafe { &*(self.0.data_union.v_obj as *const N) })
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

/// Typed dispatch implemented by a walk-layer observer.
///
/// [`crate::dispatch`] tests the implementation's `visit_*` methods in source
/// order. Borrowed node arguments use refcount-free subtype checks, owned
/// FFI-compatible arguments use exact value casts, and `&VisitValue` is a
/// catch-all. `None` reports that no handler matched: a standalone walk then
/// advances normally, while a tuple chain hands the value to the next link —
/// so a spliced visitor that "handled" a value must not return `None`.
///
/// This is the observer layer, mirroring C++ `StructuralWalk` callbacks: the
/// walker owns recursion, and a handler steers it only through the returned
/// [`WalkResult`]. A traversal that must visit children itself — selected
/// children, custom orders, explicit definition-region overrides — belongs in
/// a [`StructuralVisitor`] instead.
///
/// The definition-region state active at the dispatched value arrives as the
/// `def_region_kind` argument. A `#[dispatch(visit)]` handler opts into it by
/// declaring a trailing `DefRegionKind` parameter — the analog of a C++
/// `StructuralWalk` callback accepting `(value, def_region_kind)` instead of
/// `(value)`.
pub trait VisitDispatch: Sized {
    fn dispatch_visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult>;
}

impl<V: VisitDispatch> VisitDispatch for &mut V {
    #[inline]
    fn dispatch_visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        (**self).dispatch_visit(value, def_region_kind)
    }
}

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

#[doc(hidden)]
pub enum ByDispatch {}

impl<'a, V: VisitDispatch> IntoWalker<ByDispatch> for &'a mut V {
    type Walker = DispatchVisitor<&'a mut V>;
    fn into_walker(self, order: WalkOrder) -> Self::Walker {
        DispatchVisitor {
            visitor: self,
            order,
        }
    }
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
/// * `FnMut(T) -> impl IntoVisitResult` for an FFI-convertible `T` — exact
///   value cast via [`VisitValue::cast`]. Exact includes the value itself:
///   a numeric link matches only losslessly representable values, so an
///   `Int(300)` falls through a `u8` link to the next one rather than
///   truncating.
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

impl_chain_walker!((F0, M0, 0));
impl_chain_walker!((F0, M0, 0), (F1, M1, 1));
impl_chain_walker!((F0, M0, 0), (F1, M1, 1), (F2, M2, 2));
impl_chain_walker!((F0, M0, 0), (F1, M1, 1), (F2, M2, 2), (F3, M3, 3));
impl_chain_walker!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4)
);
impl_chain_walker!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4),
    (F5, M5, 5)
);
impl_chain_walker!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4),
    (F5, M5, 5),
    (F6, M6, 6)
);
impl_chain_walker!(
    (F0, M0, 0),
    (F1, M1, 1),
    (F2, M2, 2),
    (F3, M3, 3),
    (F4, M4, 4),
    (F5, M5, 5),
    (F6, M6, 6),
    (F7, M7, 7)
);

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
        self.visit(&VisitValue::from_raw(raw), def_region_kind)
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
        let result = visit_children_raw(
            value.0,
            &mut UserChildren { visitor: self },
            def_region_kind,
        )
        .map_err(|halt| with_value_context(halt, value.0));
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

/// Owns its walker so a closure's state stays inline and a `&mut` visitor
/// keeps a single level of indirection. Public only as an
/// [`IntoWalker::Walker`] projection.
#[doc(hidden)]
pub struct DispatchVisitor<V> {
    visitor: V,
    order: WalkOrder,
}

impl<V: VisitDispatch> NativeVisit for DispatchVisitor<V> {
    fn enter(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => self
                .visitor
                .dispatch_visit(value, def_region_kind)
                .unwrap_or(Ok(WalkResult::Advance)),
            WalkOrder::PostOrder => Ok(WalkResult::Advance),
        }
    }

    fn exit(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => Ok(WalkResult::Advance),
            WalkOrder::PostOrder => self
                .visitor
                .dispatch_visit(value, def_region_kind)
                .unwrap_or(Ok(WalkResult::Advance)),
        }
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

/// Recurse into `value` on behalf of `visitor`: fire its enter hook, walk the
/// children, fire its exit hook. The engine below is stateless — these are
/// free functions, with the only shared piece (the `__s_visit__` attribute
/// column) cached process-wide.
fn visit_raw<V: NativeVisit>(
    value: TVMFFIAny,
    visitor: &mut V,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
        return Ok(());
    }

    let visit_value = VisitValue::from_raw(value);
    let enter = match visitor.enter(&visit_value, def_region_kind) {
        Ok(flow) => flow,
        Err(error) => return Err(with_value_context(error.into(), value)),
    };
    match enter {
        WalkResult::Advance => {}
        WalkResult::Skip => return Ok(()),
        WalkResult::Interrupt => return Err(NativeHalt::Interrupt(Any::new())),
        WalkResult::InterruptWith(payload) => return Err(NativeHalt::Interrupt(payload)),
    }

    let children = &mut WalkChildren {
        visitor: &mut *visitor,
    };
    if let Err(halt) = visit_children_raw(value, children, def_region_kind) {
        return Err(with_value_context(halt, value));
    }

    let exit = match visitor.exit(&visit_value, def_region_kind) {
        Ok(flow) => flow,
        Err(error) => return Err(with_value_context(error.into(), value)),
    };
    match exit {
        WalkResult::Interrupt => Err(NativeHalt::Interrupt(Any::new())),
        WalkResult::InterruptWith(payload) => Err(NativeHalt::Interrupt(payload)),
        WalkResult::Advance | WalkResult::Skip => Ok(()),
    }
}

#[inline]
fn visit_children_raw<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    match value.type_index {
        x if x == TVMFFITypeIndex::kTVMFFIArray as i32
            || x == TVMFFITypeIndex::kTVMFFIList as i32 =>
        {
            return visit_sequence(value, visitor, def_region_kind);
        }
        x if x == TVMFFITypeIndex::kTVMFFIMap as i32
            || x == TVMFFITypeIndex::kTVMFFIDict as i32 =>
        {
            // Fast path: read the MapBaseObj storage layout directly, like
            // the SeqPrefix path for arrays — zero FFI calls per entry.
            // Dict entries are snapshotted first to keep the re-entrant
            // mutation guard. If the one-time layout validation fails
            // (e.g. an ABI-debug build), fall back to the packed-functor
            // iteration protocol.
            if map_layout_usable(value) {
                let snapshot = x == TVMFFITypeIndex::kTVMFFIDict as i32;
                return visit_map_layout(value, visitor, def_region_kind, snapshot);
            }
            return visit_map(value, visitor, def_region_kind);
        }
        _ => {}
    }

    reject_foreign_structural_visit(value.type_index)?;
    if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        Ok(())
    } else {
        visit_reflected_fields(value, visitor, def_region_kind)
    }
}

#[inline(never)]
fn visit_sequence<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    let seq = unsafe { &*(value.data_union.v_obj as *const SeqPrefix) };
    if seq.size < 0 {
        return Err(runtime_error("native visitor: sequence reports a negative size").into());
    }
    if seq.data.is_null() && seq.size != 0 {
        return Err(
            runtime_error("native visitor: non-empty sequence has a null data pointer").into(),
        );
    }
    let size = usize::try_from(seq.size)
        .map_err(|_| runtime_error("native visitor: sequence size does not fit usize"))?;
    if size == 0 {
        return Ok(());
    }

    if value.type_index == TVMFFITypeIndex::kTVMFFIList as i32 {
        // List storage may be invalidated by a re-entrant callback. Own a
        // snapshot before running the first callback.
        let children: Vec<Any> = {
            let cells = unsafe { std::slice::from_raw_parts(seq.data, size) };
            cells
                .iter()
                .map(|cell| Any::from(unsafe { view_of(cell) }))
                .collect()
        };
        for (index, child) in children.into_iter().enumerate() {
            let raw = raw_of_owned(&child);
            visitor
                .visit_child(raw, def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("sequence item [{index}]")))?;
        }
        return Ok(());
    }

    // Array is immutable, so its element cells remain stable throughout
    // recursive callbacks and need no refcounted snapshot.
    let cells = unsafe { std::slice::from_raw_parts(seq.data, size) };
    for (index, child) in cells.iter().enumerate() {
        visitor
            .visit_child(*child, def_region_kind)
            .map_err(|halt| with_error_context(halt, &format!("sequence item [{index}]")))?;
    }
    Ok(())
}

/// Walk map/dict entries by reading the `MapBaseObj` storage directly —
/// the map analog of the `SeqPrefix` array fast path. `snapshot` first
/// takes owned copies of all entries (Dict re-entrant mutation guard).
#[inline(never)]
fn visit_map_layout<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    def_region_kind: DefRegionKind,
    snapshot: bool,
) -> NativeResult {
    let map = unsafe { &*(value.data_union.v_obj as *const MapPrefix) };
    let size = map.size as usize;
    if size == 0 {
        return Ok(());
    }
    let mut cursor = unsafe { MapCursor::new(map) };

    if snapshot {
        let mut entries: Vec<(Any, Any)> = Vec::with_capacity(size);
        for _ in 0..size {
            let Some((key, val)) = (unsafe { cursor.next() }) else {
                return Err(runtime_error("native visitor: map iteration ended early").into());
            };
            entries.push((
                Any::from(unsafe { view_of(&key) }),
                Any::from(unsafe { view_of(&val) }),
            ));
        }
        for (index, (key, val)) in entries.into_iter().enumerate() {
            visitor
                .visit_child(raw_of_owned(&key), def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("dict key [{index}]")))?;
            visitor
                .visit_child(raw_of_owned(&val), def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("dict value [{index}]")))?;
        }
        return Ok(());
    }

    // Immutable map: entry cells stay stable throughout recursive
    // callbacks, so visit them in place. The `size` bound also guards the
    // dense iteration list against corruption-induced cycles.
    for index in 0..size {
        let Some((key, val)) = (unsafe { cursor.next() }) else {
            return Err(runtime_error("native visitor: map iteration ended early").into());
        };
        visitor
            .visit_child(key, def_region_kind)
            .map_err(|halt| with_error_context(halt, &format!("map key [{index}]")))?;
        visitor
            .visit_child(val, def_region_kind)
            .map_err(|halt| with_error_context(halt, &format!("map value [{index}]")))?;
    }
    Ok(())
}

/// Cold fallback used when the mirrored layout fails validation (e.g. an
/// ABI-debug build): iterate through the public packed functors. Map storage
/// is private C++; the Rust binding itself uses these iterator functors, so
/// no structural visiting or traversal control leaves Rust. Entries are
/// snapshotted before user callbacks run — required for Dict, whose mutation
/// invalidates the iterator, and harmless for immutable Map on this
/// non-performance path.
fn visit_map<C: ChildVisit>(
    value: TVMFFIAny,
    visitor: &mut C,
    def_region_kind: DefRegionKind,
) -> NativeResult {
    let is_dict = value.type_index == TVMFFITypeIndex::kTVMFFIDict as i32;
    let (size_name, iter_name, kind) = if is_dict {
        ("ffi.DictSize", "ffi.DictForwardIterFunctor", "dict")
    } else {
        ("ffi.MapSize", "ffi.MapForwardIterFunctor", "map")
    };
    let size = Function::get_global(size_name)?
        .call_packed(&[unsafe { view_of(&value) }])
        .and_then(i64::try_from)?;
    if size < 0 {
        return Err(runtime_error("native visitor: map reports a negative size").into());
    }
    let size = usize::try_from(size)
        .map_err(|_| runtime_error("native visitor: map size does not fit usize"))?;
    if size == 0 {
        return Ok(());
    }

    let iter_any = Function::get_global(iter_name)?.call_packed(&[unsafe { view_of(&value) }])?;
    let iter = Function::try_from(iter_any)?;

    let mut entries = Vec::with_capacity(size);
    for index in 0..size {
        let key = iter.call_packed(&[AnyView::from(&0i64)])?;
        let map_value = iter.call_packed(&[AnyView::from(&1i64)])?;
        entries.push((key, map_value));
        if index + 1 != size {
            iter.call_packed(&[AnyView::from(&2i64)])?;
        }
    }

    for (index, (key, map_value)) in entries.into_iter().enumerate() {
        visitor
            .visit_child(raw_of_owned(&key), def_region_kind)
            .map_err(|halt| with_error_context(halt, &format!("{kind} key [{index}]")))?;
        visitor
            .visit_child(raw_of_owned(&map_value), def_region_kind)
            .map_err(|halt| with_error_context(halt, &format!("{kind} value [{index}]")))?;
    }
    Ok(())
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

#[inline]
fn reject_foreign_structural_visit(type_index: i32) -> Result<()> {
    let Some(attr) = structural_visit_column().and_then(|column| column.get(type_index)) else {
        return Ok(());
    };
    match attr.type_index {
        x if x == TVMFFITypeIndex::kTVMFFINone as i32 => Ok(()),
        x if x == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32
            || x == TVMFFITypeIndex::kTVMFFIFunction as i32 =>
        {
            let value_type = if type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
                format!("type index {type_index}")
            } else {
                format!("type `{}`", type_key_of(type_index))
            };
            Err(runtime_error(&format!(
                "native visitor: {value_type} registers foreign `{STRUCTURAL_VISIT_ATTR}`; \
                     visit its children explicitly from a `StructuralVisitor` \
                     (`structural_visit`), or skip it with a pre-order `WalkResult::Skip` \
                     handler"
            )))
        }
        _ => Err(Error::new(
            TYPE_ERROR,
            &format!("{STRUCTURAL_VISIT_ATTR} must be an opaque function pointer or ffi.Function"),
            "",
        )),
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
    visitor.visit_child(root, DefRegionKind::None)
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
    finish(visit_raw(
        raw_of(AnyView::from(root)),
        &mut dispatch,
        DefRegionKind::None,
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
fn field_def_region(field: &TVMFFIFieldInfo, inherited: DefRegionKind) -> DefRegionKind {
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
fn free_var_child_region(inherited: DefRegionKind, structural_eq_hash_kind: i32) -> DefRegionKind {
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

/// Layout prefix shared by the C++ `MapObj` and `DictObj` (`MapBaseObj`,
/// release ABI without `TVM_FFI_DEBUG_WITH_ABI_CHANGE`).
#[repr(C)]
struct MapPrefix {
    _header: TVMFFIObject,
    data: *mut u8,
    size: u64,
    slots: u64,
    _data_deleter: Option<unsafe extern "C" fn(*mut c_void)>,
}

/// Dense-layout extension of the prefix (`DenseMapBaseObj`).
#[repr(C)]
struct DenseMapPrefix {
    base: MapPrefix,
    fib_shift: u32,
    iter_list_head: u64,
    iter_list_tail: u64,
}

const _: () = {
    assert!(std::mem::offset_of!(MapPrefix, data) == 24);
    assert!(std::mem::offset_of!(MapPrefix, size) == 32);
    assert!(std::mem::offset_of!(MapPrefix, slots) == 40);
    assert!(std::mem::offset_of!(MapPrefix, _data_deleter) == 48);
    assert!(std::mem::offset_of!(DenseMapPrefix, fib_shift) == 56);
    assert!(std::mem::offset_of!(DenseMapPrefix, iter_list_head) == 64);
};

/// MSB tag on `slots_` marking the small (inline KV array) layout.
const MAP_SMALL_TAG: u64 = 1 << 63;
/// `kInvalidIndex`: terminator of the dense iteration list.
const MAP_INVALID_INDEX: u64 = u64::MAX;
/// `kBlockCap`: entries per dense block.
const MAP_BLOCK_CAP: u64 = 16;
/// `sizeof(ItemType)`: KV pair (32 bytes) + prev/next indices (16 bytes).
const MAP_ITEM_SIZE: usize = 48;
/// `sizeof(Block)`: `kBlockCap` metadata bytes + `kBlockCap` items.
const MAP_BLOCK_SIZE: usize = 16 + 16 * MAP_ITEM_SIZE;
/// Byte offset of `ItemType::next` (after the 32-byte KV pair and `prev`).
const MAP_ITEM_NEXT_OFFSET: usize = 40;

/// Borrowed traversal cursor over either map storage layout, yielding entries
/// in the same order as the C++ iterator.
enum MapCursor {
    Small {
        kv: *const TVMFFIAny,
        index: usize,
        size: usize,
    },
    Dense {
        data: *const u8,
        index: u64,
    },
}

impl MapCursor {
    #[inline]
    unsafe fn new(map: &MapPrefix) -> MapCursor {
        if map.slots & MAP_SMALL_TAG != 0 {
            MapCursor::Small {
                kv: map.data as *const TVMFFIAny,
                index: 0,
                size: map.size as usize,
            }
        } else {
            let dense = &*(map as *const MapPrefix as *const DenseMapPrefix);
            MapCursor::Dense {
                data: map.data,
                index: dense.iter_list_head,
            }
        }
    }

    #[inline]
    unsafe fn next(&mut self) -> Option<(TVMFFIAny, TVMFFIAny)> {
        match self {
            MapCursor::Small { kv, index, size } => {
                if *index >= *size {
                    return None;
                }
                let pair = kv.add(*index * 2);
                *index += 1;
                Some((*pair, *pair.add(1)))
            }
            MapCursor::Dense { data, index } => {
                if *index == MAP_INVALID_INDEX {
                    return None;
                }
                let block = data.add((*index / MAP_BLOCK_CAP) as usize * MAP_BLOCK_SIZE);
                let item = block.add(
                    MAP_BLOCK_CAP as usize + (*index % MAP_BLOCK_CAP) as usize * MAP_ITEM_SIZE,
                );
                let key = *(item as *const TVMFFIAny);
                let val = *(item.add(16) as *const TVMFFIAny);
                *index = *(item.add(MAP_ITEM_NEXT_OFFSET) as *const u64);
                Some((key, val))
            }
        }
    }
}

/// Process-wide result of the one-time map layout validation:
/// 0 = unknown, 1 = usable, 2 = unusable.
static MAP_LAYOUT_STATE: AtomicU8 = AtomicU8::new(0);

#[inline]
fn map_layout_usable(value: TVMFFIAny) -> bool {
    match MAP_LAYOUT_STATE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let usable = validate_map_layout(value);
            MAP_LAYOUT_STATE.store(if usable { 1 } else { 2 }, Ordering::Relaxed);
            usable
        }
    }
}

/// Cross-check the mirrored `MapBaseObj` layout against the public size
/// functor once per process. An ABI-debug build inserts a state marker that
/// shifts every field by 8 bytes, which this detects: offset 32 then holds a
/// pointer value that cannot equal the reported entry count.
fn validate_map_layout(value: TVMFFIAny) -> bool {
    let expected = (|| -> Result<i64> {
        let is_dict = value.type_index == TVMFFITypeIndex::kTVMFFIDict as i32;
        let name = if is_dict {
            "ffi.DictSize"
        } else {
            "ffi.MapSize"
        };
        Function::get_global(name)?
            .call_packed(&[unsafe { view_of(&value) }])
            .and_then(i64::try_from)
    })();
    let Ok(expected) = expected else {
        return false;
    };
    let map = unsafe { &*(value.data_union.v_obj as *const MapPrefix) };
    expected >= 0 && map.size == expected as u64
}

/// Layout prefix shared by the C++ `ArrayObj` and `ListObj`.
#[repr(C)]
struct SeqPrefix {
    _header: TVMFFIObject,
    data: *const TVMFFIAny,
    size: i64,
}

const _: () = {
    assert!(std::mem::offset_of!(SeqPrefix, data) == 24);
    assert!(std::mem::offset_of!(SeqPrefix, size) == 32);
};

#[derive(Clone, Copy)]
struct TypeAttrColumn(NonNull<TVMFFITypeAttrColumn>);

impl TypeAttrColumn {
    /// Copy one borrowed cell; ownership remains with the registry.
    fn get(self, type_index: i32) -> Option<TVMFFIAny> {
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

fn type_attr_column(attr_name: &str) -> Option<TypeAttrColumn> {
    unsafe {
        let attr_name = TVMFFIByteArray::from_str(attr_name);
        NonNull::new(TVMFFIGetTypeAttrColumn(&attr_name).cast_mut()).map(TypeAttrColumn)
    }
}

/// Cached `__s_visit__` column pointer (0 = not seen yet). A registry column
/// is stable once created — C++ `DefaultVisitExpected` caches the same
/// pointer in a function-local static — while an absent column is re-queried
/// because a later attr registration may create it. The cache keeps the
/// per-value foreign-hook check free of FFI lookups.
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

fn type_key_of(type_index: i32) -> String {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            format!("<type_index {type_index}>")
        } else {
            (*info).type_key.as_str().to_string()
        }
    }
}

/// Subtype check with the base's inheritance depth supplied by the caller
/// (`ObjectCore::TYPE_DEPTH`), so only the object's type info is fetched.
#[inline]
fn is_instance_at_depth(object_type_index: i32, base_type_index: i32, base_depth: i32) -> bool {
    if object_type_index == base_type_index {
        return true;
    }
    unsafe {
        let info = TVMFFIGetTypeInfo(object_type_index);
        if info.is_null() {
            return false;
        }
        if (*info).type_depth <= base_depth {
            return false;
        }
        let ancestors = (*info).type_acenstors;
        if ancestors.is_null() {
            return false;
        }
        let ancestor = *ancestors.offset(base_depth as isize);
        !ancestor.is_null() && (*ancestor).type_index == base_type_index
    }
}

/// Visit every reflected field of `type_index` and its ancestors in the same
/// parent-to-child order as C++ `ForEachFieldInfoWithEarlyStop`.
///
/// # Safety
///
/// `type_index` must be a registered type index.
unsafe fn for_each_field<B>(
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

#[inline]
unsafe fn view_of(raw: &TVMFFIAny) -> AnyView<'_> {
    unsafe { AnyView::from_raw_ffi_any(*raw) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    struct RegionProbe(Vec<DefRegionKind>);

    impl NativeVisit for RegionProbe {
        fn enter(
            &mut self,
            _value: &VisitValue,
            def_region_kind: DefRegionKind,
        ) -> Result<WalkResult> {
            self.0.push(def_region_kind);
            Ok(WalkResult::Advance)
        }
    }

    #[derive(Default)]
    struct TypedRegionProbe {
        seen: Vec<DefRegionKind>,
    }

    #[crate::dispatch(visit)]
    impl TypedRegionProbe {
        fn visit_integer(&mut self, _value: i64, def_region_kind: DefRegionKind) -> WalkResult {
            self.seen.push(def_region_kind);
            WalkResult::Advance
        }
    }

    unsafe extern "C" fn clone_any_field(field: *mut c_void, result: *mut TVMFFIAny) -> i32 {
        let value = &*(field as *const Any);
        *result = Any::into_raw_ffi_any(value.clone());
        0
    }

    #[test]
    fn def_region_is_inherited_through_containers() {
        let root = Array::new(vec![1i64, 2]);
        let mut probe = RegionProbe(Vec::new());
        assert!(visit_raw(
            raw_of(AnyView::from(&root)),
            &mut probe,
            DefRegionKind::Recursive,
        )
        .is_ok());
        assert_eq!(probe.0, vec![DefRegionKind::Recursive; 3]);
    }

    #[test]
    fn reflected_field_def_region_reaches_typed_handler() {
        let mut probe = TypedRegionProbe::default();
        let mut dispatch = DispatchVisitor {
            visitor: &mut probe,
            order: WalkOrder::PreOrder,
        };
        let mut value = Any::from(7i64);
        let mut field: TVMFFIFieldInfo = unsafe { std::mem::zeroed() };
        field.name = unsafe { TVMFFIByteArray::from_str("value") };
        field.getter = Some(clone_any_field);
        let object = (&mut value as *mut Any).cast::<u8>();

        let mut children = WalkChildren {
            visitor: &mut dispatch,
        };
        for flags in [
            FLAG_SEQ_HASH_DEF_RECURSIVE,
            0,
            FLAG_SEQ_HASH_DEF_NON_RECURSIVE,
            FLAG_SEQ_HASH_DEF_NON_RECURSIVE | FLAG_SEQ_HASH_DEF_RECURSIVE,
            FLAG_SEQ_HASH_IGNORE,
        ] {
            field.flags = flags;
            assert!(unsafe {
                visit_reflected_field(object, &field, &mut children, DefRegionKind::None)
            }
            .is_ok());
        }
        assert_eq!(
            probe.seen,
            vec![
                DefRegionKind::Recursive,
                DefRegionKind::None,
                DefRegionKind::NonRecursive,
                DefRegionKind::NonRecursive,
            ]
        );
    }

    #[test]
    fn non_recursive_region_is_clamped_for_free_var_children_only() {
        use TVMFFISEqHashKind::{kTVMFFISEqHashKindFreeVar, kTVMFFISEqHashKindTreeNode};

        let free_var = kTVMFFISEqHashKindFreeVar as i32;
        let tree_node = kTVMFFISEqHashKindTreeNode as i32;
        assert_eq!(
            free_var_child_region(DefRegionKind::NonRecursive, free_var),
            DefRegionKind::None
        );
        assert_eq!(
            free_var_child_region(DefRegionKind::Recursive, free_var),
            DefRegionKind::Recursive
        );
        assert_eq!(
            free_var_child_region(DefRegionKind::None, free_var),
            DefRegionKind::None
        );
        assert_eq!(
            free_var_child_region(DefRegionKind::NonRecursive, tree_node),
            DefRegionKind::NonRecursive
        );
    }
}
