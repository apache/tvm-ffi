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

//! Native Rust structural mutation and mapping.
//!
//! [`structural_mutate`] lets a mutator drive recursion, while
//! [`structural_map`] applies callbacks around engine-owned recursion.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{BuildHasherDefault, Hasher};
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

use crate::any::{Any, AnyView};
use crate::collections::array::ArrayObj;
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::{self, Object, ObjectArc, ObjectCore};
use crate::reflection::TypeAttrColumn;
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashIgnore, kTVMFFIFieldFlagBitSetterIsFunctionObj,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyViewToOwnedAny, TVMFFIByteArray, TVMFFIFieldInfo, TVMFFIFieldSetter,
    TVMFFIFunctionCall, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex,
    TVMFFITypeKeyToIndex,
};
use crate::tvm_ffi_sys::{TVMFFIObjectHandle, TVMFFISEqHashKind};

use super::structural_common::{
    impl_callback_chain_tuple_arities, is_plain_inline, same_shallow,
    try_to_owned_without_normalization, with_structural_error_context, with_visit_error_context,
};
use super::structural_visit::{
    field_def_region, for_each_field_info, free_var_child_region, type_attr_column, type_key_of,
    DefRegionKind, WalkOrder,
};
use super::unchanged::is_unchanged;
pub use super::unchanged::{Unchanged, UnchangedOr};

const STRUCTURAL_MUTATE_ATTR: &str = "__s_mutate__";
const STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR: &str = "__s_maybe_inplace_mutate__";
const SHALLOW_COPY_ATTR: &str = "__ffi_shallow_copy__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SETTER_IS_FUNCTION: i64 = kTVMFFIFieldFlagBitSetterIsFunctionObj as i64;

/// Borrowed value passed to structural map and mutation callbacks.
pub use super::StructuralView;

mod policy;
pub use policy::{DefaultMutContextPolicy, MapWithContextPolicy, MutContextPolicy};

mod callback_result_sealed {
    use super::{Any, Result};

    pub trait Sealed {}

    impl<T: Into<Any>> Sealed for T {}
    impl<T: Into<Any>> Sealed for Result<T> {}
}

/// Convert an infallible or fallible map or mutation callback result into [`Result<Any>`].
///
/// A callback may return any value convertible into [`Any`], or wrap it in
/// [`Result`] to use `?`.
///
/// This trait is sealed and is not an extension point.
#[doc(hidden)]
pub trait IntoMutateResult: callback_result_sealed::Sealed {
    fn into_mutate_result(self) -> Result<Any>;
}

impl<T: Into<Any>> IntoMutateResult for T {
    #[inline]
    fn into_mutate_result(self) -> Result<Any> {
        Ok(self.into())
    }
}

impl<T: Into<Any>> IntoMutateResult for Result<T> {
    #[inline]
    fn into_mutate_result(self) -> Result<Any> {
        self.map(Into::into)
    }
}

/// Permission to attempt in-place structural mutation along an owned path.
///
/// `Allow` still requires unique ownership; it does not grant permission to
/// modify a borrowed value. Use an owned value or an engine-issued
/// [`InplaceValue`] with the mode-aware mutation helpers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InplaceMode {
    /// Use ordinary mutation, including for uniquely owned values.
    #[default]
    Disallow,
    /// Permit reuse when ownership and alias checks allow it.
    Allow,
}

impl InplaceMode {
    fn permit(self) -> Permit {
        match self {
            Self::Disallow => Permit::Copy,
            Self::Allow => Permit::MaybeInPlace,
        }
    }

    fn permit_if_unique(self, raw: TVMFFIAny) -> Permit {
        if self == Self::Allow && object_is_unique(raw) {
            Permit::MaybeInPlace
        } else {
            Permit::Copy
        }
    }
}

/// A callback value carrying the engine's in-place permission.
///
/// Unlike an owning typed argument, this handle does not increment the reference
/// count. Borrow through it to inspect the node, then consume it in
/// `default_maybe_inplace_mutate` to continue recursion. A surviving owning alias
/// still forces copying. Node borrows come from this handle, not from the
/// callback context, so the handle can safely be passed to another context.
/// `T` selects the callback's matched FFI type; `Any` matches every value.
///
/// A node borrow cannot survive consumption of the handle:
/// ```compile_fail
/// use tvm_ffi::{MutateContext, MutateValue};
/// fn invalid(value: MutateValue<'_>, ctx: &mut MutateContext<'_>) {
///     let node = value.as_node::<tvm_ffi::collections::array::ArrayObj>().unwrap();
///     ctx.default_maybe_inplace_mutate(value).unwrap();
///     println!("{}", node.size);
/// }
/// ```
///
/// Moving the handle into a nested callback cannot bypass that borrow:
/// ```compile_fail
/// use std::cell::RefCell;
/// use tvm_ffi::{structural_mutate, MutateContext, MutateValue};
/// fn invalid(value: MutateValue<'_>) {
///     let node = value.as_node::<tvm_ffi::collections::array::ArrayObj>().unwrap();
///     let pending = RefCell::new(Some(value));
///     structural_mutate(true, |_: bool, inner: &mut MutateContext<'_>| {
///         inner.default_maybe_inplace_mutate(pending.borrow_mut().take().unwrap())
///     }).unwrap();
///     println!("{}", node.size);
/// }
/// ```
pub struct MutateValue<'a, T = Any> {
    value: StructuralView,
    mode: InplaceMode,
    _scope: PhantomData<&'a StructuralView>,
    _type: PhantomData<fn() -> T>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'a> MutateValue<'a> {
    fn new(value: &'a StructuralView, mode: InplaceMode) -> Self {
        Self {
            value: StructuralView::from_raw(value.raw()),
            mode,
            _scope: PhantomData,
            _type: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Wrap a borrowed value without granting in-place permission.
    pub fn borrowed(value: &'a StructuralView) -> Self {
        Self::new(value, InplaceMode::Disallow)
    }
}

impl<'a, T> MutateValue<'a, T> {
    /// Permission established before callback arguments acquire ownership.
    pub fn inplace_mode(&self) -> InplaceMode {
        self.mode
    }

    /// Borrow the current value. The borrow must end before default mutation.
    pub fn as_value(&self) -> &StructuralView {
        &self.value
    }

    /// Narrow the matched type without acquiring an owning reference.
    pub fn try_cast<U: crate::type_traits::ContainerElement>(
        self,
    ) -> std::result::Result<MutateValue<'a, U>, Self> {
        // SAFETY: the engine keeps the borrowed value live for this handle's
        // lifetime. This only checks its type and does not acquire ownership.
        if unsafe { U::container_check_any_strict(&self.value.raw()) } {
            Ok(MutateValue {
                value: self.value,
                mode: self.mode,
                _scope: PhantomData,
                _type: PhantomData,
                _not_send_sync: PhantomData,
            })
        } else {
            Err(self)
        }
    }

    /// Erase the matched type while preserving the engine-issued permission.
    pub fn into_untyped(self) -> MutateValue<'a> {
        MutateValue {
            value: self.value,
            mode: self.mode,
            _scope: PhantomData,
            _type: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    fn permit(&self, requested: InplaceMode) -> Permit {
        if self.mode == InplaceMode::Disallow {
            Permit::Copy
        } else {
            requested.permit_if_unique(self.value.raw())
        }
    }
}

impl<T> Deref for MutateValue<'_, T> {
    type Target = StructuralView;
    fn deref(&self) -> &Self::Target {
        self.as_value()
    }
}

/// State and recursive operations shared by mutation callbacks and context policies.
///
/// A matched callback owns mutation of its value. Recursive operations
/// reborrow the mutator, so mutable state cannot remain borrowed across them.
/// The context does not store a node: inspect the callback argument and pass
/// it explicitly to default recursion.
pub struct MutateContext<'a, State = ()> {
    driver: &'a mut dyn MutateContextDriver<State>,
    def_region_kind: DefRegionKind,
    inplace_mode: InplaceMode,
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Recursion control passed to a typed `#[dispatch(mutate)]` handler.
///
/// The dispatch object owns all pass state. Recursive operations take that
/// object explicitly so Rust can safely reborrow the same `&mut self` for the
/// child call. Node access comes only from the callback argument; default
/// recursion takes that value explicitly.
pub struct Mutator {
    def_region_kind: DefRegionKind,
    inplace_mode: InplaceMode,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl Mutator {
    /// Engine-established permission for this callback's current value.
    ///
    /// This is not authority to modify a borrow: default in-place descent also
    /// requires consuming a [`MutateValue`].
    pub fn inplace_mode(&self) -> InplaceMode {
        self.inplace_mode
    }

    /// Definition-region state active at the callback's current value.
    #[inline(always)]
    pub fn def_region_kind(&self) -> DefRegionKind {
        self.def_region_kind
    }

    /// Mutate a borrowed child through the same typed dispatch object.
    #[inline(always)]
    pub fn mutate<D, T>(&mut self, dispatch: &mut D, value: &T) -> Result<Any>
    where
        D: MutateDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        StructuralMutator::mutate(dispatch, value, self.def_region_kind)
    }

    /// Mutate a borrowed child under an explicit definition-region state.
    #[inline(always)]
    pub fn mutate_with<D, T>(
        &mut self,
        dispatch: &mut D,
        value: &T,
        def_region_kind: DefRegionKind,
    ) -> Result<Any>
    where
        D: MutateDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        StructuralMutator::mutate(dispatch, value, def_region_kind)
    }

    /// Mutate an owned child and permit reuse when it remains uniquely owned.
    #[inline(always)]
    pub fn maybe_inplace_mutate<D, T>(&mut self, dispatch: &mut D, value: T) -> Result<Any>
    where
        D: MutateDispatch,
        T: Into<Any>,
    {
        self.maybe_inplace_mutate_with(dispatch, value, self.def_region_kind)
    }

    /// Mutate an owned child under an explicit definition-region state.
    #[inline(always)]
    pub fn maybe_inplace_mutate_with<D, T>(
        &mut self,
        dispatch: &mut D,
        value: T,
        def_region_kind: DefRegionKind,
    ) -> Result<Any>
    where
        D: MutateDispatch,
        T: Into<Any>,
    {
        self.maybe_inplace_mutate_with_mode(dispatch, value, def_region_kind, InplaceMode::Allow)
    }

    /// Mutate an owned child under an explicit region and in-place permission.
    ///
    /// `Disallow` keeps this input on the copy path even when uniquely owned.
    #[inline(always)]
    pub fn maybe_inplace_mutate_with_mode<D, T>(
        &mut self,
        dispatch: &mut D,
        value: T,
        def_region_kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any>
    where
        D: MutateDispatch,
        T: Into<Any>,
    {
        StructuralMutator::maybe_inplace_mutate_with_mode(dispatch, value, def_region_kind, mode)
    }

    /// Apply default copy-only mutation to an explicitly borrowed value.
    #[inline(always)]
    pub fn default_mutate<D, T>(&mut self, dispatch: &mut D, value: &T) -> Result<Any>
    where
        D: MutateDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        StructuralMutator::default_mutate(dispatch, value, self.def_region_kind)
    }

    /// Mutate a borrowed child while preserving an unchanged result.
    #[inline]
    pub fn mutate_result<D, T>(&mut self, dispatch: &mut D, value: &T) -> Result<UnchangedOr<Any>>
    where
        D: MutateDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.mutate_with_result(dispatch, value, self.def_region_kind)
    }

    /// Mutate a borrowed child under an explicit region, preserving unchanged.
    #[inline]
    pub fn mutate_with_result<D, T>(
        &mut self,
        dispatch: &mut D,
        value: &T,
        kind: DefRegionKind,
    ) -> Result<UnchangedOr<Any>>
    where
        D: MutateDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        StructuralMutator::mutate_result(dispatch, value, kind)
    }

    /// Apply default mutation without materializing an unchanged original.
    #[inline]
    pub fn default_mutate_result<D, T>(
        &mut self,
        dispatch: &mut D,
        value: &T,
    ) -> Result<UnchangedOr<Any>>
    where
        D: MutateDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        StructuralMutator::default_mutate_result(dispatch, value, self.def_region_kind)
    }

    /// Consume the handle for default descent with its existing permission.
    #[inline]
    pub fn default_maybe_inplace_mutate<D: MutateDispatch, T>(
        &mut self,
        dispatch: &mut D,
        value: MutateValue<'_, T>,
    ) -> Result<Any> {
        let mode = value.inplace_mode();
        self.default_mutate_with_mode(dispatch, value, mode)
    }

    /// Consume the handle for default descent, preserving permission and `Unchanged`.
    #[inline]
    pub fn default_maybe_inplace_mutate_result<D: MutateDispatch, T>(
        &mut self,
        dispatch: &mut D,
        value: MutateValue<'_, T>,
    ) -> Result<UnchangedOr<Any>> {
        let mode = value.inplace_mode();
        self.default_mutate_with_mode_result(dispatch, value, mode)
    }

    /// Continue default mutation after relinquishing the callback value's borrows.
    ///
    /// The requested mode can restrict, but cannot upgrade, the engine-issued
    /// permission. Retained owning aliases force copying.
    pub fn default_mutate_with_mode<D: MutateDispatch, T>(
        &mut self,
        dispatch: &mut D,
        value: MutateValue<'_, T>,
        mode: InplaceMode,
    ) -> Result<Any> {
        let raw = value.value.raw();
        let permit = value.permit(mode);
        user_default_mutate(dispatch, raw, self.def_region_kind, permit)
            .and_then(|result| resolve_result(result, raw))
    }

    /// Consume the callback value for default descent, preserving `Unchanged`.
    pub fn default_mutate_with_mode_result<D: MutateDispatch, T>(
        &mut self,
        dispatch: &mut D,
        value: MutateValue<'_, T>,
        mode: InplaceMode,
    ) -> Result<UnchangedOr<Any>> {
        let permit = value.permit(mode);
        user_default_mutate(dispatch, value.value.raw(), self.def_region_kind, permit)
            .and_then(UnchangedOr::from_carrier)
    }

    /// Look up an invocation-local identity substitution.
    #[inline(always)]
    pub fn var_remap_get<D: MutateDispatch>(
        &mut self,
        dispatch: &mut D,
        var: &StructuralView,
    ) -> Result<Option<Any>> {
        StructuralMutator::var_remap_get(dispatch, var)
    }

    /// Store an invocation-local identity substitution.
    #[inline(always)]
    pub fn var_remap_set<D: MutateDispatch>(
        &mut self,
        dispatch: &mut D,
        var: &StructuralView,
        mutated_value: &Any,
    ) -> Result<()> {
        StructuralMutator::var_remap_set(dispatch, var, mutated_value)
    }
}

/// Internal operations used by [`MutateContext`].
///
/// Borrowed inputs carry a lifetime; in-place entry requires an owned
/// value or a consumed capability, never a bare ABI value plus permission.
trait MutateContextDriver<State> {
    fn state(&self) -> &State;
    fn state_mut(&mut self) -> &mut State;
    fn mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any>;
    fn mutate_owned(&mut self, value: Any, kind: DefRegionKind, mode: InplaceMode) -> Result<Any>;
    fn default_mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any>;
    /// Consume the handle to exclude node borrows before default descent.
    fn default_mutate_value(
        &mut self,
        value: MutateValue<'_>,
        kind: DefRegionKind,
        _mode: InplaceMode,
    ) -> Result<Any> {
        self.default_mutate_borrowed(AnyView::from(value.as_value()), kind)
    }
    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>>;
    fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()>;
}

impl<State> MutateContext<'_, State> {
    /// User state shared by every callback in this mutation.
    #[inline(always)]
    pub fn state(&self) -> &State {
        self.driver.state()
    }

    /// Mutably borrow the user state.
    #[inline(always)]
    pub fn state_mut(&mut self) -> &mut State {
        self.driver.state_mut()
    }

    /// Engine-established permission for this callback's current value.
    ///
    /// This is not authority to modify a borrow: default in-place descent also
    /// requires consuming a [`MutateValue`].
    pub fn inplace_mode(&self) -> InplaceMode {
        self.inplace_mode
    }

    /// Definition-region state active at the callback's current value.
    #[inline(always)]
    pub fn def_region_kind(&self) -> DefRegionKind {
        self.def_region_kind
    }

    /// Run recursive operations in a definition region, preserving an outer Pattern.
    /// The previous context is restored on return, error, or unwinding.
    pub fn with_def_region_kind<T>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut MutateContext<'_, State>) -> Result<T>,
    ) -> Result<T> {
        with_mutation_region(kind, |kind| {
            callback(&mut MutateContext {
                driver: &mut *self.driver,
                def_region_kind: kind,
                inplace_mode: self.inplace_mode,
                _not_send_sync: PhantomData,
            })
        })
    }

    /// Mutate a borrowed value through the same callback chain. The value and
    /// its descendants begin on the non-in-place path.
    #[inline(always)]
    pub fn mutate<T>(&mut self, value: &T) -> Result<Any>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.mutate_with(value, self.def_region_kind)
    }

    /// Mutate a borrowed value under an explicit definition-region state.
    #[inline(always)]
    pub fn mutate_with<T>(&mut self, value: &T, def_region_kind: DefRegionKind) -> Result<Any>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        self.driver
            .mutate_borrowed(view, def_region_kind)
            .and_then(|result| resolve_result(result, *view.as_raw_ffi_any()))
    }

    /// Mutate an owned value, allowing an in-place attempt when it remains
    /// uniquely owned and no matched callback borrows it.
    #[inline(always)]
    pub fn maybe_inplace_mutate<T: Into<Any>>(&mut self, value: T) -> Result<Any> {
        self.maybe_inplace_mutate_with(value, self.def_region_kind)
    }

    /// Mutate an owned value under an explicit definition-region state.
    #[inline(always)]
    pub fn maybe_inplace_mutate_with<T: Into<Any>>(
        &mut self,
        value: T,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.maybe_inplace_mutate_with_mode(value, def_region_kind, InplaceMode::Allow)
    }

    /// Mutate an owned value under an explicit region and in-place permission.
    ///
    /// `Disallow` keeps this input on the copy path even when uniquely owned.
    #[inline(always)]
    pub fn maybe_inplace_mutate_with_mode<T: Into<Any>>(
        &mut self,
        value: T,
        def_region_kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        self.driver
            .mutate_owned(value.into(), def_region_kind, mode)
    }

    /// Apply default copy-only mutation to an explicitly borrowed value.
    ///
    /// This may be called repeatedly while holding node borrows. The value's
    /// children re-enter callback dispatch, but the value itself does not.
    #[inline(always)]
    pub fn default_mutate<T>(&mut self, value: &T) -> Result<Any>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        let raw = *view.as_raw_ffi_any();
        self.driver
            .default_mutate_borrowed(view, self.def_region_kind)
            .and_then(|result| resolve_result(result, raw))
    }

    /// Mutate a borrowed child while preserving an unchanged result.
    #[inline]
    pub fn mutate_result<T>(&mut self, value: &T) -> Result<UnchangedOr<Any>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.mutate_with_result(value, self.def_region_kind)
    }

    /// Mutate a borrowed child under an explicit region, preserving unchanged.
    #[inline]
    pub fn mutate_with_result<T>(
        &mut self,
        value: &T,
        kind: DefRegionKind,
    ) -> Result<UnchangedOr<Any>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        self.driver
            .mutate_borrowed(view, kind)
            .and_then(UnchangedOr::from_carrier)
    }

    /// Apply default mutation without materializing an unchanged original.
    #[inline]
    pub fn default_mutate_result<T>(&mut self, value: &T) -> Result<UnchangedOr<Any>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        self.driver
            .default_mutate_borrowed(view, self.def_region_kind)
            .and_then(UnchangedOr::from_carrier)
    }

    /// Consume the handle for default descent with its existing permission.
    #[inline]
    pub fn default_maybe_inplace_mutate<T>(&mut self, value: MutateValue<'_, T>) -> Result<Any> {
        let mode = value.inplace_mode();
        self.default_mutate_with_mode(value, mode)
    }

    /// Consume the handle for default descent, preserving permission and `Unchanged`.
    #[inline]
    pub fn default_maybe_inplace_mutate_result<T>(
        &mut self,
        value: MutateValue<'_, T>,
    ) -> Result<UnchangedOr<Any>> {
        let mode = value.inplace_mode();
        self.default_mutate_with_mode_result(value, mode)
    }

    /// Continue default mutation after relinquishing the callback value's borrows.
    ///
    /// The mode can restrict, but cannot upgrade, the handle's permission.
    /// Retained owning aliases force copying; temporary handles can be dropped
    /// before this call to preserve reuse of the current value.
    pub fn default_mutate_with_mode<T>(
        &mut self,
        value: MutateValue<'_, T>,
        mode: InplaceMode,
    ) -> Result<Any> {
        let raw = value.value.raw();
        self.driver
            .default_mutate_value(value.into_untyped(), self.def_region_kind, mode)
            .and_then(|result| resolve_result(result, raw))
    }

    /// Consume the callback value for default descent, preserving `Unchanged`.
    pub fn default_mutate_with_mode_result<T>(
        &mut self,
        value: MutateValue<'_, T>,
        mode: InplaceMode,
    ) -> Result<UnchangedOr<Any>> {
        self.driver
            .default_mutate_value(value.into_untyped(), self.def_region_kind, mode)
            .and_then(UnchangedOr::from_carrier)
    }

    /// Look up an invocation-local identity substitution.
    #[inline(always)]
    pub fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        self.driver.var_remap_get(var)
    }

    /// Store an invocation-local identity substitution.
    #[inline(always)]
    pub fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
        self.driver.var_remap_set(var, mutated_value)
    }
}

/// Conversion into the mutator argument accepted by [`structural_mutate`].
///
/// Accepts a mutable low-level [`StructuralMutator`], a generated
/// [`MutateDispatch`], or a first-match callback chain. Generated dispatch
/// objects keep mutable pass state directly on themselves; [`MutateCallbacks`]
/// remains available for closure callback chains with separate state.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a supported `structural_mutate` mutator",
    note = "accepted mutators: `&mut U` where `U: StructuralMutator`; a generated `MutateDispatch`; an `Fn` callback over an FFI value type `T`, `&N` of an object node type, or `&StructuralView`, or a consuming `MutateValue<T>`, followed by `&mut MutateContext<'_, State>`; or a tuple of up to 12 such callbacks (tuples may nest)",
    note = "callback arguments need explicit type annotations; use `MutateCallbacks::new(state, callbacks)` for ordinary mutable callback state"
)]
pub trait IntoMutator<Marker> {
    #[doc(hidden)]
    fn mutate_root(self, root: Any) -> Result<Any>;
}

impl<U: StructuralMutator> IntoMutator<U> for &mut U {
    fn mutate_root(self, root: Any) -> Result<Any> {
        run_structural_mutator(root, self)
    }
}

/// One typed callback in a callback-driven structural mutator.
pub trait MutateChainLink<State, Marker>: mutate_sealed::SealedLink<State, Marker> {
    #[doc(hidden)]
    fn try_mutate(
        &self,
        value: &mut Option<MutateValue<'_>>,
        mutator: &mut MutateContext<'_, State>,
    ) -> Option<Result<Any>>;
}

/// Ordered typed callback dispatch for [`structural_mutate`].
///
/// `None` means no handler matched, so structural mutation applies its default
/// behavior. A generated `#[dispatch(mutate)]` implementation tests
/// `mutate_*` methods in source order and passes the same [`Mutator`] to the
/// first match. The implementation owns its pass state and receives `&mut
/// self`, while [`Mutator`] controls recursion, the definition region, and
/// in-place permission. Handlers can consume [`MutateValue`] and take an
/// optional `&mut Mutator` to inspect the region and in-place permission.
pub trait MutateDispatch: Sized {
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        mutator: &mut Mutator,
    ) -> Option<Result<Any>>;

    /// Dispatch an engine-issued value. Existing borrowed dispatchers retain
    /// their copy-only default recursion; generated dispatch supports consuming it.
    fn dispatch_mutate_value(
        &mut self,
        value: MutateValue<'_>,
        mutator: &mut Mutator,
    ) -> Option<Result<Any>> {
        self.dispatch_mutate(value.as_value(), mutator)
    }

    #[doc(hidden)]
    fn on_default_mutate(&mut self, value: MutateValue<'_>, kind: DefRegionKind) -> Result<Any> {
        default_mutate_driver(
            self,
            value.value.raw(),
            kind,
            value.permit(value.inplace_mode()),
        )
    }
}

impl<D: MutateDispatch> IntoMutator<ByMutateDispatch> for D {
    #[inline]
    fn mutate_root(mut self, root: Any) -> Result<Any> {
        run_structural_mutator(root, &mut self)
    }
}

mod mutate_sealed {
    use super::{IntoMutateResult, MutateContext, MutateValue, ObjectCore, StructuralView};

    pub trait SealedLink<State, Marker> {}

    impl<F, State, T, O> SealedLink<State, super::ByMutateValue<T>> for F
    where
        F: for<'value, 'mutator, 'driver> Fn(
            MutateValue<'value, T>,
            &'mutator mut MutateContext<'driver, State>,
        ) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, State, T, O> SealedLink<State, super::ByMutateOwned<T>> for F
    where
        F: for<'mutator, 'driver> Fn(T, &'mutator mut MutateContext<'driver, State>) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, State, N: ObjectCore, O> SealedLink<State, super::ByMutateNode<N>> for F
    where
        F: for<'value, 'mutator, 'driver> Fn(
            &'value N,
            &'mutator mut MutateContext<'driver, State>,
        ) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, State, O> SealedLink<State, super::ByMutateCatchAll> for F
    where
        F: for<'value, 'mutator, 'driver> Fn(
            &'value StructuralView,
            &'mutator mut MutateContext<'driver, State>,
        ) -> O,
        O: IntoMutateResult,
    {
    }
}

#[doc(hidden)]
pub enum ByMutateDispatch {}

#[doc(hidden)]
pub struct ByMutateValue<T>(PhantomData<T>);

impl<F, State, T, O> MutateChainLink<State, ByMutateValue<T>> for F
where
    F: for<'value, 'mutator, 'driver> Fn(
        MutateValue<'value, T>,
        &'mutator mut MutateContext<'driver, State>,
    ) -> O,
    T: crate::type_traits::ContainerElement,
    O: IntoMutateResult,
{
    fn try_mutate(
        &self,
        value: &mut Option<MutateValue<'_>>,
        mutator: &mut MutateContext<'_, State>,
    ) -> Option<Result<Any>> {
        match value
            .take()
            .expect("unconsumed callback value")
            .try_cast::<T>()
        {
            Ok(typed) => Some(self(typed, mutator).into_mutate_result()),
            Err(original) => {
                *value = Some(original);
                None
            }
        }
    }
}

#[doc(hidden)]
pub struct ByMutateOwned<T>(PhantomData<T>);

impl<F, State, T, O> MutateChainLink<State, ByMutateOwned<T>> for F
where
    F: for<'mutator, 'driver> Fn(T, &'mutator mut MutateContext<'driver, State>) -> O,
    T: crate::type_traits::AnyCompatible,
    O: IntoMutateResult,
{
    fn try_mutate(
        &self,
        value: &mut Option<MutateValue<'_>>,
        mutator: &mut MutateContext<'_, State>,
    ) -> Option<Result<Any>> {
        value
            .as_ref()
            .expect("unconsumed callback value")
            .cast::<T>()
            .map(|typed| self(typed, mutator).into_mutate_result())
    }
}

#[doc(hidden)]
pub struct ByMutateNode<N>(PhantomData<N>);

impl<F, State, N, O> MutateChainLink<State, ByMutateNode<N>> for F
where
    F: for<'value, 'mutator, 'driver> Fn(
        &'value N,
        &'mutator mut MutateContext<'driver, State>,
    ) -> O,
    N: ObjectCore,
    O: IntoMutateResult,
{
    fn try_mutate(
        &self,
        value: &mut Option<MutateValue<'_>>,
        mutator: &mut MutateContext<'_, State>,
    ) -> Option<Result<Any>> {
        value
            .as_ref()
            .expect("unconsumed callback value")
            .as_node::<N>()
            .map(|node| self(node, mutator).into_mutate_result())
    }
}

#[doc(hidden)]
pub enum ByMutateCatchAll {}

impl<F, State, O> MutateChainLink<State, ByMutateCatchAll> for F
where
    F: for<'value, 'mutator, 'driver> Fn(
        &'value StructuralView,
        &'mutator mut MutateContext<'driver, State>,
    ) -> O,
    O: IntoMutateResult,
{
    fn try_mutate(
        &self,
        value: &mut Option<MutateValue<'_>>,
        mutator: &mut MutateContext<'_, State>,
    ) -> Option<Result<Any>> {
        Some(
            self(
                value
                    .as_ref()
                    .expect("unconsumed callback value")
                    .as_value(),
                mutator,
            )
            .into_mutate_result(),
        )
    }
}

#[doc(hidden)]
pub struct ByMutateChainLink<Markers>(PhantomData<fn(Markers)>);

macro_rules! impl_mutate_chain_link {
    ($(($F:ident, $M:ident, $idx:tt)),+) => {
        impl<State, $($F, $M,)+>
            mutate_sealed::SealedLink<State, ByMutateChainLink<($($M,)+)>> for ($($F,)+)
        where
            $($F: MutateChainLink<State, $M>,)+
        {
        }

        impl<State, $($F, $M,)+> MutateChainLink<State, ByMutateChainLink<($($M,)+)>>
            for ($($F,)+)
        where
            $($F: MutateChainLink<State, $M>,)+
        {
            fn try_mutate(
                &self,
                value: &mut Option<MutateValue<'_>>,
                mutator: &mut MutateContext<'_, State>,
            ) -> Option<Result<Any>> {
                $(
                    if let Some(result) = self.$idx.try_mutate(value, mutator) {
                        return Some(result);
                    }
                )+
                None
            }
        }
    };
}

impl_callback_chain_tuple_arities!(impl_mutate_chain_link);

/// A reusable typed-dispatch or callback mutator with shared user state.
pub struct MutateCallbacks<State, Link, Marker, Policy = DefaultMutContextPolicy> {
    policy: Option<Rc<Policy>>,
    state: State,
    callbacks: Rc<Link>,
    _marker: PhantomData<fn(Marker)>,
}

impl<State, Link, Marker> MutateCallbacks<State, Link, Marker>
where
    Link: MutateChainLink<State, Marker>,
{
    /// Construct a stateful callback mutator.
    pub fn new(state: State, callbacks: Link) -> Self {
        Self {
            state,
            callbacks: Rc::new(callbacks),
            policy: None,
            _marker: PhantomData,
        }
    }
}

impl<State, Link, Marker, Policy> MutateCallbacks<State, Link, Marker, Policy> {
    /// Customize default descent while sharing the callback state.
    pub fn with_policy<P: MutContextPolicy<State>>(
        self,
        policy: P,
    ) -> MutateCallbacks<State, Link, Marker, P> {
        MutateCallbacks {
            policy: Some(Rc::new(policy)),
            state: self.state,
            callbacks: self.callbacks,
            _marker: PhantomData,
        }
    }

    /// Shared access to the callback state.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Mutable access to callback state outside an active recursive call.
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Consume the mutator and return its state.
    pub fn into_state(self) -> State {
        self.state
    }
}

struct DirectMutateCallbacks<'a, Link, Marker> {
    state: (),
    callbacks: &'a Link,
    _marker: PhantomData<fn(Marker)>,
}

#[doc(hidden)]
pub trait MutateCallbackState<State> {
    fn callback_state(&self) -> &State;
    fn callback_state_mut(&mut self) -> &mut State;
}

impl<State, Link, Marker, Policy> MutateCallbackState<State>
    for MutateCallbacks<State, Link, Marker, Policy>
{
    fn callback_state(&self) -> &State {
        &self.state
    }

    fn callback_state_mut(&mut self) -> &mut State {
        &mut self.state
    }
}

impl<Link, Marker> MutateCallbackState<()> for DirectMutateCallbacks<'_, Link, Marker> {
    fn callback_state(&self) -> &() {
        &self.state
    }

    fn callback_state_mut(&mut self) -> &mut () {
        &mut self.state
    }
}

#[doc(hidden)]
pub struct ByMutateCallbacks<Marker>(PhantomData<fn(Marker)>);

impl<Link, Marker> IntoMutator<ByMutateCallbacks<Marker>> for Link
where
    Link: MutateChainLink<(), Marker>,
{
    fn mutate_root(self, root: Any) -> Result<Any> {
        let callbacks = self;
        let mut mutator = DirectMutateCallbacks::<Link, Marker> {
            state: (),
            callbacks: &callbacks,
            _marker: PhantomData,
        };
        run_structural_mutator(root, &mut mutator)
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
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>>;
}

/// Internal root-mapping protocol used by [`IntoMapper`].
#[doc(hidden)]
pub trait NativeMap: Sized {
    fn map_root(&mut self, root: Any, order: WalkOrder) -> Result<Any>;
}

impl<D: MapDispatch> NativeMap for D {
    fn map_root(&mut self, root: Any, order: WalkOrder) -> Result<Any> {
        run_native_mapper::<_, DefaultMutContextPolicy>(root, self, None, order)
    }
}

impl<V: MapDispatch> MapDispatch for &mut V {
    #[inline]
    fn dispatch_map(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        (**self).dispatch_map(value, def_region_kind)
    }
}

/// Conversion into the mapper consumed by [`structural_map`].
#[diagnostic::on_unimplemented(
    message = "unsupported structural-map callback shape",
    label = "this value cannot be used as a structural mapper",
    note = "pass `&mut` a `MapDispatch`, a supported closure or callback tuple, or a `MapWithContextPolicy`"
)]
pub trait IntoMapper<Marker> {
    type Mapper: NativeMap;
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
/// Links use first-match order and may receive an owned FFI value, borrowed
/// object node, or `&StructuralView`, optionally followed by [`DefRegionKind`].
pub trait MapChainLink<Marker>: sealed_map::SealedMapLink<Marker> {
    #[doc(hidden)]
    fn try_map(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>>;
}

mod sealed_map {
    use super::{DefRegionKind, IntoMutateResult, MapDispatch, ObjectCore, StructuralView};

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
        F: for<'a> FnMut(&'a StructuralView) -> O,
        O: IntoMutateResult,
    {
    }

    impl<F, O> SealedMapLink<super::ByMapCatchAllKind> for F
    where
        F: for<'a> FnMut(&'a StructuralView, DefRegionKind) -> O,
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
        value: &StructuralView,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .cast::<T>()
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
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .cast::<T>()
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
        value: &StructuralView,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .as_node::<N>()
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
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .as_node::<N>()
            .map(|node| self(node, def_region_kind).into_mutate_result())
    }
}

#[doc(hidden)]
pub enum ByMapCatchAll {}

impl<F, O> MapChainLink<ByMapCatchAll> for F
where
    F: for<'a> FnMut(&'a StructuralView) -> O,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: &StructuralView,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        Some(self(value).into_mutate_result())
    }
}

#[doc(hidden)]
pub enum ByMapCatchAllKind {}

impl<F, O> MapChainLink<ByMapCatchAllKind> for F
where
    F: for<'a> FnMut(&'a StructuralView, DefRegionKind) -> O,
    O: IntoMutateResult,
{
    #[inline]
    fn try_map(
        &mut self,
        value: &StructuralView,
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
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        self.dispatch_map(value, def_region_kind)
    }
}

/// Adapter from a [`MapChainLink`] to [`MapDispatch`].
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

impl<Link, Marker> MapDispatch for MapChain<Link, Marker>
where
    Link: MapChainLink<Marker>,
{
    #[inline]
    fn dispatch_map(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
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
                value: &StructuralView,
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
    F: for<'a> FnMut(&'a StructuralView) -> O,
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
    F: for<'a> FnMut(&'a StructuralView, DefRegionKind) -> O,
    O: IntoMutateResult,
{
    type Mapper = MapChain<F, ByMapCatchAllKind>;

    #[inline]
    fn into_mapper(self) -> Self::Mapper {
        MapChain::new(self)
    }
}

/// Engine-issued permission to attempt in-place mutation of one value.
///
/// The engine issues it only when the current ownership path permits reuse.
pub struct InplaceValue<'a> {
    value: StructuralView,
    _scope: PhantomData<&'a mut TVMFFIAny>,
}

impl<'a> InplaceValue<'a> {
    #[inline]
    fn from_raw(raw: &'a mut TVMFFIAny) -> Self {
        Self {
            value: StructuralView::from_raw(*raw),
            _scope: PhantomData,
        }
    }

    /// Borrow the value without its in-place capability.
    #[inline]
    pub fn as_value(&self) -> &StructuralView {
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
    type Target = StructuralView;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_value()
    }
}

/// Identity substitutions for a custom [`StructuralMutator`] remapping policy.
///
/// The map owns its keys and values so object addresses remain stable.
#[derive(Default)]
pub struct StructuralVarRemap {
    entries: HashMap<NonNull<TVMFFIObject>, MemoEntry, BuildHasherDefault<IdentityHasher>>,
}

#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }

    fn write_usize(&mut self, value: usize) {
        self.0 = value as u64;
    }

    fn finish(&self) -> u64 {
        // Mix address alignment bits into both the bucket index and control tag.
        let mut hash = self.0;
        hash = (hash ^ (hash >> 33)).wrapping_mul(0xff51afd7ed558ccd);
        hash = (hash ^ (hash >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
        hash ^ (hash >> 33)
    }
}

impl StructuralVarRemap {
    /// Look up an identity replacement previously stored for `var`.
    pub fn get(&self, var: &StructuralView) -> Result<Option<Any>> {
        let key = object_identity_key(var.raw())?;
        Ok(self.entries.get(&key).map(|entry| entry.result.clone()))
    }

    /// Store a descent result or an [`Unchanged`] marker for `var`.
    pub fn set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
        let key = object_identity_key(var.raw())?;
        self.entries.insert(
            key,
            MemoEntry {
                _original: var.to_owned(),
                result: mutated_value.clone(),
            },
        );
        Ok(())
    }

    /// Remove every recorded identity substitution.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// A low-level mutator that controls its own recursion.
///
/// Implementations descend with the `mutate` or `default_*` helpers.
/// Prefer mutation callbacks or `#[dispatch(mutate)]` for typed dispatch with
/// recursion supplied through [`Mutator`].
pub trait StructuralMutator: Sized {
    /// Dispatch one borrowed value without modifying its source storage.
    ///
    /// The structural-mutation engine calls this hook for each value.
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Result<Any>;

    /// Dispatch one value for which the engine permits an in-place attempt.
    ///
    /// The default delegates to [`Self::dispatch_mutate`] and therefore remains
    /// non-in-place. Override this method to opt into the default container
    /// reuse path.
    fn dispatch_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.dispatch_mutate(value.as_value(), def_region_kind)
    }

    #[doc(hidden)]
    fn on_default_mutate(&mut self, value: MutateValue<'_>, kind: DefRegionKind) -> Result<Any> {
        default_mutate_driver(
            self,
            value.value.raw(),
            kind,
            value.permit(value.inplace_mode()),
        )
    }

    /// Re-enter this mutator for a borrowed value. The value and all of its
    /// descendants use the non-in-place path.
    fn mutate<T>(&mut self, value: &T, def_region_kind: DefRegionKind) -> Result<Any>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        dispatch_user_raw(self, *view.as_raw_ffi_any(), def_region_kind, Permit::Copy)
            .and_then(|result| resolve_result(result, *view.as_raw_ffi_any()))
    }

    /// Re-enter this mutator for an owned value, permitting reuse only when
    /// the converted value remains uniquely owned.
    fn maybe_inplace_mutate<T>(&mut self, value: T, def_region_kind: DefRegionKind) -> Result<Any>
    where
        T: Into<Any>,
    {
        self.maybe_inplace_mutate_with_mode(value, def_region_kind, InplaceMode::Allow)
    }

    /// Re-enter for an owned value with an explicit region and in-place permission.
    ///
    /// `Allow` checks uniqueness before dispatch; `Disallow` uses ordinary
    /// mutation without checking uniqueness. An independently owned replacement
    /// or a later explicit owned entry can establish a new permission boundary.
    fn maybe_inplace_mutate_with_mode<T>(
        &mut self,
        value: T,
        def_region_kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any>
    where
        T: Into<Any>,
    {
        let value = value.into();
        let result = dispatch_user_raw(
            self,
            *value.as_raw_ffi_any(),
            def_region_kind,
            mode.permit(),
        )?;
        Ok(if is_unchanged(&result) { value } else { result })
    }

    /// Apply default non-in-place mutation to a borrowed typed value.
    ///
    /// Unlike [Self::mutate], this bypasses dispatch for the value
    /// itself while its children still re-enter this mutator. This lets a
    /// typed structural-mutate handler recurse through its current node
    /// before applying a post-order rewrite.
    fn default_mutate<T>(&mut self, value: &T, def_region_kind: DefRegionKind) -> Result<Any>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        user_default_mutate(self, *view.as_raw_ffi_any(), def_region_kind, Permit::Copy)
            .and_then(|result| resolve_result(result, *view.as_raw_ffi_any()))
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
        self.default_maybe_inplace_mutate_with_mode(value, def_region_kind, InplaceMode::Allow)
    }

    /// Apply default mutation with a capability and an explicit permission.
    ///
    /// `Disallow` overrides the capability and uses the copy path. `Allow`
    /// rechecks uniqueness because the handler may have retained an owning
    /// alias. Consuming the capability prevents a borrow from it surviving this
    /// call. Unlike the C++ default-descent API, safe Rust cannot simply trust
    /// the earlier uniqueness check after arbitrary user code has run.
    fn default_maybe_inplace_mutate_with_mode(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        let raw = value.raw();
        let permit = mode.permit_if_unique(raw);
        user_default_mutate(self, raw, def_region_kind, permit)
            .and_then(|result| resolve_result(result, raw))
    }

    /// Re-enter this mutator for a borrowed value, preserving unchanged.
    fn mutate_result<T>(&mut self, value: &T, kind: DefRegionKind) -> Result<UnchangedOr<Any>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        dispatch_user_raw(self, *view.as_raw_ffi_any(), kind, Permit::Copy)
            .and_then(UnchangedOr::from_carrier)
    }

    /// Default mutation of a borrowed typed value, preserving unchanged.
    fn default_mutate_result<T>(
        &mut self,
        value: &T,
        kind: DefRegionKind,
    ) -> Result<UnchangedOr<Any>>
    where
        for<'x> AnyView<'x>: From<&'x T>,
    {
        let view = AnyView::from(value);
        user_default_mutate(self, *view.as_raw_ffi_any(), kind, Permit::Copy)
            .and_then(UnchangedOr::from_carrier)
    }

    /// Default mutation under an engine-issued capability, preserving unchanged.
    fn default_maybe_inplace_mutate_result(
        &mut self,
        value: InplaceValue<'_>,
        kind: DefRegionKind,
    ) -> Result<UnchangedOr<Any>> {
        self.default_maybe_inplace_mutate_with_mode_result(value, kind, InplaceMode::Allow)
    }

    /// Default mutation with a capability and permission, preserving unchanged.
    ///
    /// Uses the same ownership checks as [`Self::default_maybe_inplace_mutate_with_mode`].
    fn default_maybe_inplace_mutate_with_mode_result(
        &mut self,
        value: InplaceValue<'_>,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<UnchangedOr<Any>> {
        let raw = value.raw();
        let permit = mode.permit_if_unique(raw);
        user_default_mutate(self, raw, kind, permit).and_then(UnchangedOr::from_carrier)
    }

    /// Look up a FreeVar or DAG-node substitution from the active mutation.
    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        invocation_var_remap_get(self, var)
    }

    /// Store a FreeVar or DAG-node substitution for the active mutation.
    fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
        invocation_var_remap_set(self, var, mutated_value)
    }
}

impl<D: MutateDispatch> StructuralMutator for D {
    fn on_default_mutate(&mut self, value: MutateValue<'_>, kind: DefRegionKind) -> Result<Any> {
        MutateDispatch::on_default_mutate(self, value, kind)
    }

    #[inline(always)]
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        let mut mutator = Mutator {
            def_region_kind,
            inplace_mode: InplaceMode::Disallow,
            _not_send_sync: PhantomData,
        };
        match MutateDispatch::dispatch_mutate_value(
            self,
            MutateValue::borrowed(value),
            &mut mutator,
        ) {
            Some(result) => result,
            None => user_default_mutate(self, value.raw(), def_region_kind, Permit::Copy),
        }
    }

    #[inline(always)]
    fn dispatch_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        let mut mutator = Mutator {
            def_region_kind,
            inplace_mode: InplaceMode::Allow,
            _not_send_sync: PhantomData,
        };
        match MutateDispatch::dispatch_mutate_value(
            self,
            MutateValue::new(value.as_value(), InplaceMode::Allow),
            &mut mutator,
        ) {
            Some(result) => result,
            None => {
                let raw = value.raw();
                let permit = if object_is_unique(raw) {
                    Permit::MaybeInPlace
                } else {
                    Permit::Copy
                };
                user_default_mutate(self, raw, def_region_kind, permit)
            }
        }
    }
}

#[doc(hidden)]
pub fn default_mutate_with_policy<D: MutateDispatch + MutateCallbackState<D>>(
    dispatch: &mut D,
    policy: &impl MutContextPolicy<D>,
    value: MutateValue<'_>,
    kind: DefRegionKind,
) -> Result<Any> {
    policy::mutate_with_policy(
        &mut policy::MutationDescent { driver: dispatch },
        policy,
        value,
        kind,
    )
}

// Closure callback chains use a type-erased context driver so one concrete
// function signature can recurse through the complete chain.
#[inline(always)]
fn try_mutate_callbacks<State, Link, Marker, Driver>(
    driver: &mut Driver,
    callback_ptr: *const Link,
    value: &StructuralView,
    def_region_kind: DefRegionKind,
    inplace_mode: InplaceMode,
) -> Option<Result<Any>>
where
    Link: MutateChainLink<State, Marker>,
    Driver: MutateContextDriver<State>,
{
    let mut mutator = MutateContext {
        driver,
        def_region_kind,
        inplace_mode,
        _not_send_sync: PhantomData,
    };
    // SAFETY: The owning `Rc` or the direct callback's stack slot remains live
    // and is never modified through the driver during recursive reentry.
    unsafe {
        (&*callback_ptr).try_mutate(
            &mut Some(MutateValue::new(value, inplace_mode)),
            &mut mutator,
        )
    }
}

impl<State, Link, Marker, Policy> StructuralMutator for MutateCallbacks<State, Link, Marker, Policy>
where
    Policy: MutContextPolicy<State>,
    Link: MutateChainLink<State, Marker>,
{
    #[inline(always)]
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        let callback_ptr = Rc::as_ptr(&self.callbacks);
        match try_mutate_callbacks::<State, Link, Marker, _>(
            self,
            callback_ptr,
            value,
            def_region_kind,
            InplaceMode::Disallow,
        ) {
            Some(result) => result,
            None => user_default_mutate(self, value.raw(), def_region_kind, Permit::Copy),
        }
    }

    #[inline(always)]
    fn dispatch_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        let callback_ptr = Rc::as_ptr(&self.callbacks);
        match try_mutate_callbacks::<State, Link, Marker, _>(
            self,
            callback_ptr,
            value.as_value(),
            def_region_kind,
            InplaceMode::Allow,
        ) {
            Some(result) => result,
            None => self
                .default_maybe_inplace_mutate_result(value, def_region_kind)
                .map(Any::from),
        }
    }

    fn on_default_mutate(&mut self, value: MutateValue<'_>, kind: DefRegionKind) -> Result<Any> {
        match self.policy.as_ref().map(Rc::as_ptr) {
            Some(policy) => policy::mutate_with_policy(
                &mut policy::MutationDescent { driver: self },
                // SAFETY: callbacks cannot replace the policy; its owning Rc
                // remains live throughout this recursive call.
                unsafe { &*policy },
                value,
                kind,
            ),
            None => default_mutate_driver(
                self,
                value.value.raw(),
                kind,
                value.permit(value.inplace_mode()),
            ),
        }
    }
}

impl<Link, Marker> StructuralMutator for DirectMutateCallbacks<'_, Link, Marker>
where
    Link: MutateChainLink<(), Marker>,
{
    #[inline(always)]
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        let callback_ptr = std::ptr::from_ref(self.callbacks);
        match try_mutate_callbacks::<(), Link, Marker, _>(
            self,
            callback_ptr,
            value,
            def_region_kind,
            InplaceMode::Disallow,
        ) {
            Some(result) => result,
            None => user_default_mutate(self, value.raw(), def_region_kind, Permit::Copy),
        }
    }

    #[inline(always)]
    fn dispatch_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        let callback_ptr = std::ptr::from_ref(self.callbacks);
        match try_mutate_callbacks::<(), Link, Marker, _>(
            self,
            callback_ptr,
            value.as_value(),
            def_region_kind,
            InplaceMode::Allow,
        ) {
            Some(result) => result,
            None => self
                .default_maybe_inplace_mutate_result(value, def_region_kind)
                .map(Any::from),
        }
    }
}

impl<State, Driver> MutateContextDriver<State> for Driver
where
    Driver: StructuralMutator + MutateCallbackState<State>,
{
    #[inline(always)]
    fn state(&self) -> &State {
        self.callback_state()
    }

    #[inline(always)]
    fn state_mut(&mut self) -> &mut State {
        self.callback_state_mut()
    }

    #[inline(always)]
    fn mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any> {
        dispatch_user_raw(self, *value.as_raw_ffi_any(), kind, Permit::Copy)
    }

    #[inline(always)]
    fn mutate_owned(&mut self, value: Any, kind: DefRegionKind, mode: InplaceMode) -> Result<Any> {
        StructuralMutator::maybe_inplace_mutate_with_mode(self, value, kind, mode)
    }

    #[inline(always)]
    fn default_mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any> {
        user_default_mutate(self, *value.as_raw_ffi_any(), kind, Permit::Copy)
    }

    fn default_mutate_value(
        &mut self,
        value: MutateValue<'_>,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        let permit = value.permit(mode);
        user_default_mutate(self, value.value.raw(), kind, permit)
    }

    #[inline(always)]
    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        <Self as StructuralMutator>::var_remap_get(self, var)
    }

    #[inline(always)]
    fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
        <Self as StructuralMutator>::var_remap_set(self, var, mutated_value)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Permit {
    Copy,
    MaybeInPlace,
}

impl Permit {
    fn inplace_mode(self, raw: TVMFFIAny) -> InplaceMode {
        if self == Self::MaybeInPlace && object_is_unique(raw) {
            InplaceMode::Allow
        } else {
            InplaceMode::Disallow
        }
    }
}

struct MemoEntry {
    // Keeps the pointer-valued key alive so its address cannot be reused
    // during the same mapping invocation.
    _original: Any,
    result: Any,
}

struct NativeMapper<'a, D, Policy, const PRE_ORDER: bool> {
    dispatch: &'a mut D,
    policy: Option<Rc<Policy>>,
    remap: StructuralVarRemap,
}

fn run_native_mapper<D: MapDispatch, Policy: MutContextPolicy<D>>(
    root: Any,
    dispatch: &mut D,
    policy: Option<Rc<Policy>>,
    order: WalkOrder,
) -> Result<Any> {
    match order {
        WalkOrder::PreOrder => NativeMapper::<_, _, true>::run(root, dispatch, policy),
        WalkOrder::PostOrder => NativeMapper::<_, _, false>::run(root, dispatch, policy),
    }
}

impl<D: MapDispatch, Policy: MutContextPolicy<D>, const PRE_ORDER: bool>
    NativeMapper<'_, D, Policy, PRE_ORDER>
{
    fn run(root: Any, dispatch: &mut D, policy: Option<Rc<Policy>>) -> Result<Any> {
        run_structural_mutator(
            root,
            &mut NativeMapper::<_, _, PRE_ORDER> {
                dispatch,
                policy,
                remap: StructuralVarRemap::default(),
            },
        )
    }

    #[inline(always)]
    fn map_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        // Plain inline values have no children or structural identity.  Map
        // them directly instead of routing through identity lookup and the
        // default-mutation path, whose owning conversion crosses the C ABI.
        // Raw strings, byte-array views, and ObjectRValueRef are deliberately
        // excluded because converting those borrowed special values into an
        // Any performs normalization rather than a bitwise copy.
        if self.policy.is_none() && is_plain_inline(raw.type_index) {
            let value = StructuralView::from_raw(raw);
            return match self.dispatch.dispatch_map(&value, def_region_kind) {
                Some(result) => {
                    let mapped = result?;
                    // A pre-order callback may replace an inline leaf with a subtree.
                    if PRE_ORDER && !is_plain_inline(mapped.type_index()) {
                        let descended =
                            self.map_default_root(&mapped, def_region_kind, Permit::Copy)?;
                        Ok(if is_unchanged(&descended) {
                            mapped
                        } else {
                            descended
                        })
                    } else {
                        Ok(mapped)
                    }
                }
                // SAFETY: `is_plain_inline` excludes every borrowed
                // representation that needs normalization.  These values own
                // no external resource, so their owning form is the same
                // bitwise TVMFFIAny value.
                None => Ok(unsafe { Any::from_raw_ffi_any(raw) }),
            };
        }

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
                let value = StructuralView::from_raw(raw);
                let Some(callback_result) = self.dispatch.dispatch_map(&value, def_region_kind)
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
                let original = StructuralView::from_raw(raw);
                let value = if is_unchanged(&mapped) {
                    &original
                } else {
                    StructuralView::from_any(&mapped)
                };
                // Keep child rewrites when the callback leaves its input unchanged.
                match self.dispatch.dispatch_map(value, def_region_kind) {
                    Some(Ok(result)) if is_unchanged(&result) => Ok(mapped),
                    Some(result) => result.map_err(|error| with_value_context(error, value.raw())),
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

/// Internal mutation operations shared by the native mapper and a user
/// [`StructuralMutator`].
trait MutationDriver: Sized {
    fn dispatch_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any>;

    // The ABI entry has already installed and validated the active region.
    #[inline(always)]
    fn dispatch_abi_raw<const INPLACE: bool>(
        &mut self,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> TVMFFIAny {
        let permit = if INPLACE {
            Permit::MaybeInPlace
        } else {
            Permit::Copy
        };
        result_into_raw(self.dispatch_raw(raw, kind, permit))
    }

    fn var_remap_get_raw(&mut self, raw: TVMFFIAny) -> Result<Option<Any>>;

    fn var_remap_set_raw(&mut self, raw: TVMFFIAny, replacement: &Any) -> Result<()>;

    fn call_registered_hook(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Option<Any>> {
        let (mutator, context) = checked_driver_context(self)?;
        if raw.type_index == TVMFFITypeIndex::kTVMFFIArray as i32
            && permit == Permit::MaybeInPlace
            && object_is_unique(raw)
        {
            return with_mutation_region(def_region_kind, |kind| {
                mutate_array(self, mutator, raw, kind)
            })
            .map(Some);
        }
        let Some(attr) = structural_mutate_hook(raw, permit) else {
            // No foreign call needs access to the driver on a hook miss.
            return Ok(None);
        };
        with_current_driver_context(mutator, context, || {
            call_structural_mutate_hook(mutator, raw, def_region_kind, attr).map(Some)
        })
    }

    fn default_map_current_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        default_mutate_driver(self, raw, def_region_kind, permit)
    }

    fn map_reflected(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        type_info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    ) -> Result<Any> {
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
            for_each_field_info(type_info, &mut |field| {
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
            Ok(Unchanged.into())
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
        let field_offset = usize::try_from(field.offset).map_err(|_| {
            runtime_error(&format!(
                "native structural map: reflected field `{}` has an invalid offset",
                field.name.as_str()
            ))
        })?;
        // SAFETY: registered reflection metadata guarantees that the field
        // offset lies within this object's allocation. The checked conversion
        // above also prevents truncation on 32-bit targets.
        let source_address = output_object.add(field_offset).cast::<c_void>();
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
            .dispatch_raw(child_raw, child_region, Permit::Copy)
            .map_err(|error| {
                with_error_context(error, &format!("field `{}`", field.name.as_str()))
            })?;
        if is_unchanged(&mapped) || same_shallow(child_raw, *mapped.as_raw_ffi_any()) {
            return Ok(());
        }

        call_field_setter(field, source_address, mapped.as_raw_ffi_any()).map_err(|error| {
            with_error_context(error, &format!("field `{}`", field.name.as_str()))
        })?;
        *field_changed = true;
        Ok(())
    }
}

type StructuralMutatorHandle = *mut RuntimeStructuralMutatorObj;

type FStructuralMutate =
    unsafe extern "C" fn(StructuralMutatorHandle, AnyView<'static>) -> TVMFFIAny;
type FStructuralVarRemapGet =
    unsafe extern "C" fn(StructuralMutatorHandle, AnyView<'static>) -> TVMFFIAny;
type FStructuralVarRemapSet =
    unsafe extern "C" fn(StructuralMutatorHandle, AnyView<'static>, AnyView<'static>) -> TVMFFIAny;

/// Rust mirror of the C++ `StructuralMutatorVTable` ABI.
#[repr(C)]
struct StructuralMutatorVTable {
    mutate: FStructuralMutate,
    maybe_inplace_mutate: FStructuralMutate,
    var_remap_get: FStructuralVarRemapGet,
    var_remap_set: FStructuralVarRemapSet,
}

type RuntimeVarRemapGetCallback = unsafe fn(*mut c_void, TVMFFIAny) -> Result<Option<Any>>;
type RuntimeVarRemapSetCallback = unsafe fn(*mut c_void, TVMFFIAny, &Any) -> Result<()>;

struct RuntimeMutatorCallbacks {
    var_remap_get: RuntimeVarRemapGetCallback,
    var_remap_set: RuntimeVarRemapSetCallback,
}

/// Active Rust mutator with the exact C++ `StructuralMutatorObj` prefix.
///
/// C++ type hooks read `vtable` and `def_region_mode`; Rust keeps its erased
/// driver state after that shared prefix.
#[repr(C)]
struct RuntimeStructuralMutatorObj {
    base: Object,
    vtable: *const StructuralMutatorVTable,
    def_region_mode: i32,
    // `context` is available only while a registered type hook is allowed to
    // re-enter Rust. `context_identity` is never dereferenced; it verifies
    // that a helper is being called on the mutator that started this run.
    context: *mut c_void,
    context_identity: *mut c_void,
    owner_thread: std::thread::ThreadId,
    callbacks: RuntimeMutatorCallbacks,
    // Identity substitutions for the active traversal.
    remap: RefCell<StructuralVarRemap>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

const _: () = {
    assert!(
        std::mem::offset_of!(RuntimeStructuralMutatorObj, vtable)
            == std::mem::size_of::<TVMFFIObject>()
    );
    assert!(
        std::mem::offset_of!(RuntimeStructuralMutatorObj, def_region_mode)
            == std::mem::size_of::<TVMFFIObject>() + std::mem::size_of::<*const c_void>()
    );
};

// SAFETY: `RuntimeStructuralMutatorObj` is `repr(C)` and starts with `Object`,
// so `object_header_mut` returns the allocation's actual TVMFFIObject header.
// `type_index` resolves the registered `ffi.StructuralMutator` subtype whose
// C++ prefix is checked by the compile-time offset assertions above.
unsafe impl ObjectCore for RuntimeStructuralMutatorObj {
    const TYPE_KEY: &'static str = "ffi.StructuralMutator";
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH + 1;

    fn type_index() -> i32 {
        static TYPE_INDEX: LazyLock<i32> = LazyLock::new(|| unsafe {
            let key = TVMFFIByteArray::from_str(RuntimeStructuralMutatorObj::TYPE_KEY);
            let mut type_index = 0;
            let return_code = TVMFFITypeKeyToIndex(&key, &mut type_index);
            if return_code != 0 {
                panic!(
                    "ffi.StructuralMutator is not registered: {}",
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

struct RuntimeMutatorVTable<D>(PhantomData<D>);

impl<D: MutationDriver> RuntimeMutatorVTable<D> {
    const VTABLE: StructuralMutatorVTable = StructuralMutatorVTable {
        mutate: rust_vtable_mutate::<D>,
        maybe_inplace_mutate: rust_vtable_maybe_inplace_mutate::<D>,
        var_remap_get: rust_vtable_var_remap_get,
        var_remap_set: rust_vtable_var_remap_set,
    };
}

struct RuntimeContextGuard {
    mutator: StructuralMutatorHandle,
    context: *mut c_void,
}

impl Drop for RuntimeContextGuard {
    fn drop(&mut self) {
        // SAFETY: the guard is created only for a live mutator on its owner
        // thread. Restoring the pointer makes the same registered hook able to
        // invoke another child after this callback returns.
        unsafe { (*self.mutator).context = self.context };
    }
}

/// Temporarily take the erased driver context out of the runtime object.
///
/// # Safety
///
/// `mutator` must be null or point to a live [`RuntimeStructuralMutatorObj`].
/// A non-null context must have been installed by the current run. Its runtime
/// callback table must reconstruct it according to that run's mutable-driver
/// contract.
unsafe fn take_runtime_context(mutator: StructuralMutatorHandle) -> Result<RuntimeContextGuard> {
    if !is_active_mutator(mutator) {
        return Err(inactive_mutator_error(mutator, "callback"));
    }
    let context = (*mutator).context;
    if context.is_null() {
        let message = if (*mutator).context_identity.is_null() {
            "structural mutator was retained after its active call"
        } else {
            "structural mutator may only be called by its active registered hook"
        };
        return Err(runtime_error(message));
    }
    // No raw context pointer remains callable while Rust executes the selected
    // mutable-driver entry.
    (*mutator).context = std::ptr::null_mut();
    Ok(RuntimeContextGuard { mutator, context })
}

unsafe extern "C" fn rust_vtable_mutate<D: MutationDriver>(
    mutator: StructuralMutatorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    // SAFETY: this function is installed only in the vtable of a live
    // RuntimeStructuralMutatorObj built for D; `value` is borrowed for this call.
    rust_vtable_mutate_impl::<D, false>(mutator, value)
}

unsafe extern "C" fn rust_vtable_maybe_inplace_mutate<D: MutationDriver>(
    mutator: StructuralMutatorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    // SAFETY: same vtable and borrowed-value contract as
    // `rust_vtable_mutate`.
    rust_vtable_mutate_impl::<D, true>(mutator, value)
}

/// Run one typed vtable mutation callback and convert its result to ABI form.
///
/// # Safety
///
/// `mutator` must be a live runtime mutator created for `D`, and `value`
/// must remain valid for this call.
#[inline(always)]
unsafe fn rust_vtable_mutate_impl<D: MutationDriver, const INPLACE: bool>(
    mutator: StructuralMutatorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    let context_guard = match take_runtime_context(mutator) {
        Ok(guard) => guard,
        Err(error) => return result_into_raw(Err(error)),
    };
    let context = context_guard.context;
    let raw = *value.as_raw_ffi_any();
    let outcome = catch_unwind(AssertUnwindSafe(
        #[inline(always)]
        || {
            let kind = match def_region_from_raw((*mutator).def_region_mode) {
                Ok(kind) => kind,
                Err(error) => return result_into_raw(Err(error)),
            };
            // take_runtime_context verified that this mutator is already active.
            runtime_dispatch_mutate::<D, INPLACE>(context, raw, kind)
        },
    ));
    match outcome {
        Ok(result) => result,
        Err(payload) => mutation_panic_result(mutator, payload),
    }
}

#[cold]
unsafe fn mutation_panic_result(
    mutator: StructuralMutatorHandle,
    payload: Box<dyn std::any::Any + Send>,
) -> TVMFFIAny {
    (*mutator).panic = Some(payload);
    result_into_raw(Err(runtime_error("panic in structural mutator callback")))
}

unsafe extern "C" fn rust_vtable_var_remap_get(
    mutator: StructuralMutatorHandle,
    var: AnyView<'static>,
) -> TVMFFIAny {
    let context_guard = match take_runtime_context(mutator) {
        Ok(guard) => guard,
        Err(error) => return result_into_raw(Err(error)),
    };
    let callback = (*mutator).callbacks.var_remap_get;
    let context = context_guard.context;
    let raw = *var.as_raw_ffi_any();
    match catch_unwind(AssertUnwindSafe(|| callback(context, raw))) {
        Ok(Ok(Some(replacement))) => Any::into_raw_ffi_any(replacement),
        Ok(Ok(None)) => TVMFFIAny::new(),
        Ok(Err(error)) => result_into_raw(Err(error)),
        Err(payload) => {
            (*mutator).panic = Some(payload);
            result_into_raw(Err(runtime_error("panic in structural var-remap lookup")))
        }
    }
}

unsafe extern "C" fn rust_vtable_var_remap_set(
    mutator: StructuralMutatorHandle,
    var: AnyView<'static>,
    replacement: AnyView<'static>,
) -> TVMFFIAny {
    let context_guard = match take_runtime_context(mutator) {
        Ok(guard) => guard,
        Err(error) => return result_into_raw(Err(error)),
    };
    let callback = (*mutator).callbacks.var_remap_set;
    let context = context_guard.context;
    let var_raw = *var.as_raw_ffi_any();
    let replacement_raw = *replacement.as_raw_ffi_any();
    match catch_unwind(AssertUnwindSafe(|| {
        let replacement = owned_from_raw(replacement_raw)?;
        callback(context, var_raw, &replacement)
    })) {
        Ok(Ok(())) => TVMFFIAny::new(),
        Ok(Err(error)) => result_into_raw(Err(error)),
        Err(payload) => {
            (*mutator).panic = Some(payload);
            result_into_raw(Err(runtime_error(
                "panic in structural var-remap insertion",
            )))
        }
    }
}

thread_local! {
    static ACTIVE_MUTATOR: Cell<StructuralMutatorHandle> = const {
        Cell::new(std::ptr::null_mut())
    };
}

fn with_active_mutator<T>(handle: StructuralMutatorHandle, callback: impl FnOnce() -> T) -> T {
    ACTIVE_MUTATOR.with(|active| {
        let previous = active.replace(handle);
        struct Restore<'a> {
            active: &'a Cell<StructuralMutatorHandle>,
            previous: StructuralMutatorHandle,
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

fn active_mutator() -> Result<StructuralMutatorHandle> {
    ACTIVE_MUTATOR.with(|active| {
        let handle = active.get();
        if handle.is_null() {
            Err(runtime_error(
                "structural mutator helper called outside structural_mutate",
            ))
        } else {
            Ok(handle)
        }
    })
}

fn invocation_var_remap_get<U: Sized>(
    mutator: &mut U,
    var: &StructuralView,
) -> Result<Option<Any>> {
    let active = active_mutator()?;
    let context = std::ptr::from_mut(mutator).cast::<c_void>();
    unsafe {
        if (*active).context_identity != context {
            return Err(runtime_error(
                "default structural var-remap used by a non-active mutator",
            ));
        }
        (*active).remap.borrow().get(var)
    }
}

fn invocation_var_remap_set<U: Sized>(
    mutator: &mut U,
    var: &StructuralView,
    mutated_value: &Any,
) -> Result<()> {
    let active = active_mutator()?;
    let context = std::ptr::from_mut(mutator).cast::<c_void>();
    unsafe {
        if (*active).context_identity != context {
            return Err(runtime_error(
                "default structural var-remap used by a non-active mutator",
            ));
        }
        (*active).remap.borrow_mut().set(var, mutated_value)
    }
}

#[inline]
fn is_active_mutator(handle: StructuralMutatorHandle) -> bool {
    !handle.is_null() && ACTIVE_MUTATOR.with(|active| active.get() == handle)
}

#[cold]
fn inactive_mutator_error(mutator: StructuralMutatorHandle, operation: &str) -> Error {
    if mutator.is_null() {
        return runtime_error("null active structural mutator");
    }
    // The immutable owner id lets a foreign thread be rejected before reading
    // context fields that the owner thread may be updating.
    unsafe {
        if (*mutator).owner_thread != std::thread::current().id() {
            return runtime_error(&format!(
                "structural mutator {operation} invoked from a different thread"
            ));
        }
        if (*mutator).context_identity.is_null() {
            runtime_error("structural mutator was retained after its active call")
        } else {
            runtime_error(&format!(
                "structural mutator {operation} may only be used by its active registered hook"
            ))
        }
    }
}

/// Expose the current mutable reborrow only for the duration of one registered
/// type hook. Nested vtable calls then reborrow from this pointer, and
/// [`take_runtime_context`] hides it again while Rust is executing.
fn with_current_driver_context<T>(
    mutator: StructuralMutatorHandle,
    context: *mut c_void,
    callback: impl FnOnce() -> Result<T>,
) -> Result<T> {
    // checked_driver_context validated this reborrow; no user code runs
    // between that check and exposing the pointer to the registered hook.
    unsafe {
        (*mutator).context = context;
        struct HideContext {
            mutator: StructuralMutatorHandle,
        }
        impl Drop for HideContext {
            fn drop(&mut self) {
                // SAFETY: this scope owns the temporary exposure and runs on
                // the mutator's owner thread.
                unsafe { (*self.mutator).context = std::ptr::null_mut() };
            }
        }
        let _hide = HideContext { mutator };
        callback()
    }
}

fn checked_driver_context<D>(driver: &mut D) -> Result<(StructuralMutatorHandle, *mut c_void)> {
    let mutator = active_mutator()?;
    let context = std::ptr::from_mut(driver).cast::<c_void>();
    unsafe {
        if (*mutator).context_identity != context {
            return Err(runtime_error(
                "structural mutator helper called on a non-active mutator",
            ));
        }
        if !(*mutator).context.is_null() {
            return Err(runtime_error(
                "structural mutator driver context is already exposed",
            ));
        }

        Ok((mutator, context))
    }
}

fn def_region_from_raw(kind: i32) -> Result<DefRegionKind> {
    match kind {
        x if x == DefRegionKind::None as i32 => Ok(DefRegionKind::None),
        x if x == DefRegionKind::Pattern as i32 => Ok(DefRegionKind::Pattern),
        x if x == DefRegionKind::Simple as i32 => Ok(DefRegionKind::Simple),
        _ => Err(runtime_error("invalid structural definition-region kind")),
    }
}

impl<D: MapDispatch, Policy: MutContextPolicy<D>, const PRE_ORDER: bool> MutationDriver
    for NativeMapper<'_, D, Policy, PRE_ORDER>
{
    fn dispatch_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        with_mutation_region(def_region_kind, |kind| self.map_raw(raw, kind, permit))
    }

    #[inline(always)]
    fn dispatch_abi_raw<const INPLACE: bool>(
        &mut self,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> TVMFFIAny {
        let permit = if INPLACE {
            Permit::MaybeInPlace
        } else {
            Permit::Copy
        };
        result_into_raw(self.map_raw(raw, kind, permit))
    }

    #[inline(always)]
    fn default_map_current_raw(
        &mut self,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        match self.policy.as_ref().map(Rc::as_ptr) {
            Some(policy) => {
                let view = StructuralView::from_raw(raw);
                policy::mutate_with_policy(
                    &mut policy::MutationDescent { driver: self },
                    // SAFETY: recursive mapping never replaces the owning Rc.
                    unsafe { &*policy },
                    MutateValue::new(&view, permit.inplace_mode(raw)),
                    kind,
                )
            }
            None => default_mutate_driver(self, raw, kind, permit),
        }
    }

    fn var_remap_get_raw(&mut self, raw: TVMFFIAny) -> Result<Option<Any>> {
        self.remap.get(&StructuralView::from_raw(raw))
    }

    fn var_remap_set_raw(&mut self, raw: TVMFFIAny, replacement: &Any) -> Result<()> {
        self.remap.set(&StructuralView::from_raw(raw), replacement)
    }
}

impl<U: StructuralMutator> MutationDriver for U {
    fn dispatch_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        dispatch_user_raw(self, raw, def_region_kind, permit)
    }

    #[inline(always)]
    fn dispatch_abi_raw<const INPLACE: bool>(
        &mut self,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> TVMFFIAny {
        if INPLACE && raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return result_into_raw(
                self.dispatch_mutate(&StructuralView::from_raw(raw), kind)
                    .map_err(|error| with_value_context(error, raw)),
            );
        }
        let permit = if INPLACE {
            Permit::MaybeInPlace
        } else {
            Permit::Copy
        };
        result_into_raw(dispatch_user_current_raw(self, raw, kind, permit))
    }

    fn var_remap_get_raw(&mut self, raw: TVMFFIAny) -> Result<Option<Any>> {
        self.var_remap_get(&StructuralView::from_raw(raw))
    }

    fn var_remap_set_raw(&mut self, raw: TVMFFIAny, replacement: &Any) -> Result<()> {
        self.var_remap_set(&StructuralView::from_raw(raw), replacement)
    }
}

/// Invoke the concrete Rust driver selected when the runtime mutator was built.
///
/// # Safety
///
/// `context` must come from the current mutable reborrow of a live `D`; the
/// runtime object hides that pointer until this call returns.
#[inline(always)]
unsafe fn runtime_dispatch_mutate<D: MutationDriver, const INPLACE: bool>(
    context: *mut c_void,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
) -> TVMFFIAny {
    (&mut *context.cast::<D>()).dispatch_abi_raw::<INPLACE>(raw, def_region_kind)
}

/// Dispatch a variable-remap lookup through the erased driver context.
///
/// # Safety
///
/// `context` must satisfy the same requirements as [`runtime_dispatch_mutate`].
unsafe fn runtime_var_remap_get<D: MutationDriver>(
    context: *mut c_void,
    raw: TVMFFIAny,
) -> Result<Option<Any>> {
    (&mut *context.cast::<D>()).var_remap_get_raw(raw)
}

/// Dispatch a variable-remap insertion through the erased driver context.
///
/// # Safety
///
/// `context` must satisfy the same requirements as [`runtime_dispatch_mutate`], and
/// `replacement` must remain alive for this call.
unsafe fn runtime_var_remap_set<D: MutationDriver>(
    context: *mut c_void,
    raw: TVMFFIAny,
    replacement: &Any,
) -> Result<()> {
    (&mut *context.cast::<D>()).var_remap_set_raw(raw, replacement)
}

fn run_structural_mutator<D: MutationDriver>(root: Any, driver: &mut D) -> Result<Any> {
    let context = std::ptr::from_mut(driver).cast::<c_void>();
    let callbacks = RuntimeMutatorCallbacks {
        var_remap_get: runtime_var_remap_get::<D>,
        var_remap_set: runtime_var_remap_set::<D>,
    };
    run_structural_mutator_with_context(
        root,
        context,
        callbacks,
        &RuntimeMutatorVTable::<D>::VTABLE,
    )
}

fn run_structural_mutator_with_context(
    root: Any,
    context: *mut c_void,
    callbacks: RuntimeMutatorCallbacks,
    vtable: &'static StructuralMutatorVTable,
) -> Result<Any> {
    let mut active = ObjectArc::new(RuntimeStructuralMutatorObj {
        base: Object::new(),
        vtable,
        def_region_mode: DefRegionKind::None as i32,
        context,
        context_identity: context,
        owner_thread: std::thread::current().id(),
        callbacks,
        remap: RefCell::new(StructuralVarRemap::default()),
        panic: None,
    });
    let handle = unsafe { ObjectArc::as_raw_mut(&mut active) };
    let result = with_active_mutator(handle, || {
        call_mutator(
            handle,
            *root.as_raw_ffi_any(),
            DefRegionKind::None,
            Permit::MaybeInPlace,
        )
    });
    // A structural hook may only use the active mutator synchronously on this
    // thread. Make a retained reference fail instead of exposing a dangling
    // Rust state pointer. Release invocation-local identity owners here as
    // well: foreign code may retain the ABI object after the run ends.
    unsafe {
        (*handle).remap.get_mut().clear();
        (*handle).context = std::ptr::null_mut();
        (*handle).context_identity = std::ptr::null_mut();
    }
    let panic = unsafe { (*handle).panic.take() };
    if let Some(payload) = panic {
        drop(result);
        resume_unwind(payload);
    }
    result.map(|result| if is_unchanged(&result) { root } else { result })
}

fn call_mutator(
    mutator: StructuralMutatorHandle,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    if mutator.is_null() {
        return Err(runtime_error("no active structural mutator"));
    }
    let use_inplace = permit == Permit::MaybeInPlace && object_is_unique(raw);
    let callback = unsafe {
        if use_inplace {
            (*(*mutator).vtable).maybe_inplace_mutate
        } else {
            (*(*mutator).vtable).mutate
        }
    };
    with_mutator_def_region(mutator, def_region_kind, || unsafe {
        let view = AnyView::from_raw_ffi_any(raw);
        result_from_raw(callback(mutator, view))
    })
}

#[inline(always)]
fn structural_mutate_hook(raw: TVMFFIAny, permit: Permit) -> Option<TVMFFIAny> {
    let use_inplace = permit == Permit::MaybeInPlace && object_is_unique(raw);
    if use_inplace {
        if let Some(attr) = structural_maybe_inplace_mutate_column()
            .and_then(|column| column.get_raw(raw.type_index))
        {
            if attr.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32
                || attr.type_index == TVMFFITypeIndex::kTVMFFIFunction as i32
            {
                return Some(attr);
            }
        }
    }
    let attr = structural_mutate_column().and_then(|column| column.get_raw(raw.type_index))?;
    (attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32).then_some(attr)
}

fn call_structural_mutate_hook(
    mutator: StructuralMutatorHandle,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    attr: TVMFFIAny,
) -> Result<Any> {
    with_mutator_def_region(mutator, def_region_kind, || unsafe {
        match attr.type_index {
            x if x == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 => {
                let pointer = attr.data_union.v_ptr;
                if pointer.is_null() {
                    return Err(runtime_error("structural mutation hook is null"));
                }
                // SAFETY: the `__s_mutate__`/`__s_maybe_inplace_mutate__`
                // registration protocol defines an opaque-pointer attribute
                // as exactly an FStructuralMutate function pointer.
                let hook: FStructuralMutate = std::mem::transmute(pointer);
                let value = AnyView::from_raw_ffi_any(raw);
                result_from_raw(hook(mutator, value))
            }
            x if x == TVMFFITypeIndex::kTVMFFIFunction as i32 => {
                let function = Function::try_from(AnyView::from_raw_ffi_any(attr))?;
                let mutator_value = borrowed_mutator_view(mutator);
                let value = AnyView::from_raw_ffi_any(raw);
                function.call_packed(&[mutator_value, value])
            }
            _ => Err(Error::new(
                TYPE_ERROR,
                "__s_mutate__ must be an opaque function pointer or ffi.Function",
                "",
            )),
        }
    })
}

/// Borrow a live runtime mutator as an object-valued ABI argument.
///
/// # Safety
///
/// `mutator` must point to a live object and outlive the returned view. The
/// view does not increment the object's reference count.
unsafe fn borrowed_mutator_view<'a>(mutator: StructuralMutatorHandle) -> AnyView<'a> {
    let object = mutator.cast::<TVMFFIObject>();
    let mut raw = TVMFFIAny::new();
    raw.type_index = (*object).type_index;
    raw.small_str_len = 0;
    raw.data_union.v_obj = object;
    AnyView::from_raw_ffi_any(raw)
}

#[inline(always)]
fn result_into_raw(result: Result<Any>) -> TVMFFIAny {
    unsafe {
        match result {
            Ok(value) => Any::into_raw_ffi_any(value),
            Err(error) => Any::into_raw_ffi_any(Any::from(error)),
        }
    }
}

/// Resolve only at an owning-value API boundary; internal Any carriers and
/// native hooks keep the unchanged tag to avoid acquiring the original.
#[inline(always)]
fn resolve_result(result: Any, original: TVMFFIAny) -> Result<Any> {
    if is_unchanged(&result) {
        owned_from_raw(original)
    } else {
        Ok(result)
    }
}

/// Take ownership of one value returned by a structural-mutation ABI hook.
///
/// # Safety
///
/// `raw` must contain one owning TVMFFIAny result that has not already been
/// consumed. An Error object is converted into the Rust error channel.
#[inline(always)]
unsafe fn result_from_raw(raw: TVMFFIAny) -> Result<Any> {
    let value = Any::from_raw_ffi_any(raw);
    if value.type_index() != TVMFFITypeIndex::kTVMFFIError as i32 {
        return Ok(value);
    }
    match Error::try_from(value) {
        Ok(error) | Err(error) => Err(error),
    }
}

#[inline(always)]
fn with_mutator_def_region<T>(
    mutator: StructuralMutatorHandle,
    kind: DefRegionKind,
    callback: impl FnOnce() -> T,
) -> T {
    unsafe {
        let previous = (*mutator).def_region_mode;
        // Precedence: a pattern region propagates; entering any kind inside it has no effect.
        if previous != DefRegionKind::Pattern as i32 {
            (*mutator).def_region_mode = kind as i32;
        }
        struct Restore {
            mutator: StructuralMutatorHandle,
            previous: i32,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                if self.previous != DefRegionKind::Pattern as i32 {
                    unsafe { (*self.mutator).def_region_mode = self.previous };
                }
            }
        }
        let _restore = Restore { mutator, previous };
        // One call site keeps the continuation visible to the inliner.
        callback()
    }
}

#[inline(always)]
fn with_mutation_region<T>(
    kind: DefRegionKind,
    callback: impl FnOnce(DefRegionKind) -> Result<T>,
) -> Result<T> {
    let mutator = active_mutator()?;
    with_mutator_def_region(
        mutator,
        kind,
        #[inline(always)]
        || {
            // SAFETY: the active invocation keeps this thread's ABI mutator alive.
            let effective = def_region_from_raw(unsafe { (*mutator).def_region_mode })?;
            callback(effective)
        },
    )
}

#[inline(always)]
fn dispatch_user_raw<U: StructuralMutator>(
    mutator: &mut U,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    with_mutation_region(
        def_region_kind,
        #[inline(always)]
        |kind| dispatch_user_current_raw(mutator, raw, kind, permit),
    )
}

#[inline(always)]
fn dispatch_user_current_raw<U: StructuralMutator>(
    mutator: &mut U,
    raw: TVMFFIAny,
    kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    let result = if permit == Permit::MaybeInPlace && object_is_unique(raw) {
        let mut scoped_raw = raw;
        mutator.dispatch_maybe_inplace_mutate(InplaceValue::from_raw(&mut scoped_raw), kind)
    } else {
        mutator.dispatch_mutate(&StructuralView::from_raw(raw), kind)
    };
    result.map_err(|error| with_value_context(error, raw))
}

fn user_default_mutate<U: StructuralMutator>(
    mutator: &mut U,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    with_mutation_region(def_region_kind, |kind| {
        let value = StructuralView::from_raw(raw);
        mutator.on_default_mutate(MutateValue::new(&value, permit.inplace_mode(raw)), kind)
    })
    .map_err(|error| with_value_context(error, raw))
}

#[inline(never)]
fn mutate_array<D: MutationDriver>(
    driver: &mut D,
    mutator: StructuralMutatorHandle,
    raw: TVMFFIAny,
    kind: DefRegionKind,
) -> Result<Any> {
    // Array has a fixed native hook and a shared ABI layout. Keep its loop in
    // Rust so child callbacks can inline without exposing a foreign reborrow.
    let array = unsafe { raw.data_union.v_obj.cast::<ArrayObj>() };
    if array.is_null() {
        return Err(runtime_error("structural mutation of a null array"));
    }
    let (items, len) = unsafe { ((*array).data.cast::<TVMFFIAny>(), (*array).size as usize) };
    for index in 0..len {
        // SAFETY: the active root owns this array, and both the input's
        // permission and unique ownership were checked before entering.
        let original = unsafe { *items.add(index) };
        let outcome = catch_unwind(AssertUnwindSafe(
            #[inline(always)]
            || driver.dispatch_abi_raw::<true>(original, kind),
        ));
        let mapped = unsafe {
            result_from_raw(match outcome {
                Ok(value) => value,
                Err(payload) => mutation_panic_result(mutator, payload),
            })?
        };
        if is_unchanged(&mapped) || same_shallow(original, *mapped.as_raw_ffi_any()) {
            continue;
        }
        unsafe {
            let old = std::ptr::replace(items.add(index), Any::into_raw_ffi_any(mapped));
            drop(Any::from_raw_ffi_any(old));
        }
    }
    Ok(Unchanged.into())
}

fn default_mutate_driver<D: MutationDriver>(
    driver: &mut D,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    default_mutate_driver_impl(driver, raw, def_region_kind, permit)
        .map_err(|error| with_value_context(error, raw))
}

fn default_mutate_driver_impl<D: MutationDriver>(
    driver: &mut D,
    raw: TVMFFIAny,
    def_region_kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    // Match C++ DefaultMutateExpected: a registered type hook owns any
    // identity-remap policy for that type. Automatic remapping applies only
    // to the reflected fallback below.
    if let Some(mutated) = driver.call_registered_hook(raw, def_region_kind, permit)? {
        return Ok(mutated);
    }
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return owned_from_raw(raw);
    }

    let type_info = checked_type_info(raw.type_index)?;
    let kind = unsafe {
        if (*type_info).metadata.is_null() {
            None
        } else {
            Some((*(*type_info).metadata).structural_eq_hash_kind)
        }
    };
    let is_free_var = kind == Some(TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar as i32);
    let is_dag_node = kind == Some(TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode as i32);
    if is_free_var || is_dag_node {
        if let Some(mutated) = driver.var_remap_get_raw(raw)? {
            // The ABI uses None for a miss; an Unchanged marker is a cached result.
            if mutated.type_index() != TVMFFITypeIndex::kTVMFFINone as i32 {
                return Ok(mutated);
            }
        }
    }
    // A variable use with no binding retains its identity without visiting fields.
    if is_free_var && def_region_kind == DefRegionKind::None {
        return Ok(Unchanged.into());
    }

    let result = driver.map_reflected(raw, def_region_kind, type_info)?;
    if is_dag_node
        || (is_free_var && (def_region_kind == DefRegionKind::Pattern || !is_unchanged(&result)))
    {
        // Keep markers for DAG nodes and pattern definitions. An unchanged simple
        // definition needs no binding: subsequent uses retain the original on a miss.
        driver.var_remap_set_raw(raw, &result)?;
    }
    Ok(result)
}

/// Mutate a structured value with a mutator or typed callback chain.
///
/// The root is consumed to establish the ownership boundary for optional
/// in-place mutation. A matching callback supplies the final value; unmatched
/// values use default mutation.
pub fn structural_mutate<R, M>(root: R, mutator: impl IntoMutator<M>) -> Result<Any>
where
    R: Into<Any>,
{
    mutator.mutate_root(root.into())
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

fn shallow_copy(raw: TVMFFIAny) -> Result<Any> {
    let Some(attr) = shallow_copy_column().and_then(|column| column.get_raw(raw.type_index)) else {
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
    let function = unsafe { attr.data_union.v_obj };
    if function.is_null() {
        return Err(runtime_error("shallow copy function pointer is null"));
    }
    // The registry owns this function throughout the call; borrowing it avoids
    // an atomic retain/release for every reflected node.
    let mut result = Any::new();
    let status =
        unsafe { TVMFFIFunctionCall(function.cast(), &raw, 1, Any::as_data_ptr(&mut result)) };
    if status != 0 {
        return Err(Error::from_raised());
    }
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
            // SAFETY: reflection registration requires a non-Function setter
            // pointer to use the TVMFFIFieldSetter signature.
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

fn object_identity_key(raw: TVMFFIAny) -> Result<NonNull<TVMFFIObject>> {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return Err(var_remap_key_error());
    }
    let pointer = unsafe { raw.data_union.v_obj };
    NonNull::new(pointer)
        .ok_or_else(|| runtime_error("native structural map: identity object has a null pointer"))
}

#[cold]
fn var_remap_key_error() -> Error {
    Error::new(
        TYPE_ERROR,
        "variable-remap keys must be object-backed values",
        "",
    )
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

#[inline(always)]
fn owned_from_raw(raw: TVMFFIAny) -> Result<Any> {
    if let Some(owned) = try_to_owned_without_normalization(raw) {
        return Ok(owned);
    }
    if raw.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return Err(runtime_error(
            "native structural map: object-backed value has a null pointer",
        ));
    }

    // Raw string/bytes views and ObjectRValueRef require normalization (or a
    // move) rather than a bitwise copy; keep the generic C ABI conversion for
    // those uncommon representations.
    let mut owned = Any::new();
    let return_code = unsafe { TVMFFIAnyViewToOwnedAny(&raw, Any::as_data_ptr(&mut owned)) };
    if return_code == 0 {
        Ok(owned)
    } else {
        Err(Error::from_raised())
    }
}

#[cold]
fn with_value_context(error: Error, raw: TVMFFIAny) -> Error {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        error
    } else {
        with_error_context(
            with_visit_error_context(error, raw),
            &format!("object `{}`", type_key_of(raw.type_index)),
        )
    }
}

#[cold]
fn with_error_context(error: Error, frame: &str) -> Error {
    with_structural_error_context(error, "map", frame)
}

#[cold]
#[inline(never)]
fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

fn cached_column(cache: &'static AtomicUsize, name: &'static str) -> Option<TypeAttrColumn> {
    let cached = cache.load(Ordering::Relaxed);
    if cached != 0 {
        let pointer = cached as *mut TVMFFITypeAttrColumn;
        return Some(unsafe { TypeAttrColumn::from_non_null(NonNull::new_unchecked(pointer)) });
    }
    let column = type_attr_column(name)?;
    // TypeAttrColumn is a transparent NonNull wrapper shared with the
    // structural-visit module. Registry column addresses are immortal.
    cache.store(column.as_ptr() as usize, Ordering::Relaxed);
    Some(column)
}

static STRUCTURAL_MUTATE_COLUMN: AtomicUsize = AtomicUsize::new(0);
static STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN: AtomicUsize = AtomicUsize::new(0);
static SHALLOW_COPY_COLUMN: AtomicUsize = AtomicUsize::new(0);

fn structural_mutate_column() -> Option<TypeAttrColumn> {
    cached_column(&STRUCTURAL_MUTATE_COLUMN, STRUCTURAL_MUTATE_ATTR)
}

fn structural_maybe_inplace_mutate_column() -> Option<TypeAttrColumn> {
    cached_column(
        &STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN,
        STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR,
    )
}

fn shallow_copy_column() -> Option<TypeAttrColumn> {
    cached_column(&SHALLOW_COPY_COLUMN, SHALLOW_COPY_ATTR)
}
