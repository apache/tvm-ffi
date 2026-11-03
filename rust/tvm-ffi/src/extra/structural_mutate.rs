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

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

use super::dispatch::Dispatcher;
use crate::any::{Any, AnyView};
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::{self, Object, ObjectCore};
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
    borrowed_view, impl_callback_chain_tuple_arities, initialize_scoped_runtime, is_plain_inline,
    raw_of, same_shallow, try_to_owned_without_normalization, with_structural_error_context,
    with_visit_error_context,
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
/// `Allow` still requires unique ownership. A [`MutationView`] from an exclusive
/// borrow of an owning `Any` may allow reuse; a shared borrow never grants it.
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
}

/// A non-owning, move-only view carrying in-place permission.
///
/// This view cannot be copied or cloned. Consuming it transfers the same
/// permission; narrowing its matched type does not acquire ownership.
///
/// Unlike an owning typed argument, this handle does not increment the reference
/// count. Borrow through it to inspect the node, then consume it in
/// `default_mutate` to continue recursion. A surviving owning alias
/// still forces copying. Node borrows come from this handle, not from the
/// callback context, so the handle can safely be passed to another context.
/// `T` selects the callback's matched FFI type; `Any` matches every value.
/// The same handle is passed to consuming [`Dispatcher`] handlers. A publicly
/// shared-borrow handle always disallows reuse. An exclusive `&mut Any` borrow
/// grants permission to attempt reuse while keeping its owner borrowed.
///
/// A node borrowed through a copied view cannot survive consumption of the handle:
/// ```compile_fail,E0505
/// use tvm_ffi::{StructuralMutatorView, MutationView};
/// fn invalid<D>(value: MutationView<'_>, dispatch: &mut D, mutator: &mut StructuralMutatorView<'_, D>) {
///     let view = value.as_any_view();
///     let copy = view;
///     let node = copy.downcast_ref::<tvm_ffi::collections::array::ArrayObj>().unwrap();
///     mutator.default_mutate(dispatch, value).unwrap();
///     println!("{}", node.size);
/// }
/// ```
///
/// Moving the handle into a nested callback cannot bypass that borrow:
/// ```compile_fail,E0505
/// use tvm_ffi::{dispatch, structural_mutate, Any, MutationView, StructuralMutatorView, Result};
/// struct Inner<'a>(Option<MutationView<'a>>);
/// #[dispatch(mutate, MutationView)]
/// impl<'a> Inner<'a> {
///     fn mutate_bool(&mut self, _: bool, mutator: &mut StructuralMutatorView<'_, Self>) -> Result<Any> {
///         let pending = self.0.take().unwrap();
///         mutator.default_mutate(self, pending).map(Into::into)
///     }
/// }
/// fn invalid(value: MutationView<'_>) {
///     let view = value.as_any_view();
///     let node = view.downcast_ref::<tvm_ffi::collections::array::ArrayObj>().unwrap();
///     structural_mutate(true, Inner(Some(value))).unwrap();
///     println!("{}", node.size);
/// }
/// ```
///
/// A copied inspection view also cannot survive low-level default mutation:
/// ```compile_fail,E0505
/// use tvm_ffi::{dispatch, Any, MutationView, StructuralMutatorView, Result, Unchanged};
/// struct Pass;
/// #[dispatch(mutate, MutationView)]
/// impl Pass {
///     fn mutate_any(&mut self, value: MutationView<'_>, mutator: &mut StructuralMutatorView<'_, Self>) -> Result<Any> {
///         let view = value.as_any_view();
///         let copy = view;
///         mutator.default_mutate(self, value).unwrap();
///         println!("{}", copy.type_index());
///         Ok(Any::from(Unchanged))
///     }
/// }
/// ```
pub struct MutationView<'a, T = Any> {
    value: TVMFFIAny,
    mode: InplaceMode,
    _scope: PhantomData<AnyView<'a>>,
    _type: PhantomData<fn() -> T>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'a> MutationView<'a> {
    #[inline(always)]
    fn new(value: AnyView<'a>, mode: InplaceMode) -> Self {
        Self {
            value: raw_of(value),
            mode,
            _scope: PhantomData,
            _type: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Wrap a borrowed value without granting in-place permission.
    #[inline(always)]
    pub fn borrowed(value: AnyView<'a>) -> Self {
        Self::new(value, InplaceMode::Disallow)
    }
}

impl<'a> From<&'a mut Any> for MutationView<'a> {
    /// Borrow an owner exclusively without acquiring a reference count.
    /// Reuse still requires the runtime's uniqueness and alias checks.
    #[inline(always)]
    fn from(value: &'a mut Any) -> Self {
        Self::new(AnyView::from(&*value), InplaceMode::Allow)
    }
}

impl<'a, T> MutationView<'a, T> {
    /// Permission carried independently of temporary owning aliases.
    #[inline(always)]
    pub fn inplace_mode(&self) -> InplaceMode {
        self.mode
    }

    /// Restrict permission without upgrading a shared-borrow capability.
    #[inline(always)]
    pub fn with_inplace_mode(mut self, requested: InplaceMode) -> Self {
        if requested == InplaceMode::Disallow {
            self.mode = InplaceMode::Disallow;
        }
        self
    }

    /// Borrow the current value. The view and its copies borrow this handle,
    /// so their last use must precede consuming it for default mutation.
    #[inline(always)]
    pub fn as_any_view(&self) -> AnyView<'_> {
        // SAFETY: the input owner keeps this payload live while the capability is
        // borrowed. The returned lifetime is bounded by self, not by 'a.
        unsafe { borrowed_view(&self.value) }
    }

    /// Borrow an object or subtype as `U` without changing ownership or permission.
    /// The reference borrows this capability and cannot survive its consumption.
    #[inline(always)]
    pub fn downcast_ref<U: ObjectCore>(&self) -> Option<&U> {
        if self.value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32
            || !object::is_instance_of::<U>(self.value.type_index)
        {
            return None;
        }
        // SAFETY: the type check establishes U's layout; the capability keeps
        // the allocation borrowed for the lifetime of the returned reference.
        Some(unsafe { &*self.value.data_union.v_obj.cast::<U>() })
    }

    /// Narrow the matched type without acquiring an owning reference.
    #[inline(always)]
    pub fn try_cast<U: crate::type_traits::ContainerElement>(
        self,
    ) -> std::result::Result<MutationView<'a, U>, Self> {
        // SAFETY: the input owner keeps the borrowed value live for this handle's
        // lifetime. This only checks its type and does not acquire ownership.
        if unsafe { U::container_check_any_strict(&self.value) } {
            Ok(MutationView {
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

    /// Erase the matched type while preserving the carried permission.
    #[inline(always)]
    pub fn into_untyped(self) -> MutationView<'a> {
        MutationView {
            value: self.value,
            mode: self.mode,
            _scope: PhantomData,
            _type: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    // Built-in descent rechecks ownership before using this permission.
    #[inline(always)]
    fn permit(&self) -> Permit {
        self.mode.permit()
    }
}

mod mutation_field {
    pub trait Sealed {}
    impl Sealed for crate::Any {}
    impl<T: crate::object::ObjectRefCore + crate::type_traits::AnyCompatible> Sealed for T {}
}

/// Owned storage supported by [`MutationView::field`].
/// Sealed to `Any` and owning object handles; borrowed conversion adapters do not qualify.
#[doc(hidden)]
pub trait MutationField: crate::type_traits::ContainerElement + mutation_field::Sealed {}
impl MutationField for Any {}
impl<T: object::ObjectRefCore + crate::type_traits::AnyCompatible> MutationField for T {}

impl<T: object::ObjectRefCore> std::ops::Deref for MutationView<'_, T> {
    type Target = T::ContainerType;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: typed capability construction establishes T's object layout;
        // this shared borrow cannot outlive or survive consumption of the handle.
        unsafe { &*self.value.data_union.v_obj.cast::<T::ContainerType>() }
    }
}

impl<T: object::ObjectRefCore> MutationView<'_, T> {
    /// Borrow a direct owning field while exclusively borrowing this capability.
    /// Permission also requires the parent to remain unique after selection.
    /// This does not grant writeback to the parent field.
    ///
    /// Panics if the selector returns storage outside the parent representation.
    #[inline(always)]
    pub fn field<U: MutationField>(
        &mut self,
        selector: impl for<'node> FnOnce(&'node T::ContainerType) -> &'node U,
    ) -> MutationView<'_, U> {
        let node: &T::ContainerType = self;
        let child = selector(node);
        let offset = (child as *const U)
            .addr()
            .checked_sub((node as *const T::ContainerType).addr());
        assert!(
            offset.is_some_and(|offset| {
                offset <= std::mem::size_of::<T::ContainerType>()
                    && std::mem::size_of::<U>() > 0
                    && std::mem::size_of::<U>() <= std::mem::size_of::<T::ContainerType>() - offset
            }),
            "MutationView::field must select a direct owning field"
        );
        let mut value = TVMFFIAny::new();
        // SAFETY: child is a live owning field. Copying its view acquires no owner;
        // the returned capability exclusively borrows the parent capability.
        unsafe { U::container_copy_to_any_view(child, &mut value) };
        // Selection and conversion may retain aliases, so check the parent last.
        let mode = self.permit().inplace_mode(self.value);
        MutationView {
            value,
            mode,
            _scope: PhantomData,
            _type: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

/// Recursive mutation and default descent for a dispatcher.
///
/// Pass the handler's `self` to mutation methods to recurse with the same state.
/// Full mutation re-enters dispatch; default mutation runs Base descent.
/// Both preserve [`UnchangedOr`]; resolve it only when an owning value is needed.
#[repr(transparent)]
pub struct StructuralMutatorView<'a, D> {
    runtime: StructuralMutatorHandle,
    _scope: PhantomData<&'a mut ()>,
    _dispatcher: PhantomData<fn(D) -> D>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<D> StructuralMutatorView<'_, D> {
    #[inline(always)]
    fn reborrow(&self) -> StructuralMutatorView<'_, D> {
        StructuralMutatorView {
            runtime: self.runtime,
            _scope: PhantomData,
            _dispatcher: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
    #[inline(always)]
    pub fn def_region_kind(&self) -> DefRegionKind {
        // Registered hooks preserve the definition-region ABI enum.
        unsafe { std::mem::transmute((*self.runtime).def_region_mode) }
    }

    /// Run recursive operations under a scoped definition region.
    /// An enclosing Pattern wins; the original region is restored on all exits.
    #[inline(always)]
    pub fn with_def_region_kind<R>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut StructuralMutatorView<'_, D>) -> R,
    ) -> R {
        with_mutator_def_region(self.runtime, kind, |_| callback(&mut self.reborrow()))
    }

    #[inline(always)]
    fn dispatch_raw(
        &self,
        dispatch: &mut D,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        let _binding = bind_dispatcher(self.runtime, dispatch);
        unsafe { ((*runtime_state(self.runtime)).recurse)(self.runtime, raw, kind, permit) }
    }
    #[inline(always)]
    fn default_raw(
        &self,
        dispatch: &mut D,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        let _binding = bind_dispatcher(self.runtime, dispatch);
        unsafe { ((*runtime_state(self.runtime)).default_mutate)(self.runtime, raw, kind, permit) }
    }

    /// Re-enter dispatch with the capability's carried permission.
    #[inline(always)]
    pub fn mutate<'input, T>(
        &mut self,
        dispatch: &mut D,
        value: impl Into<MutationView<'input, T>>,
    ) -> Result<UnchangedOr<Any>> {
        let value = value.into();
        let kind = self.def_region_kind();
        self.dispatch_raw(dispatch, value.value, kind, value.permit())
            .and_then(UnchangedOr::from_carrier)
    }
    /// Run Base default descent with the capability's carried permission.
    #[inline(always)]
    pub fn default_mutate<'input, T>(
        &mut self,
        dispatch: &mut D,
        value: impl Into<MutationView<'input, T>>,
    ) -> Result<UnchangedOr<Any>> {
        let value = value.into();
        let kind = self.def_region_kind();
        self.default_raw(dispatch, value.value, kind, value.permit())
            .and_then(UnchangedOr::from_carrier)
    }
    #[inline(always)]
    pub fn var_remap_get(&mut self, dispatch: &mut D, var: AnyView<'_>) -> Result<Option<Any>> {
        let _binding = bind_dispatcher(self.runtime, dispatch);
        unsafe { runtime_var_remap_get(self.runtime, raw_of(var)) }
    }
    #[inline(always)]
    pub fn var_remap_set(&mut self, dispatch: &mut D, var: AnyView<'_>, value: &Any) -> Result<()> {
        let _binding = bind_dispatcher(self.runtime, dispatch);
        unsafe { runtime_var_remap_set(self.runtime, raw_of(var), value) }
    }
}

pub(crate) trait MutateDispatch: Sized {
    fn dispatch_mutate(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Option<Result<Any>>;
}
impl<D> MutateDispatch for D
where
    for<'s, 'v, 'c, 'r> &'s mut D:
        Dispatcher<(MutationView<'v>, &'c mut StructuralMutatorView<'r, D>), Option<Result<Any>>>,
{
    #[inline(always)]
    fn dispatch_mutate(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Option<Result<Any>> {
        self.dispatch_tuple((value, mutator))
    }
}

/// A concrete mutation Base and its owned or borrowed dispatcher.
pub struct StructuralMutator<Base, D> {
    base: Base,
    dispatch: D,
}

impl<Base, D> StructuralMutator<Base, D> {
    pub fn new(base: Base, dispatch: D) -> Self {
        Self { base, dispatch }
    }
}

/// Default descent for a concrete structural-mutation Base.
///
/// # Safety
/// Self must start with StructuralMutatorObj at offset zero and preserve its
/// engine-initialized header/vtable. Callbacks are synchronous and non-escaping;
/// no reference to reentered Base state may span recursive calls. Project only
/// fields needed for each operation, never borrow the whole Base or engine.
pub unsafe trait StructuralMutatorBase: Sized {
    /// Delegate to StructuralMutatorObj::default_mutate for built-in descent.
    ///
    /// # Safety
    /// `this` is the live Base of `mutator`, valid only during this synchronous call.
    unsafe fn default_mutate<D>(
        _this: *mut Self,
        dispatch: &mut D,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, D>,
    ) -> Result<UnchangedOr<Any>> {
        StructuralMutatorObj::default_mutate(dispatch, value, mutator)
    }
}

impl Default for StructuralMutatorObj {
    fn default() -> Self {
        Self {
            base: Object::new(),
            vtable: std::ptr::null(),
            def_region_mode: DefRegionKind::None as i32,
        }
    }
}

impl StructuralMutatorObj {
    /// Delegate to built-in hooks/reflection without invoking the Base override again.
    #[inline(always)]
    pub fn default_mutate<'input, D, T>(
        dispatch: &mut D,
        value: impl Into<MutationView<'input, T>>,
        mutator: &mut StructuralMutatorView<'_, D>,
    ) -> Result<UnchangedOr<Any>> {
        let value = value.into();
        let _binding = bind_dispatcher(mutator.runtime, dispatch);
        let kind = mutator.def_region_kind();
        let permit = value.permit();
        unsafe {
            ((*runtime_state(mutator.runtime)).builtin_default)(
                mutator.runtime,
                value.value,
                kind,
                permit,
            )
        }
        .and_then(UnchangedOr::from_carrier)
    }
}

// SAFETY: this is the exact ABI prefix and delegates without holding a Base borrow.
unsafe impl StructuralMutatorBase for StructuralMutatorObj {}

unsafe fn base_default_mutate<B: StructuralMutatorBase, D>(
    runtime: StructuralMutatorHandle,
    raw: TVMFFIAny,
    kind: DefRegionKind,
    permit: Permit,
) -> Result<Any> {
    with_mutator_def_region(runtime, kind, |_| {
        let mode = permit.inplace_mode(raw);
        let dispatch = &mut *(*runtime_state(runtime)).context.cast::<D>();
        with_mutation_view(dispatch, runtime, |dispatch, mutator| {
            B::default_mutate(
                runtime.cast(),
                dispatch,
                MutationView::new(borrowed_view(&raw), mode),
                mutator,
            )
            .map(Into::into)
            .map_err(|error| with_value_context(error, raw))
        })
    })
}

#[doc(hidden)]
pub enum ByBaseMutator {}

impl<B: StructuralMutatorBase, D: MutateDispatch> IntoMutator<(ByBaseMutator, ByOwnedMutator)>
    for StructuralMutator<B, D>
{
    fn mutate_root(self, root: Any) -> Result<Any> {
        run_mutator_engine::<UserMutator<'_, D>, _, _>(
            root,
            OwnedDispatcher(self.dispatch),
            self.base,
            false,
        )
    }
}
impl<B: StructuralMutatorBase, D: MutateDispatch> IntoMutator<(ByBaseMutator, ByBorrowedMutator)>
    for StructuralMutator<B, &mut D>
{
    fn mutate_root(self, root: Any) -> Result<Any> {
        run_mutator_engine::<UserMutator<'_, D>, _, _>(
            root,
            BorrowedDispatcher(self.dispatch),
            self.base,
            false,
        )
    }
}

/// Conversion into a dispatcher-driven mutation. Closures belong to structural_map.
pub trait IntoMutator<Marker> {
    #[doc(hidden)]
    fn mutate_root(self, root: Any) -> Result<Any>;
}
#[doc(hidden)]
pub enum ByOwnedMutator {}
#[doc(hidden)]
pub enum ByBorrowedMutator {}
impl<D: MutateDispatch> IntoMutator<ByOwnedMutator> for D {
    fn mutate_root(self, root: Any) -> Result<Any> {
        run_user_mutator(root, OwnedDispatcher(self))
    }
}
impl<D: MutateDispatch> IntoMutator<ByBorrowedMutator> for &mut D {
    fn mutate_root(self, root: Any) -> Result<Any> {
        run_user_mutator(root, BorrowedDispatcher(self))
    }
}
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
        WalkOrder::PreOrder => run_mutator_engine::<NativeMapper<'_, S::Dispatch, true>, _, _>(
            root, dispatch, base, false,
        ),
        WalkOrder::PostOrder => run_mutator_engine::<NativeMapper<'_, S::Dispatch, false>, _, _>(
            root, dispatch, base, false,
        ),
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

/// Adapter from a [`MapChainLink`] to [`Dispatcher`](crate::Dispatcher).
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

/// Identity substitutions for structural mutation within one invocation.
///
/// The map owns its keys and values so object addresses remain stable.
#[derive(Default)]
pub struct StructuralVarRemap {
    entries: HashMap<NonNull<TVMFFIObject>, MemoEntry>,
}

impl StructuralVarRemap {
    /// Look up an identity replacement previously stored for `var`.
    pub fn get(&self, var: AnyView<'_>) -> Result<Option<Any>> {
        let key = object_identity_key(raw_of(var))?;
        Ok(self.entries.get(&key).map(|entry| entry.result.clone()))
    }

    /// Store a descent result or an [`Unchanged`] marker for `var`.
    pub fn set(&mut self, var: AnyView<'_>, mutated_value: &Any) -> Result<()> {
        let key = object_identity_key(raw_of(var))?;
        self.entries.insert(
            key,
            MemoEntry {
                _original: Any::from(var),
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum Permit {
    Copy,
    MaybeInPlace,
}

impl Permit {
    #[inline(always)]
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

struct NativeMapper<'a, D, const PRE_ORDER: bool> {
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
    run_mutator_engine::<NativeMapper<'_, S::Dispatch, PRE>, S, _>(
        root,
        dispatch,
        StructuralMutatorObj::default(),
        true,
    )
}

impl<D: MapDispatch, const PRE_ORDER: bool> NativeMapper<'_, D, PRE_ORDER> {
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
        if is_plain_inline(raw.type_index)
            && unsafe { (*runtime_state(self.runtime)).plain_inline_fastpath }
        {
            let value = unsafe { borrowed_view(&raw) };
            return match self.dispatch.dispatch_map(value, def_region_kind) {
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

/// Internal mutation operations shared by the native mapper and a user
/// [`StructuralMutatorView`].
trait MutationDriver: Sized {
    fn runtime(&self) -> StructuralMutatorHandle;
    /// Expose only the current dispatcher reborrow, never this driver wrapper.
    fn context(&mut self) -> *mut c_void;
    // Each static callback reconstructs a temporary driver around that reborrow.
    unsafe fn runtime_dispatch<const INPLACE: bool>(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> TVMFFIAny;
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

    unsafe fn runtime_recurse(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any>;
    unsafe fn runtime_default(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any>;

    fn var_remap_get_raw(&mut self, raw: TVMFFIAny) -> Result<Option<Any>> {
        unsafe { runtime_var_remap_get(self.runtime(), raw) }
    }
    fn var_remap_set_raw(&mut self, raw: TVMFFIAny, replacement: &Any) -> Result<()> {
        unsafe { runtime_var_remap_set(self.runtime(), raw, replacement) }
    }

    fn call_registered_hook(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Option<Any>> {
        let mutator = self.runtime();
        let context = self.context();
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
        let runtime = self.runtime();
        let context = self.context();
        with_current_driver_context(runtime, context, || unsafe {
            ((*runtime_state(runtime)).default_mutate)(runtime, raw, def_region_kind, permit)
        })
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

type StructuralMutatorHandle = *mut StructuralMutatorObj;
type FStructuralMutate =
    unsafe extern "C" fn(StructuralMutatorHandle, AnyView<'static>) -> TVMFFIAny;
type FStructuralVarRemapGet =
    unsafe extern "C" fn(StructuralMutatorHandle, AnyView<'static>) -> TVMFFIAny;
type FStructuralVarRemapSet =
    unsafe extern "C" fn(StructuralMutatorHandle, AnyView<'static>, AnyView<'static>) -> TVMFFIAny;

#[repr(C)]
struct StructuralMutatorVTable {
    mutate: FStructuralMutate,
    maybe_inplace_mutate: FStructuralMutate,
    var_remap_get: FStructuralVarRemapGet,
    var_remap_set: FStructuralVarRemapSet,
    state_offset: usize,
}

/// Exact C++ StructuralMutatorObj ABI prefix, with no Rust traversal state.
#[repr(C)]
pub struct StructuralMutatorObj {
    base: Object,
    vtable: *const StructuralMutatorVTable,
    def_region_mode: i32,
}

type MutationCallback =
    unsafe fn(StructuralMutatorHandle, TVMFFIAny, DefRegionKind, Permit) -> Result<Any>;
struct MutationState {
    context: *mut c_void,
    context_identity: *mut c_void,
    recurse: MutationCallback,
    default_mutate: MutationCallback,
    builtin_default: MutationCallback,
    plain_inline_fastpath: bool,
    remap: RefCell<StructuralVarRemap>,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

// The final inline field owns D or the explicitly borrowed dispatcher storage.
// Its offset depends on Dispatcher alignment, not just the size of Base.
#[repr(C)]
struct StructuralMutateEngine<Base, Dispatcher> {
    base: Base,
    state: MutationState,
    dispatcher: Dispatcher,
}

#[inline(always)]
unsafe fn runtime_state(runtime: StructuralMutatorHandle) -> *mut MutationState {
    // Only Rust-created engine handles use this extended vtable.
    runtime
        .cast::<u8>()
        .add((*(*runtime).vtable).state_offset)
        .cast()
}

const _: () = {
    assert!(
        std::mem::offset_of!(StructuralMutatorObj, vtable) == std::mem::size_of::<TVMFFIObject>()
    );
    assert!(
        std::mem::offset_of!(StructuralMutatorObj, def_region_mode)
            == std::mem::size_of::<TVMFFIObject>() + std::mem::size_of::<*const c_void>()
    );
};

// Scoped runtime prefixes are not independently allocatable ObjectCore values.
impl StructuralMutatorObj {
    const TYPE_KEY: &'static str = "ffi.StructuralMutator";
    fn type_index() -> i32 {
        static TYPE_INDEX: LazyLock<i32> = LazyLock::new(|| unsafe {
            let key = TVMFFIByteArray::from_str(StructuralMutatorObj::TYPE_KEY);
            let mut type_index = 0;
            if TVMFFITypeKeyToIndex(&key, &mut type_index) != 0 {
                panic!(
                    "ffi.StructuralMutator is not registered: {}",
                    Error::from_raised()
                );
            }
            type_index
        });
        *TYPE_INDEX
    }
}

struct RuntimeMutatorVTable<D, B, S>(PhantomData<(D, B, S)>);

impl<D: MutationDriver, B, S> RuntimeMutatorVTable<D, B, S> {
    const VTABLE: StructuralMutatorVTable = StructuralMutatorVTable {
        mutate: rust_vtable_mutate::<D>,
        maybe_inplace_mutate: rust_vtable_maybe_inplace_mutate::<D>,
        var_remap_get: rust_vtable_var_remap_get,
        var_remap_set: rust_vtable_var_remap_set,
        state_offset: std::mem::offset_of!(StructuralMutateEngine<B, S>, state),
    };
}

struct RuntimeContextGuard {
    runtime: StructuralMutatorHandle,
    previous: *mut c_void,
}
impl RuntimeContextGuard {
    #[inline(always)]
    unsafe fn bind(runtime: StructuralMutatorHandle, context: *mut c_void) -> Self {
        let previous = (*runtime_state(runtime)).context;
        (*runtime_state(runtime)).context = context;
        Self { runtime, previous }
    }
}
impl Drop for RuntimeContextGuard {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            (*runtime_state(self.runtime)).context = self.previous;
        }
    }
}

#[inline(always)]
fn bind_dispatcher<D>(runtime: StructuralMutatorHandle, dispatch: &mut D) -> RuntimeContextGuard {
    let context = std::ptr::from_mut(dispatch).cast();
    // D type equality does not prove instance identity or that this borrow
    // covers the runtime dispatcher. Keep this one comparison in release builds.
    if context != unsafe { (*runtime_state(runtime)).context_identity } {
        wrong_mutation_dispatcher();
    }
    unsafe { RuntimeContextGuard::bind(runtime, context) }
}
#[cold]
#[inline(never)]
fn wrong_mutation_dispatcher() -> ! {
    panic!("mutation context used with a different dispatcher instance")
}

unsafe extern "C" fn rust_vtable_mutate<D: MutationDriver>(
    mutator: StructuralMutatorHandle,
    value: AnyView<'static>,
) -> TVMFFIAny {
    // SAFETY: this function is installed only in the vtable of a live
    // StructuralMutatorObj built for D; `value` is borrowed for this call.
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
    let raw = *value.as_raw_ffi_any();
    let outcome = catch_unwind(AssertUnwindSafe(
        #[inline]
        || {
            let kind = std::mem::transmute((*mutator).def_region_mode);
            D::runtime_dispatch::<INPLACE>(mutator, raw, kind)
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
    (*runtime_state(mutator)).panic = Some(payload);
    result_into_raw(Err(runtime_error("panic in structural mutator callback")))
}

unsafe extern "C" fn rust_vtable_var_remap_get(
    mutator: StructuralMutatorHandle,
    var: AnyView<'static>,
) -> TVMFFIAny {
    let raw = *var.as_raw_ffi_any();
    match catch_unwind(AssertUnwindSafe(|| runtime_var_remap_get(mutator, raw))) {
        Ok(Ok(Some(replacement))) => Any::into_raw_ffi_any(replacement),
        Ok(Ok(None)) => TVMFFIAny::new(),
        Ok(Err(error)) => result_into_raw(Err(error)),
        Err(payload) => {
            (*runtime_state(mutator)).panic = Some(payload);
            result_into_raw(Err(runtime_error("panic in structural var-remap lookup")))
        }
    }
}

unsafe extern "C" fn rust_vtable_var_remap_set(
    mutator: StructuralMutatorHandle,
    var: AnyView<'static>,
    replacement: AnyView<'static>,
) -> TVMFFIAny {
    let var_raw = *var.as_raw_ffi_any();
    let replacement_raw = *replacement.as_raw_ffi_any();
    match catch_unwind(AssertUnwindSafe(|| {
        let replacement = owned_from_raw(replacement_raw)?;
        runtime_var_remap_set(mutator, var_raw, &replacement)
    })) {
        Ok(Ok(())) => TVMFFIAny::new(),
        Ok(Err(error)) => result_into_raw(Err(error)),
        Err(payload) => {
            (*runtime_state(mutator)).panic = Some(payload);
            result_into_raw(Err(runtime_error(
                "panic in structural var-remap insertion",
            )))
        }
    }
}

#[inline(always)]
fn with_current_driver_context<T>(
    runtime: StructuralMutatorHandle,
    context: *mut c_void,
    callback: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let _binding = unsafe { RuntimeContextGuard::bind(runtime, context) };
    callback()
}

impl<D: MapDispatch, const PRE_ORDER: bool> MutationDriver for NativeMapper<'_, D, PRE_ORDER> {
    fn runtime(&self) -> StructuralMutatorHandle {
        self.runtime
    }
    fn context(&mut self) -> *mut c_void {
        std::ptr::from_mut(&mut *self.dispatch).cast()
    }
    unsafe fn runtime_dispatch<const INPLACE: bool>(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> TVMFFIAny {
        NativeMapper::<_, PRE_ORDER> {
            dispatch: &mut *(*runtime_state(runtime)).context.cast::<D>(),
            runtime,
        }
        .dispatch_abi_raw::<INPLACE>(raw, kind)
    }
    unsafe fn runtime_recurse(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        NativeMapper::<D, PRE_ORDER> {
            dispatch: &mut *(*runtime_state(runtime)).context.cast::<D>(),
            runtime,
        }
        .dispatch_raw(raw, kind, permit)
    }
    unsafe fn runtime_default(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        with_mutation_region(runtime, kind, |kind| {
            default_mutate_driver(
                &mut NativeMapper::<D, PRE_ORDER> {
                    dispatch: &mut *(*runtime_state(runtime)).context.cast::<D>(),
                    runtime,
                },
                raw,
                kind,
                permit,
            )
        })
    }
    fn dispatch_raw(
        &mut self,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        with_mutation_region(self.runtime, def_region_kind, |kind| {
            self.map_raw(raw, kind, permit)
        })
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
}

struct UserMutator<'a, D> {
    dispatch: &'a mut D,
    runtime: StructuralMutatorHandle,
}

impl<D: MutateDispatch> MutationDriver for UserMutator<'_, D> {
    fn runtime(&self) -> StructuralMutatorHandle {
        self.runtime
    }
    fn context(&mut self) -> *mut c_void {
        std::ptr::from_mut(&mut *self.dispatch).cast()
    }

    unsafe fn runtime_dispatch<const INPLACE: bool>(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> TVMFFIAny {
        let mut driver = UserMutator {
            dispatch: &mut *(*runtime_state(runtime)).context.cast::<D>(),
            runtime,
        };
        driver.dispatch_abi_raw::<INPLACE>(raw, kind)
    }
    unsafe fn runtime_recurse(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        UserMutator {
            dispatch: &mut *(*runtime_state(runtime)).context.cast::<D>(),
            runtime,
        }
        .dispatch_raw(raw, kind, permit)
    }
    unsafe fn runtime_default(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        kind: DefRegionKind,
        permit: Permit,
    ) -> Result<Any> {
        with_mutation_region(runtime, kind, |kind| {
            default_mutate_driver(
                &mut UserMutator {
                    dispatch: &mut *(*runtime_state(runtime)).context.cast::<D>(),
                    runtime,
                },
                raw,
                kind,
                permit,
            )
        })
    }
    fn dispatch_raw(&mut self, raw: TVMFFIAny, kind: DefRegionKind, permit: Permit) -> Result<Any> {
        with_mutation_region(self.runtime, kind, |kind| {
            let mode = permit.inplace_mode(raw);
            with_mutation_view(self.dispatch, self.runtime, |dispatch, mutator| {
                let value = MutationView::new(unsafe { borrowed_view(&raw) }, mode);
                match dispatch.dispatch_mutate(value, mutator) {
                    Some(result) => result,
                    None => UserMutator {
                        dispatch,
                        runtime: self.runtime,
                    }
                    .default_map_current_raw(raw, kind, mode.permit()),
                }
            })
        })
        .map_err(|error| with_value_context(error, raw))
    }
}

struct MutationScope {
    runtime: StructuralMutatorHandle,
}
impl MutationScope {
    #[inline(always)]
    fn context<D>(&mut self) -> StructuralMutatorView<'_, D> {
        StructuralMutatorView {
            runtime: self.runtime,
            _scope: PhantomData,
            _dispatcher: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

#[inline(always)]
fn with_mutation_view<D, T>(
    dispatch: &mut D,
    runtime: StructuralMutatorHandle,
    callback: impl FnOnce(&mut D, &mut StructuralMutatorView<'_, D>) -> T,
) -> T {
    let mut scope = MutationScope { runtime };
    callback(dispatch, &mut scope.context())
}

fn run_user_mutator<S: DispatcherStorage>(root: Any, dispatch: S) -> Result<Any>
where
    S::Dispatch: MutateDispatch,
{
    run_mutator_engine::<UserMutator<'_, S::Dispatch>, S, _>(
        root,
        dispatch,
        StructuralMutatorObj::default(),
        true,
    )
}

unsafe fn runtime_var_remap_get(
    runtime: StructuralMutatorHandle,
    raw: TVMFFIAny,
) -> Result<Option<Any>> {
    (*runtime_state(runtime))
        .remap
        .borrow()
        .get(borrowed_view(&raw))
}
unsafe fn runtime_var_remap_set(
    runtime: StructuralMutatorHandle,
    raw: TVMFFIAny,
    value: &Any,
) -> Result<()> {
    (*runtime_state(runtime))
        .remap
        .borrow_mut()
        .set(borrowed_view(&raw), value)
}
trait DispatcherStorage {
    type Dispatch;
    /// Project only during startup, after the engine has reached its stable allocation.
    unsafe fn project(storage: *mut Self) -> *mut Self::Dispatch;
}
#[repr(transparent)]
struct OwnedDispatcher<D>(D);
#[repr(transparent)]
struct BorrowedDispatcher<'a, D>(&'a mut D);
impl<D> DispatcherStorage for OwnedDispatcher<D> {
    type Dispatch = D;
    unsafe fn project(storage: *mut Self) -> *mut D {
        std::ptr::addr_of_mut!((*storage).0)
    }
}
impl<D> DispatcherStorage for BorrowedDispatcher<'_, D> {
    type Dispatch = D;
    unsafe fn project(storage: *mut Self) -> *mut D {
        std::ptr::from_mut(&mut *(*storage).0)
    }
}

fn run_mutator_engine<D: MutationDriver, S: DispatcherStorage, B: StructuralMutatorBase>(
    root: Any,
    dispatcher: S,
    base: B,
    plain_inline_fastpath: bool,
) -> Result<Any> {
    let mut engine = Box::new(StructuralMutateEngine {
        base,
        state: MutationState {
            context: std::ptr::null_mut(),
            context_identity: std::ptr::null_mut(),
            recurse: D::runtime_recurse,
            default_mutate: base_default_mutate::<B, S::Dispatch>,
            builtin_default: D::runtime_default,
            plain_inline_fastpath,
            remap: RefCell::new(StructuralVarRemap::default()),
            panic: None,
        },
        dispatcher,
    });
    let allocation = std::ptr::from_mut(&mut *engine);
    let handle = allocation.cast::<StructuralMutatorObj>();
    unsafe {
        // The scoped Box owns the reserved strong/weak counts; trusted hooks
        // balance temporary references before returning and cannot retain them.
        (*handle).vtable = &RuntimeMutatorVTable::<D, B, S>::VTABLE;
        (*handle).def_region_mode = DefRegionKind::None as i32;
        initialize_scoped_runtime(handle.cast(), StructuralMutatorObj::type_index());
        // Project the tail only here, after the Box is stable. Recursive calls
        // use the current D reborrow in runtime metadata, never this projection.
        let storage = allocation
            .cast::<u8>()
            .add(std::mem::offset_of!(StructuralMutateEngine<B, S>, dispatcher))
            .cast::<S>();
        let context = S::project(storage).cast::<c_void>();
        (*runtime_state(handle)).context = context;
        (*runtime_state(handle)).context_identity = context;
    }
    let result = call_mutator(
        handle,
        *root.as_raw_ffi_any(),
        DefRegionKind::None,
        Permit::MaybeInPlace,
    );
    let panic = unsafe { (*runtime_state(handle)).panic.take() };
    drop(engine);
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
    let use_inplace = permit == Permit::MaybeInPlace && object_is_unique(raw);
    let callback = unsafe {
        if use_inplace {
            (*(*mutator).vtable).maybe_inplace_mutate
        } else {
            (*(*mutator).vtable).mutate
        }
    };
    with_mutator_def_region(mutator, def_region_kind, |_| unsafe {
        let view = AnyView::from_raw_ffi_any(raw);
        result_from_raw(callback(mutator, view))
    })
}

#[inline]
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
    with_mutator_def_region(mutator, def_region_kind, |_| unsafe {
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
    callback: impl FnOnce(DefRegionKind) -> T,
) -> T {
    unsafe {
        let previous = (*mutator).def_region_mode;
        // Precedence: a pattern region propagates; entering any kind inside it has no effect.
        let effective = if previous == DefRegionKind::Pattern as i32 {
            DefRegionKind::Pattern
        } else {
            (*mutator).def_region_mode = kind as i32;
            kind
        };
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
        callback(effective)
    }
}

#[inline(always)]
fn with_mutation_region<T>(
    runtime: StructuralMutatorHandle,
    kind: DefRegionKind,
    callback: impl FnOnce(DefRegionKind) -> Result<T>,
) -> Result<T> {
    with_mutator_def_region(runtime, kind, callback)
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

/// Mutate a structured value with an owned or borrowed [`Dispatcher`].
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

#[inline]
fn checked_type_info(type_index: i32) -> Result<*const crate::tvm_ffi_sys::TVMFFITypeInfo> {
    let info = unsafe { TVMFFIGetTypeInfo(type_index) };
    if info.is_null() {
        Err(unregistered_type_error(type_index))
    } else {
        Ok(info)
    }
}

#[cold]
fn unregistered_type_error(type_index: i32) -> Error {
    runtime_error(&format!(
        "native structural map: unregistered type index {type_index}"
    ))
}

#[inline(always)]
fn object_is_unique(raw: TVMFFIAny) -> bool {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return false;
    }
    let pointer = unsafe { raw.data_union.v_obj };
    !pointer.is_null() && unsafe { object::unsafe_::strong_count(pointer) == 1 }
}

#[inline]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_abi_ownership_normalizes_views_and_rejects_null_objects() {
        let text = std::ffi::CString::new("borrowed traversal string").unwrap();
        let mut raw = TVMFFIAny::new();
        raw.type_index = TVMFFITypeIndex::kTVMFFIRawStr as i32;
        raw.data_union.v_c_str = text.as_ptr().cast();
        let owned = owned_from_raw(raw).unwrap();
        drop(text);
        assert_eq!(
            owned.try_as::<crate::String>().unwrap().as_str(),
            "borrowed traversal string"
        );

        let bytes = vec![1_u8, 0, 255];
        let data = TVMFFIByteArray::new(bytes.as_ptr(), bytes.len());
        raw.type_index = TVMFFITypeIndex::kTVMFFIByteArrayPtr as i32;
        raw.data_union.v_ptr = std::ptr::from_ref(&data).cast_mut().cast();
        let owned = owned_from_raw(raw).unwrap();
        drop(bytes);
        assert_eq!(
            owned.try_as::<crate::Bytes>().unwrap().as_slice(),
            &[1, 0, 255]
        );

        let owner = Any::from(crate::Array::new(vec![1_i64]));
        let mut slot = unsafe { Any::into_raw_ffi_any(owner).data_union.v_obj };
        raw.type_index = TVMFFITypeIndex::kTVMFFIObjectRValueRef as i32;
        raw.data_union.v_ptr = std::ptr::from_mut(&mut slot).cast();
        let moved = owned_from_raw(raw).unwrap();
        assert!(slot.is_null());
        assert_eq!(moved.debug_strong_count(), Some(1));
        assert_eq!(
            moved.try_as::<crate::Array<i64>>().unwrap().get(0).unwrap(),
            1
        );

        raw.type_index = TVMFFITypeIndex::kTVMFFIArray as i32;
        raw.data_union.v_obj = std::ptr::null_mut();
        let error = owned_from_raw(raw)
            .err()
            .expect("null objects must be rejected");
        assert!(error.message().contains("null pointer"));
    }
}
