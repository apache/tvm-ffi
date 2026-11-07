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

// `_raw` functions move one owned `TVMFFIAny`: forward it once, or import it into `Any` before
// anything can unwind. Private `extern "C-unwind"` helpers return it in registers on AArch64.

use std::cell::{RefCell, UnsafeCell};
use std::collections::{hash_map::Entry, HashMap};
use std::ffi::c_void;
use std::marker::PhantomData;

use super::cold_init_lazy_lock::ColdInitLazyLock;
use super::dispatch::IntoDispatcher;
use std::borrow::BorrowMut;
use std::ops::ControlFlow;
use std::ptr::NonNull;
use std::rc::Rc;

pub use super::structural_map::*;

use super::dispatch::Dispatcher;
use crate::any::{is_plain_inline, Any, AnyView};
use crate::error::{Error, Result, TYPE_ERROR};
use crate::function::Function;
use crate::object::{self, Object, ObjectArc};
use crate::reflection::TypeAttrColumn;
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyViewToOwnedAny, TVMFFIFieldInfo, TVMFFIFieldSetter, TVMFFIFunctionCall,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeIndex,
};
use crate::tvm_ffi_sys::{TVMFFIObjectHandle, TVMFFISEqHashKind};

use super::structural_common::{
    field_def_region, for_each_field_info, free_var_child_region, missing_field_getter_error,
    raw_ffi_any_into_result, raw_ffi_any_same_as, registered_type_index, result_into_raw_ffi_any,
    runtime_error, seq_hash_kind, with_structural_error_context,
    with_structural_field_error_context, wrong_dispatcher, FLAG_SEQ_HASH_IGNORE,
    FLAG_SETTER_IS_FUNCTION,
};
use super::structural_visit::DefRegionKind;
use super::unchanged::is_unchanged;
pub use super::unchanged::{Unchanged, UnchangedOr};
use super::visit_error_context::update_visit_error_context;

const STRUCTURAL_MUTATE_ATTR: &str = "__s_mutate__";
const STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR: &str = "__s_maybe_inplace_mutate__";
const SHALLOW_COPY_ATTR: &str = "__ffi_shallow_copy__";

// -----------------------------------------------------------------------------
// Data structures and FFI layout
// -----------------------------------------------------------------------------

pub(super) type StructuralMutatorHandle = *mut StructuralMutatorObj;

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
}

/// The native mutator state, laid out as the prefix of the C++
/// `StructuralMutatorObj`, with no Rust traversal state.
#[repr(C)]
pub struct StructuralMutatorObj {
    base_: Object,
    vtable: *const StructuralMutatorVTable,
    def_region_mode: UnsafeCell<i32>,
}

// Scoped runtime prefixes are not independently allocatable ObjectCore values.
impl StructuralMutatorObj {
    const TYPE_KEY: &'static str = "ffi.StructuralMutator";
}

impl Default for StructuralMutatorObj {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base_: Object::new(),
            vtable: std::ptr::null(),
            def_region_mode: UnsafeCell::new(DefRegionKind::None as i32),
        }
    }
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

/// Native state and default mutation for [`StructuralMutateEngine`].
/// The engine owns allocation, scoped dispatch, identity remaps and results.
///
/// # Safety
///
/// `Self` must have a complete initialized C layout with [`StructuralMutatorObj`]
/// at offset zero, matching its registered native type and every field hooks access.
/// Native-mutated fields require interior mutability; shared references must not
/// freeze them during recursion. Hooks must preserve the vtable and valid region state.
///
/// The base must not implement [`crate::ObjectCore`] or otherwise allow callers
/// to retain the runtime. Hooks may access the runtime and borrowed mutation
/// state only synchronously, on the calling thread.
///
/// `default_mutate_raw` returns one owning ABI replacement/Unchanged/Error and recurses
/// only through the installed vtable, preserving the input's in-place mode.
/// No whole-base mutable reference may span recursion. Failures use the result
/// carrier; panics must not cross native callbacks.
pub unsafe trait StructuralMutatorBase: Sized {
    /// Initialize the complete base before the engine installs its runtime header.
    fn new() -> Result<Self>;

    /// Native type registered for this complete base layout.
    fn type_index() -> i32;

    /// Descend from a borrowed view without dispatching that value again.
    ///
    /// # Safety
    /// `runtime` is the base of a live `StructuralMutateEngine<Self>` with its
    /// matching installed vtable.
    /// The input and runtime remain borrowed for this synchronous call. The
    /// caller has published the current exclusive dispatcher reborrow.
    unsafe fn default_mutate_raw(runtime: *mut Self, value: MutationView<'_>) -> TVMFFIAny;
}

// SAFETY: this prefix exactly matches the ordinary native mutator. Its mutable
// region is interior mutable, and it exposes no safe ownership conversion.
unsafe impl StructuralMutatorBase for StructuralMutatorObj {
    #[inline(always)]
    fn new() -> Result<Self> {
        Ok(Self::default())
    }

    fn type_index() -> i32 {
        static TYPE_INDEX: ColdInitLazyLock<i32> =
            ColdInitLazyLock::new(|| registered_type_index(StructuralMutatorObj::TYPE_KEY));
        *TYPE_INDEX
    }

    #[inline(always)]
    unsafe fn default_mutate_raw(runtime: *mut Self, value: MutationView<'_>) -> TVMFFIAny {
        // Prefix-only: remaps go through the vtable, so any base may reuse this.
        let raw = value.value;
        // Registered hooks own their remapping policy; reflection handles its own.
        if let Some(attr) = Self::structural_mutate_hook(raw, value.mode) {
            return Self::call_structural_mutate_hook_raw(runtime, raw, attr);
        }
        Self::mutate_without_hook_raw(runtime, raw)
    }
}

/// Scoped Rust mutator runtime with a statically selected native base.
///
/// Enter through [`StructuralMutateEngine::mutate_root`]. The engine owns the complete base,
/// dispatcher reborrow handle and identity remaps. The dispatcher stays in the
/// outer call; callbacks receive only a scoped [`StructuralMutatorView`].
#[repr(C)]
pub struct StructuralMutateEngine<Base = StructuralMutatorObj> {
    base: Base,
    dispatcher: *mut c_void,
    remap: RefCell<StructuralVarRemap>,
}

/// Identity substitutions for structural mutation within one invocation.
///
/// Internal to the engine: handlers reach the remap through
/// [`StructuralMutatorView::var_remap_get`] and [`StructuralMutatorView::var_remap_set`].
///
/// Each pointer key owns one strong reference to its original object, preventing
/// address reuse until removal. Values are owning replacements, including
/// [`Unchanged`] markers. Replacing a value does not acquire another key reference;
/// lookup clones the replacement, and dropping releases every owner.
// Perf note: Store only the replacement Any; raw identity keys avoid a second
// Any payload, but still own one retained reference per distinct key. Replacing
// a value does not retain the key again; Drop releases each key exactly once.
#[derive(Default)]
pub(crate) struct StructuralVarRemap {
    entries: HashMap<NonNull<TVMFFIObject>, Any>,
}

impl StructuralVarRemap {
    /// Look up an identity replacement previously stored for `var`.
    pub(crate) fn get(&self, var: AnyView<'_>) -> Result<Option<Any>> {
        let key = object_identity_key(*var.as_raw_ffi_any())?;
        Ok(self.entries.get(&key).cloned())
    }

    /// Store a descent result or an [`Unchanged`] marker for `var`.
    ///
    /// A new key acquires one strong reference; replacing its value preserves
    /// that reference. The replacement is cloned before modifying the table.
    pub(crate) fn set(&mut self, var: AnyView<'_>, mutated_value: &Any) -> Result<()> {
        let key = object_identity_key(*var.as_raw_ffi_any())?;
        let value = mutated_value.clone();
        match self.entries.entry(key) {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
                // Insertion may allocate, so retain only after it succeeds.
                // The borrowed var still keeps key live; IncRef cannot unwind.
                unsafe { object::unsafe_::inc_ref(key.as_ptr()) };
            }
        }
        Ok(())
    }
}

impl Drop for StructuralVarRemap {
    fn drop(&mut self) {
        // Release keys first: their C ABI deleters cannot unwind. Field Drop
        // then releases values and buckets without resetting capacity for reuse.
        // The remaining NonNull key bits are dropped without dereferencing them.
        for key in self.entries.keys() {
            unsafe { object::unsafe_::dec_ref(key.as_ptr()) };
        }
    }
}

/// Whether a [`MutationView`] may reuse its value's storage in place.
///
/// `Allow` is an upper bound: a value is reused in place only when it is also
/// uniquely owned. A view made from `&mut Any` starts as `Allow`, and a view
/// from [`MutationView::borrowed`] is always `Disallow`.
///
/// See also [`structural_mutate`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InplaceMode {
    /// Use ordinary mutation, including for uniquely owned values.
    #[default]
    Disallow,
    /// Permit reuse when ownership and alias checks allow it.
    Allow,
}

impl InplaceMode {
    // The mode is only an upper bound; ownership must still be unique here.
    #[inline(always)]
    fn for_value(self, raw: TVMFFIAny) -> Self {
        if self == Self::Allow && object_is_unique(raw) {
            Self::Allow
        } else {
            Self::Disallow
        }
    }
}

/// A borrowed value that a structural mutation may rewrite in place.
///
/// Mutation handlers receive a `MutationView` instead of an owned value, so
/// matching a node does not retain it. Inspect the node through the view,
/// then pass the view to [`StructuralMutatorView::default_mutate`] to rewrite
/// its children, or rewrite one field with [`field`](MutationView::field).
/// When the node is uniquely owned and the view carries
/// [`InplaceMode::Allow`], its storage is reused instead of copied.
///
/// `T` is the matched type: `MutationView<'_, Array<Any>>` derefs to the
/// array's node, and the default `MutationView<'_>` matches every value. The
/// view is move-only: it cannot be copied or cloned, and consuming it passes
/// on the same mode. A view made from `&mut Any` may reuse storage;
/// one made with [`borrowed`](MutationView::borrowed) never does.
///
/// See also [`structural_mutate`].
pub struct MutationView<'a, T = Any> {
    value: TVMFFIAny,
    mode: InplaceMode,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<(AnyView<'a>, fn() -> T, Rc<()>)>,
}

impl<T> MutationView<'_, T> {
    /// The viewed cell, borrowed in place by `match_any!` reference patterns.
    #[inline(always)]
    pub(crate) fn cell(&self) -> &TVMFFIAny {
        &self.value
    }
}

impl<'a> MutationView<'a> {
    #[inline(always)]
    pub(super) fn new(value: AnyView<'a>, mode: InplaceMode) -> Self {
        Self::from_raw(*value.as_raw_ffi_any(), mode)
    }

    /// Create a view of a borrowed value that never reuses its storage.
    ///
    /// The mutation result is always a new value or [`Unchanged`]. To allow reuse,
    /// make the view from `&mut Any` instead.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to view.
    ///
    /// # Returns
    ///
    /// A view with [`InplaceMode::Disallow`].
    #[inline(always)]
    pub fn borrowed(value: AnyView<'a>) -> Self {
        Self::new(value, InplaceMode::Disallow)
    }
}

impl<'a> From<&'a mut Any> for MutationView<'a> {
    /// Borrow an owner exclusively without acquiring a reference count.
    ///
    /// Reuse still requires the runtime's uniqueness and alias checks.
    #[inline(always)]
    fn from(value: &'a mut Any) -> Self {
        Self::new(AnyView::from(&*value), InplaceMode::Allow)
    }
}

impl<'a, T> MutationView<'a, T> {
    #[inline(always)]
    fn from_raw(value: TVMFFIAny, mode: InplaceMode) -> Self {
        Self {
            value,
            mode,
            _marker: PhantomData,
        }
    }

    // A chain may lend its view to one link at a time. The shorter lifetime
    // prevents a matched link from retaining a borrow across the next link.
    #[inline(always)]
    pub(crate) fn reborrow(&mut self) -> MutationView<'_, T> {
        MutationView::from_raw(self.value, self.mode)
    }

    /// Return the in-place mode this view carries.
    ///
    /// # Returns
    ///
    /// Whether this view may reuse its value's storage.
    #[inline(always)]
    pub fn inplace_mode(&self) -> InplaceMode {
        self.mode
    }

    /// Restrict the mode, without ever upgrading it.
    ///
    /// `Disallow` turns off reuse. `Allow` keeps the current mode, so it never
    /// grants reuse to a view that does not have it, such as a borrowed one.
    ///
    /// # Arguments
    ///
    /// * `requested` - The mode to restrict to.
    ///
    /// # Returns
    ///
    /// The view with the restricted mode.
    #[inline(always)]
    pub fn with_inplace_mode(mut self, requested: InplaceMode) -> Self {
        if requested == InplaceMode::Disallow {
            self.mode = InplaceMode::Disallow;
        }
        self
    }

    /// Borrow the current value.
    ///
    /// The view and its copies borrow this handle, so their last use must precede
    /// consuming it for default mutation.
    ///
    /// # Returns
    ///
    /// The value as an [`AnyView`].
    #[inline(always)]
    pub fn as_any_view(&self) -> AnyView<'_> {
        // SAFETY: the input owner keeps this payload live while the view is
        // borrowed. The returned lifetime is bounded by self, not by 'a.
        unsafe { AnyView::from_raw_ffi_any(self.value) }
    }

    /// Borrow the value as `U` without changing ownership or mode.
    ///
    /// `U` is anything [`AnyView::downcast_ref`] borrows: a managed reference such as
    /// `Array<i64>`, an object type such as `ArrayObj`, or [`Any`], borrowed in place
    /// without a reference-count bump. The check is strict, and the reference borrows
    /// this view and cannot survive its consumption.
    ///
    /// # Returns
    ///
    /// The borrowed value, or `None` when the value is not a `U`.
    #[inline(always)]
    pub fn downcast_ref<U: crate::type_traits::DowncastRefFromAny>(&self) -> Option<&U> {
        // SAFETY: the view holds a valid cell, borrowed for the returned lifetime,
        // and the input owner keeps its payload live.
        unsafe { U::downcast_ref_from_any(&self.value) }
    }

    /// Narrow the matched type without acquiring an owning reference.
    ///
    /// The check is the strict one of [`Any::downcast_ref`], and the view keeps its
    /// mode.
    ///
    /// # Returns
    ///
    /// The view typed as `MutationView<'_, U>`.
    ///
    /// # Errors
    ///
    /// Returns the view unchanged when the value is not a `U`, so another type can
    /// be tried.
    #[inline(always)]
    pub fn downcast<U: crate::type_traits::ContainerElement>(
        self,
    ) -> std::result::Result<MutationView<'a, U>, Self> {
        // SAFETY: the input owner keeps the borrowed value live for this handle's
        // lifetime. This only checks its type and does not acquire ownership.
        if unsafe { U::container_check_any_strict(&self.value) } {
            Ok(MutationView::from_raw(self.value, self.mode))
        } else {
            Err(self)
        }
    }

    /// Erase the matched type while preserving the carried mode.
    ///
    /// # Returns
    ///
    /// The view as `MutationView<'_>`.
    #[inline(always)]
    pub fn into_untyped(self) -> MutationView<'a> {
        MutationView::from_raw(self.value, self.mode)
    }
}

mod mutation_field {
    pub trait Sealed {}
    impl Sealed for crate::Any {}
    impl<T: crate::object::ObjectRefCore> Sealed for T {}
}

/// Owned storage supported by [`MutationView::field`].
/// Sealed to `Any` and owning object handles; borrowed conversion adapters do not qualify.
#[doc(hidden)]
pub trait MutationField: crate::type_traits::ContainerElement + mutation_field::Sealed {}

impl MutationField for Any {}

impl<T: object::ObjectRefCore> MutationField for T {}

impl<T: object::ObjectRefCore> std::ops::Deref for MutationView<'_, T> {
    type Target = T::ContainerType;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: typed view construction establishes T's object layout;
        // this shared borrow cannot outlive or survive consumption of the handle.
        unsafe { &*self.value.data_union.v_obj.cast::<T::ContainerType>() }
    }
}

impl<T: object::ObjectRefCore> MutationView<'_, T> {
    /// Borrow a direct owning field of the node as a view of its own, to rewrite it
    /// with [`StructuralMutatorView::mutate`].
    ///
    /// The field view exclusively borrows this view. It may reuse the field's
    /// storage only when this view may and the node remains uniquely owned. This
    /// does not write the result back into the node: keep the node [`Unchanged`] or
    /// rebuild it from the result.
    ///
    /// # Arguments
    ///
    /// * `selector` - Returns a reference to one of the node's own owning fields,
    ///   an [`Any`] or a managed reference.
    ///
    /// # Returns
    ///
    /// A view of the field.
    ///
    /// # Panics
    ///
    /// Panics if `selector` returns a reference outside the node.
    ///
    /// ```
    /// # use tvm_ffi::object::ObjectRef;
    /// # use tvm_ffi::*;
    /// # #[rs_object("my_ext.MyPair", final)]
    /// # pub struct MyPairObj { base_: Object, pub first: Array<i64>, pub second: Array<i64> }
    /// # #[derive(Clone, tvm_ffi::derive::ObjectRef)]
    /// # pub struct MyPair { data: ObjectArc<MyPairObj> }
    /// # struct Pass;
    /// # #[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
    /// # impl Pass {
    /// #     fn mutate_pair(&mut self, mut pair: MutationView<'_, MyPair>, mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
    /// let second = mutator.mutate(self, pair.field(|pair| &pair.second))?;
    /// #         Ok(second)
    /// #     }
    /// # }
    /// # let pair = MyPairObj::new(Object::new(), Array::from_vec(vec![1_i64]), Array::from_vec(vec![2_i64]));
    /// # structural_mutate(MyPair::from_object(pair), Pass)?;
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`structural_mutate`].
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
        // the returned view exclusively borrows the parent view.
        unsafe { U::container_copy_to_any_view(child, &mut value) };
        // Selection and conversion may retain aliases, so check the parent last.
        let mode = self.mode.for_value(self.value);
        MutationView::from_raw(value, mode)
    }
}

impl<'source, 'value, T: crate::type_traits::ContainerElement>
    super::dispatch::DispatchRecovery<crate::match_any_internal::MutationSource<'source>>
    for crate::MutationView<'value, T>
{
    type Recovery = crate::MutationView<'source>;
}

/// The running structural mutation, passed to each handler.
///
/// A handler that matches a node decides how its children are rewritten: call
/// [`default_mutate`](Self::default_mutate) to rewrite every child through the
/// dispatcher, [`mutate`](Self::mutate) to rewrite one value, or neither to
/// leave the children as they are. Pass the handler's own `self` as `dispatch`
/// to every method.
///
/// See also [`structural_mutate`].
pub struct StructuralMutatorView<'a, D, Base = StructuralMutatorObj> {
    runtime: StructuralMutatorHandle,
    #[allow(clippy::type_complexity)]
    _marker: PhantomData<(&'a mut (), fn(&mut D) -> Base, Rc<()>)>,
}

// -----------------------------------------------------------------------------
// Public mutation API and entrypoint
// -----------------------------------------------------------------------------

impl<D, Base: StructuralMutatorBase> StructuralMutatorView<'_, D, Base> {
    /// Inspect the base state without retaining the scoped mutator runtime.
    ///
    /// # Returns
    ///
    /// The base state, borrowed for the lifetime of this view.
    #[inline(always)]
    pub fn base(&self) -> &Base {
        // SAFETY: the unsafe base contract permits shared inspection, including
        // native mutation through interior-mutable fields during recursion.
        unsafe { &*self.runtime.cast::<Base>() }
    }

    #[inline(always)]
    fn reborrow(&self) -> StructuralMutatorView<'_, D, Base> {
        StructuralMutatorView {
            runtime: self.runtime,
            _marker: PhantomData,
        }
    }

    /// Return the definition region the current value is in.
    ///
    /// Reflected fields marked as definitions set it while their values are
    /// mutated; it is [`DefRegionKind::None`] elsewhere.
    ///
    /// # Returns
    ///
    /// The current definition region.
    #[inline(always)]
    pub fn def_region_kind(&self) -> DefRegionKind {
        // SAFETY: the view's runtime is live.
        unsafe { std::mem::transmute(*(*self.runtime).def_region_mode.get()) }
    }

    /// Run `callback` under a scoped definition region.
    ///
    /// An enclosing [`DefRegionKind::Pattern`] region keeps precedence, since a
    /// pattern applies to everything inside it. The previous region is restored
    /// afterwards.
    ///
    /// # Arguments
    ///
    /// * `kind` - The definition region to enter.
    /// * `callback` - Runs inside the region, with this view.
    ///
    /// # Returns
    ///
    /// The result of `callback`.
    #[inline(always)]
    pub fn with_def_region_kind<R>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut StructuralMutatorView<'_, D, Base>) -> R,
    ) -> R {
        // SAFETY: the view's runtime outlives the callback.
        StructuralMutatorObj::with_def_region_kind(self.runtime, kind, |_| {
            callback(&mut self.reborrow())
        })
    }

    /// Mutate `value` through the dispatcher, as if the traversal had reached it.
    ///
    /// `value` keeps the mode it carries. Use it to rewrite a value other
    /// than the current node, such as a field selected with
    /// [`MutationView::field`].
    ///
    /// # Arguments
    ///
    /// * `dispatch` - The handler's own `self`.
    /// * `value` - The value to mutate.
    ///
    /// # Returns
    ///
    /// The replacement, or [`Unchanged`].
    ///
    /// # Errors
    ///
    /// Returns the error of the handler that `value` reaches.
    #[inline(always)]
    pub fn mutate<'input, T>(
        &mut self,
        dispatch: &mut D,
        value: impl Into<MutationView<'input, T>>,
    ) -> Result<UnchangedOr<Any>> {
        let value = value.into();
        unsafe {
            let context =
                StructuralMutateEngine::<Base>::checked_dispatcher(self.runtime, dispatch);
            (*StructuralMutateEngine::<Base>::from_handle(self.runtime)).dispatcher = context;
            StructuralMutatorObj::call_mutator(self.runtime, value.value, value.mode)
        }
        .and_then(UnchangedOr::from_carrier)
    }

    /// Rewrite the children of `value` through the dispatcher, without dispatching
    /// `value` itself again.
    ///
    /// Children are found by the type's registered mutation hook or its reflected
    /// fields. The result is unchanged when no child changed; otherwise it is a
    /// node with the new children, made in place when `value` allows it and is
    /// uniquely owned.
    ///
    /// # Arguments
    ///
    /// * `dispatch` - The handler's own `self`.
    /// * `value` - The value whose children to mutate, usually the view the
    ///   handler received.
    ///
    /// # Returns
    ///
    /// The rewritten node, or [`Unchanged`].
    ///
    /// # Errors
    ///
    /// Returns the first error of a child's handler.
    #[inline(always)]
    pub fn default_mutate<'input, T>(
        &mut self,
        dispatch: &mut D,
        value: impl Into<MutationView<'input, T>>,
    ) -> Result<UnchangedOr<Any>> {
        let value = value.into();
        unsafe {
            let context =
                StructuralMutateEngine::<Base>::checked_dispatcher(self.runtime, dispatch);
            raw_ffi_any_into_result(StructuralMutateEngine::<Base>::publish_and_descend_raw(
                self.runtime,
                context,
                value.value,
                value.mode,
            ))
        }
        .and_then(UnchangedOr::from_carrier)
    }

    /// Look up the replacement recorded for a variable in this mutation.
    ///
    /// Default mutation records the result of every free variable and DAG node it
    /// rewrites, and replaces each later occurrence with it.
    ///
    /// # Arguments
    ///
    /// * `dispatch` - The handler's own `self`.
    /// * `var` - The variable, an object.
    ///
    /// # Returns
    ///
    /// The recorded replacement, or `None`.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `var` is not an object.
    #[inline(always)]
    pub fn var_remap_get(&mut self, dispatch: &mut D, var: AnyView<'_>) -> Result<Option<Any>> {
        unsafe {
            StructuralMutateEngine::<Base>::checked_dispatcher(self.runtime, dispatch);
            StructuralMutateEngine::<Base>::runtime_var_remap_get(
                self.runtime,
                *var.as_raw_ffi_any(),
            )
        }
    }

    /// Record a replacement for a variable for the rest of this mutation.
    ///
    /// Default mutation then replaces each later occurrence of `var`, a free
    /// variable or DAG node, with `value`.
    ///
    /// # Arguments
    ///
    /// * `dispatch` - The handler's own `self`.
    /// * `var` - The variable, an object.
    /// * `value` - The replacement.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the replacement is recorded.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `var` is not an object.
    #[inline(always)]
    pub fn var_remap_set(&mut self, dispatch: &mut D, var: AnyView<'_>, value: &Any) -> Result<()> {
        unsafe {
            StructuralMutateEngine::<Base>::checked_dispatcher(self.runtime, dispatch);
            StructuralMutateEngine::<Base>::runtime_var_remap_set(
                self.runtime,
                *var.as_raw_ffi_any(),
                value,
            )
        }
    }
}

/// Mutate a structured value with an owned or borrowed [`Dispatcher`].
///
/// The root is consumed to establish the ownership boundary for optional
/// in-place mutation. A matching callback supplies the final value; unmatched
/// values use default mutation.
///
/// # Arguments
///
/// * `root` - Owned value to mutate, convertible into an [`Any`]. A uniquely
///   owned container may be reused in place; clone first to preserve the original.
/// * `dispatcher` - Owned or mutably borrowed dispatcher implementing [`IntoDispatcher`].
///   Matching handlers return replacements and control recursion through
///   [`StructuralMutatorView`]; unmatched values use default mutation. For closure
///   callbacks with automatic recursion, use [`structural_map`].
///
/// # Examples
///
/// Double the integers in nested arrays; default mutation descends into the
/// arrays:
///
/// ```
/// use tvm_ffi::{
///     dispatch, structural_mutate, Any, Array, MutationView, Result, StructuralMutatorView,
///     UnchangedOr,
/// };
///
/// struct Double;
///
/// #[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
/// impl Double {
///     fn mutate_int(
///         &mut self,
///         value: i64,
///         _: &mut StructuralMutatorView<'_, Self>,
///     ) -> Result<UnchangedOr<Any>> {
///         Ok((value * 2).into())
///     }
/// }
///
/// let root = Array::from_vec(vec![Array::from_vec(vec![1_i64, 2]), Array::from_vec(vec![3_i64])]);
/// let result = Array::<Array<i64>>::try_from(structural_mutate(root, Double)?)?;
/// assert_eq!(result.get(0)?.iter().copied().collect::<Vec<_>>(), vec![2, 4]);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub fn structural_mutate<R, H, M, D, O>(root: R, dispatcher: H) -> Result<Any>
where
    R: Into<Any>,
    O: BorrowMut<D>,
    for<'v, 'c, 'r> H: IntoDispatcher<
        (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, D, StructuralMutatorObj>,
        ),
        Option<Result<UnchangedOr<Any>>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
        (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, D, StructuralMutatorObj>,
        ),
        Option<Result<UnchangedOr<Any>>>,
    >,
{
    StructuralMutateEngine::mutate_root(root, dispatcher)
}

mod callback_result_sealed {
    use super::{Any, Result};

    pub trait Sealed {}

    impl<T: Into<Any>> Sealed for T {}
    impl<T: Into<Any>> Sealed for Result<T> {}
}

/// Convert an infallible or fallible map or mutation callback result into
/// [`Result<UnchangedOr<Any>>`].
///
/// A callback may return any value convertible into [`Any`], including
/// [`Unchanged`] and [`UnchangedOr`], or wrap it in [`Result`] to use `?`.
///
/// This trait is sealed and is not an extension point.
#[doc(hidden)]
pub trait IntoMutateResult: callback_result_sealed::Sealed {
    fn into_mutate_result(self) -> Result<UnchangedOr<Any>>;
}

impl<T: Into<Any>> IntoMutateResult for T {
    #[inline(always)]
    fn into_mutate_result(self) -> Result<UnchangedOr<Any>> {
        Ok(UnchangedOr::from_any(self.into()))
    }
}

impl<T: Into<Any>> IntoMutateResult for Result<T> {
    #[inline(always)]
    fn into_mutate_result(self) -> Result<UnchangedOr<Any>> {
        self.map(|value| UnchangedOr::from_any(value.into()))
    }
}

// -----------------------------------------------------------------------------
// StructuralMutateEngine implementation
// -----------------------------------------------------------------------------

// Perf note: Raw carriers avoid materializing an owning Any on the success path.
// Private C-unwind boundaries return this 16-byte carrier in x0/x1 on AArch64,
// avoiding the hidden result slot observed with the Rust ABI while allowing
// Rust cleanup to unwind. Native vtables remain extern "C".
// Raw bits provide no ownership guard: import an owning result into Any before
// fallible work or destruction of an observer; borrowed inputs must stay live.
impl<Base: StructuralMutatorBase> StructuralMutateEngine<Base> {
    /// Mutate `root` with `dispatcher` on the native base `Base`. See also [`structural_mutate`].
    #[inline(always)]
    pub fn mutate_root<R, H, M, D, O>(root: R, dispatcher: H) -> Result<Any>
    where
        R: Into<Any>,
        O: BorrowMut<D>,
        for<'v, 'c, 'r> H: IntoDispatcher<
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D, Base>),
            Option<Result<UnchangedOr<Any>>>,
            M,
            Target = D,
            Output = O,
        >,
        for<'s, 'v, 'c, 'r> &'s mut D: crate::Dispatcher<
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D, Base>),
            Option<Result<UnchangedOr<Any>>>,
        >,
    {
        // Normalize the input once. Named dispatchers keep their own recursive view
        // type; callback and tuple targets are their aggregate adapters.
        let mut storage = dispatcher.into_dispatcher();
        Self::run_mutator::<D, true>(root.into(), storage.borrow_mut())
    }

    /// The engine whose first field is the native prefix `runtime`.
    ///
    /// # Safety
    ///
    /// `runtime` is this live Rust engine, not an arbitrary native mutator; only raw
    /// projections of the result may cross recursion.
    #[inline(always)]
    unsafe fn from_handle(runtime: StructuralMutatorHandle) -> *mut Self {
        runtime.cast()
    }

    /// Validate a recursive caller and return its current exclusive reborrow.
    ///
    /// # Safety
    /// `runtime` must belong to the live invocation of `dispatch` with matching `D`.
    /// The identity check does not replace that lifetime/type contract; mismatch panics.
    #[inline(always)]
    unsafe fn checked_dispatcher<D>(
        runtime: StructuralMutatorHandle,
        dispatch: &mut D,
    ) -> *mut c_void {
        let context = std::ptr::from_mut(dispatch).cast();
        if context != (*Self::from_handle(runtime)).dispatcher {
            core::hint::cold_path();
            wrong_dispatcher();
        }
        context
    }

    /// Build native callbacks for dispatcher `D`. `CALLBACK_CONTEXT` selects
    /// whether direct callback failures receive input context; map adapters add their own.
    #[inline(always)]
    fn vtable<D, const CALLBACK_CONTEXT: bool>() -> &'static StructuralMutatorVTable
    where
        for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D, Base>),
            Option<Result<UnchangedOr<Any>>>,
        >,
    {
        &StructuralMutatorVTable {
            mutate: Self::rust_vtable_mutate::<D, false, CALLBACK_CONTEXT>,
            maybe_inplace_mutate: Self::rust_vtable_mutate::<D, true, CALLBACK_CONTEXT>,
            var_remap_get: Self::rust_vtable_var_remap_get,
            var_remap_set: Self::rust_vtable_var_remap_set,
        }
    }

    /// Enter a native callback with borrowed `value` and return one owning raw result.
    /// `INPLACE` sets the mode only; uniqueness is checked before dispatch.
    ///
    /// # Safety
    /// `runtime` must have this `D` vtable and a live published exclusive reborrow.
    /// The ABI lifetime on `value` is erased: it remains valid only during this call.
    /// Callbacks report errors through results and must not unwind across this C boundary.
    unsafe extern "C" fn rust_vtable_mutate<D, const INPLACE: bool, const CALLBACK_CONTEXT: bool>(
        runtime: StructuralMutatorHandle,
        value: AnyView<'static>,
    ) -> TVMFFIAny
    where
        for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D, Base>),
            Option<Result<UnchangedOr<Any>>>,
        >,
    {
        let context = (*Self::from_handle(runtime)).dispatcher;
        let raw = *value.as_raw_ffi_any();
        let mode = if INPLACE {
            InplaceMode::Allow
        } else {
            InplaceMode::Disallow
        };
        let mode = mode.for_value(raw);
        let dispatch = &mut *context.cast::<D>();
        let mut mutator = StructuralMutatorView {
            runtime,
            _marker: PhantomData,
        };
        let result = match Dispatcher::dispatch_tuple(
            &mut *dispatch,
            (MutationView::new(value, mode), &mut mutator),
        ) {
            // Map adapters attribute errors to their possibly replaced value;
            // direct mutation attributes callback failures to this input.
            Some(result) => {
                let result = result.map(Any::from);
                result_into_raw_ffi_any(if CALLBACK_CONTEXT {
                    result.map_err(|error| with_mutation_error_context(error, raw, "mutate"))
                } else {
                    result
                })
            }
            None => Self::publish_and_descend_raw(
                runtime,
                std::ptr::from_mut(dispatch).cast(),
                raw,
                mode,
            ),
        };
        // All callback reborrows have ended before restoring the caller's provenance.
        (*Self::from_handle(runtime)).dispatcher = context;
        result
    }

    /// Publish `context`, then descend through hooks or reflection for borrowed `raw`.
    /// `mode` is an upper bound, not proof of uniqueness. Returns one owning token.
    ///
    /// # Safety
    /// `runtime` must be live; `context` must be its current exclusive dispatcher
    /// reborrow, and `raw` must remain valid throughout synchronous descent.
    #[inline(always)]
    unsafe fn publish_and_descend_raw(
        runtime: StructuralMutatorHandle,
        context: *mut c_void,
        raw: TVMFFIAny,
        mode: InplaceMode,
    ) -> TVMFFIAny {
        (*Self::from_handle(runtime)).dispatcher = context;
        <Base as StructuralMutatorBase>::default_mutate_raw(
            runtime.cast(),
            MutationView::new(AnyView::from_raw_ffi_any(raw), mode),
        )
    }

    /// Look up borrowed identity `raw` in the invocation remap, cloning its value.
    ///
    /// # Safety
    /// `runtime` and the object represented by `raw` must remain live for this lookup.
    /// The RefCell borrow ends before the owning result reaches the caller.
    #[inline(always)]
    unsafe fn runtime_var_remap_get(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
    ) -> Result<Option<Any>> {
        (*Self::from_handle(runtime))
            .remap
            .borrow()
            .get(AnyView::from_raw_ffi_any(raw))
    }

    /// Record borrowed identity `raw` and clone `value` into the invocation remap.
    ///
    /// # Safety
    /// `runtime`, `raw`, and `value` must remain live through insertion. The remap
    /// retains each new key once; no RefCell borrow escapes this call.
    #[inline(always)]
    unsafe fn runtime_var_remap_set(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        value: &Any,
    ) -> Result<()> {
        (*Self::from_handle(runtime))
            .remap
            .borrow_mut()
            .set(AnyView::from_raw_ffi_any(raw), value)
    }

    /// Native remap lookup: borrow `var`, return an owning value, None, or Error.
    ///
    /// # Safety
    /// `runtime` must be this live engine and `var` valid for the call despite its
    /// erased ABI lifetime. Rust panics cannot cross this native C boundary.
    unsafe extern "C" fn rust_vtable_var_remap_get(
        runtime: StructuralMutatorHandle,
        var: AnyView<'static>,
    ) -> TVMFFIAny {
        match Self::runtime_var_remap_get(runtime, *var.as_raw_ffi_any()) {
            Ok(Some(value)) => Any::into_raw_ffi_any(value),
            Ok(None) => TVMFFIAny::new(),
            Err(error) => result_into_raw_ffi_any(Err(error)),
        }
    }

    /// Native remap update borrowing `var` and `replacement`; return owning None/Error.
    ///
    /// # Safety
    /// Both views and this Rust `runtime` must remain live for the synchronous call.
    /// Replacement ownership is acquired before insertion; panics cannot cross C.
    unsafe extern "C" fn rust_vtable_var_remap_set(
        runtime: StructuralMutatorHandle,
        var: AnyView<'static>,
        replacement: AnyView<'static>,
    ) -> TVMFFIAny {
        // Finish replacement cleanup before constructing the ABI result.
        let result = raw_ffi_any_to_owned_any(*replacement.as_raw_ffi_any())
            .and_then(|value| Self::runtime_var_remap_set(runtime, *var.as_raw_ffi_any(), &value));
        result_into_raw_ffi_any(result.map(|()| Any::new()))
    }

    /// Run the public mutate/map adapter with owned `root` and borrowed `dispatch`.
    /// `CALLBACK_CONTEXT` distinguishes direct mutation from map error attribution.
    /// Keep the owning result and root live across engine/remap destruction, including
    /// unwind, then replace Unchanged with the original root. Hooks may not retain
    /// the scoped runtime or dispatcher after this call.
    pub(super) fn run_mutator<D, const CALLBACK_CONTEXT: bool>(
        root: Any,
        dispatch: &mut D,
    ) -> Result<Any>
    where
        for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D, Base>),
            Option<Result<UnchangedOr<Any>>>,
        >,
    {
        let mut base = Base::new()?;
        let type_index = Base::type_index();
        // Initialize the native prefix before allocation, so the ordinary base's
        // default region value and this reset can be folded into one write.
        unsafe {
            let prefix = std::ptr::from_mut(&mut base).cast::<StructuralMutatorObj>();
            *(*prefix).def_region_mode.get() = DefRegionKind::None as i32;
        }
        let mut engine = unsafe {
            ObjectArc::<Object>::new_opaque_unchecked(
                Self {
                    base,
                    dispatcher: std::ptr::from_mut(dispatch).cast(),
                    remap: RefCell::new(StructuralVarRemap::default()),
                },
                type_index,
            )
        };
        let runtime = ObjectArc::as_mut_ptr(&mut engine).cast::<StructuralMutatorObj>();
        unsafe { (*runtime).vtable = Self::vtable::<D, CALLBACK_CONTEXT>() };
        let result =
            StructuralMutatorObj::call_mutator(runtime, *root.as_raw_ffi_any(), InplaceMode::Allow);
        // Native hooks release temporary runtime owners before this synchronous run ends.
        drop(engine);
        result.map(|value| if is_unchanged(&value) { root } else { value })
    }
}

// -----------------------------------------------------------------------------
// Prefix-only default traversal and native ABI operations
// -----------------------------------------------------------------------------

// Prefix operations use only the native layout and installed vtable.
impl StructuralMutatorObj {
    /// Consume a native hook's `result`, naming the borrowed `raw` in it when it is an Error.
    ///
    /// # Safety
    ///
    /// `result` owns one valid value and `raw` outlives the call.
    #[inline(always)]
    unsafe fn with_native_mutation_context_raw(result: TVMFFIAny, raw: TVMFFIAny) -> TVMFFIAny {
        if result.type_index != TVMFFITypeIndex::kTVMFFIError as i32 {
            result
        } else {
            core::hint::cold_path();
            Self::with_mutation_context_raw(result, raw)
        }
    }

    /// The cold arm of [`Self::with_native_mutation_context_raw`].
    ///
    /// # Safety
    ///
    /// `result` owns one Error, and `raw` is a live borrowed value.
    #[cold]
    #[inline(never)]
    unsafe extern "C-unwind" fn with_mutation_context_raw(
        result: TVMFFIAny,
        raw: TVMFFIAny,
    ) -> TVMFFIAny {
        result_into_raw_ffi_any(
            raw_ffi_any_into_result(result)
                .map_err(|error| with_mutation_error_context(error, raw, "mutate")),
        )
    }

    /// Mutate the reflected fields of a shallow copy of borrowed `raw`; `Unchanged` when no
    /// field changed.
    fn map_reflected(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
        type_info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    ) -> Result<Any> {
        let inherited_region =
            free_var_child_region(def_region_kind, unsafe { seq_hash_kind(type_info) });
        // Match the C++ reflected-mutation contract: resolve and invoke the
        // shallow-copy hook before inspecting any fields. Besides providing
        // isolated setter storage, this means a missing or failing hook is an
        // error even when no field eventually changes.
        let output = shallow_copy(raw)?;
        let output_raw = *output.as_raw_ffi_any();
        let output_object = unsafe { output_raw.data_union.v_obj.cast::<u8>() };
        if output_object.is_null() {
            return Err(runtime_error(
                "native structural mutate: shallow copy has a null object pointer",
            ));
        }

        let mut field_changed = false;
        let mut failure: Option<Error> = None;
        unsafe {
            for_each_field_info(type_info, &mut |field| {
                if field.flags & FLAG_SEQ_HASH_IGNORE != 0 {
                    return ControlFlow::Continue(());
                }
                match Self::map_reflected_field(
                    runtime,
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

    /// Mutate one reflected `field` of `output_object`, setting `field_changed` on a
    /// replacement.
    ///
    /// # Safety
    ///
    /// `runtime` is live, and `field` describes `output_object`'s storage.
    unsafe fn map_reflected_field(
        runtime: StructuralMutatorHandle,
        output_object: *mut u8,
        field: &TVMFFIFieldInfo,
        inherited_region: DefRegionKind,
        field_changed: &mut bool,
    ) -> Result<()> {
        let Some(getter) = field.getter else {
            return Err(missing_field_getter_error(
                "structural mutate",
                field.name.as_str(),
            ));
        };
        // Read every field from the copy so earlier setters' side effects are
        // visible to later field mappings, exactly as in the C++ fallback.
        let field_offset = usize::try_from(field.offset)
            .map_err(|_| invalid_field_offset_error(field.name.as_str()))?;
        // SAFETY: registered reflection metadata guarantees that the field
        // offset lies within this object's allocation. The checked conversion
        // above also prevents truncation on 32-bit targets.
        let source_address = output_object.add(field_offset).cast::<c_void>();
        // Own the output slot before entering foreign code. A getter may
        // populate an owning result and still report an error.
        let mut child = Any::new();
        if getter(source_address, Any::as_raw_ffi_any_mut(&mut child)) != 0 {
            return Err(with_structural_field_error_context(
                Error::from_raised(),
                "mutate",
                field.name.as_str(),
            ));
        }
        // Reflection getters return owning values. Keep the child alive for
        // the complete recursive call, then let normal Drop release it.
        let child_raw = *child.as_raw_ffi_any();
        let child_region = field_def_region(field, inherited_region);
        let mapped = Self::with_def_region_kind(runtime, child_region, |_| {
            Self::call_mutator(runtime, child_raw, InplaceMode::Disallow)
        })
        .map_err(|error| {
            with_structural_field_error_context(error, "mutate", field.name.as_str())
        })?;
        if is_unchanged(&mapped) || raw_ffi_any_same_as(child_raw, *mapped.as_raw_ffi_any()) {
            return Ok(());
        }

        call_field_setter(field, source_address, mapped.as_raw_ffi_any()).map_err(|error| {
            with_structural_field_error_context(error, "mutate", field.name.as_str())
        })?;
        *field_changed = true;
        Ok(())
    }

    /// Dispatch borrowed `raw` through the runtime's vtable, in place when `mode` allows
    /// it and `raw` is unique.
    #[inline(always)]
    fn call_mutator(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        mode: InplaceMode,
    ) -> Result<Any> {
        let callback = unsafe {
            if mode.for_value(raw) == InplaceMode::Allow {
                (*(*runtime).vtable).maybe_inplace_mutate
            } else {
                (*(*runtime).vtable).mutate
            }
        };
        unsafe {
            let view = AnyView::from_raw_ffi_any(raw);
            raw_ffi_any_into_result(callback(runtime, view))
        }
    }

    /// The registered mutation hook of `raw`: the in-place one when `mode` allows it and
    /// `raw` is unique, otherwise the copying one.
    #[inline(always)]
    fn structural_mutate_hook(raw: TVMFFIAny, mode: InplaceMode) -> Option<TVMFFIAny> {
        if mode.for_value(raw) == InplaceMode::Allow {
            if let Some(attr) = structural_mutate_column::<true>().get_raw_ffi_any(raw.type_index) {
                if attr.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32
                    || attr.type_index == TVMFFITypeIndex::kTVMFFIFunction as i32
                {
                    return Some(attr);
                }
            }
        }
        let attr = structural_mutate_column::<false>().get_raw_ffi_any(raw.type_index)?;
        (attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32).then_some(attr)
    }

    /// Call the registered mutation `hook` (native pointer or packed function) on `value`.
    ///
    /// # Safety
    ///
    /// `runtime` is live with its dispatcher published, `value` and `hook` outlive the call,
    /// the caller's mode allows any in-place mutation the hook performs, and a native hook
    /// has the mutation ABI for this runtime's layout and obeys [`StructuralMutatorBase`]'s
    /// non-escaping contract.
    #[inline(always)]
    pub unsafe fn call_structural_mutate_hook_raw(
        runtime: *mut Self,
        value: TVMFFIAny,
        hook: TVMFFIAny,
    ) -> TVMFFIAny {
        unsafe {
            if hook.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
                let callback: FStructuralMutate = std::mem::transmute(hook.data_union.v_ptr);
                Self::with_native_mutation_context_raw(
                    callback(runtime, AnyView::from_raw_ffi_any(value)),
                    value,
                )
            } else {
                Self::call_structural_mutate_hook_slow_raw(runtime, value, hook)
            }
        }
    }

    /// The packed-function arm of [`Self::call_structural_mutate_hook_raw`], outlined as cold.
    ///
    /// # Safety
    ///
    /// As [`Self::call_structural_mutate_hook_raw`].
    #[cold]
    #[inline(never)]
    unsafe extern "C-unwind" fn call_structural_mutate_hook_slow_raw(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        attr: TVMFFIAny,
    ) -> TVMFFIAny {
        let result = if attr.type_index == TVMFFITypeIndex::kTVMFFIFunction as i32 {
            Function::try_from(AnyView::from_raw_ffi_any(attr)).and_then(|function| {
                function.call_packed(&[
                    Self::borrowed_mutator_view(runtime),
                    AnyView::from_raw_ffi_any(raw),
                ])
            })
        } else {
            Err(invalid_mutation_hook_error())
        };
        // Preserve successful Error-valued returns until the public conversion.
        result_into_raw_ffi_any(
            result.map_err(|error| with_mutation_error_context(error, raw, "mutate")),
        )
    }

    /// Borrow the live `runtime` as an object argument, without a reference count.
    ///
    /// # Safety
    ///
    /// `runtime` outlives the returned view.
    #[inline(always)]
    unsafe fn borrowed_mutator_view<'a>(runtime: StructuralMutatorHandle) -> AnyView<'a> {
        let object = runtime.cast::<TVMFFIObject>();
        let mut raw = TVMFFIAny::new();
        raw.type_index = (*object).type_index;
        raw.small_str_len = 0;
        raw.data_union.v_obj = object;
        AnyView::from_raw_ffi_any(raw)
    }

    /// Run `callback` under requested `kind` on live borrowed `mutator`.
    /// Pattern precedence is retained; pass the effective kind to the callback and
    /// restore prior state on normal return or Rust unwind. Internal callers keep
    /// the runtime live and do not retain a reference to its mutable region field.
    #[inline(always)]
    fn with_def_region_kind<T>(
        mutator: StructuralMutatorHandle,
        kind: DefRegionKind,
        callback: impl FnOnce(DefRegionKind) -> T,
    ) -> T {
        unsafe {
            let previous = *(*mutator).def_region_mode.get();
            if previous == kind as i32 || previous == DefRegionKind::Pattern as i32 {
                // Inherited descent does not change state; Pattern keeps precedence.
                return callback(std::mem::transmute::<i32, DefRegionKind>(previous));
            }
            *(*mutator).def_region_mode.get() = kind as i32;
            struct Restore {
                mutator: StructuralMutatorHandle,
                previous: i32,
            }
            impl Drop for Restore {
                #[inline(always)]
                fn drop(&mut self) {
                    unsafe { *(*self.mutator).def_region_mode.get() = self.previous };
                }
            }
            let _restore = Restore { mutator, previous };
            // One call site keeps the continuation visible to the inliner.
            callback(kind)
        }
    }

    /// Mutate borrowed `raw` by reflection, caching free variables and DAG nodes
    /// through the vtable remap.
    ///
    /// # Safety
    ///
    /// `runtime` is a live mutator prefix with its vtable, and `raw` outlives the call.
    /// Outlined (frequent, not cold) to keep reflection scratch out of recursive frames.
    #[inline(never)]
    unsafe extern "C-unwind" fn mutate_without_hook_raw(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
    ) -> TVMFFIAny {
        // Read the region here rather than on the native-hook path, which does not use it.
        let def_region_kind =
            std::mem::transmute::<i32, DefRegionKind>(*(*runtime).def_region_mode.get());
        result_into_raw_ffi_any(
            Self::mutate_without_hook(runtime, raw, def_region_kind)
                .map_err(|error| with_mutation_error_context(error, raw, "mutate")),
        )
    }

    /// Borrow the native remap interface without assuming the complete engine layout.
    /// The vtable owns its returned replacement, including Unchanged markers.
    #[inline(always)]
    unsafe fn var_remap_get(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
    ) -> Result<Option<Any>> {
        let callback = (*(*runtime).vtable).var_remap_get;
        let value = raw_ffi_any_into_result(callback(runtime, AnyView::from_raw_ffi_any(raw)))?;
        Ok((value.type_index() != TVMFFITypeIndex::kTVMFFINone as i32).then_some(value))
    }

    /// Borrow both inputs through the native remap interface; it retains the key
    /// and replacement. Import the owning None/Error result before returning.
    #[inline(always)]
    unsafe fn var_remap_set(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        value: &Any,
    ) -> Result<()> {
        let callback = (*(*runtime).vtable).var_remap_set;
        raw_ffi_any_into_result(callback(
            runtime,
            AnyView::from_raw_ffi_any(raw),
            AnyView::from_raw_ffi_any(*value.as_raw_ffi_any()),
        ))
        .map(|_| ())
    }

    /// The reflective mutation of [`Self::mutate_without_hook_raw`], on owned results.
    #[inline(always)]
    fn mutate_without_hook(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return raw_ffi_any_to_owned_any(raw);
        }

        let type_info = checked_type_info(raw.type_index)?;
        let kind = unsafe { seq_hash_kind(type_info) };
        let is_free_var = kind == TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar as i32;
        let is_dag_node = kind == TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode as i32;
        if is_free_var || is_dag_node {
            // The ABI uses None for a miss; an Unchanged marker is a cached result.
            if let Some(mutated) = unsafe { Self::var_remap_get(runtime, raw)? } {
                return Ok(mutated);
            }
        }
        // A variable use with no binding retains its identity without visiting fields.
        if is_free_var && def_region_kind == DefRegionKind::None {
            return Ok(Unchanged.into());
        }

        let result = Self::map_reflected(runtime, raw, def_region_kind, type_info)?;
        if is_dag_node
            || (is_free_var
                && (def_region_kind == DefRegionKind::Pattern || !is_unchanged(&result)))
        {
            // Keep markers for DAG nodes and pattern definitions. An unchanged simple
            // definition needs no binding: subsequent uses retain the original on a miss.
            unsafe { Self::var_remap_set(runtime, raw, &result)? };
        }
        Ok(result)
    }
}

// -----------------------------------------------------------------------------
// Private utility functions
// -----------------------------------------------------------------------------

fn shallow_copy(raw: TVMFFIAny) -> Result<Any> {
    let Some(attr) = SHALLOW_COPY_COLUMN.get_raw_ffi_any(raw.type_index) else {
        return Err(missing_shallow_copy_error(
            raw.type_index,
            SHALLOW_COPY_ATTR,
        ));
    };
    if attr.type_index != TVMFFITypeIndex::kTVMFFIFunction as i32 {
        return Err(invalid_shallow_copy_hook_error(SHALLOW_COPY_ATTR));
    }
    let function = unsafe { attr.data_union.v_obj };
    if function.is_null() {
        return Err(runtime_error("shallow copy function pointer is null"));
    }
    // The registry owns this function throughout the call; borrowing it avoids
    // an atomic retain/release for every reflected node.
    let mut result = Any::new();
    let status = unsafe {
        TVMFFIFunctionCall(
            function.cast(),
            &raw,
            1,
            Any::as_raw_ffi_any_mut(&mut result),
        )
    };
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
        return Err(invalid_shallow_copy_result_error());
    }
    Ok(result)
}

fn call_field_setter(
    field: &TVMFFIFieldInfo,
    field_address: *mut c_void,
    value: &TVMFFIAny,
) -> Result<()> {
    if field.setter.is_null() {
        // SAFETY: registered field names are UTF-8 text that lives as long as the process.
        return Err(missing_field_setter_error(unsafe { field.name.as_str() }));
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
                Any::as_raw_ffi_any_mut(&mut result),
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
        core::hint::cold_path();
        return Err(var_remap_key_error());
    }
    let pointer = unsafe { raw.data_union.v_obj };
    NonNull::new(pointer).ok_or_else(|| {
        core::hint::cold_path();
        runtime_error("native structural mutate: identity object has a null pointer")
    })
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

#[inline(always)]
fn object_is_unique(raw: TVMFFIAny) -> bool {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return false;
    }
    let pointer = unsafe { raw.data_union.v_obj };
    !pointer.is_null() && unsafe { object::unsafe_::is_unique(pointer) }
}

/// Acquire an owning value from a valid borrowed carrier.
///
/// Inline values are copied and objects are retained once, including Error objects
/// as ordinary values. String/byte views are normalized by the runtime; an rvalue
/// slot transfers its reference. Internal callers keep borrowed resources live and
/// uniquely own each writable rvalue slot, converting it at most once. A null
/// object pointer or failed runtime conversion returns an error; successful Error
/// objects remain ordinary values rather than entering the error channel.
fn raw_ffi_any_to_owned_any(raw: TVMFFIAny) -> Result<Any> {
    if is_plain_inline(raw.type_index) {
        return Ok(unsafe { Any::from_raw_ffi_any(raw) });
    }
    if raw.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        let object = unsafe { raw.data_union.v_obj };
        if object.is_null() {
            return Err(runtime_error(
                "native structural mutate: object-backed value has a null pointer",
            ));
        }
        unsafe { object::unsafe_::inc_ref(object) };
        return Ok(unsafe { Any::from_raw_ffi_any(raw) });
    }

    // Raw string/bytes views and ObjectRValueRef require normalization (or a
    // move) rather than a bitwise copy; keep the generic C ABI conversion for
    // those uncommon representations.
    let mut owned = Any::new();
    let return_code = unsafe { TVMFFIAnyViewToOwnedAny(&raw, Any::as_raw_ffi_any_mut(&mut owned)) };
    if return_code == 0 {
        Ok(owned)
    } else {
        Err(Error::from_raised())
    }
}

// Cache registry columns, not their optional per-type cells. Absent cells retain
// the normal hook/reflection fallback.
static STRUCTURAL_MUTATE_COLUMN: ColdInitLazyLock<TypeAttrColumn> = ColdInitLazyLock::new(|| {
    TypeAttrColumn::lookup(STRUCTURAL_MUTATE_ATTR).expect("missing __s_mutate__ column")
});

static STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN: ColdInitLazyLock<TypeAttrColumn> =
    ColdInitLazyLock::new(|| {
        TypeAttrColumn::lookup(STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR)
            .expect("missing __s_maybe_inplace_mutate__ column")
    });

/// Return the shared column for normal or permitted in-place mutation hooks.
/// Cache the column, not its per-type cells, so absent hooks still use fallback.
#[inline(always)]
fn structural_mutate_column<const INPLACE: bool>() -> TypeAttrColumn {
    if INPLACE {
        *STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN
    } else {
        *STRUCTURAL_MUTATE_COLUMN
    }
}

static SHALLOW_COPY_COLUMN: ColdInitLazyLock<TypeAttrColumn> = ColdInitLazyLock::new(|| {
    TypeAttrColumn::lookup(SHALLOW_COPY_ATTR).expect("missing __ffi_shallow_copy__ column")
});

// -----------------------------------------------------------------------------
// Outlined errors
// -----------------------------------------------------------------------------

#[cold]
#[inline(never)]
fn missing_shallow_copy_error(type_index: i32, attribute: &str) -> Error {
    Error::new(
        TYPE_ERROR,
        &format!(
            "type `{}` cannot use reflected structural mutation because it does not define `{attribute}`",
            crate::object::type_key_of_index(type_index)
        ),
        "",
    )
}

#[cold]
#[inline(never)]
fn invalid_shallow_copy_hook_error(attribute: &str) -> Error {
    Error::new(
        TYPE_ERROR,
        &format!("{attribute} must be an ffi.Function"),
        "",
    )
}

#[cold]
#[inline(never)]
fn invalid_shallow_copy_result_error() -> Error {
    Error::new(
        TYPE_ERROR,
        "shallow copy callback must return a distinct object with the same type as its input",
        "",
    )
}

#[cold]
#[inline(never)]
fn missing_field_setter_error(name: &str) -> Error {
    Error::new(
        TYPE_ERROR,
        &format!("cannot structurally mutate field `{name}` because it does not define a setter"),
        "",
    )
}

#[cold]
#[inline(never)]
fn invalid_field_offset_error(name: &str) -> Error {
    runtime_error(&format!(
        "native structural mutate: reflected field `{name}` has an invalid offset"
    ))
}

#[cold]
#[inline(never)]
fn invalid_mutation_hook_error() -> Error {
    Error::new(
        TYPE_ERROR,
        "__s_mutate__ must be an opaque function pointer or ffi.Function",
        "",
    )
}

#[cold]
#[inline(never)]
fn var_remap_key_error() -> Error {
    Error::new(
        TYPE_ERROR,
        "variable-remap keys must be object-backed values",
        "",
    )
}

#[cold]
#[inline(never)]
fn unregistered_type_error(type_index: i32) -> Error {
    runtime_error(&format!(
        "native structural mutate: unregistered type index {type_index}"
    ))
}

#[cold]
pub(super) fn with_mutation_error_context(error: Error, raw: TVMFFIAny, operation: &str) -> Error {
    if raw.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        error
    } else {
        with_structural_error_context(
            update_visit_error_context(error, raw),
            operation,
            &format!(
                "object `{}`",
                crate::object::type_key_of_index(raw.type_index)
            ),
        )
    }
}
