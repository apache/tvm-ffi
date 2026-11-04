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
use std::collections::{hash_map::Entry, HashMap};
use std::ffi::c_void;
use std::marker::PhantomData;

use super::marker::{Borrowed, Owned};
use std::ops::ControlFlow;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::LazyLock;

pub use super::structural_map::*;

use super::dispatch::Dispatcher;
use crate::any::{Any, AnyView};
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::{self, Object, ObjectArc, ObjectCore};
use crate::reflection::TypeAttrColumn;
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashIgnore, kTVMFFIFieldFlagBitSetterIsFunctionObj,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIAnyViewToOwnedAny, TVMFFIByteArray, TVMFFIFieldInfo,
    TVMFFIFieldSetter, TVMFFIFunctionCall, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeIndex,
    TVMFFITypeKeyToIndex,
};
use crate::tvm_ffi_sys::{TVMFFIObjectHandle, TVMFFISEqHashKind};

use super::structural_common::{
    borrowed_view, raw_of, same_shallow, structural_error_from_raw,
    try_to_owned_without_normalization, type_key_of, with_structural_error_context,
    with_visit_error_context,
};
use super::structural_visit::{
    field_def_region, for_each_field_info, free_var_child_region, DefRegionKind,
};
use super::unchanged::is_unchanged;
pub use super::unchanged::{Unchanged, UnchangedOr};

const STRUCTURAL_MUTATE_ATTR: &str = "__s_mutate__";
const STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR: &str = "__s_maybe_inplace_mutate__";
const SHALLOW_COPY_ATTR: &str = "__ffi_shallow_copy__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SETTER_IS_FUNCTION: i64 = kTVMFFIFieldFlagBitSetterIsFunctionObj as i64;

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

/// Exact C++ StructuralMutatorObj ABI prefix, with no Rust traversal state.
#[repr(C)]
pub struct StructuralMutatorObj {
    base: Object,
    vtable: *const StructuralMutatorVTable,
    def_region_mode: i32,
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

#[repr(C)]
pub(super) struct StructuralMutateEngine {
    base: StructuralMutatorObj,
    dispatcher: *mut c_void,
    remap: RefCell<StructuralVarRemap>,
}

impl Default for StructuralMutatorObj {
    #[inline(always)]
    fn default() -> Self {
        Self {
            base: Object::new(),
            vtable: std::ptr::null(),
            def_region_mode: DefRegionKind::None as i32,
        }
    }
}

/// Identity substitutions for structural mutation within one invocation.
///
/// Each pointer key owns one strong reference to its original object, preventing
/// address reuse until removal. Values are owning replacements, including
/// [`Unchanged`] markers. Replacing a value does not acquire another key reference;
/// lookup clones the replacement, and clearing or dropping releases every owner.
#[derive(Default)]
pub struct StructuralVarRemap {
    entries: HashMap<NonNull<TVMFFIObject>, Any>,
}

impl StructuralVarRemap {
    /// Look up an identity replacement previously stored for `var`.
    pub fn get(&self, var: AnyView<'_>) -> Result<Option<Any>> {
        let key = object_identity_key(raw_of(var))?;
        Ok(self.entries.get(&key).cloned())
    }

    /// Store a descent result or an [`Unchanged`] marker for `var`.
    ///
    /// A new key acquires one strong reference; replacing its value preserves
    /// that reference. The replacement is cloned before modifying the table.
    pub fn set(&mut self, var: AnyView<'_>, mutated_value: &Any) -> Result<()> {
        let key = object_identity_key(raw_of(var))?;
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

    /// Remove every substitution, releasing each key and replacement owner.
    pub fn clear(&mut self) {
        let empty = HashMap::with_hasher(self.entries.hasher().clone());
        let mut entries = std::mem::replace(&mut self.entries, empty);
        // Release all key references before dropping values. Object deleters
        // use the non-unwinding C ABI, so this pass cannot leave keys unreleased.
        // Subsequent cleanup only drops pointer bits, never dereferences them.
        for key in entries.keys() {
            unsafe { object::unsafe_::dec_ref(key.as_ptr()) };
        }
        entries.clear();
        // Reuse capacity without replacing any entries created during cleanup.
        if self.entries.is_empty() {
            self.entries = entries;
        }
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

impl<'source, 'value, T: crate::type_traits::ContainerElement>
    super::dispatch::DispatchRecovery<crate::match_any_internal::CapabilitySource<'source>>
    for crate::MutationView<'value, T>
{
    type Recovery = crate::MutationView<'source>;
}

// -----------------------------------------------------------------------------
// Public mutation API
// -----------------------------------------------------------------------------

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
    #[inline(always)]
    fn into_mutate_result(self) -> Result<Any> {
        Ok(self.into())
    }
}

impl<T: Into<Any>> IntoMutateResult for Result<T> {
    #[inline(always)]
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
    // Permission is only an upper bound; ownership must still be unique here.
    #[inline(always)]
    fn for_value(self, raw: TVMFFIAny) -> Self {
        if self == Self::Allow && object_is_unique(raw) {
            Self::Allow
        } else {
            Self::Disallow
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
/// #[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<Any>)]
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
/// #[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<Any>)]
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
    pub(super) fn new(value: AnyView<'a>, mode: InplaceMode) -> Self {
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
        let mode = self.mode.for_value(self.value);
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
/// Pass the handler's `self` to recursive methods with the same dispatcher identity.
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
        unsafe { std::mem::transmute((*self.runtime).def_region_mode) }
    }

    /// Scope definition-region state; an enclosing Pattern keeps precedence.
    #[inline(always)]
    pub fn with_def_region_kind<R>(
        &mut self,
        kind: DefRegionKind,
        callback: impl FnOnce(&mut StructuralMutatorView<'_, D>) -> R,
    ) -> R {
        StructuralMutateEngine::with_mutator_def_region(self.runtime, kind, |_| {
            callback(&mut self.reborrow())
        })
    }

    /// Re-enter dispatch with the capability's carried permission.
    #[inline(always)]
    pub fn mutate<'input, T>(
        &mut self,
        dispatch: &mut D,
        value: impl Into<MutationView<'input, T>>,
    ) -> Result<UnchangedOr<Any>> {
        let value = value.into();
        unsafe {
            let context = StructuralMutateEngine::checked_dispatcher(self.runtime, dispatch);
            (*StructuralMutateEngine::from_handle(self.runtime)).dispatcher = context;
            StructuralMutateEngine::call_mutator(self.runtime, value.value, value.mode)
        }
        .and_then(UnchangedOr::from_carrier)
    }

    /// Descend through the native hook or reflected fields without redispatching the root.
    #[inline(always)]
    pub fn default_mutate<'input, T>(
        &mut self,
        dispatch: &mut D,
        value: impl Into<MutationView<'input, T>>,
    ) -> Result<UnchangedOr<Any>> {
        let value = value.into();
        unsafe {
            let context = StructuralMutateEngine::checked_dispatcher(self.runtime, dispatch);
            result_from_raw(StructuralMutateEngine::default_mutate_raw(
                self.runtime,
                context,
                value.value,
                value.mode,
            ))
        }
        .and_then(UnchangedOr::from_carrier)
    }

    #[inline(always)]
    pub fn var_remap_get(&mut self, dispatch: &mut D, var: AnyView<'_>) -> Result<Option<Any>> {
        unsafe {
            StructuralMutateEngine::checked_dispatcher(self.runtime, dispatch);
            StructuralMutateEngine::runtime_var_remap_get(self.runtime, raw_of(var))
        }
    }

    #[inline(always)]
    pub fn var_remap_set(&mut self, dispatch: &mut D, var: AnyView<'_>, value: &Any) -> Result<()> {
        unsafe {
            StructuralMutateEngine::checked_dispatcher(self.runtime, dispatch);
            StructuralMutateEngine::runtime_var_remap_set(self.runtime, raw_of(var), value)
        }
    }
}

/// Conversion into a dispatcher-driven mutation. Closures belong to structural_map.
pub trait IntoMutator<Marker> {
    #[doc(hidden)]
    fn mutate_root(self, root: Any) -> Result<Any>;
}

impl<D> IntoMutator<Owned> for D
where
    for<'s, 'v, 'c, 'r> &'s mut D:
        Dispatcher<(MutationView<'v>, &'c mut StructuralMutatorView<'r, D>), Option<Result<Any>>>,
{
    #[inline(always)]
    fn mutate_root(mut self, root: Any) -> Result<Any> {
        StructuralMutateEngine::run_mutator::<D, true>(root, &mut self)
    }
}

impl<D> IntoMutator<Borrowed> for &mut D
where
    for<'s, 'v, 'c, 'r> &'s mut D:
        Dispatcher<(MutationView<'v>, &'c mut StructuralMutatorView<'r, D>), Option<Result<Any>>>,
{
    #[inline(always)]
    fn mutate_root(self, root: Any) -> Result<Any> {
        StructuralMutateEngine::run_mutator::<D, true>(root, self)
    }
}

// -----------------------------------------------------------------------------
// Private utility functions
// -----------------------------------------------------------------------------

// Keep panic construction out of the successful dispatcher identity check.
#[cold]
#[inline(never)]
fn wrong_mutation_dispatcher() -> ! {
    panic!("mutation context used with a different dispatcher instance")
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
    if raw.type_index != TVMFFITypeIndex::kTVMFFIError as i32 {
        return Ok(Any::from_raw_ffi_any(raw));
    }
    core::hint::cold_path();
    Err(structural_error_from_raw(
        raw.type_index,
        raw.small_str_len,
        raw.data_union,
    ))
}

fn shallow_copy(raw: TVMFFIAny) -> Result<Any> {
    let Some(attr) = SHALLOW_COPY_COLUMN.get_raw(raw.type_index) else {
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
        core::hint::cold_path();
        return Err(var_remap_key_error());
    }
    let pointer = unsafe { raw.data_union.v_obj };
    NonNull::new(pointer).ok_or_else(|| {
        core::hint::cold_path();
        runtime_error("native structural map: identity object has a null pointer")
    })
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
pub(super) fn with_mutation_error_context(error: Error, raw: TVMFFIAny) -> Error {
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

// Error allocation is exceptional and does not belong in recursive hot frames.
#[cold]
#[inline(never)]
fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

// Cache registry columns, not their optional per-type cells. Absent cells retain
// the normal hook/reflection fallback.
static STRUCTURAL_MUTATE_COLUMN: LazyLock<TypeAttrColumn> = LazyLock::new(|| {
    TypeAttrColumn::new(STRUCTURAL_MUTATE_ATTR).expect("missing __s_mutate__ column")
});
static STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN: LazyLock<TypeAttrColumn> = LazyLock::new(|| {
    TypeAttrColumn::new(STRUCTURAL_MAYBE_INPLACE_MUTATE_ATTR)
        .expect("missing __s_maybe_inplace_mutate__ column")
});
/// Return the shared column for normal or permitted in-place mutation hooks.
/// Cache the column, not its per-type cells, so absent hooks still use fallback.
/// The ready check stays inline; initialization uses the same LazyLock through
/// an outlined tail to keep Once descriptors out of recursive caller frames.
#[inline(always)]
fn structural_mutate_column<const INPLACE: bool>() -> TypeAttrColumn {
    let column = if INPLACE {
        &STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN
    } else {
        &STRUCTURAL_MUTATE_COLUMN
    };
    if let Some(column) = LazyLock::get(column) {
        *column
    } else {
        initialize_structural_mutate_column::<INPLACE>()
    }
}

// LazyLock owns blocking/poison semantics; outline its first-use Once scratch.
#[cold]
#[inline(never)]
fn initialize_structural_mutate_column<const INPLACE: bool>() -> TypeAttrColumn {
    if INPLACE {
        *LazyLock::force(&STRUCTURAL_MAYBE_INPLACE_MUTATE_COLUMN)
    } else {
        *LazyLock::force(&STRUCTURAL_MUTATE_COLUMN)
    }
}

static SHALLOW_COPY_COLUMN: LazyLock<TypeAttrColumn> = LazyLock::new(|| {
    TypeAttrColumn::new(SHALLOW_COPY_ATTR).expect("missing __ffi_shallow_copy__ column")
});

// -----------------------------------------------------------------------------
// StructuralMutateEngine implementation
// -----------------------------------------------------------------------------

impl StructuralMutateEngine {
    /// Recover the engine prefix for internal runtime projections.
    ///
    /// # Safety
    /// `runtime` must be this live Rust engine, not an arbitrary native mutator.
    /// The returned pointer borrows it; no whole-engine reference may span recursion.
    // Unsafe internals preserve a safe API: the matching runtime and dispatcher
    // remain live for this synchronous call, and each recursive entry publishes
    // its actual exclusive reborrow. No whole-engine reference spans recursion.
    // Callbacks use Result for failures and must not unwind across the C ABI.
    // The live handle starts this exact Rust engine; only raw projections cross recursion.
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
            wrong_mutation_dispatcher();
        }
        context
    }

    /// Build native callbacks for dispatcher `D`. `CALLBACK_CONTEXT` selects
    /// whether direct callback failures receive input context; map adapters add their own.
    #[inline(always)]
    fn vtable<D, const CALLBACK_CONTEXT: bool>() -> &'static StructuralMutatorVTable
    where
        for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D>),
            Option<Result<Any>>,
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
    /// `INPLACE` grants permission only; uniqueness is checked before dispatch.
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
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D>),
            Option<Result<Any>>,
        >,
    {
        let context = (*Self::from_handle(runtime)).dispatcher;
        let raw = raw_of(value);
        let mode = if INPLACE {
            InplaceMode::Allow
        } else {
            InplaceMode::Disallow
        };
        let mode = mode.for_value(raw);
        let dispatch = &mut *context.cast::<D>();
        let mut mutator = StructuralMutatorView {
            runtime,
            _scope: PhantomData,
            _dispatcher: PhantomData,
            _not_send_sync: PhantomData,
        };
        let result = match Dispatcher::dispatch_tuple(
            &mut *dispatch,
            (MutationView::new(value, mode), &mut mutator),
        ) {
            // Map adapters attribute errors to their possibly replaced value;
            // direct mutation attributes callback failures to this input.
            Some(result) => result_into_raw(if CALLBACK_CONTEXT {
                result.map_err(|error| with_mutation_error_context(error, raw))
            } else {
                result
            }),
            None => {
                Self::default_mutate_raw(runtime, std::ptr::from_mut(dispatch).cast(), raw, mode)
            }
        };
        // All callback reborrows have ended before restoring the caller's provenance.
        (*Self::from_handle(runtime)).dispatcher = context;
        result
    }

    /// Publish `context`, then descend through hooks or reflection for borrowed `raw`.
    /// `mode` grants reuse permission, not proof of uniqueness. Returns one owning token.
    ///
    /// # Safety
    /// `runtime` must be live; `context` must be its current exclusive dispatcher
    /// reborrow, and `raw` must remain valid throughout synchronous descent.
    #[inline(always)]
    unsafe fn default_mutate_raw(
        runtime: StructuralMutatorHandle,
        context: *mut c_void,
        raw: TVMFFIAny,
        mode: InplaceMode,
    ) -> TVMFFIAny {
        (*Self::from_handle(runtime)).dispatcher = context;
        Self::mutate_children_raw(runtime, raw, mode)
    }

    /// Consume the native hook owning `result`, annotating errors with borrowed `raw`.
    /// Nonerrors transfer unchanged; packed successful Error values do not use this path.
    ///
    /// # Safety
    /// `result` must own one valid ABI value and `raw` must remain live during annotation.
    #[inline(always)]
    unsafe fn with_native_mutation_context_raw(result: TVMFFIAny, raw: TVMFFIAny) -> TVMFFIAny {
        if result.type_index != TVMFFITypeIndex::kTVMFFIError as i32 {
            result
        } else {
            core::hint::cold_path();
            Self::with_mutation_context_raw(
                result.type_index,
                result.small_str_len,
                result.data_union,
                raw.type_index,
                raw.small_str_len,
                raw.data_union,
            )
        }
    }

    /// Reconstruct an owning error from `result_*` and borrowed input from the other
    /// scalar fields, then transfer one context-annotated owning raw result.
    /// This private C-unwind helper uses AArch64 register returns and permits Rust
    /// cleanup on unwind; it changes no public or native callback ABI.
    ///
    /// # Safety
    /// The result fields must describe one owned Error; the input fields must describe
    /// a live borrowed value. No caller may release the transferred error token.
    // Outline error annotation scratch; the AArch64 C ABI returns the raw
    // carrier in registers while C-unwind preserves Rust cleanup on panic.
    #[cold]
    #[inline(never)]
    unsafe extern "C-unwind" fn with_mutation_context_raw(
        result_type_index: i32,
        result_small_str_len: u32,
        result_data: TVMFFIAnyDataUnion,
        type_index: i32,
        small_str_len: u32,
        data_union: TVMFFIAnyDataUnion,
    ) -> TVMFFIAny {
        let result = TVMFFIAny {
            type_index: result_type_index,
            small_str_len: result_small_str_len,
            data_union: result_data,
        };
        let raw = TVMFFIAny {
            type_index,
            small_str_len,
            data_union,
        };
        result_into_raw(
            result_from_raw(result).map_err(|error| with_mutation_error_context(error, raw)),
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
            .get(borrowed_view(&raw))
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
            .set(borrowed_view(&raw), value)
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
        match Self::runtime_var_remap_get(runtime, raw_of(var)) {
            Ok(Some(value)) => Any::into_raw_ffi_any(value),
            Ok(None) => TVMFFIAny::new(),
            Err(error) => result_into_raw(Err(error)),
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
        result_into_raw(owned_from_raw(raw_of(replacement)).and_then(|value| {
            Self::runtime_var_remap_set(runtime, raw_of(var), &value).map(|()| Any::new())
        }))
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
            (MutationView<'v>, &'c mut StructuralMutatorView<'r, D>),
            Option<Result<Any>>,
        >,
    {
        let type_index = StructuralMutatorObj::type_index();
        let mut engine = unsafe {
            ObjectArc::<Object>::new_opaque_unchecked(
                Self {
                    base: StructuralMutatorObj::default(),
                    dispatcher: std::ptr::from_mut(dispatch).cast(),
                    remap: RefCell::new(StructuralVarRemap::default()),
                },
                type_index,
            )
        };
        let handle = unsafe { ObjectArc::as_raw_mut(&mut engine).cast::<StructuralMutatorObj>() };
        unsafe {
            (*handle).vtable = Self::vtable::<D, CALLBACK_CONTEXT>();
        }
        let result = Self::call_mutator(handle, *root.as_raw_ffi_any(), InplaceMode::Allow);
        // Native hooks release temporary runtime owners before this synchronous run ends.
        drop(engine);
        result.map(|value| if is_unchanged(&value) { root } else { value })
    }

    /// Map registered fields of borrowed `raw` using live `runtime` and `type_info`.
    /// `def_region_kind` is the fallback-entry snapshot. Own the shallow copy and
    /// getter results through recursion/setters; return a replacement or Unchanged.
    /// Internal callers provide valid registered metadata and a live input owner.
    fn map_reflected(
        runtime: StructuralMutatorHandle,
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

    /// Map one field of the owned shallow copy and set `field_changed` on replacement.
    /// `inherited_region` supplies the parent definition policy. Getter/result owners
    /// survive recursion and setters; errors retain field context.
    ///
    /// # Safety
    /// `runtime` must be live; `output_object` and registered `field` metadata must
    /// describe a valid live copy, including its getter/setter offsets and ABIs.
    unsafe fn map_reflected_field(
        runtime: StructuralMutatorHandle,
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
        let mapped = Self::with_mutator_def_region(runtime, child_region, |_| {
            Self::call_mutator(runtime, child_raw, InplaceMode::Disallow)
        })
        .map_err(|error| with_error_context(error, &format!("field `{}`", field.name.as_str())))?;
        if is_unchanged(&mapped) || same_shallow(child_raw, *mapped.as_raw_ffi_any()) {
            return Ok(());
        }

        call_field_setter(field, source_address, mapped.as_raw_ffi_any()).map_err(|error| {
            with_error_context(error, &format!("field `{}`", field.name.as_str()))
        })?;
        *field_changed = true;
        Ok(())
    }

    /// Invoke a live mutator vtable on borrowed `raw` and import its owning result.
    /// Internal callers keep both live; `mode` permits reuse only after a fresh
    /// uniqueness check. Native callbacks cannot unwind through their C ABI.
    #[inline(always)]
    fn call_mutator(
        mutator: StructuralMutatorHandle,
        raw: TVMFFIAny,
        mode: InplaceMode,
    ) -> Result<Any> {
        let use_inplace = mode == InplaceMode::Allow && object_is_unique(raw);
        let callback = unsafe {
            if use_inplace {
                (*(*mutator).vtable).maybe_inplace_mutate
            } else {
                (*(*mutator).vtable).mutate
            }
        };
        unsafe {
            let view = AnyView::from_raw_ffi_any(raw);
            result_from_raw(callback(mutator, view))
        }
    }

    /// Borrow the registered hook cell for live `raw`. Recheck uniqueness for
    /// `mode` before selecting an in-place hook, otherwise use the normal column.
    /// The registry owns the returned attribute; no reference count is acquired.
    #[inline(always)]
    fn structural_mutate_hook(raw: TVMFFIAny, mode: InplaceMode) -> Option<TVMFFIAny> {
        let use_inplace = mode == InplaceMode::Allow && object_is_unique(raw);
        if use_inplace {
            if let Some(attr) = structural_mutate_column::<true>().get_raw(raw.type_index) {
                if attr.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32
                    || attr.type_index == TVMFFITypeIndex::kTVMFFIFunction as i32
                {
                    return Some(attr);
                }
            }
        }
        let attr = structural_mutate_column::<false>().get_raw(raw.type_index)?;
        (attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32).then_some(attr)
    }

    /// Call borrowed registered `attr` for live `mutator` and borrowed `raw`.
    /// Internal callers uphold native hook registration/lifetime contracts. Transfer
    /// one owning result; only native Error results receive raw-tag context here.
    #[inline(always)]
    fn call_structural_mutate_hook_raw(
        mutator: StructuralMutatorHandle,
        raw: TVMFFIAny,
        attr: TVMFFIAny,
    ) -> TVMFFIAny {
        unsafe {
            if attr.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
                let hook: FStructuralMutate = std::mem::transmute(attr.data_union.v_ptr);
                Self::with_native_mutation_context_raw(
                    hook(mutator, AnyView::from_raw_ffi_any(raw)),
                    raw,
                )
            } else {
                Self::call_structural_mutate_hook_slow_raw(
                    mutator,
                    raw.type_index,
                    raw.small_str_len,
                    raw.data_union,
                    attr.type_index,
                    attr.small_str_len,
                    attr.data_union,
                )
            }
        }
    }

    /// Reconstruct borrowed input and attribute from their scalar fields for the
    /// packed/malformed-hook fallback. `mutator` is borrowed for the call. Return
    /// one owning token after Function/argument cleanup; annotate actual Err only.
    /// The private C-unwind ABI preserves Rust cleanup and AArch64 register returns
    /// without changing public or native callback ABIs.
    ///
    /// # Safety
    /// `mutator`, input fields, and registered attribute fields must remain valid
    /// through this synchronous call; none of these inputs transfers ownership.
    // Outline packed-call scratch; the AArch64 C ABI keeps raw results in
    // registers while C-unwind preserves Rust cleanup inside this private tail.
    #[cold]
    #[inline(never)]
    unsafe extern "C-unwind" fn call_structural_mutate_hook_slow_raw(
        mutator: StructuralMutatorHandle,
        type_index: i32,
        small_str_len: u32,
        data_union: TVMFFIAnyDataUnion,
        attr_type_index: i32,
        attr_small_str_len: u32,
        attr_data_union: TVMFFIAnyDataUnion,
    ) -> TVMFFIAny {
        let raw = TVMFFIAny {
            type_index,
            small_str_len,
            data_union,
        };
        let attr = TVMFFIAny {
            type_index: attr_type_index,
            small_str_len: attr_small_str_len,
            data_union: attr_data_union,
        };
        let result = if attr.type_index == TVMFFITypeIndex::kTVMFFIFunction as i32 {
            Function::try_from(AnyView::from_raw_ffi_any(attr)).and_then(|function| {
                function.call_packed(&[
                    Self::borrowed_mutator_view(mutator),
                    AnyView::from_raw_ffi_any(raw),
                ])
            })
        } else {
            Err(Error::new(
                TYPE_ERROR,
                "__s_mutate__ must be an opaque function pointer or ffi.Function",
                "",
            ))
        };
        // Preserve successful Error-valued returns until the public conversion.
        result_into_raw(result.map_err(|error| with_mutation_error_context(error, raw)))
    }

    /// Borrow a live runtime mutator as an object-valued ABI argument.
    ///
    /// # Safety
    ///
    /// `mutator` must point to a live object and outlive the returned view. The
    /// view does not increment the object's reference count.
    #[inline(always)]
    unsafe fn borrowed_mutator_view<'a>(mutator: StructuralMutatorHandle) -> AnyView<'a> {
        let object = mutator.cast::<TVMFFIObject>();
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
    fn with_mutator_def_region<T>(
        mutator: StructuralMutatorHandle,
        kind: DefRegionKind,
        callback: impl FnOnce(DefRegionKind) -> T,
    ) -> T {
        unsafe {
            let previous = (*mutator).def_region_mode;
            if previous == kind as i32 || previous == DefRegionKind::Pattern as i32 {
                // Inherited descent does not change state; Pattern keeps precedence.
                return callback(std::mem::transmute::<i32, DefRegionKind>(previous));
            }
            (*mutator).def_region_mode = kind as i32;
            struct Restore {
                mutator: StructuralMutatorHandle,
                previous: i32,
            }
            impl Drop for Restore {
                #[inline(always)]
                fn drop(&mut self) {
                    unsafe { (*self.mutator).def_region_mode = self.previous };
                }
            }
            let _restore = Restore { mutator, previous };
            // One call site keeps the continuation visible to the inliner.
            callback(kind)
        }
    }

    /// Default descent for borrowed `raw` on live `runtime`, with reuse permission
    /// `mode`. Internal callers publish the current dispatcher reborrow first.
    /// Return one owning token from a hook or the outlined reflection fallback.
    // Operate in the raw path to optimize stack frame cost. Each returned raw
    // carrier transfers one owning result; fallible/destructor-bearing code
    // imports it into Any before performing work that may unwind.
    #[inline(always)]
    fn mutate_children_raw(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        mode: InplaceMode,
    ) -> TVMFFIAny {
        // Registered hooks own their remapping policy; reflection handles its own.
        if let Some(attr) = Self::structural_mutate_hook(raw, mode) {
            return Self::call_structural_mutate_hook_raw(runtime, raw, attr);
        }
        unsafe {
            Self::mutate_without_hook_raw(
                runtime,
                raw.type_index,
                raw.small_str_len,
                raw.data_union,
            )
        }
    }

    /// Snapshot the live `runtime` region before fallback work, reconstruct borrowed
    /// input from scalar fields, and transfer one owning replacement/Unchanged/Error.
    /// The private C-unwind ABI keeps AArch64 results in registers while permitting
    /// Rust cleanup; public/native callback ABIs remain unchanged.
    ///
    /// # Safety
    /// `runtime` must be this live engine and the input fields a valid borrowed value.
    /// The input owner must survive reflection, remap updates, and error annotation.
    // Outline reflection scratch without marking this normal fallback cold.
    // The AArch64 C ABI keeps raw results in registers; C-unwind preserves
    // Rust unwinding inside this private helper.
    #[inline(never)]
    unsafe extern "C-unwind" fn mutate_without_hook_raw(
        runtime: StructuralMutatorHandle,
        type_index: i32,
        small_str_len: u32,
        data_union: TVMFFIAnyDataUnion,
    ) -> TVMFFIAny {
        let def_region_kind = std::mem::transmute::<i32, DefRegionKind>((*runtime).def_region_mode);
        let raw = TVMFFIAny {
            type_index,
            small_str_len,
            data_union,
        };
        result_into_raw(
            Self::mutate_without_hook(runtime, raw, def_region_kind)
                .map_err(|error| with_mutation_error_context(error, raw)),
        )
    }

    /// Resolve fallback mutation for live borrowed `raw` and `runtime`, using the
    /// entry snapshot `def_region_kind`. Preserve cached identities, free-variable
    /// policy and owning reflected results. Internal callers guarantee valid inputs;
    /// normal Rust owners clean up partial work if reflection fails or unwinds.
    fn mutate_without_hook(
        runtime: StructuralMutatorHandle,
        raw: TVMFFIAny,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
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
            if let Some(mutated) = unsafe { Self::runtime_var_remap_get(runtime, raw)? } {
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

        let result = Self::map_reflected(runtime, raw, def_region_kind, type_info)?;
        if is_dag_node
            || (is_free_var
                && (def_region_kind == DefRegionKind::Pattern || !is_unchanged(&result)))
        {
            // Keep markers for DAG nodes and pattern definitions. An unchanged simple
            // definition needs no binding: subsequent uses retain the original on a miss.
            unsafe { Self::runtime_var_remap_set(runtime, raw, &result)? };
        }
        Ok(result)
    }
}

// -----------------------------------------------------------------------------
// Public entrypoint
// -----------------------------------------------------------------------------

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
/// * `mutator` - Owned or mutably borrowed dispatcher implementing [`IntoMutator`].
///   Matching handlers return replacements and control recursion through
///   [`StructuralMutatorView`]; unmatched values use default mutation. For closure
///   callbacks with automatic recursion, use [`structural_map`].
///
/// # Examples
///
/// Double integer leaves with a typed handler; default mutation visits the array:
///
/// ```
/// # pub use tvm_ffi::{extra, match_any, match_any_internal, Dispatcher};
/// # fn main() -> tvm_ffi::Result<()> {
/// use tvm_ffi::{
///     dispatch, structural_mutate, Any, Array, MutationView, Result, StructuralMutatorView,
/// };
///
/// struct Double;
/// #[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<Any>)]
/// impl Double {
///     fn mutate_integer(
///         &mut self,
///         value: i64,
///         _: &mut StructuralMutatorView<'_, Self>,
///     ) -> Result<Any> {
///         Ok(Any::from(value * 2))
///     }
/// }
///
/// let result = structural_mutate(Array::new(vec![1_i64, 2, 3]), Double)?;
/// let result = Array::<i64>::try_from(result)?;
/// assert_eq!(result.iter().collect::<Vec<_>>(), vec![2, 4, 6]);
/// # Ok(())
/// # }
/// ```
pub fn structural_mutate<R, M>(root: R, mutator: impl IntoMutator<M>) -> Result<Any>
where
    R: Into<Any>,
{
    mutator.mutate_root(root.into())
}
