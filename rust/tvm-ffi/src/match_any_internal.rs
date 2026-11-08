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

//! Runtime support for the `match_any!` and `#[dispatch]` expansions. Only
//! macro output names these items.

use std::any::TypeId;
use std::marker::PhantomData;

use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

use crate::{AnyCompatible, AnyView};

/// Per-call-site arm number assigned by `match_any!`.
#[doc(hidden)]
pub type ArmId = u32;

impl<'a, T> From<&'a crate::MutationView<'_, T>> for AnyView<'a> {
    #[inline(always)]
    fn from(value: &'a crate::MutationView<'_, T>) -> Self {
        value.as_any_view()
    }
}

/// Holds the matched value. The expansion calls `prepare()`; inherent impls for
/// `AnyView`, `&T` and `MutationView` take priority over the [`PrepareSource`]
/// fallback, which selects the source by type at compile time.
#[doc(hidden)]
pub struct SourceProbe<T>(Option<T>);
impl<T> SourceProbe<T> {
    #[inline(always)]
    pub fn new(value: T) -> Self {
        Self(Some(value))
    }
}

/// A source read through an `AnyView`.
#[doc(hidden)]
pub struct ViewSource<'a>(AnyView<'a>);
/// A source read through a mutation view, kept so a `MutationView<T>` arm can
/// narrow it.
#[doc(hidden)]
pub struct MutationSource<'a>(crate::MutationView<'a>);

/// Fallback `prepare()` for an owned value: it stays in the probe for the whole
/// match and is viewed in place, without a move or refcount change.
#[doc(hidden)]
pub trait PrepareSource {
    fn prepare(&mut self) -> ViewSource<'_>;
}
impl<T> PrepareSource for SourceProbe<T>
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    #[inline(always)]
    fn prepare(&mut self) -> ViewSource<'_> {
        ViewSource(self.0.as_ref().unwrap().into())
    }
}
impl<'a> SourceProbe<AnyView<'a>> {
    #[inline(always)]
    pub fn prepare(&mut self) -> ViewSource<'a> {
        ViewSource(self.0.take().unwrap())
    }
}
impl<'a, T: ?Sized> SourceProbe<&'a T> {
    #[inline(always)]
    pub fn prepare<'out>(&mut self) -> ViewSource<'out>
    where
        AnyView<'out>: From<&'a T>,
    {
        ViewSource(self.0.take().unwrap().into())
    }
}
impl<'a, T> SourceProbe<crate::MutationView<'a, T>> {
    #[inline(always)]
    pub fn prepare(&mut self) -> MutationSource<'a> {
        MutationSource(self.0.take().unwrap().into_untyped())
    }
}

/// Read access arms use on a source: the view, or its cell for in-place borrows.
#[doc(hidden)]
pub trait SourceView {
    fn view(&self) -> AnyView<'_>;
    /// The source's own cell, which a reference pattern borrows in place.
    fn cell(&self) -> &crate::tvm_ffi_sys::TVMFFIAny;
}
impl SourceView for ViewSource<'_> {
    #[inline(always)]
    fn view(&self) -> AnyView<'_> {
        self.0
    }
    #[inline(always)]
    fn cell(&self) -> &crate::tvm_ffi_sys::TVMFFIAny {
        self.0.as_raw_ffi_any()
    }
}
impl SourceView for MutationSource<'_> {
    #[inline(always)]
    fn view(&self) -> AnyView<'_> {
        self.0.as_any_view()
    }
    #[inline(always)]
    fn cell(&self) -> &crate::tvm_ffi_sys::TVMFFIAny {
        self.0.cell()
    }
}

/// A value arm `Type(name)`: if the source strictly holds a `Self`, return it
/// owned. Never consumes the source, so a rejected guard falls through.
#[doc(hidden)]
pub trait ValuePattern<S>: Sized {
    /// `None` leaves the source untouched for the next arm. A typed mutation
    /// view aliases the source, which the expansion never uses after an arm
    /// accepts.
    fn try_match(source: S) -> Option<Self>;
}
impl<'b, T: crate::type_traits::ContainerElement, S: SourceView> ValuePattern<&'b mut S> for T {
    #[inline(always)]
    fn try_match(source: &'b mut S) -> Option<Self> {
        let view = source.view();
        let data = view.as_raw_ffi_any();
        unsafe {
            (T::container_check_any_strict(data) && T::container_match_constraints(data))
                .then(|| T::container_copy_from_any_view_after_check(data))
        }
    }
}
impl<'a, 'b, 'out> ValuePattern<&'b mut ViewSource<'a>> for AnyView<'out>
where
    'a: 'out,
{
    #[inline(always)]
    fn try_match(source: &'b mut ViewSource<'a>) -> Option<Self> {
        Some(source.0)
    }
}
impl<'a, 'b> ValuePattern<&'b mut MutationSource<'a>> for AnyView<'b> {
    #[inline(always)]
    fn try_match(source: &'b mut MutationSource<'a>) -> Option<Self> {
        Some(source.view())
    }
}
impl<'a, 'b, T: crate::type_traits::ContainerElement> ValuePattern<&'b mut MutationSource<'a>>
    for crate::MutationView<'a, T>
{
    #[inline(always)]
    fn try_match(source: &'b mut MutationSource<'a>) -> Option<Self> {
        // SAFETY: the macro-private source is never used after an arm accepts
        // this view, so the alias stays the only live handle.
        unsafe { source.0.alias() }.downcast::<T>().ok()
    }
}

/// Autoref probe for a reference arm: `(&&&probe).try_match_ref(..)` selects the
/// borrowing impl when `T: DowncastRefFromAny`, else the owned fallback.
#[doc(hidden)]
pub struct RefPatternProbe<T>(PhantomData<fn() -> T>);
impl<T> RefPatternProbe<T> {
    #[doc(hidden)]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// A reference arm `Type(ref name)`: return a holder that borrows as `&T`, in
/// place when possible (no refcount change), else an owned `T` from
/// [`ValuePattern`].
#[doc(hidden)]
pub trait RefPattern<'b, S, T> {
    type Holder: std::borrow::Borrow<T>;
    /// `None` leaves the source untouched for the next arm.
    fn try_match_ref(&self, source: &'b mut S) -> Option<Self::Holder>;
}
impl<'b, T: crate::type_traits::DowncastRefFromAny + 'static, S: SourceView> RefPattern<'b, S, T>
    for &&RefPatternProbe<T>
{
    type Holder = &'b T;
    #[inline(always)]
    fn try_match_ref(&self, source: &'b mut S) -> Option<Self::Holder> {
        // Borrow the source's own cell, which lives for 'b: a reference type
        // such as `&Add` points into the cell, an object type into the heap.
        let node = unsafe { T::downcast_ref_from_any(source.cell()) }? as *const T;
        // SAFETY: the checked value is borrowed from source for 'b, and the arm
        // does not modify a list or dict holding it (the `match_any!`
        // contract). No owning conversion or view transfer can occur while this
        // borrow lives.
        Some(unsafe { &*node })
    }
}
impl<'b, S: 'b, T: ValuePattern<&'b mut S>> RefPattern<'b, S, T> for &RefPatternProbe<T> {
    type Holder = T;
    #[inline(always)]
    fn try_match_ref(&self, source: &'b mut S) -> Option<T> {
        T::try_match(source)
    }
}

// ----------------------------------------------------------------------------
// Jump table: arms that are all exact final types, no guards, 20+ arms
// ----------------------------------------------------------------------------

/// Type index → arm table for arms that are all exact final types. Built and
/// validated once per call site; each match is one lookup.
#[doc(hidden)]
pub struct JumpTable {
    type_list_id: TypeId,
    base: i32,
    arm_ids: Box<[ArmId]>,
}

const NO_ARM_ID: ArmId = ArmId::MAX;

impl JumpTable {
    #[doc(hidden)]
    #[cold]
    #[inline(never)]
    pub fn build(type_list_id: TypeId, type_indices: &[i32]) -> Self {
        assert!(
            !type_indices.is_empty(),
            "match_any! jump table requires at least one arm"
        );
        assert!(
            type_indices
                .iter()
                .all(|&type_index| type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32),
            "match_any! jump table arm has a non-object type index"
        );
        let min_type_index = *type_indices.iter().min().unwrap();
        let max_type_index = *type_indices.iter().max().unwrap();
        let span = usize::try_from(i64::from(max_type_index) - i64::from(min_type_index) + 1)
            .expect("match_any! jump table has an invalid type-index span");
        assert!(
            type_indices.len() <= NO_ARM_ID as usize,
            "match_any! jump table has too many arms"
        );
        let mut arm_ids = vec![NO_ARM_ID; span];
        for (arm_id, &type_index) in type_indices.iter().enumerate() {
            let offset = (type_index - min_type_index) as usize;
            let slot = &mut arm_ids[offset];
            // Keep the first source arm for duplicate runtime indices.
            if *slot == NO_ARM_ID {
                *slot = arm_id as ArmId;
            }
        }

        Self {
            type_list_id,
            base: min_type_index,
            arm_ids: arm_ids.into_boxed_slice(),
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    #[allow(clippy::result_unit_err)]
    pub fn lookup(&self, type_list_id: TypeId, type_index: i32) -> Result<Option<ArmId>, ()> {
        if self.type_list_id != type_list_id {
            return Err(());
        }
        let offset = type_index.wrapping_sub(self.base) as usize;
        let arm_id = self.arm_ids.get(offset).copied().unwrap_or(NO_ARM_ID);
        Ok((arm_id != NO_ARM_ID).then_some(arm_id))
    }
}

/// Compile-time list of the arms' exact types, used to build the jump table.
/// Its key uses the final object types, so arm types need not be `'static`.
#[doc(hidden)]
pub trait JumpTableTypes {
    #[doc(hidden)]
    type Key: 'static;

    #[doc(hidden)]
    const ALL_EXACT_FINAL: bool;

    /// Validate that the final payload identity describes this exact matcher.
    #[doc(hidden)]
    fn matches_key() -> bool;

    #[doc(hidden)]
    fn fill_type_indices(out: &mut [i32]);
}

impl JumpTableTypes for () {
    type Key = ();
    const ALL_EXACT_FINAL: bool = true;

    #[inline(always)]
    fn matches_key() -> bool {
        true
    }

    #[inline(always)]
    fn fill_type_indices(out: &mut [i32]) {
        debug_assert!(out.is_empty());
    }
}

impl<Head, Tail> JumpTableTypes for (Head, Tail)
where
    Head: AnyCompatible + crate::object::ObjectRefCore,
    Tail: JumpTableTypes,
{
    type Key = (
        <Head as crate::object::ObjectRefCore>::ContainerType,
        Tail::Key,
    );
    const ALL_EXACT_FINAL: bool = Head::MATCH_ANY_EXACT
        && <Head::ContainerType as crate::object::ObjectCore>::TYPE_FINAL
        && Tail::ALL_EXACT_FINAL;

    #[inline(always)]
    fn matches_key() -> bool {
        Head::match_any_exact_type_index()
            == <Head::ContainerType as crate::object::ObjectCore>::type_index()
            && Tail::matches_key()
    }

    #[inline(always)]
    fn fill_type_indices(out: &mut [i32]) {
        let (head, tail) = out
            .split_first_mut()
            .expect("match_any! pattern metadata length mismatch");
        *head = Head::match_any_exact_type_index();
        Tail::fill_type_indices(tail);
    }
}

/// A typed mutation view arm routes as its value type.
impl<'a, Head, Tail> JumpTableTypes for (crate::MutationView<'a, Head>, Tail)
where
    Head: AnyCompatible + crate::object::ObjectRefCore,
    Tail: JumpTableTypes,
{
    type Key = <(Head, Tail) as JumpTableTypes>::Key;
    const ALL_EXACT_FINAL: bool = <(Head, Tail) as JumpTableTypes>::ALL_EXACT_FINAL;
    fn matches_key() -> bool {
        <(Head, Tail) as JumpTableTypes>::matches_key()
    }
    fn fill_type_indices(out: &mut [i32]) {
        <(Head, Tail) as JumpTableTypes>::fill_type_indices(out);
    }
}

/// A reference arm's entry in a [`JumpTableTypes`] list.
#[doc(hidden)]
pub struct RefJumpTableType<T>(PhantomData<fn() -> T>);

impl<Head: crate::object::ObjectCore, Tail: JumpTableTypes> JumpTableTypes
    for (RefJumpTableType<Head>, Tail)
{
    type Key = (Head, Tail::Key);
    const ALL_EXACT_FINAL: bool = Head::TYPE_FINAL && Tail::ALL_EXACT_FINAL;

    #[inline(always)]
    fn matches_key() -> bool {
        Tail::matches_key()
    }

    #[inline(always)]
    fn fill_type_indices(out: &mut [i32]) {
        let (head, tail) = out
            .split_first_mut()
            .expect("match_any! pattern metadata length mismatch");
        *head = Head::type_index();
        Tail::fill_type_indices(tail);
    }
}

/// Autoref probe that decides at compile time whether every arm qualifies for
/// the jump table; if not, the expansion tries arms in order.
#[doc(hidden)]
pub struct JumpTableProbe<T>(PhantomData<fn() -> T>);

impl<T> JumpTableProbe<T> {
    #[doc(hidden)]
    pub fn prepend<Head>(self, _: PhantomData<Head>) -> JumpTableProbe<(Head, T)> {
        JumpTableProbe::new()
    }

    #[doc(hidden)]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// The probe's jump-table inputs: the type list id, the key check and the arms'
/// type indices.
#[doc(hidden)]
pub trait JumpTableMetadata {
    #[doc(hidden)]
    fn type_list_id(&self) -> Option<TypeId> {
        None
    }

    #[doc(hidden)]
    fn matches_key(&self) -> bool {
        false
    }

    #[doc(hidden)]
    fn fill_type_indices(&self, _out: &mut [i32]) {}
}

impl<T> JumpTableMetadata for &JumpTableProbe<T> {}

impl<T: JumpTableTypes> JumpTableMetadata for JumpTableProbe<T> {
    #[inline(always)]
    fn type_list_id(&self) -> Option<TypeId> {
        T::ALL_EXACT_FINAL.then(TypeId::of::<T::Key>)
    }

    #[inline(always)]
    fn matches_key(&self) -> bool {
        T::matches_key()
    }

    #[inline(always)]
    fn fill_type_indices(&self, out: &mut [i32]) {
        T::fill_type_indices(out);
    }
}

/// A reference arm's entry in the jump table's type list, selected by the same
/// autoref probe as [`RefPattern`].
#[doc(hidden)]
pub trait RefJumpTableEntry {
    type Entry;
    fn table_entry(&self) -> PhantomData<Self::Entry> {
        PhantomData
    }
}
impl<T: crate::object::ObjectCore> RefJumpTableEntry for &&RefPatternProbe<T> {
    type Entry = RefJumpTableType<T>;
}
impl<T> RefJumpTableEntry for &RefPatternProbe<T> {
    type Entry = T;
}
