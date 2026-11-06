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

use super::structural_common::raw_ffi_any_same_as;
use super::structural_mutate::{
    with_mutation_error_context, IntoMutateResult, MutationView, StructuralMutateEngine,
    StructuralMutatorView,
};
use super::structural_visit::DefRegionKind;
use super::structural_walk::WalkOrder;
use super::unchanged::{is_unchanged, UnchangedOr};
use crate::{Any, AnyView, Dispatcher, IntoDispatcher, Result};
use std::borrow::BorrowMut;

// -----------------------------------------------------------------------------
// Map adapter to Dispatcher.
// -----------------------------------------------------------------------------

impl<O: IntoMutateResult> super::dispatch::CallbackResult<Result<Any>> for O {
    #[inline(always)]
    fn callback_result(self) -> Result<Any> {
        self.into_mutate_result().map(Any::from)
    }
}

impl<O: IntoMutateResult> super::dispatch::CallbackResult<Result<UnchangedOr<Any>>> for O {
    #[inline(always)]
    fn callback_result(self) -> Result<UnchangedOr<Any>> {
        self.into_mutate_result()
    }
}

// -----------------------------------------------------------------------------
// Implementation of map.
// -----------------------------------------------------------------------------

struct MapMutator<'a, D, const PRE_ORDER: bool> {
    dispatch: &'a mut D,
}

fn run_mapper<D>(root: Any, dispatch: &mut D, order: WalkOrder) -> Result<Any>
where
    for<'s, 'v> &'s mut D:
        Dispatcher<(AnyView<'v>, DefRegionKind), Option<Result<UnchangedOr<Any>>>>,
{
    match order {
        WalkOrder::PreOrder => {
            StructuralMutateEngine::run_mutator::<MapMutator<'_, D, true>, false>(
                root,
                &mut MapMutator::<D, true> { dispatch },
            )
        }
        WalkOrder::PostOrder => StructuralMutateEngine::run_mutator::<
            MapMutator<'_, D, false>,
            false,
        >(root, &mut MapMutator::<D, false> { dispatch }),
    }
}

impl<'v, 'c, 'r, 'a, D, const PRE_ORDER: bool>
    Dispatcher<
        (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, MapMutator<'a, D, PRE_ORDER>>,
        ),
        Option<Result<UnchangedOr<Any>>>,
    > for &mut MapMutator<'a, D, PRE_ORDER>
where
    for<'s, 'value> &'s mut D:
        Dispatcher<(AnyView<'value>, DefRegionKind), Option<Result<UnchangedOr<Any>>>>,
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, mutator): (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, MapMutator<'a, D, PRE_ORDER>>,
        ),
    ) -> Option<Result<UnchangedOr<Any>>> {
        Some(self.map_current(value, mutator).map(UnchangedOr::from_any))
    }
}

impl<D, const PRE_ORDER: bool> MapMutator<'_, D, PRE_ORDER>
where
    for<'s, 'v> &'s mut D:
        Dispatcher<(AnyView<'v>, DefRegionKind), Option<Result<UnchangedOr<Any>>>>,
{
    #[inline(always)]
    fn map_current(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        let mode = value.inplace_mode();
        let raw = *value.as_any_view().as_raw_ffi_any();
        let kind = mutator.def_region_kind();
        if PRE_ORDER {
            // Snapshot the mode before a callback can retain or release ownership.
            // SAFETY: the input owner remains live through this synchronous callback;
            // its higher-ranked view cannot escape into the owning result.
            let Some(callback_result) = self
                .dispatch
                .dispatch_tuple((unsafe { AnyView::from_raw_ffi_any(raw) }, kind))
            else {
                return mutator.default_mutate(self, value).map(Into::into);
            };
            let mapped: Any = callback_result
                .map_err(|error| with_mutation_error_context(error, raw, "map"))?
                .into();
            if is_unchanged(&mapped) || raw_ffi_any_same_as(raw, *mapped.as_raw_ffi_any()) {
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
                // SAFETY: Unchanged leaves the original input owner in place. Borrow
                // it only through the callback; the other branch borrows mapped.
                unsafe { AnyView::from_raw_ffi_any(raw) }
            } else {
                AnyView::from(&mapped)
            };
            match self.dispatch.dispatch_tuple((current, kind)) {
                Some(Ok(result)) if result.is_unchanged() => Ok(mapped),
                Some(result) => result.map(Any::from).map_err(|error| {
                    with_mutation_error_context(error, *current.as_raw_ffi_any(), "map")
                }),
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
///   implementing [`IntoDispatcher`]. Callbacks return replacements; unmatched
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
/// let root = Array::from_vec(vec![1_i64, 2, 3]);
/// let mapped = structural_map(root.clone(), |value: i64| value + 1, WalkOrder::PostOrder)?;
/// let mapped = Array::<i64>::try_from(mapped)?;
/// assert_eq!(mapped.iter().copied().collect::<Vec<_>>(), vec![2, 3, 4]);
/// assert_eq!(root.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub fn structural_map<R, M, H, D, O>(root: R, mapper: H, order: WalkOrder) -> Result<Any>
where
    R: Into<Any>,
    O: BorrowMut<D>,
    for<'v> H: IntoDispatcher<
        (AnyView<'v>, DefRegionKind),
        Option<Result<UnchangedOr<Any>>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v> &'s mut D:
        Dispatcher<(AnyView<'v>, DefRegionKind), Option<Result<UnchangedOr<Any>>>>,
{
    run_mapper(root.into(), mapper.into_dispatcher().borrow_mut(), order)
}
