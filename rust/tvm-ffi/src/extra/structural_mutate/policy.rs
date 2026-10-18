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

//! Context management around default structural mutation.

use super::*;

/// Default-recursion policy for [`MutateCallbacks::with_policy`] and [`MapWithContextPolicy`].
///
/// `ctx.default_maybe_inplace_mutate_result(value)` continues to the next policy,
/// then hooks or reflected fields; children re-enter callback dispatch.
/// Policies share callback state and compose as `(outer, inner)`.
/// Restore user state before returning, including on errors; use
/// [`MutateContext::with_def_region_kind`] for scoped definition regions.
pub trait MutContextPolicy<State> {
    /// Customize default descent, preserving the input permission and result marker.
    fn default_mutate(
        &self,
        value: MutateValue<'_>,
        ctx: &mut MutateContext<'_, State>,
    ) -> Result<UnchangedOr<Any>>;
}

/// Default descent through registered hooks or reflected structural fields.
pub struct DefaultMutContextPolicy;

impl<State> MutContextPolicy<State> for DefaultMutContextPolicy {
    fn default_mutate(
        &self,
        value: MutateValue<'_>,
        ctx: &mut MutateContext<'_, State>,
    ) -> Result<UnchangedOr<Any>> {
        ctx.default_maybe_inplace_mutate_result(value)
    }
}

impl<State, Outer: MutContextPolicy<State>, Inner: MutContextPolicy<State>> MutContextPolicy<State>
    for (Outer, Inner)
{
    fn default_mutate(
        &self,
        value: MutateValue<'_>,
        ctx: &mut MutateContext<'_, State>,
    ) -> Result<UnchangedOr<Any>> {
        let kind = ctx.def_region_kind();
        let mutator = ctx.active_mutator()?;
        mutate_with_policy(
            mutator,
            &mut NextPolicy {
                driver: &mut *ctx.driver,
                policy: &self.1,
            },
            &self.0,
            value,
            kind,
        )
        .and_then(UnchangedOr::from_carrier)
    }
}

#[inline(always)]
pub(super) fn mutate_with_policy<State>(
    mutator: StructuralMutatorHandle,
    driver: &mut dyn MutateContextDriver<State>,
    policy: &impl MutContextPolicy<State>,
    value: MutateValue<'_>,
    kind: DefRegionKind,
) -> Result<Any> {
    let raw = value.value.raw();
    with_mutator_def_region(
        mutator,
        kind,
        #[inline(always)]
        |kind| {
            let mut ctx = MutateContext {
                driver,
                mutator,
                def_region_kind: kind,
                inplace_mode: value.inplace_mode(),
                _not_send_sync: PhantomData,
            };
            policy.default_mutate(value, &mut ctx).map(Any::from)
        },
    )
    .map_err(|error| with_value_context(error, raw))
}

struct NextPolicy<'a, State, Policy> {
    driver: &'a mut dyn MutateContextDriver<State>,
    policy: &'a Policy,
}

impl<State, Policy: MutContextPolicy<State>> MutateContextDriver<State>
    for NextPolicy<'_, State, Policy>
{
    #[inline]
    fn state(&self) -> &State {
        self.driver.state()
    }
    #[inline]
    fn state_mut(&mut self) -> &mut State {
        self.driver.state_mut()
    }
    #[inline]
    fn mutate_borrowed(
        &mut self,
        mutator: StructuralMutatorHandle,
        value: AnyView<'_>,
        kind: DefRegionKind,
    ) -> Result<Any> {
        self.driver.mutate_borrowed(mutator, value, kind)
    }
    #[inline]
    fn mutate_owned(
        &mut self,
        mutator: StructuralMutatorHandle,
        value: Any,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        self.driver.mutate_owned(mutator, value, kind, mode)
    }
    fn default_mutate_borrowed(
        &mut self,
        mutator: StructuralMutatorHandle,
        value: AnyView<'_>,
        kind: DefRegionKind,
    ) -> Result<Any> {
        let view = StructuralView::from_raw(*value.as_raw_ffi_any());
        mutate_with_policy(
            mutator,
            self.driver,
            self.policy,
            MutateValue::new_at(mutator, &view, InplaceMode::Disallow),
            kind,
        )
    }
    fn default_mutate_value(
        &mut self,
        mutator: StructuralMutatorHandle,
        mut value: MutateValue<'_>,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        value.mode = value.permit(mode).inplace_mode(value.value.raw());
        mutate_with_policy(mutator, self.driver, self.policy, value, kind)
    }
    #[inline]
    fn var_remap_get(
        &mut self,
        mutator: StructuralMutatorHandle,
        var: &StructuralView,
    ) -> Result<Option<Any>> {
        self.driver.var_remap_get(mutator, var)
    }
    #[inline]
    fn var_remap_set(
        &mut self,
        mutator: StructuralMutatorHandle,
        var: &StructuralView,
        replacement: &Any,
    ) -> Result<()> {
        self.driver.var_remap_set(mutator, var, replacement)
    }
}

pub(super) struct MutationDescent<'a, Driver> {
    pub(super) driver: &'a mut Driver,
}

impl<State, Driver: MutationDriver + MutateCallbackState<State>> MutateContextDriver<State>
    for MutationDescent<'_, Driver>
{
    #[inline]
    fn state(&self) -> &State {
        self.driver.callback_state()
    }
    #[inline]
    fn state_mut(&mut self) -> &mut State {
        self.driver.callback_state_mut()
    }
    #[inline]
    fn mutate_borrowed(
        &mut self,
        mutator: StructuralMutatorHandle,
        value: AnyView<'_>,
        kind: DefRegionKind,
    ) -> Result<Any> {
        self.driver
            .dispatch_raw(mutator, *value.as_raw_ffi_any(), kind, Permit::Copy)
    }
    #[inline]
    fn mutate_owned(
        &mut self,
        mutator: StructuralMutatorHandle,
        value: Any,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        let raw = *value.as_raw_ffi_any();
        let result = self
            .driver
            .dispatch_raw(mutator, raw, kind, mode.permit())?;
        Ok(if is_unchanged(&result) { value } else { result })
    }
    #[inline]
    fn default_mutate_borrowed(
        &mut self,
        mutator: StructuralMutatorHandle,
        value: AnyView<'_>,
        kind: DefRegionKind,
    ) -> Result<Any> {
        // The policy context has already installed this definition region.
        default_mutate_driver(
            mutator,
            self.driver,
            *value.as_raw_ffi_any(),
            kind,
            Permit::Copy,
        )
    }
    #[inline]
    fn default_mutate_value(
        &mut self,
        mutator: StructuralMutatorHandle,
        value: MutateValue<'_>,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        let permit = value.permit(mode);
        default_mutate_driver(mutator, self.driver, value.value.raw(), kind, permit)
    }
    #[inline]
    fn var_remap_get(
        &mut self,
        mutator: StructuralMutatorHandle,
        var: &StructuralView,
    ) -> Result<Option<Any>> {
        self.driver.var_remap_get_raw(mutator, var.raw())
    }
    #[inline]
    fn var_remap_set(
        &mut self,
        mutator: StructuralMutatorHandle,
        var: &StructuralView,
        replacement: &Any,
    ) -> Result<()> {
        self.driver
            .var_remap_set_raw(mutator, var.raw(), replacement)
    }
}

impl<D, Policy, const PRE_ORDER: bool> MutateCallbackState<D>
    for NativeMapper<'_, D, Policy, PRE_ORDER>
{
    fn callback_state(&self) -> &D {
        self.dispatch
    }
    fn callback_state_mut(&mut self) -> &mut D {
        self.dispatch
    }
}

/// A [`MapDispatch`] with a [`MutContextPolicy`], sharing the dispatcher's state.
///
/// Pass this value or a mutable reference to [`structural_map`]. Each node's
/// map callback runs outside its policy scope; its children run inside.
/// This mapper cannot be a callback tuple member or another wrapper's dispatcher.
/// Compose policies as `(outer, inner)` within one wrapper.
pub struct MapWithContextPolicy<Mapper, Policy> {
    mapper: Mapper,
    policy: Rc<Policy>,
}

impl<Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> MapWithContextPolicy<Mapper, Policy> {
    /// Combine a dispatcher and a default-recursion policy.
    pub fn new(mapper: Mapper, policy: Policy) -> Self {
        Self {
            mapper,
            policy: Rc::new(policy),
        }
    }

    /// Access the dispatcher and shared state.
    pub fn state(&self) -> &Mapper {
        &self.mapper
    }

    /// Mutably access the dispatcher outside recursive calls.
    pub fn state_mut(&mut self) -> &mut Mapper {
        &mut self.mapper
    }

    /// Recover the dispatcher and its state.
    pub fn into_state(self) -> Mapper {
        self.mapper
    }
}

impl<Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> NativeMap
    for MapWithContextPolicy<Mapper, Policy>
{
    fn map_root(&mut self, root: Any, order: WalkOrder) -> Result<Any> {
        run_native_mapper(root, &mut self.mapper, Some(self.policy.clone()), order)
    }
}

impl<Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> NativeMap
    for &mut MapWithContextPolicy<Mapper, Policy>
{
    fn map_root(&mut self, root: Any, order: WalkOrder) -> Result<Any> {
        (**self).map_root(root, order)
    }
}

#[doc(hidden)]
pub enum ByPolicyMap {}

impl<Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> IntoMapper<ByPolicyMap>
    for MapWithContextPolicy<Mapper, Policy>
{
    type Mapper = Self;
    fn into_mapper(self) -> Self {
        self
    }
}

impl<'a, Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> IntoMapper<ByPolicyMap>
    for &'a mut MapWithContextPolicy<Mapper, Policy>
{
    type Mapper = Self;
    fn into_mapper(self) -> Self {
        self
    }
}
