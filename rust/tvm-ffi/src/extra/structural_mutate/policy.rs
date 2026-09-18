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

/// A reusable policy around default map and mutation recursion.
///
/// Consume `value` with `default_maybe_inplace_mutate_result()` to continue to
/// the next policy, then registered hooks or reflected fields. Children re-enter
/// the full callback engine. A tuple `(outer, inner)` composes policies.
/// Policies share callback state; restore scoped state before forwarding errors.
///
/// ```
/// use tvm_ffi::{
///     structural_mutate, Any, Array, CallbackMutator, DefRegionKind, MutateCallbacks,
///     MutateValue, MutContextPolicy, Result, UnchangedOr,
/// };
/// struct AsPattern;
/// impl<State> MutContextPolicy<State> for AsPattern {
///     fn default_mutate(
///         &self, value: MutateValue<'_>, ctx: &mut CallbackMutator<State>,
///     ) -> Result<UnchangedOr<Any>> {
///         ctx.with_def_region_kind(DefRegionKind::Pattern, |ctx| {
///             ctx.default_maybe_inplace_mutate_result(value)
///         })
///     }
/// }
/// let mut mutator = MutateCallbacks::new((), |x: i64, ctx: &mut CallbackMutator| {
///     assert_eq!(ctx.def_region_kind(), DefRegionKind::Pattern);
///     x + 1
/// }).with_policy(AsPattern);
/// let result = structural_mutate(Array::new(vec![1_i64]), &mut mutator)?;
/// assert_eq!(Array::<i64>::try_from(result)?.get(0)?, 2);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
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
        mutate_with_policy(
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

pub(super) fn mutate_with_policy<State>(
    driver: &mut dyn MutateContextDriver<State>,
    policy: &impl MutContextPolicy<State>,
    value: MutateValue<'_>,
    kind: DefRegionKind,
) -> Result<Any> {
    with_mutation_region(kind, |kind| {
        let mut ctx = MutateContext {
            driver,
            def_region_kind: kind,
            inplace_mode: value.inplace_mode(),
            _state: PhantomData,
            _not_send_sync: PhantomData,
        };
        policy.default_mutate(value, &mut ctx).map(Any::from)
    })
}

struct NextPolicy<'a, State, Policy> {
    driver: &'a mut dyn MutateContextDriver<State>,
    policy: &'a Policy,
}

impl<State, Policy: MutContextPolicy<State>> MutateContextDriver<State>
    for NextPolicy<'_, State, Policy>
{
    fn state(&self) -> &State {
        self.driver.state()
    }
    fn state_mut(&mut self) -> &mut State {
        self.driver.state_mut()
    }
    fn mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any> {
        self.driver.mutate_borrowed(value, kind)
    }
    fn mutate_owned(&mut self, value: Any, kind: DefRegionKind, mode: InplaceMode) -> Result<Any> {
        self.driver.mutate_owned(value, kind, mode)
    }
    fn default_mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any> {
        let view = StructuralView::from_raw(*value.as_raw_ffi_any());
        mutate_with_policy(self.driver, self.policy, MutateValue::borrowed(&view), kind)
    }
    fn default_mutate_value(
        &mut self,
        mut value: MutateValue<'_>,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        value.mode = value.permit(mode).inplace_mode(value.value.raw());
        mutate_with_policy(self.driver, self.policy, value, kind)
    }
    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        self.driver.var_remap_get(var)
    }
    fn var_remap_set(&mut self, var: &StructuralView, replacement: &Any) -> Result<()> {
        self.driver.var_remap_set(var, replacement)
    }
}

pub(super) struct MutationDescent<'a, Driver> {
    pub(super) driver: &'a mut Driver,
}

impl<State, Driver: MutationDriver + MutateCallbackState<State>> MutateContextDriver<State>
    for MutationDescent<'_, Driver>
{
    fn state(&self) -> &State {
        self.driver.callback_state()
    }
    fn state_mut(&mut self) -> &mut State {
        self.driver.callback_state_mut()
    }
    fn mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any> {
        with_mutation_region(kind, |kind| {
            self.driver
                .dispatch_raw(*value.as_raw_ffi_any(), kind, Permit::Copy)
        })
    }
    fn mutate_owned(&mut self, value: Any, kind: DefRegionKind, mode: InplaceMode) -> Result<Any> {
        let raw = *value.as_raw_ffi_any();
        let result = with_mutation_region(kind, |kind| {
            self.driver.dispatch_raw(raw, kind, mode.permit())
        })?;
        Ok(if is_unchanged(&result) { value } else { result })
    }
    fn default_mutate_borrowed(&mut self, value: AnyView<'_>, kind: DefRegionKind) -> Result<Any> {
        with_mutation_region(kind, |kind| {
            default_mutate_driver(self.driver, *value.as_raw_ffi_any(), kind, Permit::Copy)
        })
    }
    fn default_mutate_value(
        &mut self,
        value: MutateValue<'_>,
        kind: DefRegionKind,
        mode: InplaceMode,
    ) -> Result<Any> {
        let permit = value.permit(mode);
        with_mutation_region(kind, |kind| {
            default_mutate_driver(self.driver, value.value.raw(), kind, permit)
        })
    }
    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        self.driver.var_remap_get_raw(var.raw())
    }
    fn var_remap_set(&mut self, var: &StructuralView, replacement: &Any) -> Result<()> {
        self.driver.var_remap_set_raw(var.raw(), replacement)
    }
}

impl<D, Policy> MutateCallbackState<D> for NativeMapper<'_, D, Policy> {
    fn callback_state(&self) -> &D {
        self.dispatch
    }
    fn callback_state_mut(&mut self) -> &mut D {
        self.dispatch
    }
}

/// A map dispatcher combined with a reusable default-recursion policy.
///
/// The dispatcher is the policy's shared state. Define its callbacks with
/// `#[dispatch(map)]` or [`MapDispatch`], then call [`Self::map`] or pass the
/// adapter to [`structural_map`]. Map callbacks run outside their own node's
/// policy scope; children see the context established during descent.
pub struct MapWithPolicy<Mapper, Policy> {
    mapper: Mapper,
    policy: Rc<Policy>,
}

impl<Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> MapWithPolicy<Mapper, Policy> {
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

    /// Map an owning root with fresh invocation-local identity substitutions.
    pub fn map<R: Into<Any>>(&mut self, root: R, order: WalkOrder) -> Result<Any> {
        run_structural_mutator(
            root.into(),
            &mut NativeMapper {
                dispatch: &mut self.mapper,
                order,
                policy: Some(self.policy.clone()),
                remap: StructuralVarRemap::default(),
            },
        )
    }
}

impl<Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> MapDispatch
    for MapWithPolicy<Mapper, Policy>
{
    fn dispatch_map(&mut self, value: &StructuralView, kind: DefRegionKind) -> Option<MapResult> {
        self.mapper.dispatch_map(value, kind)
    }
    fn map_root(&mut self, root: Any, order: WalkOrder) -> Result<Any> {
        self.map(root, order)
    }
}

#[doc(hidden)]
pub enum ByPolicyMap {}

impl<Mapper: MapDispatch, Policy: MutContextPolicy<Mapper>> IntoMapper<ByPolicyMap>
    for MapWithPolicy<Mapper, Policy>
{
    type Mapper = Self;
    fn into_mapper(self) -> Self {
        self
    }
}
