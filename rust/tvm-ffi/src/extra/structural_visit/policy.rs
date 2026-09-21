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

//! Reusable customization of default structural descent.

use super::*;

/// A reusable policy for managing context around default visit and walk recursion.
///
/// `visit_children()` on this policy's context continues with the next policy
/// (or the built-in hooks and reflected fields). `visit()` re-enters the full
/// callback engine for a child. A tuple `(outer, inner)` composes two policies;
/// tuples may nest. Policies receive `&self`; mutable pass data belongs in the
/// context's state. Save and restore scoped state around descent.
pub trait ContextPolicy<State> {
    /// Customize default descent for the current value.
    ///
    /// Return interrupts explicitly, and restore any scoped state before
    /// returning an interrupt or error. Walk invokes this between its pre- and
    /// post-order callback positions; visit invokes it only on callback miss
    /// or when a matched callback requests default descent.
    fn default_visit(
        &self,
        value: &StructuralView,
        visitor: &mut VisitContext<'_, State>,
    ) -> Result<Option<VisitInterrupt>>;
}

/// Default descent through registered hooks or reflected structural fields.
pub struct DefaultContextPolicy;

impl<State> ContextPolicy<State> for DefaultContextPolicy {
    fn default_visit(
        &self,
        _value: &StructuralView,
        visitor: &mut VisitContext<'_, State>,
    ) -> Result<Option<VisitInterrupt>> {
        visitor.visit_children()
    }
}

impl<State, Outer: ContextPolicy<State>, Inner: ContextPolicy<State>> ContextPolicy<State>
    for (Outer, Inner)
{
    fn default_visit(
        &self,
        value: &StructuralView,
        visitor: &mut VisitContext<'_, State>,
    ) -> Result<Option<VisitInterrupt>> {
        let kind = visitor.def_region_kind();
        visit_with_policy(
            &mut NextPolicy {
                driver: &mut *visitor.driver,
                policy: &self.1,
            },
            &self.0,
            value,
            kind,
        )
    }
}

#[inline(always)]
pub(super) fn visit_with_policy<State>(
    driver: &mut dyn VisitContextDriver<State>,
    policy: &impl ContextPolicy<State>,
    value: &StructuralView,
    def_region_kind: DefRegionKind,
) -> Result<Option<VisitInterrupt>> {
    with_visit_region(
        def_region_kind,
        #[inline(always)]
        |def_region_kind| {
            policy.default_visit(
                value,
                &mut VisitContext {
                    driver,
                    current: StructuralView::from_raw(value.raw()),
                    def_region_kind,
                    _not_send_sync: PhantomData,
                },
            )
        },
    )
}

struct NextPolicy<'a, State, Policy> {
    driver: &'a mut dyn VisitContextDriver<State>,
    policy: &'a Policy,
}

impl<State, Policy: ContextPolicy<State>> VisitContextDriver<State>
    for NextPolicy<'_, State, Policy>
{
    fn state(&self) -> &State {
        self.driver.state()
    }
    fn state_mut(&mut self) -> &mut State {
        self.driver.state_mut()
    }
    fn visit_raw(&mut self, raw: TVMFFIAny, kind: DefRegionKind) -> Result<Option<VisitInterrupt>> {
        self.driver.visit_raw(raw, kind)
    }
    fn visit_children_raw(
        &mut self,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        visit_with_policy(
            self.driver,
            self.policy,
            &StructuralView::from_raw(raw),
            kind,
        )
    }
}

pub(super) struct VisitDescent<'a, V> {
    pub(super) visitor: &'a mut V,
}

impl<State, V: StructuralVisitor + VisitCallbackState<State>> VisitContextDriver<State>
    for VisitDescent<'_, V>
{
    fn state(&self) -> &State {
        self.visitor.callback_state()
    }
    fn state_mut(&mut self) -> &mut State {
        self.visitor.callback_state_mut()
    }
    fn visit_raw(&mut self, raw: TVMFFIAny, kind: DefRegionKind) -> Result<Option<VisitInterrupt>> {
        VisitContextDriver::visit_raw(self.visitor, raw, kind)
    }
    fn visit_children_raw(
        &mut self,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        // Bypass the current policy; children still re-enter the complete visitor.
        default_user_visit_children(self.visitor, &StructuralView::from_raw(raw), kind)
    }
}

/// A walk dispatcher combined with a reusable default-recursion policy.
///
/// The dispatcher is also the state visible through the policy's context. Use
/// `#[dispatch(walk)]` or implement [`WalkDispatch`] to define its callbacks.
/// Pass this value or a mutable reference to [`structural_walk`].
pub struct WalkWithContextPolicy<Walker, Policy> {
    walker: Walker,
    policy: Rc<Policy>,
}

impl<Walker: WalkDispatch, Policy: ContextPolicy<Walker>> WalkWithContextPolicy<Walker, Policy> {
    /// Combine a dispatcher and a default-recursion policy.
    pub fn new(walker: Walker, policy: Policy) -> Self {
        Self {
            walker,
            policy: Rc::new(policy),
        }
    }

    /// Access the dispatcher and its traversal state.
    pub fn state(&self) -> &Walker {
        &self.walker
    }

    /// Mutably access the dispatcher outside an active walk.
    pub fn state_mut(&mut self) -> &mut Walker {
        &mut self.walker
    }

    /// Recover the dispatcher and its state.
    pub fn into_state(self) -> Walker {
        self.walker
    }
}

#[doc(hidden)]
pub enum ByPolicyWalk {}

impl<Walker: WalkDispatch, Policy: ContextPolicy<Walker>> IntoWalker<ByPolicyWalk>
    for WalkWithContextPolicy<Walker, Policy>
{
    type Walker = Self;
    fn into_walker(self) -> Self {
        self
    }
}

impl<Walker: WalkDispatch, Policy: ContextPolicy<Walker>> IntoWalker<ByPolicyWalk>
    for &mut WalkWithContextPolicy<Walker, Policy>
{
    type Walker = Self;
    fn into_walker(self) -> Self {
        self
    }
}

impl<Walker: WalkDispatch, Policy: ContextPolicy<Walker>> NativeVisit
    for WalkWithContextPolicy<Walker, Policy>
{
    const CUSTOM_DESCENT: bool = true;

    fn visit(&mut self, value: &StructuralView, kind: DefRegionKind) -> Result<WalkResult> {
        self.walker
            .dispatch_walk(value, kind)
            .unwrap_or_else(|| Ok(WalkResult::Advance))
    }

    fn default_visit_children<const PRE_ORDER: bool>(
        &mut self,
        value: &StructuralView,
        kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        let policy = Rc::clone(&self.policy);
        visit_with_policy(
            &mut WalkDescent::<_, _, PRE_ORDER> { visitor: self },
            &*policy,
            value,
            kind,
        )
    }
}

struct WalkDescent<'a, Walker, Policy, const PRE_ORDER: bool> {
    visitor: &'a mut WalkWithContextPolicy<Walker, Policy>,
}

impl<Walker: WalkDispatch, Policy: ContextPolicy<Walker>, const PRE_ORDER: bool>
    VisitContextDriver<Walker> for WalkDescent<'_, Walker, Policy, PRE_ORDER>
{
    fn state(&self) -> &Walker {
        &self.visitor.walker
    }
    fn state_mut(&mut self) -> &mut Walker {
        &mut self.visitor.walker
    }
    fn visit_raw(&mut self, raw: TVMFFIAny, kind: DefRegionKind) -> Result<Option<VisitInterrupt>> {
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(None);
        }
        let active = active_structural_visitor()?;
        let context = std::ptr::from_mut(&mut *self.visitor).cast::<c_void>();
        finish(with_current_visitor_context(active, context, || {
            call_visitor(active, raw, kind)
        }))
    }
    fn visit_children_raw(
        &mut self,
        raw: TVMFFIAny,
        kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        default_walk_children::<_, PRE_ORDER>(self.visitor, &StructuralView::from_raw(raw), kind)
    }
}
