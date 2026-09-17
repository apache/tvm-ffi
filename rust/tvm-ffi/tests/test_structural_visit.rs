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

use std::cell::{Cell, RefCell};
use tvm_ffi::object::ObjectRef;
use tvm_ffi::{
    dispatch, get_type_attr, structural_visit, structural_walk, Any, Array, DLDataType,
    DLDataTypeCode, DefRegionKind, Error, FieldGetter, Function, Map, Object, ObjectRefCore,
    Result, String as FfiString, StructuralVisitor, TypeIndex, VisitCallbacks, VisitContext,
    VisitInterrupt, VisitPolicy, VisitValue, WalkOrder, WalkResult, WalkWithPolicy, RUNTIME_ERROR,
};

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[test]
fn composed_policies_share_array_scope_with_visit_and_walk_callbacks() {
    #[derive(Default)]
    struct CollectIntegers {
        depth: usize,
        integers: Vec<(i64, usize)>,
        descent_depths: Vec<usize>,
    }

    impl CollectIntegers {
        fn record(&mut self, value: i64) {
            self.integers.push((value, self.depth));
        }
    }

    // The outer policy establishes a scope around an array's children.
    // It does not know what the callbacks will do with the current depth.
    struct ArrayScope;

    impl VisitPolicy<CollectIntegers> for ArrayScope {
        fn default_visit(
            &self,
            value: &VisitValue,
            visitor: &mut VisitContext<'_, CollectIntegers>,
        ) -> Result<Option<VisitInterrupt>> {
            if value.cast::<Array<Any>>().is_none() {
                return visitor.visit_children();
            }
            let outer_depth = visitor.state().depth;
            let depth = outer_depth + 1;
            visitor.state_mut().depth = depth;

            // Continue to RecordDescent, not directly to the children.
            let result = visitor.visit_children();

            visitor.state_mut().depth = outer_depth;
            result
        }
    }

    // The inner policy observes the scope set by ArrayScope. Its continuation
    // reaches the built-in Array hook, whose children re-enter the full engine.
    struct RecordDescent;

    impl VisitPolicy<CollectIntegers> for RecordDescent {
        fn default_visit(
            &self,
            _value: &VisitValue,
            visitor: &mut VisitContext<'_, CollectIntegers>,
        ) -> Result<Option<VisitInterrupt>> {
            let state = visitor.state_mut();
            state.descent_depths.push(state.depth);
            visitor.visit_children()
        }
    }

    // [1, [2, 3], 4]: the final sibling must see the restored outer depth.
    let root = Array::new(vec![
        Any::from(1_i64),
        Any::from(Array::new(vec![2_i64, 3])),
        Any::from(4_i64),
    ]);
    let expected = vec![(1, 1), (2, 2), (3, 2), (4, 1)];

    let mut visitor = VisitCallbacks::new(
        CollectIntegers::default(),
        |value: i64, visitor: &mut VisitContext<'_, CollectIntegers>| {
            visitor.state_mut().record(value);
            // Matched visit callbacks own recursion: an integer needs no descent.
        },
    )
    .with_policy((ArrayScope, RecordDescent));
    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.state().integers, expected);
    assert_eq!(visitor.state().depth, 0);
    // Only unmatched arrays enter the default policies in this visit.
    assert_eq!(visitor.state().descent_depths, vec![1, 2]);

    #[dispatch(walk)]
    impl CollectIntegers {
        fn walk_integer(&mut self, value: i64) -> WalkResult {
            self.record(value);
            WalkResult::Advance
        }
    }

    // Reuse the policies with a macro-generated walk dispatcher. Walk manages
    // recursion, so default policies also run for matched integer leaves.
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut walker =
            WalkWithPolicy::new(CollectIntegers::default(), (ArrayScope, RecordDescent));
        assert!(walker.walk(&root, order).unwrap().is_none());
        assert_eq!(walker.state().integers, expected);
        assert_eq!(walker.state().depth, 0);
        assert_eq!(walker.state().descent_depths, vec![1, 1, 2, 2, 2, 1]);
    }
}

// Enter Simple, then Pattern, then attempt to downgrade to None. Verify each
// enclosing region is restored even when a child interrupts or returns an error.
#[test]
fn policy_continuation_scopes_regions_and_restores_after_halts() {
    #[derive(Clone, Copy)]
    enum Outcome {
        Finish,
        Interrupt,
        Error,
    }
    struct Probe {
        outcome: Outcome,
        seen: Vec<(i64, DefRegionKind)>,
        regions: Vec<DefRegionKind>,
    }
    impl Probe {
        fn record(&mut self, value: i64, kind: DefRegionKind) -> Result<Option<VisitInterrupt>> {
            self.seen.push((value, kind));
            if value == 1 {
                match self.outcome {
                    Outcome::Interrupt => return Ok(Some(VisitInterrupt::with(42_i64))),
                    Outcome::Error => return Err(runtime_error("continuation failed")),
                    Outcome::Finish => {}
                }
            }
            Ok(None)
        }
    }
    #[dispatch(walk)]
    impl Probe {
        fn walk_integer(&mut self, value: i64, kind: DefRegionKind) -> Result<WalkResult> {
            Ok(match self.record(value, kind)? {
                Some(interrupt) => WalkResult::InterruptWith(interrupt.value),
                None => WalkResult::Advance,
            })
        }
    }
    struct Scope(DefRegionKind, i64);
    impl VisitPolicy<Probe> for Scope {
        fn default_visit(
            &self,
            value: &VisitValue,
            ctx: &mut VisitContext<'_, Probe>,
        ) -> Result<Option<VisitInterrupt>> {
            let Some(array) = value.cast::<Array<i64>>() else {
                return ctx.visit_children();
            };
            let outer = ctx.def_region_kind();
            ctx.state_mut().regions.push(outer);
            let result = ctx.default_visit_children(&array, self.0);
            assert_eq!(ctx.def_region_kind(), outer);
            // Re-dispatch checks the restored ABI region, not only the context.
            assert!(ctx.visit(&self.1)?.is_none());
            result
        }
    }
    use DefRegionKind::{None as Use, Pattern, Simple};
    let root = Array::new(vec![1_i64, 5]);
    for outcome in [Outcome::Finish, Outcome::Interrupt, Outcome::Error] {
        for order in [None, Some(WalkOrder::PreOrder), Some(WalkOrder::PostOrder)] {
            let state = Probe {
                outcome,
                seen: vec![],
                regions: vec![],
            };
            let policies = (Scope(Simple, 4), (Scope(Pattern, 3), Scope(Use, 2)));
            let (result, state) = if let Some(order) = order {
                let mut walker = WalkWithPolicy::new(state, policies);
                let result = walker.walk(&root, order);
                (result, walker.into_state())
            } else {
                let mut visitor =
                    VisitCallbacks::new(state, |value: i64, ctx: &mut VisitContext<'_, Probe>| {
                        let kind = ctx.def_region_kind();
                        ctx.state_mut().record(value, kind)
                    })
                    .with_policy(policies);
                let result = structural_visit(&root, &mut visitor);
                (result, visitor.into_state())
            };
            match outcome {
                Outcome::Finish => assert!(result.unwrap().is_none()),
                Outcome::Interrupt => {
                    assert_eq!(i64::try_from(result.unwrap().unwrap().value).unwrap(), 42)
                }
                Outcome::Error => assert!(result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("continuation failed")),
            }
            assert_eq!(state.regions, vec![Use, Simple, Pattern]);
            let mut expected = vec![(1, Pattern)];
            if matches!(outcome, Outcome::Finish) {
                expected.push((5, Pattern)); // Halts must skip the remaining child.
            }
            // Unwind the innermost, middle, and outer scopes.
            expected.extend([(2, Pattern), (3, Simple), (4, Use)]);
            assert_eq!(state.seen, expected);
        }
    }
}

fn visit_region_graph(with_hook: bool) -> Any {
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    Function::get_global("testing.make_visit_region_graph")
        .unwrap()
        .call_tuple((with_hook,))
        .unwrap()
}

#[derive(Default)]
struct PolicyRegionTrace(Vec<(i64, DefRegionKind)>);
#[dispatch(walk)]
impl PolicyRegionTrace {
    fn walk_integer(&mut self, value: i64, kind: DefRegionKind) -> WalkResult {
        self.0.push((value, kind));
        WalkResult::Advance
    }
}

// Default descent may target a child container without dispatching its own
// callback, while keeping subsequent policies and child callback dispatch.
#[test]
fn policy_continuation_retargets_without_dispatching_the_container() {
    #[derive(Default)]
    struct Probe(Vec<(&'static str, i64, DefRegionKind)>, bool);

    #[dispatch(walk)]
    impl Probe {
        fn walk_array(&mut self, value: Array<Any>, kind: DefRegionKind) -> WalkResult {
            self.0.push(("array callback", value.len() as i64, kind));
            if self.1 {
                WalkResult::Skip
            } else {
                WalkResult::Advance
            }
        }
        fn walk_integer(&mut self, value: i64, kind: DefRegionKind) -> WalkResult {
            self.0.push(("integer callback", value, kind));
            WalkResult::Advance
        }
    }

    struct Redirect;
    impl VisitPolicy<Probe> for Redirect {
        fn default_visit(
            &self,
            value: &VisitValue,
            ctx: &mut VisitContext<'_, Probe>,
        ) -> Result<Option<VisitInterrupt>> {
            let Some(array) = value.cast::<Array<Any>>() else {
                return ctx.visit_children();
            };
            assert_eq!(
                array.len(),
                2,
                "the redirected array must not restart this policy"
            );
            let kind = ctx.def_region_kind();
            ctx.state_mut().0.push(("before", 2, kind));
            let target = array.get(0).unwrap();
            let result = ctx.default_visit_children(&target, DefRegionKind::Pattern);
            assert_eq!(ctx.current().cast::<Array<Any>>().unwrap().len(), 2);
            assert_eq!(ctx.def_region_kind(), DefRegionKind::None);
            let kind = ctx.def_region_kind();
            ctx.state_mut().0.push(("after", 2, kind));
            result
        }
    }

    struct Observe;
    impl VisitPolicy<Probe> for Observe {
        fn default_visit(
            &self,
            value: &VisitValue,
            ctx: &mut VisitContext<'_, Probe>,
        ) -> Result<Option<VisitInterrupt>> {
            if let Some(array) = value.cast::<Array<Any>>() {
                assert_eq!(ctx.current().cast::<Array<Any>>().unwrap().len(), 1);
                let kind = ctx.def_region_kind();
                ctx.state_mut()
                    .0
                    .push(("next policy", array.len() as i64, kind));
            } else if value.cast::<i64>().is_none() {
                let kind = ctx.def_region_kind();
                ctx.state_mut().0.push(("reflected policy", 0, kind));
            }
            ctx.visit_children()
        }
    }

    let root = Array::new(vec![Any::from(Array::new(vec![7_i64])), Any::from(99_i64)]);
    let descent = vec![
        ("before", 2, DefRegionKind::None),
        ("next policy", 1, DefRegionKind::Pattern),
        ("integer callback", 7, DefRegionKind::Pattern),
        ("after", 2, DefRegionKind::None),
    ];
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        for skip in [false, true] {
            let mut walker = WalkWithPolicy::new(Probe(vec![], skip), (Redirect, Observe));
            assert!(walker.walk(&root, order).unwrap().is_none());
            let mut expected = descent.clone();
            let array_callback = ("array callback", 2, DefRegionKind::None);
            match order {
                WalkOrder::PreOrder => {
                    if skip {
                        expected.clear(); // Skip prevents policy entry as well as child traversal.
                    }
                    expected.insert(0, array_callback);
                }
                WalkOrder::PostOrder => expected.push(array_callback),
            }
            assert_eq!(walker.state().0, expected);
        }
    }
    let mut visitor = VisitCallbacks::new(
        Probe::default(),
        |value: &VisitValue, ctx: &mut VisitContext<'_, Probe>| {
            let kind = ctx.def_region_kind();
            if let Some(array) = value.cast::<Array<Any>>() {
                ctx.state_mut()
                    .0
                    .push(("array callback", array.len() as i64, kind));
                ctx.visit_children()
            } else {
                ctx.state_mut()
                    .0
                    .push(("integer callback", value.cast::<i64>().unwrap(), kind));
                Ok(None)
            }
        },
    )
    .with_policy((Redirect, Observe));
    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    let mut expected = descent;
    expected.insert(0, ("array callback", 2, DefRegionKind::None));
    assert_eq!(visitor.state().0, expected);

    // A callback can also retarget default descent, entering the configured
    // policy. This target uses reflection instead of the Array hook above.
    let target = visit_region_graph(false);
    let mut visitor = VisitCallbacks::new(
        Probe::default(),
        |value: &VisitValue, ctx: &mut VisitContext<'_, Probe>| {
            if value.cast::<i64>() == Some(-1) {
                assert!(ctx
                    .default_visit_children(&Any::new(), DefRegionKind::Pattern)?
                    .is_none());
                let result = ctx.default_visit_children(&target, DefRegionKind::Pattern);
                assert_eq!(ctx.current().cast::<i64>(), Some(-1));
                assert_eq!(ctx.def_region_kind(), DefRegionKind::None);
                return result;
            }
            // A callback on the reflected target itself would fail this cast.
            let integer = value.cast::<i64>().unwrap();
            let kind = ctx.def_region_kind();
            ctx.state_mut().0.push(("integer callback", integer, kind));
            Ok(None)
        },
    )
    .with_policy(Observe);
    assert!(structural_visit(&-1_i64, &mut visitor).unwrap().is_none());
    assert_eq!(
        visitor.state().0,
        vec![
            ("reflected policy", 0, DefRegionKind::Pattern),
            ("integer callback", 1, DefRegionKind::Pattern),
            ("integer callback", 2, DefRegionKind::Pattern),
            ("integer callback", 3, DefRegionKind::Pattern),
        ]
    );
}

#[test]
fn policy_halts_skip_remaining_policies_and_restore_outer_state() {
    #[derive(Default)]
    struct Probe {
        events: Vec<&'static str>,
        depth: usize,
    }
    #[dispatch(walk)]
    impl Probe {
        fn walk_any(&mut self, value: &VisitValue, kind: DefRegionKind) -> WalkResult {
            assert!(
                value.cast::<Array<i64>>().is_some(),
                "children must not be visited"
            );
            assert_eq!(kind, DefRegionKind::None);
            self.events.push("callback");
            WalkResult::Advance
        }
    }
    struct Scope;
    impl VisitPolicy<Probe> for Scope {
        fn default_visit(
            &self,
            value: &VisitValue,
            ctx: &mut VisitContext<'_, Probe>,
        ) -> Result<Option<VisitInterrupt>> {
            ctx.state_mut().events.push("enter");
            ctx.state_mut().depth += 1;
            let result = ctx.default_visit_children(&value.to_owned(), DefRegionKind::Pattern);
            assert_eq!(ctx.def_region_kind(), DefRegionKind::None);
            ctx.state_mut().depth -= 1;
            ctx.state_mut().events.push("exit");
            result
        }
    }
    struct Stop(bool);
    impl VisitPolicy<Probe> for Stop {
        fn default_visit(
            &self,
            _: &VisitValue,
            ctx: &mut VisitContext<'_, Probe>,
        ) -> Result<Option<VisitInterrupt>> {
            assert_eq!(ctx.def_region_kind(), DefRegionKind::Pattern);
            assert_eq!(ctx.state().depth, 1);
            ctx.state_mut().events.push("stop");
            if self.0 {
                Err(runtime_error("policy failed"))
            } else {
                Ok(Some(VisitInterrupt::with(42_i64)))
            }
        }
    }
    struct Unreachable;
    impl VisitPolicy<Probe> for Unreachable {
        fn default_visit(
            &self,
            _: &VisitValue,
            _: &mut VisitContext<'_, Probe>,
        ) -> Result<Option<VisitInterrupt>> {
            panic!("the policy after Stop must not run")
        }
    }
    let root = Array::new(vec![1_i64]);
    for error in [false, true] {
        for order in [None, Some(WalkOrder::PreOrder), Some(WalkOrder::PostOrder)] {
            let policies = (Scope, (Stop(error), Unreachable));
            let (result, state) = if let Some(order) = order {
                let mut walker = WalkWithPolicy::new(Probe::default(), policies);
                let result = walker.walk(&root, order);
                (result, walker.into_state())
            } else {
                let mut visitor = VisitCallbacks::new(
                    Probe::default(),
                    |value: &VisitValue, ctx: &mut VisitContext<'_, Probe>| {
                        let kind = ctx.def_region_kind();
                        ctx.state_mut().walk_any(value, kind);
                        ctx.visit_children()
                    },
                )
                .with_policy(policies);
                let result = structural_visit(&root, &mut visitor);
                (result, visitor.into_state())
            };
            if error {
                assert!(result.err().unwrap().to_string().contains("policy failed"));
            } else {
                assert_eq!(i64::try_from(result.unwrap().unwrap().value).unwrap(), 42);
            }
            let expected = if order == Some(WalkOrder::PostOrder) {
                vec!["enter", "stop", "exit"]
            } else {
                vec!["callback", "enter", "stop", "exit"]
            };
            assert_eq!(state.events, expected);
            assert_eq!(state.depth, 0);
        }
    }
}

#[test]
fn policy_regions_compose_with_field_flags_and_function_hooks() {
    struct SetRootRegion(i32, DefRegionKind);
    impl VisitPolicy<PolicyRegionTrace> for SetRootRegion {
        fn default_visit(
            &self,
            value: &VisitValue,
            ctx: &mut VisitContext<'_, PolicyRegionTrace>,
        ) -> Result<Option<VisitInterrupt>> {
            if value.type_index() == self.0 {
                ctx.default_visit_children(&value.to_owned(), self.1)
            } else {
                ctx.visit_children()
            }
        }
    }
    use DefRegionKind::{None as Use, Pattern, Simple};
    for with_hook in [false, true] {
        let root = visit_region_graph(with_hook);
        if with_hook {
            // Exercise the Function-valued __s_visit__ path, not an opaque pointer hook.
            assert!(
                Function::try_from(get_type_attr(root.type_index(), "__s_visit__").unwrap())
                    .is_ok()
            );
        }
        for region in [Use, Simple, Pattern] {
            let ordinary = if with_hook && region == Use {
                Simple
            } else {
                region
            };
            let mut expected = vec![
                (1, if region == Pattern { Pattern } else { Simple }),
                (2, Pattern),
                (3, ordinary),
            ];
            if with_hook {
                expected.push((4, region));
            }
            for order in [None, Some(WalkOrder::PreOrder), Some(WalkOrder::PostOrder)] {
                let policy = SetRootRegion(root.type_index(), region);
                let state = if let Some(order) = order {
                    let mut walker = WalkWithPolicy::new(PolicyRegionTrace::default(), policy);
                    assert!(walker.walk(&root, order).unwrap().is_none());
                    walker.into_state()
                } else {
                    let mut visitor = VisitCallbacks::new(
                        PolicyRegionTrace::default(),
                        |value: i64, ctx: &mut VisitContext<'_, PolicyRegionTrace>| {
                            let kind = ctx.def_region_kind();
                            ctx.state_mut().0.push((value, kind));
                        },
                    )
                    .with_policy(policy);
                    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
                    visitor.into_state()
                };
                assert_eq!(state.0, expected);
            }
        }
    }
}

#[test]
fn walk_policy_preserves_reflected_pattern_before_default_descent() {
    let root = visit_region_graph(false);
    struct Reenter(DefRegionKind);
    impl VisitPolicy<PolicyRegionTrace> for Reenter {
        fn default_visit(
            &self,
            value: &VisitValue,
            ctx: &mut VisitContext<'_, PolicyRegionTrace>,
        ) -> Result<Option<VisitInterrupt>> {
            match value.cast::<i64>() {
                Some(2) => {
                    assert_eq!(ctx.def_region_kind(), DefRegionKind::Pattern);
                    // Re-dispatch immediately: visit_children() must not be needed
                    // to synchronize the ABI visitor with the reflected field region.
                    ctx.visit_with(&99_i64, self.0)
                }
                Some(3) => {
                    assert_eq!(ctx.def_region_kind(), DefRegionKind::None);
                    // The preceding field's Pattern scope must not leak into a sibling.
                    ctx.visit(&100_i64)
                }
                _ => ctx.visit_children(),
            }
        }
    }
    use DefRegionKind::{None as Use, Pattern, Simple};
    for requested in [Use, Simple] {
        for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
            let mut walker = WalkWithPolicy::new(PolicyRegionTrace::default(), Reenter(requested));
            assert!(walker.walk(&root, order).unwrap().is_none());
            let expected = match order {
                WalkOrder::PreOrder => vec![
                    (1, Simple),
                    (2, Pattern),
                    (99, Pattern),
                    (3, Use),
                    (100, Use),
                ],
                WalkOrder::PostOrder => vec![
                    (1, Simple),
                    (99, Pattern),
                    (2, Pattern),
                    (100, Use),
                    (3, Use),
                ],
            };
            assert_eq!(
                walker.state().0,
                expected,
                "order={order:?}, requested={requested:?}"
            );
        }
    }
}

#[test]
fn public_reflection_access_uses_registered_field_and_type_attr() {
    // Keep the existing C++ test library linked so its startup registrations
    // are available even when this test is run by itself.
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    let root = Function::get_global("ffi.MakeObjectFromPackedArgs")
        .unwrap()
        .call_tuple((
            FfiString::from("testing.TestObjectBase"),
            FfiString::from("v_str"),
            FfiString::from("an owning C++ reflected field value"),
        ))
        .unwrap();
    let type_index = root.type_index();
    let root = ObjectRef::try_from(root).unwrap();

    let getter = FieldGetter::new(type_index, "v_str").unwrap();
    let selected = getter
        .get::<_, FfiString>(&**ObjectRef::data(&root))
        .unwrap();
    drop(root);
    assert_eq!(selected.as_str(), "an owning C++ reflected field value");

    let wrong_type = Array::new(vec![0i64]);
    assert!(getter.get_any(&**Array::data(&wrong_type)).is_err());
    assert!(FieldGetter::new(type_index, "missing").is_err());

    // ObjectDef registers this Function-valued attribute for copyable C++ types.
    assert!(Function::try_from(get_type_attr(type_index, "__ffi_shallow_copy__").unwrap()).is_ok());
    assert!(Function::from_type_attr(type_index, "__ffi_shallow_copy__").is_ok());
    assert!(get_type_attr(type_index, "missing").is_none());
}

#[test]
fn plain_walk_uses_registered_array_hook() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                integers += 1;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, 3);
}

#[test]
fn plain_walk_visits_map_values_without_visiting_keys() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
        .into_iter()
        .collect();
    let mut integers = 0;
    let mut strings = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                integers += 1;
            } else if value.cast::<FfiString>().is_some() {
                strings += 1;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, 2);
    assert_eq!(strings, 0);
}

#[test]
fn primitive_values_are_leaves_in_pre_and_post_order() {
    assert!(get_type_attr(TypeIndex::kTVMFFIDataType as i32, "__s_visit__").is_none());
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);

    let mut pre = Vec::new();
    assert!(structural_walk(
        &dtype,
        |value: &VisitValue| {
            if value.cast::<DLDataType>().is_some() {
                pre.push("dtype");
            } else if value.cast::<i64>().is_some() {
                pre.push("child");
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(pre, ["dtype"]);

    let mut skipped = Vec::new();
    assert!(structural_walk(
        &dtype,
        |value: &VisitValue| {
            if value.cast::<DLDataType>().is_some() {
                skipped.push("dtype");
                WalkResult::Skip
            } else {
                skipped.push("child");
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(skipped, ["dtype"]);

    let mut post = Vec::new();
    assert!(structural_walk(
        &dtype,
        |value: &VisitValue| {
            if value.cast::<DLDataType>().is_some() {
                post.push("dtype");
            } else if value.cast::<i64>().is_some() {
                post.push("child");
            }
            WalkResult::Advance
        },
        WalkOrder::PostOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(post, ["dtype"]);
}

#[test]
fn primitive_fast_path_preserves_none_interrupt_and_error() {
    let mut none_calls = 0;
    assert!(structural_walk(
        &Any::new(),
        |_value: &VisitValue| {
            none_calls += 1;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(none_calls, 0);

    let interrupt = structural_walk(
        &1i64,
        |_value: i64| WalkResult::interrupt_with(9i64),
        WalkOrder::PostOrder,
    )
    .unwrap()
    .unwrap();
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 9);

    let error = match structural_walk(
        &1i64,
        |_value: i64| -> Result<WalkResult> { Err(runtime_error("primitive failed")) },
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("failing primitive callback unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "primitive failed");
}

#[test]
fn registered_map_hook_visits_all_values_without_visiting_keys() {
    // More than 4 entries forces the dense (block + iteration list) layout.
    let root: Map<FfiString, i64> = (0..9)
        .map(|i| (FfiString::from(format!("k{i}")), i as i64))
        .collect();
    let mut sum = 0;
    let mut strings = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if let Some(integer) = value.cast::<i64>() {
                sum += integer;
            } else if value.cast::<FfiString>().is_some() {
                strings += 1;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(sum, (0..9).sum::<i64>());
    assert_eq!(strings, 0);
}

#[test]
fn interrupt_payload_crosses_map_traversal() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
        .into_iter()
        .collect();
    let outcome = structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                return WalkResult::interrupt_with(99i64);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("map walk unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 99);
}

#[test]
fn handler_error_crosses_map_traversal() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64)].into_iter().collect();
    let error = match structural_walk(
        &root,
        |value: &VisitValue| -> Result<WalkResult> {
            if value.cast::<i64>().is_some() {
                Err(runtime_error("map handler failed"))
            } else {
                Ok(WalkResult::Advance)
            }
        },
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("map handler unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "map handler failed");
    assert!(error.backtrace().contains("object `ffi.Map`"));
}

#[test]
fn interrupt_stops_without_running_remaining_callbacks() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = 0;
    let outcome = structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                integers += 1;
                return WalkResult::Interrupt;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert!(outcome.is_some());
    assert_eq!(integers, 1);
}

/// Visitor-layer traversal that overrides the def-region for one child and
/// inherits it for the next, mirroring a C++ visitor using
/// `WithDefRegionKind`.
#[derive(Default)]
struct ManualRegionVisitor {
    seen: Vec<DefRegionKind>,
}

impl StructuralVisitor for ManualRegionVisitor {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(array) = value.cast::<Array<i64>>() {
            // Override the state for exactly this child's subtree...
            let overridden = array.get(0).unwrap();
            if let Some(interrupt) = self.visit_child(&overridden, DefRegionKind::Simple)? {
                return Ok(Some(interrupt));
            }
            // ...and forward the received state to inherit it.
            let inherited = array.get(1).unwrap();
            return self.visit_child(&inherited, def_region_kind);
        }
        if value.cast::<i64>().is_some() {
            self.seen.push(def_region_kind);
        }
        Ok(None)
    }
}

#[test]
fn manual_child_visit_can_override_def_region() {
    let root = Array::new(vec![7i64, 8]);
    let mut probe = ManualRegionVisitor::default();
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(probe.seen, vec![DefRegionKind::Simple, DefRegionKind::None]);
}

#[derive(Default)]
struct GeneratedLeafVisitor {
    integers: Vec<(i64, DefRegionKind)>,
}

#[dispatch(visit)]
impl GeneratedLeafVisitor {
    fn visit_integer(&mut self, value: i64, kind: DefRegionKind) {
        self.integers.push((value, kind));
    }
}

#[test]
fn generated_visitor_defaults_unmatched_values() {
    let root = Array::new(vec![1i64, 2]);
    let mut visitor = GeneratedLeafVisitor::default();
    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(
        visitor.integers,
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None)]
    );
}

#[derive(Default)]
struct GeneratedRecursiveVisitor {
    events: Vec<String>,
}

#[dispatch(visit)]
impl GeneratedRecursiveVisitor {
    fn visit_array(
        &mut self,
        array: Array<i64>,
        kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        self.events.push("enter:array".to_string());
        for value in array.iter() {
            if let Some(interrupt) = self.visit_child(&value, kind)? {
                return Ok(Some(interrupt));
            }
        }
        self.events.push("exit:array".to_string());
        Ok(None)
    }

    fn visit_integer(&mut self, value: i64) -> Option<VisitInterrupt> {
        self.events.push(format!("int:{value}"));
        (value == 2).then(|| VisitInterrupt::with(value))
    }

    fn visit_other_object(&mut self, _value: &Object) {}
}

#[test]
fn generated_visitor_can_drive_recursion_through_mut_self() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut visitor = GeneratedRecursiveVisitor::default();
    let interrupt = structural_visit(&root, &mut visitor).unwrap().unwrap();
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 2);
    assert_eq!(visitor.events, vec!["enter:array", "int:1", "int:2"]);
}

#[derive(Default)]
struct GenericDispatchProbe {
    integers: Vec<i64>,
    objects: usize,
    catch_all: usize,
}

#[dispatch(walk)]
impl GenericDispatchProbe {
    fn walk_integer(&mut self, value: i64) -> WalkResult {
        self.integers.push(value);
        WalkResult::Advance
    }

    // Trailing DefRegionKind: handlers may mix arities within one impl.
    fn walk_object(&mut self, _value: &tvm_ffi::Object, kind: DefRegionKind) -> WalkResult {
        assert_eq!(kind, DefRegionKind::None);
        self.objects += 1;
        WalkResult::Advance
    }

    fn walk_any(&mut self, _value: &tvm_ffi::StructuralView) -> WalkResult {
        self.catch_all += 1;
        WalkResult::Advance
    }
}

#[test]
fn generated_dispatch_supports_pod_and_ordered_catch_all() {
    let root = Array::new(vec![1i64, 2]);
    let mut probe = GenericDispatchProbe::default();
    assert!(structural_walk(&root, &mut probe, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.integers, vec![1, 2]);
    assert_eq!(probe.objects, 1);

    let floats = Array::new(vec![1.0f64, 2.0]);
    assert!(structural_walk(&floats, &mut probe, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.objects, 2);
    assert_eq!(probe.catch_all, 2);
}

/// Visitor-layer enter/exit straddling: run enter logic, delegate the
/// default child recursion, then run exit logic with the same locals in
/// scope — the C++ `DefaultVisitExpected` pattern.
#[derive(Default)]
struct StraddleVisitor {
    events: Vec<String>,
}

impl StructuralVisitor for StraddleVisitor {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        let label = match value.cast::<i64>() {
            Some(integer) => format!("int:{integer}"),
            None => "node".to_string(),
        };
        self.events.push(format!("enter:{label}"));
        if let Some(interrupt) = self.default_visit_children(value, def_region_kind)? {
            return Ok(Some(interrupt));
        }
        self.events.push(format!("exit:{label}"));
        Ok(None)
    }
}

#[test]
fn visitor_can_straddle_default_children() {
    let root = Array::new(vec![1i64, 2]);
    let mut probe = StraddleVisitor::default();
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(
        probe.events,
        vec![
            "enter:node",
            "enter:int:1",
            "exit:int:1",
            "enter:int:2",
            "exit:int:2",
            "exit:node",
        ]
    );
}

#[derive(Default)]
struct OrderProbe {
    events: Vec<String>,
}

#[dispatch(walk)]
impl OrderProbe {
    fn walk_array(&mut self, _array: Array<i64>) -> WalkResult {
        self.events.push("array".to_string());
        WalkResult::Advance
    }

    fn walk_integer(&mut self, value: i64) -> WalkResult {
        self.events.push(format!("int:{value}"));
        WalkResult::Advance
    }
}

#[test]
fn stateful_structural_walk_supports_post_order() {
    let root = Array::new(vec![1i64, 2]);
    let mut probe = OrderProbe::default();
    assert!(structural_walk(&root, &mut probe, WalkOrder::PostOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.events, vec!["int:1", "int:2", "array"]);
}

#[test]
fn nested_walk_restores_the_outer_active_visitor() {
    let outer = Array::new(vec![10i64, 20]);
    let inner = Array::new(vec![1i64, 2]);
    let mut entered_inner = false;
    let mut outer_values = Vec::new();
    let mut inner_values = Vec::new();

    assert!(structural_walk(
        &outer,
        |value: &VisitValue| -> Result<WalkResult> {
            if let Some(value) = value.cast::<i64>() {
                outer_values.push(value);
            }
            if !entered_inner {
                entered_inner = true;
                structural_walk(
                    &inner,
                    |value: &VisitValue| {
                        if let Some(value) = value.cast::<i64>() {
                            inner_values.push(value);
                        }
                        WalkResult::Advance
                    },
                    WalkOrder::PreOrder,
                )?;
            }
            Ok(WalkResult::Advance)
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(outer_values, vec![10, 20]);
    assert_eq!(inner_values, vec![1, 2]);
}

#[test]
fn interrupt_payload_is_returned_to_the_caller() {
    let root = Array::new(vec![1i64, 2]);
    let outcome = structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>() == Some(1) {
                return WalkResult::interrupt_with(42i64);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("walk unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 42);
}

#[test]
fn handler_errors_include_native_visit_path() {
    let root = Array::new(vec![1i64]);
    let error = match structural_walk(
        &root,
        |value: &VisitValue| -> Result<WalkResult> {
            if value.cast::<i64>().is_some() {
                Err(runtime_error("handler failed"))
            } else {
                Ok(WalkResult::Advance)
            }
        },
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("handler unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "handler failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn visitor_errors_include_native_visit_path() {
    struct FailingVisitor;

    impl StructuralVisitor for FailingVisitor {
        fn visit(
            &mut self,
            value: &VisitValue,
            def_region_kind: DefRegionKind,
        ) -> Result<Option<VisitInterrupt>> {
            if value.cast::<i64>().is_some() {
                return Err(runtime_error("visitor failed"));
            }
            self.default_visit_children(value, def_region_kind)
        }
    }

    let root = Array::new(vec![1i64]);
    let error = match structural_visit(&root, &mut FailingVisitor) {
        Err(error) => error,
        Ok(_) => panic!("visitor unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "visitor failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));

    let error = match structural_visit(
        &root,
        |_value: i64, _visitor: &mut VisitContext<'_, ()>| -> Result<()> {
            Err(runtime_error("callback visitor failed"))
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("callback visitor unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "callback visitor failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn callback_panics_resume_after_the_registered_hook_returns() {
    let root = Array::new(vec![1i64]);
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        structural_walk(
            &root,
            |_value: i64| -> WalkResult { panic!("visitor panic") },
            WalkOrder::PreOrder,
        )
    })) {
        Err(panic) => panic,
        Ok(_) => panic!("panicking visitor unexpectedly returned"),
    };
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some("visitor panic"));
}

#[test]
fn visitor_interrupt_propagates_through_default_children() {
    struct InterruptingVisitor;

    impl StructuralVisitor for InterruptingVisitor {
        fn visit(
            &mut self,
            value: &VisitValue,
            def_region_kind: DefRegionKind,
        ) -> Result<Option<VisitInterrupt>> {
            if value.cast::<i64>() == Some(2) {
                return Ok(Some(VisitInterrupt::with(7i64)));
            }
            self.default_visit_children(value, def_region_kind)
        }
    }

    let root = Array::new(vec![1i64, 2, 3]);
    let outcome =
        structural_visit::<Array<i64>, InterruptingVisitor>(&root, &mut InterruptingVisitor)
            .unwrap();
    let Some(interrupt) = outcome else {
        panic!("visitor traversal unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 7);
}

#[test]
fn closure_walk_receives_def_region_kind() {
    // C++: StructuralWalk<kPreOrder>(root,
    //          [&](const TVarObj* var, TVMFFIDefRegionKind kind) { ... })
    let root = Array::new(vec![1i64, 2]);
    let mut kinds = Vec::new();
    assert!(structural_walk(
        &root,
        |value: &VisitValue, kind: DefRegionKind| {
            if value.cast::<i64>().is_some() {
                kinds.push(kind);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(kinds, vec![DefRegionKind::None; 2]);
}

#[test]
fn closure_walk_supports_post_order_and_skip() {
    let root = Array::new(vec![1i64, 2]);
    let mut order_probe = Vec::new();
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            order_probe.push(value.cast::<i64>());
            WalkResult::Advance
        },
        WalkOrder::PostOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(order_probe, vec![Some(1), Some(2), None]);

    let mut visited = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            visited += 1;
            if value.cast::<i64>().is_none() {
                WalkResult::Skip
            } else {
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(visited, 1);
}

// ---------------------------------------------------------------------------
// Tuple walkers: structural_walk(root, (link1, link2, ...), order) — links
// are tried in order and the first whose argument type matches the value
// runs, the Rust analog of the variadic C++ StructuralWalk callback chain.
// ---------------------------------------------------------------------------
#[test]
fn chain_accepts_owned_object_ref_links() {
    let root = Array::new(vec![Array::new(vec![1i64]), Array::new(vec![2i64, 3])]);
    let mut lengths = Vec::new();
    assert!(structural_walk(
        &root,
        (
            |array: Array<i64>| {
                lengths.push(array.len());
                WalkResult::Advance
            },
            |_value: i64| WalkResult::Advance,
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    // The outer Array<Array<i64>> fails the strict element check and falls
    // through the chain; only the inner arrays match the typed link.
    assert_eq!(lengths, vec![1, 2]);
}

#[test]
fn chain_links_may_mix_def_region_arity() {
    let root = Array::new(vec![1i64, 2]);
    let mut kinds = Vec::new();
    let mut objects = 0;
    assert!(structural_walk(
        &root,
        (
            |_value: i64, kind: DefRegionKind| {
                kinds.push(kind);
                WalkResult::Advance
            },
            |_value: &VisitValue, kind: DefRegionKind| {
                assert_eq!(kind, DefRegionKind::None);
                objects += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(kinds, vec![DefRegionKind::None; 2]);
    assert_eq!(objects, 1);
}

#[test]
fn chain_links_can_skip_children() {
    let root = Array::new(vec![Array::new(vec![1i64]), Array::new(vec![2i64])]);
    let mut arrays = 0;
    let mut integers = 0;
    assert!(structural_walk(
        &root,
        (
            |_array: Array<i64>| {
                arrays += 1;
                WalkResult::Skip
            },
            |_value: i64| {
                integers += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(arrays, 2);
    assert_eq!(integers, 0); // both inner arrays were skipped
}

#[test]
fn chain_link_errors_include_native_visit_path() {
    let root = Array::new(vec![1i64]);
    let error = match structural_walk(
        &root,
        (
            |_value: i64| -> Result<WalkResult> { Err(runtime_error("link failed")) },
            |_value: &VisitValue| WalkResult::Advance,
        ),
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("link unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "link failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn chain_supports_post_order() {
    let root = Array::new(vec![1i64, 2]);
    let events = std::cell::RefCell::new(Vec::new());
    assert!(structural_walk(
        &root,
        (
            |value: i64| {
                events.borrow_mut().push(format!("int:{value}"));
                WalkResult::Advance
            },
            |_object: &Object| {
                events.borrow_mut().push("array".to_string());
                WalkResult::Advance
            },
        ),
        WalkOrder::PostOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(events.into_inner(), vec!["int:1", "int:2", "array"]);
}

#[derive(Default)]
struct ObjectCounter {
    objects: usize,
}

#[dispatch(walk)]
impl ObjectCounter {
    fn walk_object(&mut self, _value: &Object) -> WalkResult {
        self.objects += 1;
        WalkResult::Advance
    }
}

#[test]
fn chain_splices_dispatch_walkers_between_closures() {
    let root = Array::new(vec![1i64, 2]);
    let mut counter = ObjectCounter::default();
    let mut integers = 0;
    assert!(structural_walk(
        &root,
        (&mut counter, |_value: i64| {
            integers += 1;
            WalkResult::Advance
        },),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(counter.objects, 1);
    assert_eq!(integers, 2);
}

#[test]
fn chain_supports_full_arity() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = Vec::new();
    let mut objects = 0;
    let mut others = 0;
    assert!(structural_walk(
        &root,
        (
            |_value: f64| WalkResult::Advance,
            |_value: bool| WalkResult::Advance,
            |_value: tvm_ffi::String| WalkResult::Advance,
            |_value: Array<f64>| WalkResult::Advance,
            |_value: f32| WalkResult::Advance,
            |_value: tvm_ffi::DLDevice| WalkResult::Advance,
            |_value: Array<bool>| WalkResult::Advance,
            |_value: Array<tvm_ffi::String>| WalkResult::Advance,
            |_value: Map<FfiString, i64>, _kind: DefRegionKind| WalkResult::Advance,
            |value: i64| {
                integers.push(value);
                WalkResult::Advance
            },
            |_object: &Object, _kind: DefRegionKind| {
                objects += 1;
                WalkResult::Advance
            },
            |_value: &VisitValue, _kind: DefRegionKind| {
                others += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, vec![1, 2, 3]);
    assert_eq!(objects, 1); // the array itself; integers matched earlier
    assert_eq!(others, 0); // every value matched an earlier link
}

#[test]
fn typed_lambda_walks_bare_and_as_single_link_tuple() {
    // A lone typed handler needs no tuple: unmatched values (the array
    // itself) advance normally. The 1-tuple spelling routes through the
    // chain impls instead and must agree.
    let root = Array::new(vec![1i64, 2, 3]);
    let mut bare = 0;
    assert!(structural_walk(
        &root,
        |value: i64| {
            bare += value;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    let mut tupled = 0;
    assert!(structural_walk(
        &root,
        (|value: i64| {
            tupled += value;
            WalkResult::Advance
        },),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!((bare, tupled), (6, 6));
}

#[test]
fn bare_node_lambda_takes_def_region_kind() {
    let root = Array::new(vec![1i64, 2]);
    let mut objects = 0;
    assert!(structural_walk(
        &root,
        |_object: &Object, kind: DefRegionKind| {
            assert_eq!(kind, DefRegionKind::None);
            objects += 1;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(objects, 1);
}

struct InheritedRegionProbe {
    at_root: bool,
    seen: Vec<DefRegionKind>,
}

impl StructuralVisitor for InheritedRegionProbe {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if self.at_root {
            self.at_root = false;
            let outer = value.cast::<Array<Array<i64>>>().unwrap();
            let inner = outer.get(0).unwrap();
            return self.visit_child(&inner, DefRegionKind::Pattern);
        }
        self.seen.push(def_region_kind);
        self.default_visit_children(value, def_region_kind)
    }
}

#[test]
fn def_region_is_inherited_through_containers() {
    let root = Array::new(vec![Array::new(vec![1i64, 2])]);
    let mut probe = InheritedRegionProbe {
        at_root: true,
        seen: Vec::new(),
    };
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(probe.seen, vec![DefRegionKind::Pattern; 3]);
}

#[test]
fn reflected_fields_reach_typed_handlers() {
    // Reference the existing test library so its C++ startup registrations are linked.
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    let root = Function::get_global("ffi.MakeObjectFromPackedArgs")
        .unwrap()
        .call_tuple((
            FfiString::from("testing.TestObjectBase"),
            FfiString::from("v_i64"),
            1i64,
            FfiString::from("v_f64"),
            2.5f64,
            FfiString::from("v_str"),
            FfiString::from("a reflected string"),
        ))
        .unwrap();
    let mut integers = Vec::new();
    let mut floats = Vec::new();
    let mut strings = Vec::new();
    assert!(structural_walk(
        &root,
        (
            |value: i64, kind: DefRegionKind| {
                integers.push((value, kind));
                WalkResult::Advance
            },
            |value: f64| {
                floats.push(value);
                WalkResult::Advance
            },
            |value: FfiString| {
                strings.push(value);
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, vec![(1, DefRegionKind::None)]);
    assert_eq!(floats, vec![2.5]);
    assert_eq!(strings.len(), 1);
    assert_eq!(strings[0].as_str(), "a reflected string");
}

#[test]
fn nested_tuple_chain_exceeds_flat_arity() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = Vec::new();
    let mut objects = 0;
    let mut others = 0;
    assert!(structural_walk(
        &root,
        (
            (
                |_value: f64| WalkResult::Advance,
                |_value: bool| WalkResult::Advance,
                |_value: tvm_ffi::String| WalkResult::Advance,
                |_value: Array<f64>| WalkResult::Advance,
                |_value: f32| WalkResult::Advance,
                |_value: tvm_ffi::DLDevice| WalkResult::Advance,
                |_value: Array<bool>| WalkResult::Advance,
                |_value: Array<tvm_ffi::String>| WalkResult::Advance,
                |_value: Map<FfiString, i64>| WalkResult::Advance,
                |_value: Map<i64, i64>, _kind: DefRegionKind| WalkResult::Advance,
                |_value: Array<Array<i64>>| WalkResult::Advance,
                |_value: tvm_ffi::Function| WalkResult::Advance,
            ),
            (
                |value: i64| {
                    integers.push(value);
                    WalkResult::Advance
                },
                (
                    |_object: &Object, _kind: DefRegionKind| {
                        objects += 1;
                        WalkResult::Advance
                    },
                    (|_value: &VisitValue, _kind: DefRegionKind| {
                        others += 1;
                        WalkResult::Advance
                    },),
                ),
                |_value: &VisitValue| WalkResult::Advance,
            ),
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, vec![1, 2, 3]);
    assert_eq!(objects, 1);
    assert_eq!(others, 0);
}

#[test]
fn nested_tuple_first_match_order_is_flattened() {
    let root = Array::new(vec![1i64, 2]);
    let mut first = 0;
    let mut second = 0;
    assert!(structural_walk(
        &root,
        (
            (|_value: &VisitValue| {
                first += 1;
                WalkResult::Advance
            },),
            |_value: i64| {
                second += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(first, 3);
    assert_eq!(second, 0);
}

#[test]
fn callback_visit_defaults_only_when_no_link_matches() {
    let root = Array::new(vec![1i64, 2]);
    let integers = Cell::new(0);
    assert!(
        structural_visit(&root, |value: i64, _visitor: &mut VisitContext<'_, ()>| {
            integers.set(integers.get() + value);
        })
        .unwrap()
        .is_none()
    );
    assert_eq!(integers.get(), 3);

    let integers = Cell::new(0);
    assert!(structural_visit(
        &root,
        (
            |_array: Array<i64>, _visitor: &mut VisitContext<'_, ()>| {},
            |value: i64, _visitor: &mut VisitContext<'_, ()>| {
                integers.set(integers.get() + value);
            },
        ),
    )
    .unwrap()
    .is_none());
    assert_eq!(integers.get(), 0);
}

#[derive(Default)]
struct StatefulVisitStats {
    arrays: usize,
    integer_sum: i64,
}

fn stateful_visit_array(
    _array: Array<i64>,
    visitor: &mut VisitContext<'_, StatefulVisitStats>,
) -> Result<Option<VisitInterrupt>> {
    visitor.state_mut().arrays += 1;
    assert!(visitor.current().cast::<Array<i64>>().is_some());
    visitor.visit_children()
}

fn stateful_visit_integer(value: i64, visitor: &mut VisitContext<'_, StatefulVisitStats>) {
    visitor.state_mut().integer_sum += value;
}

#[test]
fn stateful_callback_visit_uses_ordinary_mutable_state() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut visitor = VisitCallbacks::new(
        StatefulVisitStats::default(),
        (stateful_visit_array, stateful_visit_integer),
    );

    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.state().arrays, 1);
    assert_eq!(visitor.state().integer_sum, 6);

    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.state().arrays, 2);
    assert_eq!(visitor.into_state().integer_sum, 12);
}

#[derive(Default)]
struct StatefulVisitDepth {
    current: usize,
    maximum: usize,
    calls: usize,
}

#[test]
fn stateful_callback_visit_reborrows_visitor_during_recursion() {
    let root = Array::new(vec![Array::new(vec![1i64, 2])]);
    let mut visitor = VisitCallbacks::new(
        StatefulVisitDepth::default(),
        |_value: &VisitValue, visitor: &mut VisitContext<'_, StatefulVisitDepth>| {
            visitor.state_mut().current += 1;
            visitor.state_mut().calls += 1;
            let current = visitor.state().current;
            visitor.state_mut().maximum = visitor.state().maximum.max(current);

            let outcome = visitor.visit_children();
            visitor.state_mut().current -= 1;
            outcome
        },
    );

    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.state().current, 0);
    assert_eq!(visitor.state().maximum, 3);
    assert_eq!(visitor.state().calls, 4);
}

#[test]
fn callback_visit_can_reenter_the_same_fn_through_visitor() {
    let root = Array::new(vec![1i64, 2]);
    let visits = Cell::new(0);
    assert!(structural_visit(
        &root,
        |_value: &VisitValue, visitor: &mut VisitContext<'_, ()>| {
            visits.set(visits.get() + 1);
            visitor.visit_children()
        },
    )
    .unwrap()
    .is_none());
    assert_eq!(visits.get(), 3);
}

#[test]
fn callback_visit_tuple_is_first_match_and_can_interrupt() {
    let root = Array::new(vec![1i64, 2, 3]);
    let fallback = Cell::new(0);
    let interrupted = structural_visit(
        &root,
        (
            |value: i64, _visitor: &mut VisitContext<'_, ()>| {
                (value == 2).then(|| VisitInterrupt::with(value))
            },
            |_value: &VisitValue, visitor: &mut VisitContext<'_, ()>| {
                fallback.set(fallback.get() + 1);
                visitor.visit_children()
            },
        ),
    )
    .unwrap()
    .unwrap();
    assert_eq!(i64::try_from(interrupted.value).unwrap(), 2);
    assert_eq!(fallback.get(), 1);
}

#[test]
fn callback_visit_supports_node_links_and_nested_tuples() {
    let root = Array::new(vec![1i64, 2]);
    let seen = RefCell::new(Vec::new());
    assert!(structural_visit(
        &root,
        (
            (
                |_value: f64, _visitor: &mut VisitContext<'_, ()>| {},
                |_node: &tvm_ffi::collections::array::ArrayObj,
                 visitor: &mut VisitContext<'_, ()>| {
                    assert_eq!(visitor.def_region_kind(), DefRegionKind::None);
                    visitor.visit_children()
                },
            ),
            |value: i64, visitor: &mut VisitContext<'_, ()>| {
                seen.borrow_mut().push((value, visitor.def_region_kind()));
            },
        ),
    )
    .unwrap()
    .is_none());
    assert_eq!(
        *seen.borrow(),
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None)]
    );
}

#[test]
fn callback_visit_with_overrides_child_def_region() {
    let root = Array::new(vec![1i64, 2]);
    let seen = RefCell::new(Vec::new());
    assert!(structural_visit(
        &root,
        (
            |array: Array<i64>, visitor: &mut VisitContext<'_, ()>| {
                for value in array.iter() {
                    if let Some(interrupt) = visitor.visit_with(&value, DefRegionKind::Pattern)? {
                        return Ok(Some(interrupt));
                    }
                }
                Ok(None)
            },
            |value: i64, visitor: &mut VisitContext<'_, ()>| {
                seen.borrow_mut().push((value, visitor.def_region_kind()));
            },
        ),
    )
    .unwrap()
    .is_none());
    assert_eq!(
        *seen.borrow(),
        vec![(1, DefRegionKind::Pattern), (2, DefRegionKind::Pattern),]
    );
}

#[test]
fn nested_callback_visit_restores_the_outer_active_visitor() {
    let outer = Array::new(vec![10i64, 20]);
    let inner = Array::new(vec![1i64, 2]);
    let entered_inner = Cell::new(false);
    let outer_values = RefCell::new(Vec::new());
    let inner_values = RefCell::new(Vec::new());

    assert!(structural_visit(
        &outer,
        |value: &VisitValue, visitor: &mut VisitContext<'_, ()>| {
            if let Some(value) = value.cast::<i64>() {
                outer_values.borrow_mut().push(value);
            }
            if !entered_inner.replace(true) {
                structural_visit(&inner, |value: i64, _visitor: &mut VisitContext<'_, ()>| {
                    inner_values.borrow_mut().push(value);
                })?;
            }
            visitor.visit_children()
        },
    )
    .unwrap()
    .is_none());
    assert_eq!(*outer_values.borrow(), vec![10, 20]);
    assert_eq!(*inner_values.borrow(), vec![1, 2]);
}

#[test]
fn callback_visit_panics_resume_and_leave_the_next_run_usable() {
    let root = Array::new(vec![1i64]);
    let entered_callback = Cell::new(false);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        structural_visit(
            &root,
            |_value: i64, _visitor: &mut VisitContext<'_, ()>| -> () {
                entered_callback.set(true);
                panic!("callback visitor panic")
            },
        )
    }));
    assert!(
        entered_callback.get(),
        "panicking callback visitor was never called"
    );
    let panic = match outcome {
        Err(panic) => panic,
        Ok(Err(error)) => panic!("panicking callback visitor returned an error: {error}"),
        Ok(Ok(_)) => panic!("panicking callback visitor unexpectedly returned"),
    };
    assert_eq!(
        panic.downcast_ref::<&str>().copied(),
        Some("callback visitor panic")
    );

    let calls = Cell::new(0);
    assert!(
        structural_visit(&root, |_value: i64, _visitor: &mut VisitContext<'_, ()>| {
            calls.set(calls.get() + 1);
        })
        .unwrap()
        .is_none()
    );
    assert_eq!(calls.get(), 1);
}
