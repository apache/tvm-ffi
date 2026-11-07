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
use tvm_ffi::reflection::{FieldGetter, TypeAttrColumn};
use tvm_ffi::ObjectRefCore;
use tvm_ffi::{
    dispatch, structural_visit, structural_walk, Any, AnyView, Array, DLDataType, DLDataTypeCode,
    DefRegionKind, Dispatcher, Error, Function, Map, Object, Result, String as FfiString,
    StructuralVisitorView, TypeIndex, VisitInterrupt, WalkOrder, WalkResult, RUNTIME_ERROR,
};

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

// Nested recursion restores regions after normal, interrupt, error and panic exits.
#[test]
fn default_descent_scopes_regions_and_restores_after_halts() {
    #[derive(Clone, Copy)]
    enum Outcome {
        Finish,
        Interrupt,
        Error,
    }
    struct Probe {
        outcome: Outcome,
        depth: usize,
        seen: Vec<(i64, DefRegionKind)>,
        regions: Vec<DefRegionKind>,
    }
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl Probe {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let outer = visitor.def_region_kind();
            if let Some(integer) = value.downcast_ref::<i64>().copied() {
                self.seen.push((integer, outer));
                return match (integer, self.outcome) {
                    (1, Outcome::Interrupt) => Ok(Some(VisitInterrupt::new(42_i64))),
                    (1, Outcome::Error) => Err(runtime_error("descent failed")),
                    _ => Ok(None),
                };
            }
            use DefRegionKind::{None as Use, Pattern, Simple};
            assert_eq!(
                visitor.with_def_region_kind(Pattern, |visitor| visitor.def_region_kind()),
                Pattern
            );
            assert_eq!(visitor.def_region_kind(), outer);
            let level = self.depth;
            if level == 0 && matches!(self.outcome, Outcome::Finish) {
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    visitor.with_def_region_kind(Pattern, |visitor| -> Result<()> {
                        assert_eq!(visitor.def_region_kind(), Pattern);
                        panic!("scoped visit panic");
                    })
                }));
                assert!(panic.is_err());
                assert_eq!(visitor.def_region_kind(), outer);
            }
            self.depth += 1;
            self.regions.push(outer);
            let result = if level < 2 {
                visitor.with_def_region_kind([Simple, Pattern][level], |visitor| {
                    visitor.visit(self, value)
                })
            } else {
                visitor.with_def_region_kind(Use, |visitor| {
                    let current = value;
                    let result = visitor.default_visit(self, current);
                    assert_eq!(
                        current.downcast_ref::<Array<i64>>().cloned().unwrap().len(),
                        2
                    );
                    result
                })
            };
            self.depth -= 1;
            assert_eq!(visitor.def_region_kind(), outer);
            assert_eq!(
                value.downcast_ref::<Array<i64>>().cloned().unwrap().len(),
                2
            );
            assert!(visitor.visit(self, &[4_i64, 3, 2][level])?.is_none());
            result
        }
    }
    use DefRegionKind::{None as Use, Pattern, Simple};
    for outcome in [Outcome::Finish, Outcome::Interrupt, Outcome::Error] {
        let mut visitor = Probe {
            outcome,
            depth: 0,
            seen: vec![],
            regions: vec![],
        };
        let result = structural_visit(&Array::from_vec(vec![1_i64, 5]), &mut visitor);

        match outcome {
            Outcome::Finish => assert!(result.unwrap().is_none()),
            Outcome::Interrupt => {
                assert_eq!(
                    result
                        .unwrap()
                        .unwrap()
                        .value
                        .downcast_ref::<i64>()
                        .copied()
                        .unwrap(),
                    42
                )
            }
            Outcome::Error => assert!(result.err().unwrap().to_string().contains("descent failed")),
        }
        assert_eq!(visitor.regions, vec![Use, Simple, Pattern]);
        assert_eq!(visitor.depth, 0);
        let mut expected = vec![(1, Pattern)];
        if matches!(outcome, Outcome::Finish) {
            expected.push((5, Pattern));
        }
        expected.extend([(2, Pattern), (3, Simple), (4, Use)]);
        assert!(structural_visit(&99_i64, &mut visitor).unwrap().is_none());
        expected.push((99, Use));
        assert_eq!(visitor.seen, expected);
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

#[test]
fn default_descent_retargets_without_dispatching_the_container() {
    #[derive(Default)]
    struct Probe(Vec<(&'static str, i64, DefRegionKind)>, Option<Any>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl Probe {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let kind = visitor.def_region_kind();
            if value.downcast_ref::<i64>().copied() == Some(-1) {
                assert!(visitor
                    .with_def_region_kind(DefRegionKind::Pattern, |visitor| visitor
                        .default_visit(self, &Any::new()))?
                    .is_none());
                let target = self.1.as_ref().unwrap().clone();
                let result = visitor.with_def_region_kind(DefRegionKind::Pattern, |visitor| {
                    visitor.default_visit(self, &target)
                });
                assert_eq!(value.downcast_ref::<i64>().copied(), Some(-1));
                assert_eq!(visitor.def_region_kind(), DefRegionKind::None);
                return result;
            }
            if let Some(array) = value.downcast_ref::<Array<Any>>().cloned() {
                assert_eq!(
                    array.len(),
                    2,
                    "default descent must skip the target container's handler"
                );
                self.0.push(("array", 2, kind));
                let result = visitor.with_def_region_kind(DefRegionKind::Pattern, |visitor| {
                    visitor.default_visit(self, &array.get(0).unwrap())
                });
                assert_eq!(
                    value.downcast_ref::<Array<Any>>().cloned().unwrap().len(),
                    2
                );
                assert_eq!(visitor.def_region_kind(), DefRegionKind::None);
                self.0.push(("after", 2, visitor.def_region_kind()));
                result
            } else {
                self.0.push((
                    "integer",
                    value.downcast_ref::<i64>().copied().unwrap(),
                    kind,
                ));
                Ok(None)
            }
        }
    }
    let root = Array::from_vec(vec![
        Any::from(Array::from_vec(vec![7_i64])),
        Any::from(99_i64),
    ]);
    let mut visitor = Probe::default();
    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(
        visitor.0,
        vec![
            ("array", 2, DefRegionKind::None),
            ("integer", 7, DefRegionKind::Pattern),
            ("after", 2, DefRegionKind::None)
        ]
    );
    let mut visitor = Probe(vec![], Some(visit_region_graph(false)));
    assert!(structural_visit(&-1_i64, &mut visitor).unwrap().is_none());
    assert_eq!(
        visitor.0,
        vec![
            ("integer", 1, DefRegionKind::Pattern),
            ("integer", 2, DefRegionKind::Pattern),
            ("integer", 3, DefRegionKind::Pattern)
        ]
    );
}

#[test]
fn regions_compose_with_field_flags_and_function_hooks() {
    struct Trace {
        root_type: i32,
        region: DefRegionKind,
        seen: Vec<(i64, DefRegionKind)>,
    }
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl Trace {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let kind = visitor.def_region_kind();
            if let Some(integer) = value.downcast_ref::<i64>().copied() {
                self.seen.push((integer, kind));
                Ok(None)
            } else {
                let requested = if value.type_index() == self.root_type {
                    self.region
                } else {
                    kind
                };
                visitor
                    .with_def_region_kind(requested, |visitor| visitor.default_visit(self, value))
            }
        }
    }
    use DefRegionKind::{None as Use, Pattern, Simple};
    for with_hook in [false, true] {
        let root = visit_region_graph(with_hook);
        if with_hook {
            assert!(Function::try_from(
                TypeAttrColumn::lookup("__s_visit__")
                    .and_then(|column| column.get(root.type_index()))
                    .unwrap()
            )
            .is_ok());
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
            let mut visitor = Trace {
                root_type: root.type_index(),
                region,
                seen: vec![],
            };
            assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
            assert_eq!(visitor.seen, expected);
            if region == Use {
                for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
                    let mut seen = vec![];
                    assert!(structural_walk(
                        &root,
                        |value: i64, kind: DefRegionKind| {
                            seen.push((value, kind));
                            WalkResult::Advance
                        },
                        order
                    )
                    .unwrap()
                    .is_none());
                    assert_eq!(seen, expected);
                }
            }
        }
    }
}

#[test]
fn reflected_pattern_precedes_default_descent() {
    struct Trace(Vec<(i64, DefRegionKind)>, DefRegionKind);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl Trace {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let kind = visitor.def_region_kind();
            match value.downcast_ref::<i64>().copied() {
                Some(integer) => {
                    self.0.push((integer, kind));
                    match integer {
                        2 => {
                            let requested = self.1;
                            visitor.with_def_region_kind(requested, |visitor| {
                                visitor.visit(self, &99_i64)
                            })
                        }
                        3 => visitor.visit(self, &100_i64),
                        _ => Ok(None),
                    }
                }
                None => {
                    visitor.with_def_region_kind(kind, |visitor| visitor.default_visit(self, value))
                }
            }
        }
    }
    let root = visit_region_graph(false);
    use DefRegionKind::{None as Use, Pattern, Simple};
    for requested in [Use, Simple] {
        let mut visitor = Trace(vec![], requested);
        assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
        assert_eq!(
            visitor.0,
            vec![
                (1, Simple),
                (2, Pattern),
                (99, Pattern),
                (3, Use),
                (100, Use)
            ]
        );
    }
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut seen = vec![];
        assert!(structural_walk(
            &root,
            |value: i64, kind: DefRegionKind| {
                seen.push((value, kind));
                WalkResult::Advance
            },
            order
        )
        .unwrap()
        .is_none());
        assert_eq!(seen, vec![(1, Simple), (2, Pattern), (3, Use)]);
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

    let wrong_type = Array::from_vec(vec![0i64]);
    assert!(getter.get_any(&**Array::data(&wrong_type)).is_err());
    assert!(FieldGetter::new(type_index, "missing").is_err());

    // ObjectDef registers this Function-valued attribute for copyable C++ types.
    assert!(Function::try_from(
        TypeAttrColumn::lookup("__ffi_shallow_copy__")
            .and_then(|column| column.get(type_index))
            .unwrap()
    )
    .is_ok());
    assert!(TypeAttrColumn::lookup("missing")
        .and_then(|column| column.get(type_index))
        .is_none());
}

#[test]
fn plain_walk_uses_registered_array_hook() {
    let root = Array::from_vec(vec![1i64, 2, 3]);
    let mut integers = 0;
    assert!(structural_walk(
        &root,
        |value: AnyView<'_>| {
            if value.downcast_ref::<i64>().copied().is_some() {
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
        |value: AnyView<'_>| {
            if value.downcast_ref::<i64>().copied().is_some() {
                integers += 1;
            } else if value.downcast_ref::<FfiString>().cloned().is_some() {
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
    assert!(TypeAttrColumn::lookup("__s_visit__")
        .and_then(|column| column.get(TypeIndex::kTVMFFIDataType as i32))
        .is_none());
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);

    let mut pre = Vec::new();
    assert!(structural_walk(
        &dtype,
        |value: AnyView<'_>| {
            if value.downcast_ref::<DLDataType>().copied().is_some() {
                pre.push("dtype");
            } else if value.downcast_ref::<i64>().copied().is_some() {
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
        |value: AnyView<'_>| {
            if value.downcast_ref::<DLDataType>().copied().is_some() {
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
        |value: AnyView<'_>| {
            if value.downcast_ref::<DLDataType>().copied().is_some() {
                post.push("dtype");
            } else if value.downcast_ref::<i64>().copied().is_some() {
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
        |_value: AnyView<'_>| {
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
        |_value: i64| WalkResult::InterruptWith(9i64.into()),
        WalkOrder::PostOrder,
    )
    .unwrap()
    .unwrap();
    assert_eq!(interrupt.value.downcast_ref::<i64>().copied().unwrap(), 9);

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
        |value: AnyView<'_>| {
            if let Some(integer) = value.downcast_ref::<i64>().copied() {
                sum += integer;
            } else if value.downcast_ref::<FfiString>().cloned().is_some() {
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
        |value: AnyView<'_>| {
            if value.downcast_ref::<i64>().copied().is_some() {
                return WalkResult::InterruptWith(99i64.into());
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("map walk unexpectedly completed");
    };
    assert_eq!(interrupt.value.downcast_ref::<i64>().copied().unwrap(), 99);
}

#[test]
fn handler_error_crosses_map_traversal() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64)].into_iter().collect();
    let error = match structural_walk(
        &root,
        |value: AnyView<'_>| -> Result<WalkResult> {
            if value.downcast_ref::<i64>().copied().is_some() {
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
    let root = Array::from_vec(vec![1i64, 2, 3]);
    let mut integers = 0;
    let outcome = structural_walk(
        &root,
        |value: AnyView<'_>| {
            if value.downcast_ref::<i64>().copied().is_some() {
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

/// StructuralVisitorView-layer traversal that overrides the def-region for one child and
/// inherits it for the next, mirroring a C++ visitor using
/// `WithDefRegionKind`.
#[derive(Default)]
struct ManualRegionVisitor {
    seen: Vec<DefRegionKind>,
}

#[dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl ManualRegionVisitor {
    fn visit_any(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let def_region_kind = visitor.def_region_kind();
        if let Some(array) = value.downcast_ref::<Array<i64>>().cloned() {
            // Override the state for exactly this child's subtree...
            let overridden = array.get(0).unwrap();
            if let Some(interrupt) = visitor
                .with_def_region_kind(DefRegionKind::Simple, |visitor| {
                    visitor.visit(self, &overridden)
                })?
            {
                return Ok(Some(interrupt));
            }
            // ...and forward the received state to inherit it.
            let inherited = array.get(1).unwrap();
            return visitor
                .with_def_region_kind(def_region_kind, |visitor| visitor.visit(self, &inherited));
        }
        if value.downcast_ref::<i64>().copied().is_some() {
            self.seen.push(def_region_kind);
        }
        Ok(None)
    }
}

#[test]
fn manual_child_visit_can_override_def_region() {
    let root = Array::from_vec(vec![7i64, 8]);
    let mut probe = ManualRegionVisitor::default();
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(probe.seen, vec![DefRegionKind::Simple, DefRegionKind::None]);
}

#[derive(Default)]
struct GeneratedLeafVisitor {
    integers: Vec<(i64, DefRegionKind)>,
}

#[dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl GeneratedLeafVisitor {
    fn visit_integer(
        &mut self,
        value: i64,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let kind = visitor.def_region_kind();
        if value == -1 {
            let root = Array::from_vec(vec![1_i64, 2]);
            assert!(
                Dispatcher::dispatch_tuple(&mut *self, (AnyView::from(&root), &mut *visitor))
                    .is_none()
            );
            assert!(
                Dispatcher::dispatch_tuple(&mut *self, (AnyView::from(&0_i64), &mut *visitor))
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            return Ok(None);
        }
        self.integers.push((value, kind));
        Ok(None)
    }
}

#[test]
fn generated_visitor_defaults_unmatched_values() {
    let root = Array::from_vec(vec![1i64, 2]);
    let mut visitor = GeneratedLeafVisitor::default();
    assert!(structural_visit(&-1_i64, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.integers, vec![(0, DefRegionKind::None)]);
    visitor.integers.clear();
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

#[dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl GeneratedRecursiveVisitor {
    fn visit_array(
        &mut self,
        array: Array<i64>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let kind = visitor.def_region_kind();
        self.events.push("enter:array".to_string());
        for value in array.iter() {
            if let Some(interrupt) =
                visitor.with_def_region_kind(kind, |visitor| visitor.visit(self, value))?
            {
                return Ok(Some(interrupt));
            }
        }
        self.events.push("exit:array".to_string());
        Ok(None)
    }

    fn visit_integer(&mut self, value: i64) -> Result<Option<VisitInterrupt>> {
        self.events.push(format!("int:{value}"));
        Ok((value == 2).then(|| VisitInterrupt::new(value)))
    }

    fn visit_other_object(&mut self, _value: &Object) -> Result<Option<VisitInterrupt>> {
        Ok(None)
    }
}

#[test]
fn generated_visitor_can_drive_recursion_through_mut_self() {
    let root = Array::from_vec(vec![1i64, 2, 3]);
    let mut visitor = GeneratedRecursiveVisitor::default();
    let interrupt = structural_visit(&root, &mut visitor).unwrap().unwrap();
    assert_eq!(interrupt.value.downcast_ref::<i64>().copied().unwrap(), 2);
    assert_eq!(visitor.events, vec!["enter:array", "int:1", "int:2"]);
}

#[derive(Default)]
struct GenericDispatchProbe {
    integers: Vec<i64>,
    objects: usize,
    catch_all: usize,
}

#[dispatch(walk, (AnyView<'_>, DefRegionKind), Result<WalkResult>)]
impl GenericDispatchProbe {
    fn walk_integer(&mut self, value: i64) -> Result<WalkResult> {
        self.integers.push(value);
        Ok(WalkResult::Advance)
    }

    // Trailing DefRegionKind: handlers may mix arities within one impl.
    fn walk_object(&mut self, _value: &tvm_ffi::Object, kind: DefRegionKind) -> Result<WalkResult> {
        assert_eq!(kind, DefRegionKind::None);
        self.objects += 1;
        Ok(WalkResult::Advance)
    }

    fn walk_any(&mut self, _value: tvm_ffi::AnyView<'_>) -> Result<WalkResult> {
        self.catch_all += 1;
        Ok(WalkResult::Advance)
    }
}

struct DispatchLabel(String);
struct DescribeDispatchProbe;

#[dispatch(describe, (AnyView<'_>, DispatchLabel, &str), String)]
impl DescribeDispatchProbe {
    fn describe_integer(&mut self, value: i64, prefix: DispatchLabel) -> String {
        format!("{}{value}", prefix.0)
    }

    fn describe_string(&mut self, value: tvm_ffi::String, prefix: DispatchLabel) -> String {
        format!("{}{}", prefix.0, value.as_str())
    }

    fn describe_object(&mut self, _: &Object, prefix: DispatchLabel, suffix: &str) -> String {
        format!("{}object{suffix}", prefix.0)
    }

    fn describe_any(&mut self, _: AnyView<'_>) -> String {
        "other".to_owned()
    }

    // Configuration on an ordinary helper does not make it a dispatch handler.
    #[cfg(any())]
    fn dispatch_describe(&self) {}
}

#[dispatch(describe, (AnyView<'_>, MissingType), MissingType)]
#[cfg_attr(all(), cfg_attr(all(), cfg(any())))]
impl MissingWholeImpl {
    fn describe_disabled(&mut self, _: MissingType, _: MissingType) -> MissingType {
        unreachable!()
    }
}

#[test]
fn generated_dispatch_supports_pod_and_ordered_catch_all() {
    let root = Array::from_vec(vec![1i64, 2]);
    let mut probe = GenericDispatchProbe::default();
    assert!(structural_walk(&root, &mut probe, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.integers, vec![1, 2]);
    assert_eq!(probe.objects, 1);

    let floats = Array::from_vec(vec![1.0f64, 2.0]);
    assert!(structural_walk(&floats, &mut probe, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.objects, 2);
    assert_eq!(probe.catch_all, 2);

    let mut describe = DescribeDispatchProbe;
    let label = || DispatchLabel("value:".to_owned());
    assert_eq!(
        describe.dispatch(AnyView::from(&3_i64), label(), "!"),
        Some("value:3".to_owned())
    );
    let text = tvm_ffi::String::from("hello");
    assert_eq!(
        Dispatcher::<(_, _, _), Option<String>>::dispatch_tuple(
            &mut describe,
            (AnyView::from(&text), label(), "!")
        ),
        Some("value:hello".to_owned())
    );
    assert_eq!(
        Dispatcher::<(_, _, _), Option<String>>::dispatch_tuple(
            &mut describe,
            (AnyView::from(&root), label(), "!")
        ),
        Some("value:object!".to_owned())
    );
    assert_eq!(
        Dispatcher::<(_, _, _), Option<String>>::dispatch_tuple(
            &mut describe,
            (AnyView::from(&true), label(), "!")
        ),
        Some("other".to_owned())
    );
}

/// StructuralVisitorView-layer enter/exit straddling: run enter logic, delegate the
/// default child recursion, then run exit logic with the same locals in
/// scope — the C++ `DefaultVisitExpected` pattern.
#[derive(Default)]
struct StraddleVisitor {
    events: Vec<String>,
}

#[dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl StraddleVisitor {
    fn visit_any(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let def_region_kind = visitor.def_region_kind();
        let label = match value.downcast_ref::<i64>().copied() {
            Some(integer) => format!("int:{integer}"),
            None => "node".to_string(),
        };
        self.events.push(format!("enter:{label}"));
        if let Some(interrupt) = visitor.with_def_region_kind(def_region_kind, |visitor| {
            visitor.default_visit(self, value)
        })? {
            return Ok(Some(interrupt));
        }
        self.events.push(format!("exit:{label}"));
        Ok(None)
    }
}

#[test]
fn visitor_can_straddle_default_children() {
    let root = Array::from_vec(vec![1i64, 2]);
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

#[dispatch(walk, (AnyView<'_>, DefRegionKind), Result<WalkResult>)]
impl OrderProbe {
    fn walk_array(&mut self, _array: Array<i64>, _kind: DefRegionKind) -> Result<WalkResult> {
        self.events.push("array".to_string());
        Ok(WalkResult::Advance)
    }

    fn walk_integer(&mut self, value: i64) -> Result<WalkResult> {
        self.events.push(format!("int:{value}"));
        Ok(WalkResult::Advance)
    }
}

#[test]
fn stateful_structural_walk_supports_post_order() {
    let root = Array::from_vec(vec![1i64, 2]);
    let mut probe = OrderProbe::default();
    assert!(structural_walk(&root, &mut probe, WalkOrder::PostOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.events, vec!["int:1", "int:2", "array"]);
}

#[test]
fn nested_walk_restores_the_outer_active_visitor() {
    let outer = Array::from_vec(vec![10i64, 20]);
    let inner = Array::from_vec(vec![1i64, 2]);
    let mut entered_inner = false;
    let mut outer_values = Vec::new();
    let mut inner_values = Vec::new();

    assert!(structural_walk(
        &outer,
        |value: AnyView<'_>| -> Result<WalkResult> {
            if let Some(value) = value.downcast_ref::<i64>().copied() {
                outer_values.push(value);
            }
            if !entered_inner {
                entered_inner = true;
                structural_walk(
                    &inner,
                    |value: AnyView<'_>| {
                        if let Some(value) = value.downcast_ref::<i64>().copied() {
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
    let root = Array::from_vec(vec![1i64, 2]);
    let outcome = structural_walk(
        &root,
        |value: AnyView<'_>| {
            if value.downcast_ref::<i64>().copied() == Some(1) {
                return WalkResult::InterruptWith(42i64.into());
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("walk unexpectedly completed");
    };
    assert_eq!(interrupt.value.downcast_ref::<i64>().copied().unwrap(), 42);
}

#[test]
fn handler_errors_include_native_visit_path() {
    let root = Array::from_vec(vec![1i64]);
    let error = match structural_walk(
        &root,
        |value: AnyView<'_>| -> Result<WalkResult> {
            if value.downcast_ref::<i64>().copied().is_some() {
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

    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl FailingVisitor {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let def_region_kind = visitor.def_region_kind();
            if value.downcast_ref::<i64>().copied().is_some() {
                return Err(runtime_error("visitor failed"));
            }
            visitor.with_def_region_kind(def_region_kind, |visitor| {
                visitor.default_visit(self, value)
            })
        }
    }

    let root = Array::from_vec(vec![1i64]);
    let error = match structural_visit(&root, &mut FailingVisitor) {
        Err(error) => error,
        Ok(_) => panic!("visitor unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "visitor failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));

    struct FailingLeaf;
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl FailingLeaf {
        fn visit_integer(
            &mut self,
            _: i64,
            _: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            Err(runtime_error("callback visitor failed"))
        }
    }
    let error = match structural_visit(&root, FailingLeaf) {
        Err(error) => error,
        Ok(_) => panic!("callback visitor unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "callback visitor failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn visitor_interrupt_propagates_through_default_children() {
    struct InterruptingVisitor;

    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl InterruptingVisitor {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let def_region_kind = visitor.def_region_kind();
            if value.downcast_ref::<i64>().copied() == Some(2) {
                return Ok(Some(VisitInterrupt::new(7i64)));
            }
            visitor.with_def_region_kind(def_region_kind, |visitor| {
                visitor.default_visit(self, value)
            })
        }
    }

    let root = Array::from_vec(vec![1i64, 2, 3]);
    let outcome =
        structural_visit::<&Array<i64>, _, _, _, _>(&root, &mut InterruptingVisitor).unwrap();
    let Some(interrupt) = outcome else {
        panic!("visitor traversal unexpectedly completed");
    };
    assert_eq!(interrupt.value.downcast_ref::<i64>().copied().unwrap(), 7);
}

#[test]
fn closure_walk_receives_def_region_kind() {
    // C++: StructuralWalk<kPreOrder>(root,
    //          [&](const TVarObj* var, TVMFFIDefRegionKind kind) { ... })
    let root = Array::from_vec(vec![1i64, 2]);
    let mut kinds = Vec::new();
    assert!(structural_walk(
        &root,
        |value: AnyView<'_>, kind: DefRegionKind| {
            if value.downcast_ref::<i64>().copied().is_some() {
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
    let root = Array::from_vec(vec![1i64, 2]);
    let mut order_probe = Vec::new();
    assert!(structural_walk(
        &root,
        |value: AnyView<'_>| {
            order_probe.push(value.downcast_ref::<i64>().copied());
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
        |value: AnyView<'_>| {
            visited += 1;
            if value.downcast_ref::<i64>().copied().is_none() {
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
    let root = Array::from_vec(vec![
        Array::from_vec(vec![1i64]),
        Array::from_vec(vec![2i64, 3]),
    ]);
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
    let root = Array::from_vec(vec![1i64, 2]);
    let mut kinds = Vec::new();
    let mut objects = 0;
    assert!(structural_walk(
        &root,
        (
            |_value: i64, kind: DefRegionKind| {
                kinds.push(kind);
                WalkResult::Advance
            },
            |_value: AnyView<'_>, kind: DefRegionKind| {
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
    let root = Array::from_vec(vec![
        Array::from_vec(vec![1i64]),
        Array::from_vec(vec![2i64]),
    ]);
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
    let root = Array::from_vec(vec![1i64]);
    let error = match structural_walk(
        &root,
        (
            |_value: i64| -> Result<WalkResult> { Err(runtime_error("link failed")) },
            |_value: AnyView<'_>| WalkResult::Advance,
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
    let root = Array::from_vec(vec![1i64, 2]);
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

#[dispatch(walk, (AnyView<'_>, DefRegionKind), Result<WalkResult>)]
impl ObjectCounter {
    fn walk_object(&mut self, _value: &Object, _kind: DefRegionKind) -> Result<WalkResult> {
        self.objects += 1;
        Ok(WalkResult::Advance)
    }
}

#[test]
fn chain_splices_dispatch_walkers_between_closures() {
    let root = Array::from_vec(vec![1i64, 2]);
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
    let root = Array::from_vec(vec![1i64, 2, 3]);
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
            |_value: AnyView<'_>, _kind: DefRegionKind| {
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
    let root = Array::from_vec(vec![1i64, 2, 3]);
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
    let root = Array::from_vec(vec![1i64, 2]);
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

#[dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl InheritedRegionProbe {
    fn visit_any(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let def_region_kind = visitor.def_region_kind();
        if self.at_root {
            self.at_root = false;
            let outer = value.downcast_ref::<Array<Array<i64>>>().cloned().unwrap();
            let inner = outer.get(0).unwrap();
            return visitor.with_def_region_kind(DefRegionKind::Pattern, |visitor| {
                visitor.visit(self, &inner)
            });
        }
        self.seen.push(def_region_kind);
        visitor.with_def_region_kind(def_region_kind, |visitor| {
            visitor.default_visit(self, value)
        })
    }
}

#[test]
fn def_region_is_inherited_through_containers() {
    let root = Array::from_vec(vec![Array::from_vec(vec![1i64, 2])]);
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
    let root = Array::from_vec(vec![1i64, 2, 3]);
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
                    (|_value: AnyView<'_>, _kind: DefRegionKind| {
                        others += 1;
                        WalkResult::Advance
                    },),
                ),
                |_value: AnyView<'_>| WalkResult::Advance,
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
    let root = Array::from_vec(vec![1i64, 2]);
    let mut first = 0;
    let mut second = 0;
    assert!(structural_walk(
        &root,
        (
            (|_value: AnyView<'_>| {
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
    struct Integers<'a>(&'a Cell<i64>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> Integers<'a> {
        fn visit_integer(
            &mut self,
            value: i64,
            _: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.0.set(self.0.get() + value);
            Ok(None)
        }
    }
    struct SkipArray<'a>(&'a Cell<i64>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> SkipArray<'a> {
        fn visit_array(
            &mut self,
            _: Array<i64>,
            _: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            Ok(None)
        }
        fn visit_integer(&mut self, value: i64) -> Result<Option<VisitInterrupt>> {
            self.0.set(self.0.get() + value);
            Ok(None)
        }
    }
    let root = Array::from_vec(vec![1i64, 2]);
    let integers = Cell::new(0);
    assert!(structural_visit(&root, Integers(&integers),)
        .unwrap()
        .is_none());

    assert_eq!(integers.get(), 3);
    let integers = Cell::new(0);
    assert!(structural_visit(&root, SkipArray(&integers),)
        .unwrap()
        .is_none());

    assert_eq!(integers.get(), 0);
}

#[derive(Default)]
struct StatefulVisitStats {
    arrays: usize,
    integer_sum: i64,
}

#[dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl StatefulVisitStats {
    fn visit_array(
        &mut self,
        array: Array<i64>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        self.arrays += 1;
        assert!(AnyView::from(&array)
            .downcast_ref::<Array<i64>>()
            .cloned()
            .is_some());
        visitor.default_visit(self, &array)
    }
    fn visit_integer(&mut self, value: i64) -> Result<Option<VisitInterrupt>> {
        self.integer_sum += value;
        Ok(None)
    }
}

#[test]
fn stateful_callback_visit_uses_ordinary_mutable_state() {
    let root = Array::from_vec(vec![1i64, 2, 3]);
    let mut visitor = StatefulVisitStats::default();

    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.arrays, 1);
    assert_eq!(visitor.integer_sum, 6);

    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.arrays, 2);
    assert_eq!(visitor.integer_sum, 12);
}

#[derive(Default)]
struct StatefulVisitDepth {
    current: usize,
    maximum: usize,
    calls: usize,
}

#[dispatch(
    visit,
    (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
    Result<Option<VisitInterrupt>>
)]
impl StatefulVisitDepth {
    fn visit_any(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        self.current += 1;
        self.calls += 1;
        let current = self.current;
        self.maximum = self.maximum.max(current);
        let outcome = visitor.default_visit(self, value);
        self.current -= 1;
        outcome
    }
}

#[test]
fn stateful_callback_visit_reborrows_visitor_during_recursion() {
    let root = Array::from_vec(vec![Array::from_vec(vec![1i64, 2])]);
    let mut visitor = StatefulVisitDepth::default();

    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.current, 0);
    assert_eq!(visitor.maximum, 3);
    assert_eq!(visitor.calls, 4);
}

#[test]
fn callback_visit_can_reenter_the_same_fn_through_visitor() {
    struct Reenter<'a>(&'a Cell<i32>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> Reenter<'a> {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.0.set(self.0.get() + 1);
            visitor.default_visit(self, value)
        }
    }
    let root = Array::from_vec(vec![1i64, 2]);
    let visits = Cell::new(0);
    assert!(structural_visit(&root, Reenter(&visits)).unwrap().is_none());
    assert_eq!(visits.get(), 3);
}

#[test]
fn callback_visit_tuple_is_first_match_and_can_interrupt() {
    type View<'value> = AnyView<'value>;
    struct Stop<'a>(&'a Cell<i32>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> Stop<'a> {
        fn visit_integer(
            &mut self,
            value: i64,
            _: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            Ok((value == 2).then(|| VisitInterrupt::new(value)))
        }
        fn visit_any<'value>(
            &mut self,
            value: View<'value>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.0.set(self.0.get() + 1);
            visitor.default_visit(self, value)
        }
    }
    let root = Array::from_vec(vec![1i64, 2, 3]);
    let fallback = Cell::new(0);
    let interrupted = structural_visit(&root, Stop(&fallback)).unwrap().unwrap();
    assert_eq!(interrupted.value.downcast_ref::<i64>().copied().unwrap(), 2);
    assert_eq!(fallback.get(), 1);
}

#[test]
fn callback_visit_supports_node_links_and_nested_tuples() {
    struct Nodes<'a>(&'a RefCell<Vec<(i64, DefRegionKind)>>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> Nodes<'a> {
        fn visit_float(
            &mut self,
            _: f64,
            _: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            Ok(None)
        }
        fn visit_integer(
            &mut self,
            value: i64,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.0.borrow_mut().push((value, visitor.def_region_kind()));
            Ok(None)
        }
        fn visit_node(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let _node = value
                .downcast_ref::<tvm_ffi::collections::array::ArrayObj>()
                .unwrap();
            assert_eq!(visitor.def_region_kind(), DefRegionKind::None);
            visitor.default_visit(self, value)
        }
    }
    let root = Array::from_vec(vec![
        Any::from(Array::from_vec(vec![1_i64])),
        Any::from(2_i64),
    ]);
    let seen = RefCell::new(Vec::new());
    assert!(structural_visit(&root, Nodes(&seen)).unwrap().is_none());
    assert_eq!(
        *seen.borrow(),
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None)]
    );
}

#[test]
fn callback_visit_with_overrides_child_def_region() {
    struct Regions<'a>(&'a RefCell<Vec<(i64, DefRegionKind)>>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> Regions<'a> {
        fn visit_array(
            &mut self,
            array: Array<i64>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            for value in array.iter() {
                if let Some(interrupt) = visitor
                    .with_def_region_kind(DefRegionKind::Pattern, |visitor| {
                        visitor.visit(self, value)
                    })?
                {
                    return Ok(Some(interrupt));
                }
            }
            Ok(None)
        }
        fn visit_integer(
            &mut self,
            value: i64,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.0.borrow_mut().push((value, visitor.def_region_kind()));
            Ok(None)
        }
    }
    let root = Array::from_vec(vec![1i64, 2]);
    let seen = RefCell::new(Vec::new());
    assert!(structural_visit(&root, Regions(&seen)).unwrap().is_none());
    assert_eq!(
        *seen.borrow(),
        vec![(1, DefRegionKind::Pattern), (2, DefRegionKind::Pattern)]
    );
}

#[test]
fn nested_callback_visit_restores_the_outer_active_visitor() {
    #[repr(align(128))]
    struct Aligned {
        checked: bool,
        calls: usize,
    }
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl Aligned {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            assert_eq!(std::ptr::from_ref(self).addr() % 128, 0);
            self.calls += 1;
            if !self.checked {
                self.checked = true;
                let mut other = Aligned {
                    checked: true,
                    calls: 0,
                };
                for default in [false, true] {
                    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if default {
                            visitor.default_visit(&mut other, value)
                        } else {
                            visitor.visit(&mut other, value)
                        }
                    }));
                    assert!(rejected.is_err());
                    assert_eq!(other.calls, 0);
                }
            }
            visitor.default_visit(self, value)
        }
    }
    let aligned_root = Array::from_vec(vec![3_i64, 4]);
    let mut borrowed = Aligned {
        checked: false,
        calls: 0,
    };
    structural_visit(&aligned_root, &mut borrowed).unwrap();
    assert_eq!(borrowed.calls, 3);

    structural_visit(
        &aligned_root,
        Aligned {
            checked: false,
            calls: 0,
        },
    )
    .unwrap();

    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let observed = RefCell::new(Vec::new());
        structural_walk(
            &Array::from_vec(vec![Any::from(Array::from_vec(vec![1_i64]))]),
            |value: i64| {
                observed.borrow_mut().push(value);
                WalkResult::Advance
            },
            order,
        )
        .unwrap();
        assert_eq!(*observed.borrow(), vec![1]);
    }

    struct Inner<'a>(&'a RefCell<Vec<i64>>);
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> Inner<'a> {
        fn visit_integer(
            &mut self,
            value: i64,
            _: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.0.borrow_mut().push(value);
            Ok(None)
        }
    }
    struct Outer<'a> {
        inner: &'a Array<i64>,
        entered: &'a Cell<bool>,
        outer_values: &'a RefCell<Vec<i64>>,
        inner_values: &'a RefCell<Vec<i64>>,
    }
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl<'a> Outer<'a> {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            if let Some(value) = value.downcast_ref::<i64>().copied() {
                self.outer_values.borrow_mut().push(value);
            }
            if !self.entered.replace(true) {
                structural_visit(self.inner, Inner(self.inner_values))?;
            }
            visitor.default_visit(self, value)
        }
    }
    let outer = Array::from_vec(vec![10i64, 20]);
    let inner = Array::from_vec(vec![1i64, 2]);
    let entered_inner = Cell::new(false);
    let outer_values = RefCell::new(Vec::new());
    let inner_values = RefCell::new(Vec::new());
    assert!(structural_visit(
        &outer,
        Outer {
            inner: &inner,
            entered: &entered_inner,
            outer_values: &outer_values,
            inner_values: &inner_values,
        }
    )
    .unwrap()
    .is_none());
    assert_eq!(*outer_values.borrow(), vec![10, 20]);
    assert_eq!(*inner_values.borrow(), vec![1, 2]);
}

#[test]
fn copied_views_forward_borrows_and_retain_only_explicit_owners() {
    let root = Any::from(Array::from_vec(vec![1_i64, 2]));
    let view = AnyView::from(&root);
    struct Identity;
    #[dispatch(identity, (AnyView<'a>,), AnyView<'a>)]
    impl Identity {
        fn identity_any<'a>(&self, value: AnyView<'a>) -> AnyView<'a> {
            value
        }
    }
    let copy = Identity.dispatch_tuple((view,)).unwrap();
    assert_eq!(copy.debug_strong_count(), Some(1));
    let mut retained = None;
    structural_walk(
        copy,
        |value: AnyView<'_>| {
            let copy = value;
            if let Some(node) = copy.downcast_ref::<tvm_ffi::collections::array::ArrayObj>() {
                let borrowed: AnyView<'_> = node.into();
                let base: AnyView<'_> = copy.downcast_ref::<Object>().unwrap().into();
                assert_eq!(borrowed.type_index(), copy.type_index());
                assert_eq!(base.type_index(), copy.type_index());
                assert_eq!(borrowed.debug_strong_count(), Some(1));
                let typed = copy.downcast_ref::<Array<i64>>().cloned().unwrap();
                assert_eq!(copy.debug_strong_count(), Some(2));
                drop(typed);
                retained = Some(Any::from(copy));
                assert_eq!(value.debug_strong_count(), Some(2));
            } else {
                assert!(copy.downcast_ref::<tvm_ffi::Object>().is_none());
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    struct Descend;
    #[dispatch(
        visit,
        (AnyView<'_>, &mut StructuralVisitorView<'_, Self>),
        Result<Option<VisitInterrupt>>
    )]
    impl Descend {
        fn visit_array(
            &mut self,
            node: &tvm_ffi::collections::array::ArrayObj,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            visitor.with_def_region_kind(DefRegionKind::None, |visitor| {
                visitor.default_visit(self, node)
            })
        }
    }
    structural_visit(
        view.downcast_ref::<tvm_ffi::collections::array::ArrayObj>()
            .unwrap(),
        Descend,
    )
    .unwrap();
    drop(root);
    let retained = Array::<i64>::try_from(retained.unwrap()).unwrap();
    assert_eq!(retained.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
}

#[test]
fn generic_dispatch_distinguishes_no_match_from_handler_outcomes() {
    struct Outcomes;
    #[dispatch(walk, (AnyView<'_>,), std::result::Result<Option<i64>, &'static str>)]
    impl Outcomes {
        fn walk_integer(&self, _: i64) -> std::result::Result<Option<i64>, &'static str> {
            Err("matched error")
        }
        fn walk_integer_again(&self, _: i64) -> std::result::Result<Option<i64>, &'static str> {
            panic!("a matched error must stop dispatch")
        }
        fn walk_bool(&self, value: bool) -> std::result::Result<Option<i64>, &'static str> {
            Ok(value.then_some(1))
        }
        fn walk_bool_again(&self, _: bool) -> std::result::Result<Option<i64>, &'static str> {
            panic!("a matched inner None must stop dispatch")
        }
    }
    assert_eq!(
        Outcomes.dispatch(AnyView::from(&1_i64)),
        Some(Err("matched error"))
    );
    assert_eq!(Outcomes.dispatch(AnyView::from(&false)), Some(Ok(None)));
    assert_eq!(
        Outcomes.dispatch_tuple((AnyView::from(&true),)),
        Some(Ok(Some(1)))
    );
    assert_eq!(Outcomes.dispatch_tuple((AnyView::from(&1.0_f64),)), None);
}

#[test]
fn managed_walk_preserves_observer_address_and_drop_on_all_exits() {
    #[repr(align(64))]
    struct Storage<'a, const N: usize> {
        _padding: [u8; N],
        address: &'a Cell<usize>,
        drops: &'a Cell<usize>,
    }
    impl<const N: usize> Drop for Storage<'_, N> {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }
    fn run<const N: usize>(mode: usize) {
        let address = Cell::new(0);
        let drops = Cell::new(0);
        let root = Array::from_vec(vec![1_i64, 2]);
        let storage = Storage {
            _padding: [0_u8; N],
            address: &address,
            drops: &drops,
        };
        let outcome = structural_walk(
            &root,
            move |value: i64| -> Result<WalkResult> {
                let current = std::ptr::from_ref(&storage).addr();
                assert_eq!(current % 64, 0);
                if storage.address.get() == 0 {
                    storage.address.set(current);
                }
                assert_eq!(storage.address.get(), current);
                if value != 2 {
                    return Ok(WalkResult::Advance);
                }
                match mode {
                    2 => Err(runtime_error("storage error")),
                    3 => Ok(WalkResult::InterruptWith(Any::new())),
                    4 => Ok(WalkResult::InterruptWith(
                        runtime_error("payload error").into(),
                    )),
                    _ => Ok(WalkResult::Advance),
                }
            },
            WalkOrder::PreOrder,
        );
        match outcome {
            Err(error) => {
                assert_eq!(mode, 2);
                assert!(error.message().contains("storage error"));
            }
            Ok(Some(interrupt)) => {
                assert!(mode == 3 || mode == 4);
                assert_eq!(
                    interrupt.value.type_index(),
                    if mode == 3 {
                        TypeIndex::kTVMFFINone
                    } else {
                        TypeIndex::kTVMFFIError
                    } as i32
                );
            }
            Ok(None) => assert_eq!(mode, 0),
        }
        assert_ne!(address.get(), 0);
        assert_eq!(drops.get(), 1);
    }
    for mode in [0, 2, 3, 4] {
        run::<0>(mode);
        run::<4096>(mode);
    }
}

#[test]
fn visitor_preserves_post_recursion_state() {
    let root = Array::from_vec(vec![
        Array::from_vec(vec![1_i64, 2]),
        Array::from_vec(vec![3, 4]),
    ]);
    #[derive(Default)]
    struct PostState {
        depth: usize,
        leaves: Vec<i64>,
        completed: usize,
    }
    #[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorView<'_, Self>), Result<Option<VisitInterrupt>>)]
    impl PostState {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let depth = self.depth;
            self.depth += 1;
            let result = visitor.default_visit(self, value)?;
            self.depth -= 1;
            assert_eq!(self.depth, depth);
            self.completed += 1;
            if let Some(value) = value.downcast_ref::<i64>().copied() {
                self.leaves.push(value);
            }
            Ok(result)
        }
    }
    let mut state = PostState::default();
    structural_visit(&root, &mut state).unwrap();
    assert_eq!(state.depth, 0);
    assert_eq!(state.completed, 7);
    assert_eq!(state.leaves, vec![1, 2, 3, 4]);
}

#[test]
fn nested_walk_returns_one_owned_interrupt_payload_and_releases_it() {
    let root = Array::from_vec(vec![
        Array::from_vec(vec![1_i64, 2]),
        Array::from_vec(vec![3]),
    ]);
    let payload = Any::from(Array::from_vec(vec![42_i64]));
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut visited = Vec::new();
        let interrupt = structural_walk(
            &root,
            |value: i64| {
                visited.push(value);
                WalkResult::InterruptWith(payload.clone())
            },
            order,
        )
        .unwrap()
        .unwrap();
        assert_eq!(visited, vec![1]);
        assert_eq!(payload.debug_strong_count(), Some(2));
        let original = payload.downcast_ref::<ObjectRef>().cloned().unwrap();
        let returned = interrupt
            .value
            .downcast_ref::<ObjectRef>()
            .cloned()
            .unwrap();
        assert!(original.same_as(&returned));
        drop((original, returned, interrupt));
        assert_eq!(payload.debug_strong_count(), Some(1));
    }
}

#[test]
fn nested_walk_error_keeps_context_and_releases_callback_owner() {
    let root = Array::from_vec(vec![Array::from_vec(vec![1_i64, 2])]);
    let owner = Any::from(Array::from_vec(vec![42_i64]));
    let callback_owner = owner.clone();
    let error = structural_walk(
        &root,
        move |_: i64| -> Result<WalkResult> {
            assert_eq!(callback_owner.debug_strong_count(), Some(2));
            Err(runtime_error("nested owned failure"))
        },
        WalkOrder::PreOrder,
    )
    .err()
    .unwrap();
    assert_eq!(error.message(), "nested owned failure");
    assert!(error.backtrace().contains("ffi.Array"));
    assert_eq!(owner.debug_strong_count(), Some(1));
}
