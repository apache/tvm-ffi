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

use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeIndex, TVMFFITypeRegisterAttr};
use tvm_ffi::{
    dispatch, structural_visit, structural_walk, Any, AnyView, Array, DefRegionKind, Error,
    Function, Map, Object, Result, Shape, String as FfiString, StructuralVisitor, VisitInterrupt,
    VisitValue, WalkOrder, WalkResult, RUNTIME_ERROR,
};

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[test]
fn plain_walk_uses_native_sequence_fallback() {
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
fn plain_walk_uses_native_map_fallback() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
        .into_iter()
        .collect();
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
    assert_eq!(integers, 2);
}

#[derive(Default)]
struct SkipForeignShape {}

#[dispatch(visit)]
impl SkipForeignShape {
    fn visit_shape(&mut self, _shape: Shape) -> WalkResult {
        WalkResult::Skip
    }
}

/// Visitor-layer handling of the foreign type: `visit` enumerates the
/// children itself (none, for a shape) instead of the default recursion.
#[derive(Default)]
struct ForeignShapeVisitor {
    shapes: usize,
}

impl StructuralVisitor for ForeignShapeVisitor {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if value.cast::<Shape>().is_some() {
            self.shapes += 1;
            return Ok(None);
        }
        self.default_visit_children(value, def_region_kind)
    }
}

#[test]
fn foreign_structural_visit_requires_explicit_rust_override() {
    let hook = Function::get_global("ffi.ArraySize").unwrap();
    let attr_name = unsafe { TVMFFIByteArray::from_str("__s_visit__") };
    let mut attr_value = Any::from(hook);
    assert_eq!(
        unsafe {
            TVMFFITypeRegisterAttr(
                TVMFFITypeIndex::kTVMFFIShape as i32,
                &attr_name,
                Any::as_data_ptr(&mut attr_value),
            )
        },
        0
    );

    let root = Shape::from([2i64, 3]);
    let error = match structural_walk(
        &root,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("foreign structural visit unexpectedly used reflection"),
    };
    assert!(error.message().contains("registers foreign `__s_visit__`"));
    assert!(error.message().contains("StructuralVisitor"));

    // Walk layer: a pre-order handler skips the foreign type.
    assert!(
        structural_walk(&root, &mut SkipForeignShape::default(), WalkOrder::PreOrder)
            .unwrap()
            .is_none()
    );

    // Visitor layer: take over the type's children explicitly instead.
    let mut takeover = ForeignShapeVisitor::default();
    assert!(structural_visit(&root, &mut takeover).unwrap().is_none());
    assert_eq!(takeover.shapes, 1);
}

#[test]
fn mutable_list_is_snapshotted_before_callbacks() {
    let root = Function::get_global("ffi.List")
        .unwrap()
        .call_packed(&[AnyView::from(&1i64), AnyView::from(&2i64)])
        .unwrap();
    let captured = root.clone();
    let append = Function::get_global("ffi.ListAppend").unwrap();
    let mut appended = false;
    let mut integers = Vec::new();

    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if let Some(integer) = value.cast::<i64>() {
                integers.push(integer);
                if !appended {
                    append
                        .call_packed(&[AnyView::from(&captured), AnyView::from(&3i64)])
                        .unwrap();
                    appended = true;
                }
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());

    assert_eq!(integers, vec![1, 2]);
    let size = Function::get_global("ffi.ListSize")
        .unwrap()
        .call_packed(&[AnyView::from(&root)])
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(size, 3);
}

#[test]
fn mutable_dict_is_snapshotted_before_callbacks() {
    let root = Function::get_global("ffi.Dict")
        .unwrap()
        .call_packed(&[
            AnyView::from(&FfiString::from("a")),
            AnyView::from(&1i64),
            AnyView::from(&FfiString::from("b")),
            AnyView::from(&2i64),
        ])
        .unwrap();
    let captured = root.clone();
    let set_item = Function::get_global("ffi.DictSetItem").unwrap();
    let mut inserted = false;
    let mut integers = Vec::new();

    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if let Some(integer) = value.cast::<i64>() {
                integers.push(integer);
                if !inserted {
                    set_item
                        .call_packed(&[
                            AnyView::from(&captured),
                            AnyView::from(&FfiString::from("c")),
                            AnyView::from(&3i64),
                        ])
                        .unwrap();
                    inserted = true;
                }
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());

    integers.sort_unstable();
    assert_eq!(integers, vec![1, 2]);
    let size = Function::get_global("ffi.DictSize")
        .unwrap()
        .call_packed(&[AnyView::from(&root)])
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(size, 3);
}

#[test]
fn dense_map_layout_is_traversed_completely() {
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
    assert_eq!(strings, 9);
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
            if let Some(interrupt) = self.visit_child(&overridden, DefRegionKind::NonRecursive)? {
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
    assert_eq!(
        probe.seen,
        vec![DefRegionKind::NonRecursive, DefRegionKind::None]
    );
}

#[derive(Default)]
struct GenericDispatchProbe {
    integers: Vec<i64>,
    objects: usize,
    catch_all: usize,
}

#[dispatch(visit)]
impl GenericDispatchProbe {
    fn visit_integer(&mut self, value: i64) -> WalkResult {
        self.integers.push(value);
        WalkResult::Advance
    }

    // Trailing DefRegionKind: handlers may mix arities within one impl.
    fn visit_object(&mut self, _value: &tvm_ffi::Object, kind: DefRegionKind) -> WalkResult {
        assert_eq!(kind, DefRegionKind::None);
        self.objects += 1;
        WalkResult::Advance
    }

    fn visit_any(&mut self, _value: &VisitValue) -> WalkResult {
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

#[dispatch(visit)]
impl OrderProbe {
    fn visit_array(&mut self, _array: Array<i64>) -> WalkResult {
        self.events.push("array".to_string());
        WalkResult::Advance
    }

    fn visit_integer(&mut self, value: i64) -> WalkResult {
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
    assert!(error.backtrace().contains("sequence item [0]"));
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
    assert!(error.backtrace().contains("sequence item [0]"));
    assert!(error.backtrace().contains("object `ffi.Array`"));
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
    let outcome = structural_visit(&root, &mut InterruptingVisitor).unwrap();
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
fn chain_dispatches_first_matching_link() {
    // C++: StructuralWalk<kPreOrder>(root, [&](int64_t v) {...},
    //          [&](const ObjectRef& o) {...}, [&](AnyView v) {...})
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = Vec::new();
    let mut objects = 0;
    let mut others = 0;
    assert!(structural_walk(
        &root,
        (
            |value: i64| {
                integers.push(value);
                WalkResult::Advance
            },
            |_object: &Object| {
                objects += 1;
                WalkResult::Advance
            },
            |_value: &VisitValue| {
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
    // Like #[dispatch(visit)] handlers, each link independently opts into
    // the trailing DefRegionKind argument.
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
fn chain_links_can_interrupt_with_payload() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut seen = 0;
    let outcome = structural_walk(
        &root,
        (
            |value: i64| {
                seen += 1;
                if value == 2 {
                    return WalkResult::interrupt_with(value * 10);
                }
                WalkResult::Advance
            },
            |_value: &VisitValue| WalkResult::Advance,
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("walk unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 20);
    assert_eq!(seen, 2);
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
    assert!(error.backtrace().contains("sequence item [0]"));
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn chain_supports_post_order() {
    // Rust borrow rules apply per link: state shared across links goes
    // through a RefCell (or a single #[dispatch(visit)] visitor).
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

#[test]
fn single_link_tuple_walks() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut total = 0;
    assert!(structural_walk(
        &root,
        (|value: i64| {
            total += value;
            WalkResult::Advance
        },),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(total, 6);
}

#[derive(Default)]
struct ObjectCounter {
    objects: usize,
}

#[dispatch(visit)]
impl ObjectCounter {
    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        self.objects += 1;
        WalkResult::Advance
    }
}

#[test]
fn chain_splices_dispatch_visitors_between_closures() {
    // A `&mut` typed visitor participates in the chain like any other link,
    // keeping its own no-match fall-through semantics.
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
    let root = Array::new(vec![1i64]);
    let mut int_hits = 0;
    let mut object_hits = 0;
    assert!(structural_walk(
        &root,
        (
            |_value: f64| WalkResult::Advance,
            |_value: bool| WalkResult::Advance,
            |_value: tvm_ffi::String| WalkResult::Advance,
            |_value: Array<f64>| WalkResult::Advance,
            |_value: i64| {
                int_hits += 1;
                WalkResult::Advance
            },
            |_value: &Object, _kind: DefRegionKind| {
                object_hits += 1;
                WalkResult::Advance
            },
            |_value: &VisitValue, _kind: DefRegionKind| WalkResult::Advance,
            |_value: &VisitValue| WalkResult::Advance,
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(int_hits, 1);
    assert_eq!(object_hits, 1);
}

#[test]
fn bare_typed_lambda_walks_without_tuple() {
    // A lone typed handler needs no tuple: unmatched values (the array
    // itself) advance normally.
    let root = Array::new(vec![1i64, 2, 3]);
    let mut total = 0;
    assert!(structural_walk(
        &root,
        |value: i64| {
            total += value;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(total, 6);
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

#[test]
fn bare_owned_object_lambda_interrupts() {
    let root = Array::new(vec![Array::new(vec![1i64]), Array::new(vec![2i64])]);
    let outcome = structural_walk(
        &root,
        |array: Array<i64>| {
            if array.len() == 1 {
                return WalkResult::interrupt_with(array.len() as i64);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("walk unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 1);
}
