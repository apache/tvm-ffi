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
use std::rc::Rc;

use tvm_ffi::derive::{Object as DeriveObject, ObjectRef as DeriveObjectRef};
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITestingDummyTarget};
use tvm_ffi::{
    dispatch, get_type_attr, structural_visit, structural_walk, Any, AnyView, Array, DLDataType,
    DLDataTypeCode, DefRegionKind, Error, FieldGetter, Function, Map, Object, ObjectArc,
    ObjectCore, ObjectRefCast, Result, String as FfiString, StructuralVisitor, TypeIndex,
    VisitCallbacks, VisitContext, VisitInterrupt, VisitValue, WalkOrder, WalkResult, RUNTIME_ERROR,
};

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.StructuralVisitDefRegion"]
#[type_final]
struct CxxVisitDefRegionObj {
    base: Object,
    recursive: Any,
    plain: Any,
    non_recursive: Any,
    both: Any,
    ignored: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct CxxVisitDefRegion {
    data: ObjectArc<CxxVisitDefRegionObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.TestIntPair"]
#[type_final]
struct TestIntPairObj {
    base: Object,
    a: i64,
    b: i64,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct TestIntPair {
    data: ObjectArc<TestIntPairObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.StructuralVisitFailingGetter"]
#[type_final]
struct CxxVisitFailingGetterObj {
    base: Object,
    value: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct CxxVisitFailingGetter {
    data: ObjectArc<CxxVisitFailingGetterObj>,
}

// Referencing the testing library keeps it linked, so its static registrations
// run before the Rust test harness, even when one test is selected in isolation.
fn fixture_constructor(type_key: &str) -> Function {
    assert_eq!(unsafe { TVMFFITestingDummyTarget() }, 0);
    Function::from_type_key_method(type_key, "__ffi_init__").unwrap()
}

fn test_int_pair(a: i64, b: i64) -> TestIntPair {
    fixture_constructor(TestIntPairObj::TYPE_KEY)
        .call_tuple((a, b))
        .unwrap()
        .try_into()
        .unwrap()
}

// Only hook integration tests need this fixture. Rust supplies an observer;
// the traversal hook itself is already registered by the C++ testing library.
fn cxx_visit_hook(selected: impl Into<Any>, ignored: impl Into<Any>, observer: Function) -> Any {
    fixture_constructor("testing.StructuralVisitHook")
        .call_tuple((selected.into(), ignored.into(), observer))
        .unwrap()
}

fn cxx_visit_failing_getter(value: impl Into<Any>) -> CxxVisitFailingGetter {
    fixture_constructor(CxxVisitFailingGetterObj::TYPE_KEY)
        .call_tuple((value.into(),))
        .unwrap()
        .try_into()
        .unwrap()
}

fn cxx_visit_def_region(
    recursive: Any,
    plain: Any,
    non_recursive: Any,
    both: Any,
    ignored: Any,
) -> CxxVisitDefRegion {
    fixture_constructor(CxxVisitDefRegionObj::TYPE_KEY)
        .call_tuple((recursive, plain, non_recursive, both, ignored))
        .unwrap()
        .try_into()
        .unwrap()
}

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[test]
fn public_reflection_access_uses_registered_field_and_type_attr() {
    let root = test_int_pair(11, 99);
    let type_index = TestIntPairObj::type_index();
    let getter = FieldGetter::new(type_index, "a").unwrap();
    assert_eq!(getter.get::<_, i64>(&*root.data).unwrap(), 11);

    let wrong_type = cxx_visit_failing_getter(1i64);
    assert!(getter.get_any(&*wrong_type.data).is_err());
    assert!(FieldGetter::new(type_index, "missing").is_err());

    // Use an existing function-valued C++ type attribute; no test registration.
    let attr = get_type_attr(type_index, "__ffi_convert__").unwrap();
    assert!(Function::try_from(attr).is_ok());
    let convert = Function::from_type_attr(type_index, "__ffi_convert__").unwrap();
    let converted: TestIntPair = convert.call_tuple((root,)).unwrap().try_into().unwrap();
    assert_eq!(getter.get::<_, i64>(&*converted.data).unwrap(), 11);
    assert!(get_type_attr(type_index, "missing").is_none());
}

#[test]
fn reflected_getter_keeps_owned_field_alive() {
    let root = cxx_visit_def_region(
        Any::new(),
        Any::from(FfiString::from("owned field")),
        Any::new(),
        Any::new(),
        Any::new(),
    );
    let getter = FieldGetter::new(CxxVisitDefRegionObj::type_index(), "plain").unwrap();
    let field = getter.get::<_, FfiString>(&*root.data).unwrap();
    drop(root);
    assert_eq!(field.as_str(), "owned field");
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
fn reflected_getter_releases_partial_result_on_error() {
    let tracked = FfiString::from("a reference-counted reflected visit field");
    let root = cxx_visit_failing_getter(tracked.clone());
    let count_before = AnyView::from(&tracked).debug_strong_count();

    let error = match structural_walk(
        &root,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    ) {
        Ok(_) => panic!("failing getter unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_eq!(
        error.message(),
        "visit getter failed after writing an owning result"
    );
    assert_eq!(AnyView::from(&tracked).debug_strong_count(), count_before);
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
fn reflected_fields_are_visited_without_a_hook() {
    let root = test_int_pair(11, 99);
    assert!(get_type_attr(TestIntPairObj::type_index(), "__s_visit__").is_none());

    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut integers = Vec::new();
        assert!(structural_walk(
            &root,
            |value: i64| {
                integers.push(value);
                WalkResult::Advance
            },
            order,
        )
        .unwrap()
        .is_none());
        assert_eq!(integers, vec![11, 99]);
    }

    let integers = RefCell::new(Vec::new());
    assert!(
        structural_visit(&root, |value: i64, _visitor: &mut VisitContext<'_, ()>| {
            integers.borrow_mut().push(value)
        },)
        .unwrap()
        .is_none()
    );
    assert_eq!(*integers.borrow(), vec![11, 99]);
}

#[test]
fn cxx_function_hook_controls_children_and_interrupts() {
    let regions = Rc::new(RefCell::new(Vec::new()));
    let observed_regions = Rc::clone(&regions);
    let observer = Function::from_packed(move |args| {
        assert_eq!(args.len(), 1);
        let kind =
            Function::get_global("ffi.StructuralVisitorDefRegionKind")?.call_packed(&[args[0]])?;
        observed_regions.borrow_mut().push(i64::try_from(kind)?);
        Ok(Any::new())
    });
    let root = cxx_visit_hook(11i64, 99i64, observer);
    assert!(Function::from_type_attr(root.type_index(), "__s_visit__").is_ok());

    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut integers = Vec::new();
        assert!(structural_walk(
            &root,
            |value: i64| {
                integers.push(value);
                WalkResult::Advance
            },
            order,
        )
        .unwrap()
        .is_none());
        // Reflection would visit both fields; the C++ hook selects only one.
        assert_eq!(integers, vec![11]);
    }

    #[derive(Default)]
    struct RecordingVisitor {
        integers: Vec<i64>,
    }
    impl StructuralVisitor for RecordingVisitor {
        fn visit(
            &mut self,
            value: &VisitValue,
            kind: DefRegionKind,
        ) -> Result<Option<VisitInterrupt>> {
            if let Some(integer) = value.cast::<i64>() {
                self.integers.push(integer);
            }
            self.default_visit_children(value, kind)
        }
    }
    let mut visitor = RecordingVisitor::default();
    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.integers, vec![11]);

    let integers = RefCell::new(Vec::new());
    assert!(
        structural_visit(&root, |value: i64, _visitor: &mut VisitContext<'_, ()>| {
            integers.borrow_mut().push(value)
        },)
        .unwrap()
        .is_none()
    );
    assert_eq!(*integers.borrow(), vec![11]);
    assert_eq!(*regions.borrow(), vec![DefRegionKind::None as i64; 4]);

    regions.borrow_mut().clear();
    let wrapped =
        cxx_visit_def_region(root.clone(), Any::new(), Any::new(), Any::new(), Any::new());
    structural_walk(
        &wrapped,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert_eq!(*regions.borrow(), vec![DefRegionKind::Recursive as i64]);

    let interrupt = structural_walk(
        &root,
        |_value: i64| WalkResult::interrupt_with(FfiString::from("stop")),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        FfiString::try_from(interrupt.value).unwrap().as_str(),
        "stop"
    );
}

#[test]
fn retained_visitor_from_cxx_hook_rejects_calls_after_traversal() {
    let retained = Rc::new(RefCell::new(None));
    let observed_visitor = Rc::clone(&retained);
    let observer = Function::from_packed(move |args| {
        assert_eq!(args.len(), 1);
        observed_visitor.replace(Some(Any::from(args[0])));
        Ok(Any::new())
    });
    let root = cxx_visit_hook(1i64, 99i64, observer);
    structural_walk(
        &root,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    )
    .unwrap();

    let error = Function::get_global("ffi.StructuralVisitorVisit")
        .unwrap()
        .call_tuple((retained.take().unwrap(), 1i64))
        .err()
        .expect("retained structural visitor unexpectedly remained active");
    assert!(error.message().contains("retained after its active call"));
}

#[test]
fn cxx_hook_rejects_foreign_thread_visitor_calls() {
    let checked = Rc::new(Cell::new(false));
    let observer_checked = Rc::clone(&checked);
    let observer = Function::from_packed(move |args| {
        assert_eq!(args.len(), 1);
        // Keep an owning reference on this thread until the worker joins.
        // The worker only borrows the visitor, without transferring ownership.
        let mut owner = Any::from(args[0]);
        let raw = unsafe { *Any::as_data_ptr(&mut owner) };
        let type_index = raw.type_index;
        let address = unsafe { raw.data_union.v_obj } as usize;
        let message = std::thread::spawn(move || {
            let mut raw = TVMFFIAny::new();
            raw.type_index = type_index;
            raw.data_union.v_obj = address as *mut _;
            let visitor = std::mem::ManuallyDrop::new(unsafe { Any::from_raw_ffi_any(raw) });
            Function::get_global("ffi.StructuralVisitorVisit")
                .unwrap()
                .call_packed(&[AnyView::from(&*visitor), AnyView::from(&1i64)])
                .err()
                .expect("foreign-thread visitor call unexpectedly succeeded")
                .message()
                .to_string()
        })
        .join()
        .unwrap();
        assert!(message.contains("invoked from a different thread"));
        observer_checked.set(true);
        Ok(Any::new())
    });
    let root = cxx_visit_hook(1i64, 99i64, observer);
    structural_walk(
        &root,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert!(checked.get(), "C++ hook observer was never called");
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

    fn walk_any(&mut self, _value: &VisitValue) -> WalkResult {
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
            return self.visit_child(&inner, DefRegionKind::Recursive);
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
    assert_eq!(probe.seen, vec![DefRegionKind::Recursive; 3]);
}

#[test]
fn reflected_field_def_region_reaches_typed_handler() {
    let root = cxx_visit_def_region(
        Any::from(1i64),
        Any::from(2i64),
        Any::from(3i64),
        Any::from(4i64),
        Any::from(5i64),
    );
    let mut seen = Vec::new();
    assert!(structural_walk(
        &root,
        |_value: i64, kind: DefRegionKind| {
            seen.push(kind);
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(
        seen,
        vec![
            DefRegionKind::Recursive,
            DefRegionKind::None,
            DefRegionKind::NonRecursive,
            DefRegionKind::NonRecursive,
        ]
    );
}

struct FreeVarClampProbe {
    at_root: bool,
    seen: Vec<(&'static str, DefRegionKind)>,
}

impl StructuralVisitor for FreeVarClampProbe {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if self.at_root {
            self.at_root = false;
            let root = value.cast::<Array<ObjectRef>>().unwrap();
            for index in 0..root.len() {
                let child = root.get(index).unwrap();
                if let Some(interrupt) = self.visit_child(&child, DefRegionKind::NonRecursive)? {
                    return Ok(Some(interrupt));
                }
            }
            return Ok(None);
        }

        if value.as_node::<CxxVisitDefRegionObj>().is_some() {
            self.seen.push(("free_var", def_region_kind));
        } else if value.cast::<Array<i64>>().is_some() {
            self.seen.push(("array", def_region_kind));
        } else if let Some(integer) = value.cast::<i64>() {
            self.seen.push((
                if integer == 6 {
                    "free_child"
                } else {
                    "array_child"
                },
                def_region_kind,
            ));
        }
        self.default_visit_children(value, def_region_kind)
    }
}

#[test]
fn non_recursive_region_is_clamped_for_free_var_children_only() {
    let free_var = cxx_visit_def_region(
        Any::new(),
        Any::from(6i64),
        Any::new(),
        Any::new(),
        Any::new(),
    );
    let free_var: ObjectRef = free_var.try_cast().unwrap();
    let array: ObjectRef = Array::new(vec![7i64]).try_cast().unwrap();
    let root = Array::new(vec![free_var, array]);
    let mut probe = FreeVarClampProbe {
        at_root: true,
        seen: Vec::new(),
    };
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(
        probe.seen,
        vec![
            ("free_var", DefRegionKind::NonRecursive),
            ("free_child", DefRegionKind::None),
            ("array", DefRegionKind::NonRecursive),
            ("array_child", DefRegionKind::NonRecursive),
        ]
    );
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
fn callback_visit_supports_node_links_nested_tuples_and_def_regions() {
    let root = cxx_visit_def_region(
        Any::from(1i64),
        Any::from(2i64),
        Any::from(3i64),
        Any::from(4i64),
        Any::from(5i64),
    );
    let seen = RefCell::new(Vec::new());

    assert!(structural_visit(
        &root,
        (
            (
                |_value: f64, _visitor: &mut VisitContext<'_, ()>| {},
                |_node: &CxxVisitDefRegionObj, visitor: &mut VisitContext<'_, ()>| {
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
        vec![
            (1, DefRegionKind::Recursive),
            (2, DefRegionKind::None),
            (3, DefRegionKind::NonRecursive),
            (4, DefRegionKind::NonRecursive),
        ]
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
                    if let Some(interrupt) = visitor.visit_with(&value, DefRegionKind::Recursive)? {
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
        vec![(1, DefRegionKind::Recursive), (2, DefRegionKind::Recursive),]
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
