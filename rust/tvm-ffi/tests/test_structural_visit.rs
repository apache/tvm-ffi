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

use std::sync::LazyLock;

use tvm_ffi::derive::{Object as DeriveObject, ObjectRef as DeriveObjectRef};
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyViewToOwnedAny, TVMFFIByteArray, TVMFFIFieldFlagBitMask, TVMFFIFieldInfo,
    TVMFFISEqHashKind, TVMFFITypeIndex, TVMFFITypeMetadata, TVMFFITypeRegisterAttr,
};
use tvm_ffi::{
    dispatch, structural_visit, structural_walk, Any, AnyView, Array, DefRegionKind, Error,
    Function, Map, Object, ObjectArc, ObjectCore, ObjectRefCast, Result, Shape,
    String as FfiString, StructuralVisitor, TypeIndex, VisitInterrupt, VisitValue, WalkOrder,
    WalkResult, RUNTIME_ERROR,
};

unsafe extern "C" {
    fn TVMFFITypeGetOrAllocIndex(
        type_key: *const TVMFFIByteArray,
        static_type_index: i32,
        type_depth: i32,
        num_child_slots: i32,
        child_slots_can_overflow: i32,
        parent_type_index: i32,
    ) -> i32;
    fn TVMFFITypeRegisterField(type_index: i32, info: *const TVMFFIFieldInfo) -> i32;
    fn TVMFFITypeRegisterMetadata(type_index: i32, metadata: *const TVMFFITypeMetadata) -> i32;
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralVisitDefRegion"]
#[type_final]
struct RustVisitDefRegionObj {
    base: Object,
    recursive: Any,
    plain: Any,
    non_recursive: Any,
    both: Any,
    ignored: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustVisitDefRegion {
    data: ObjectArc<RustVisitDefRegionObj>,
}

unsafe extern "C" fn clone_any_field(field: *mut std::ffi::c_void, result: *mut TVMFFIAny) -> i32 {
    TVMFFIAnyViewToOwnedAny(field.cast(), result)
}

fn register_any_field(type_index: i32, name: &'static str, offset: usize, flags: i64) {
    let field = TVMFFIFieldInfo {
        name: unsafe { TVMFFIByteArray::from_str(name) },
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-visit test field") },
        metadata: unsafe { TVMFFIByteArray::from_str("") },
        flags,
        size: std::mem::size_of::<Any>() as i64,
        alignment: std::mem::align_of::<Any>() as i64,
        offset: offset as i64,
        getter: Some(clone_any_field),
        setter: std::ptr::null_mut(),
        default_value_or_factory: TVMFFIAny::new(),
        field_static_type_index: -1,
    };
    assert_eq!(unsafe { TVMFFITypeRegisterField(type_index, &field) }, 0);
}

fn register_visit_type(type_key: &'static str, total_size: usize, kind: TVMFFISEqHashKind) -> i32 {
    let type_key = unsafe { TVMFFIByteArray::from_str(type_key) };
    let type_index = unsafe {
        TVMFFITypeGetOrAllocIndex(
            &type_key,
            -1,
            Object::TYPE_DEPTH + 1,
            0,
            1,
            Object::type_index(),
        )
    };
    assert!(type_index >= TypeIndex::kTVMFFIDynObjectBegin as i32);
    let metadata = TVMFFITypeMetadata {
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-visit test object") },
        creator: None,
        total_size: i32::try_from(total_size).unwrap(),
        structural_eq_hash_kind: kind as i32,
    };
    assert_eq!(
        unsafe { TVMFFITypeRegisterMetadata(type_index, &metadata) },
        0
    );
    type_index
}

static REGISTER_REGION_TYPES: LazyLock<()> = LazyLock::new(|| {
    let type_index = register_visit_type(
        RustVisitDefRegionObj::TYPE_KEY,
        std::mem::size_of::<RustVisitDefRegionObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar,
    );
    for (name, offset, flags) in [
        (
            "recursive",
            std::mem::offset_of!(RustVisitDefRegionObj, recursive),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64,
        ),
        (
            "plain",
            std::mem::offset_of!(RustVisitDefRegionObj, plain),
            0,
        ),
        (
            "non_recursive",
            std::mem::offset_of!(RustVisitDefRegionObj, non_recursive),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive as i64,
        ),
        (
            "both",
            std::mem::offset_of!(RustVisitDefRegionObj, both),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64
                | TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive as i64,
        ),
        (
            "ignored",
            std::mem::offset_of!(RustVisitDefRegionObj, ignored),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64,
        ),
    ] {
        register_any_field(type_index, name, offset, flags);
    }
});

fn ensure_region_types_registered() {
    LazyLock::force(&REGISTER_REGION_TYPES);
}

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
fn dense_map_layout_visits_all_values_without_visiting_keys() {
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
    // Doubles as the first-match ordering probe: earlier misses fall
    // through, the first matching link claims the value, later links
    // never run.
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
            |_value: &VisitValue| WalkResult::Advance,
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
    ensure_region_types_registered();
    let root = RustVisitDefRegion {
        data: ObjectArc::new(RustVisitDefRegionObj {
            base: Object::new(),
            recursive: Any::from(1i64),
            plain: Any::from(2i64),
            non_recursive: Any::from(3i64),
            both: Any::from(4i64),
            ignored: Any::from(5i64),
        }),
    };
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

        if value.as_node::<RustVisitDefRegionObj>().is_some() {
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
    ensure_region_types_registered();
    let free_var = RustVisitDefRegion {
        data: ObjectArc::new(RustVisitDefRegionObj {
            base: Object::new(),
            recursive: Any::new(),
            plain: Any::from(6i64),
            non_recursive: Any::new(),
            both: Any::new(),
            ignored: Any::new(),
        }),
    };
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
