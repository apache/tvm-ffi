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

use std::ops::ControlFlow;

use tvm_ffi::tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeIndex, TVMFFITypeRegisterAttr};
use tvm_ffi::{
    dispatch, structural_visit, structural_walk, walk, walk_with_context, Any, AnyView, Array,
    DefRegionKind, Error, Function, Map, Phase, Shape, String as FfiString, VisitCtx, VisitValue,
    WalkOrder, WalkResult, RUNTIME_ERROR,
};

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[test]
fn plain_walk_uses_native_sequence_fallback() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = 0;
    assert!(walk(&root, |value, phase| {
        if phase == Phase::Enter && value.cast::<i64>().is_some() {
            integers += 1;
        }
        WalkResult::Advance
    })
    .unwrap()
    .is_continue());
    assert_eq!(integers, 3);
}

#[test]
fn plain_walk_uses_native_map_fallback() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
        .into_iter()
        .collect();
    let mut integers = 0;
    assert!(walk(&root, |value, phase| {
        if phase == Phase::Enter && value.cast::<i64>().is_some() {
            integers += 1;
        }
        WalkResult::Advance
    })
    .unwrap()
    .is_continue());
    assert_eq!(integers, 2);
}

struct SkipForeignShape;

#[dispatch(visit)]
impl SkipForeignShape {
    fn visit_shape(&mut self, _shape: Shape, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        WalkResult::Skip
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
    let error = match walk(&root, |_value, _phase| WalkResult::Advance) {
        Err(error) => error,
        Ok(_) => panic!("foreign structural visit unexpectedly used reflection"),
    };
    assert!(error.message().contains("registers foreign `__s_visit__`"));
    assert!(error.message().contains("return `WalkResult::Skip`"));

    assert!(structural_visit(&root, &mut SkipForeignShape)
        .unwrap()
        .is_continue());
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

    assert!(walk(&root, |value, phase| {
        if phase == Phase::Enter {
            if let Some(integer) = value.cast::<i64>() {
                integers.push(integer);
                if !appended {
                    append
                        .call_packed(&[AnyView::from(&captured), AnyView::from(&3i64)])
                        .unwrap();
                    appended = true;
                }
            }
        }
        WalkResult::Advance
    })
    .unwrap()
    .is_continue());

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

    assert!(walk(&root, |value, phase| {
        if phase == Phase::Enter {
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
        }
        WalkResult::Advance
    })
    .unwrap()
    .is_continue());

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
fn interrupt_stops_without_running_remaining_callbacks() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = 0;
    let outcome = walk(&root, |value, phase| {
        if phase == Phase::Enter && value.cast::<i64>().is_some() {
            integers += 1;
            return WalkResult::Interrupt;
        }
        WalkResult::Advance
    })
    .unwrap();
    assert!(outcome.is_break());
    assert_eq!(integers, 1);
}

#[derive(Default)]
struct ManualRegionProbe {
    seen: Vec<DefRegionKind>,
}

#[dispatch(visit)]
impl ManualRegionProbe {
    fn visit_array(&mut self, array: Array<i64>, ctx: &mut VisitCtx<'_>) -> WalkResult {
        let overridden = array.get(0).unwrap();
        if !ctx.visit_with_def_region(self, &overridden, DefRegionKind::NonRecursive) {
            return WalkResult::Interrupt;
        }
        let inherited = array.get(1).unwrap();
        if !ctx.visit(self, &inherited) {
            return WalkResult::Interrupt;
        }
        WalkResult::Skip
    }

    fn visit_integer(&mut self, _value: i64, ctx: &mut VisitCtx<'_>) -> WalkResult {
        self.seen.push(ctx.def_region_kind());
        WalkResult::Advance
    }
}

#[test]
fn manual_child_visit_can_override_def_region() {
    let root = Array::new(vec![7i64, 8]);
    let mut probe = ManualRegionProbe::default();
    assert!(structural_visit(&root, &mut probe).unwrap().is_continue());
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
    fn visit_integer(&mut self, value: i64, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        self.integers.push(value);
        WalkResult::Advance
    }

    fn visit_object(&mut self, _value: &tvm_ffi::Object, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        self.objects += 1;
        WalkResult::Advance
    }

    fn visit_any(&mut self, _value: &VisitValue, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        self.catch_all += 1;
        WalkResult::Advance
    }
}

#[test]
fn generated_dispatch_supports_pod_and_ordered_catch_all() {
    let root = Array::new(vec![1i64, 2]);
    let mut probe = GenericDispatchProbe::default();
    assert!(structural_visit(&root, &mut probe).unwrap().is_continue());
    assert_eq!(probe.integers, vec![1, 2]);
    assert_eq!(probe.objects, 1);

    let floats = Array::new(vec![1.0f64, 2.0]);
    assert!(structural_visit(&floats, &mut probe).unwrap().is_continue());
    assert_eq!(probe.objects, 2);
    assert_eq!(probe.catch_all, 2);
}

#[derive(Default)]
struct OrderProbe {
    events: Vec<String>,
}

#[dispatch(visit)]
impl OrderProbe {
    fn visit_array(&mut self, _array: Array<i64>, _ctx: &mut VisitCtx<'_>) -> WalkResult {
        self.events.push("array".to_string());
        WalkResult::Advance
    }

    fn visit_integer(&mut self, value: i64, _ctx: &mut VisitCtx<'_>) -> WalkResult {
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
        .is_continue());
    assert_eq!(probe.events, vec!["int:1", "int:2", "array"]);
}

#[test]
fn interrupt_payload_is_returned_to_the_caller() {
    let root = Array::new(vec![1i64, 2]);
    let outcome = walk(&root, |value, phase| {
        if phase == Phase::Enter && value.cast::<i64>() == Some(1) {
            return WalkResult::interrupt_with(42i64);
        }
        WalkResult::Advance
    })
    .unwrap();
    let ControlFlow::Break(payload) = outcome else {
        panic!("walk unexpectedly completed");
    };
    assert_eq!(i64::try_from(payload).unwrap(), 42);
}

#[test]
fn handler_errors_include_native_visit_path() {
    let root = Array::new(vec![1i64]);
    let error = match walk(&root, |value, phase| {
        if phase == Phase::Enter && value.cast::<i64>().is_some() {
            Err(runtime_error("handler failed"))
        } else {
            Ok(WalkResult::Advance)
        }
    }) {
        Err(error) => error,
        Ok(_) => panic!("handler unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "handler failed");
    assert!(error.backtrace().contains("sequence item [0]"));
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn raw_walk_context_receives_def_region() {
    let root = Array::new(vec![1i64]);
    let mut regions = Vec::new();
    assert!(walk_with_context(&root, |_value, phase, region| {
        if phase == Phase::Enter {
            regions.push(region);
        }
        WalkResult::Advance
    })
    .unwrap()
    .is_continue());
    assert_eq!(regions, vec![DefRegionKind::None; 2]);
}
