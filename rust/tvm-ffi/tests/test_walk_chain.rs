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

//! Tuple walkers: `structural_walk(root, (link1, link2, ...), order)`, the
//! Rust analog of the C++ variadic `StructuralWalk(root, callbacks...)`
//! chain. Links are tried in order and the first whose argument type matches
//! the value runs.

use tvm_ffi::{
    dispatch, structural_walk, Array, DefRegionKind, Error, Object, Result, VisitValue, WalkOrder,
    WalkResult, RUNTIME_ERROR,
};

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

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
fn narrow_numeric_links_match_only_exact_values() {
    // An out-of-range Int falls through a narrow numeric link to the next
    // one instead of silently truncating into it.
    let root = Array::new(vec![200i64, 300, -1]);
    let mut narrow = Vec::new();
    let mut wide = Vec::new();
    assert!(structural_walk(
        &root,
        (
            |value: u8| {
                narrow.push(value);
                WalkResult::Advance
            },
            |value: i64| {
                wide.push(value);
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(narrow, vec![200u8]);
    assert_eq!(wide, vec![300, -1]);
}

#[test]
fn f32_links_match_only_lossless_values() {
    let root = Array::new(vec![1.5f64, 1e300]);
    let mut narrow = Vec::new();
    let mut wide = Vec::new();
    assert!(structural_walk(
        &root,
        (
            |value: f32| {
                narrow.push(value);
                WalkResult::Advance
            },
            |value: f64| {
                wide.push(value);
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(narrow, vec![1.5f32]);
    assert_eq!(wide, vec![1e300]);
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
