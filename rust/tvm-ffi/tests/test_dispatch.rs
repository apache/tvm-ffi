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

//! This integration test compiles as a downstream crate, checking every public
//! path emitted by the dispatch macro outside `tvm_ffi` itself.

use tvm_ffi::{
    dispatch, structural_walk, Array, DefRegionKind, Object, VisitDispatch, WalkOrder, WalkResult,
};

#[derive(Default)]
struct ExternalCounter {
    objects: usize,
}

#[dispatch(visit)]
impl ExternalCounter {
    #[cfg(any(unix, windows))]
    #[cfg_attr(all(), inline)]
    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        self.objects += 1;
        WalkResult::Advance
    }
}

fn assert_visit_dispatch<T: VisitDispatch>() {}

const _: fn() = assert_visit_dispatch::<ExternalCounter>;

#[derive(Default)]
struct CfgAttrCounter {}

#[dispatch(visit)]
impl CfgAttrCounter {
    #[cfg(any())]
    fn visit_disabled_catch_all(&mut self, _value: &tvm_ffi::VisitValue) -> WalkResult {
        WalkResult::Advance
    }

    #[cfg_attr(all(), cfg(any()))]
    fn visit_disabled(&mut self, _value: &Object) -> WalkResult {
        WalkResult::Advance
    }

    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        WalkResult::Advance
    }
}

const _: fn() = assert_visit_dispatch::<CfgAttrCounter>;

struct DisabledCounter;
const _: usize = std::mem::size_of::<DisabledCounter>();

#[dispatch(visit)]
#[cfg(any())]
impl DisabledCounter {
    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        WalkResult::Advance
    }
}

struct CfgAttrDisabledCounter;
const _: usize = std::mem::size_of::<CfgAttrDisabledCounter>();

#[dispatch(visit)]
#[cfg_attr(all(), cfg(any()))]
impl CfgAttrDisabledCounter {
    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        WalkResult::Advance
    }
}

#[derive(Default)]
struct MixedArityCounter {
    kinds: Vec<DefRegionKind>,
    objects: usize,
}

#[dispatch(visit)]
impl MixedArityCounter {
    fn visit_int(&mut self, _value: i64, kind: DefRegionKind) -> WalkResult {
        self.kinds.push(kind);
        WalkResult::Advance
    }

    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        self.objects += 1;
        WalkResult::Advance
    }
}

#[test]
fn handlers_may_mix_def_region_arity() {
    let root = Array::new(vec![1i64, 2]);
    let mut visitor = MixedArityCounter::default();
    assert!(structural_walk(&root, &mut visitor, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(visitor.objects, 1);
    assert_eq!(visitor.kinds, vec![DefRegionKind::None; 2]);
}

#[test]
fn generated_dispatch_uses_public_downstream_paths() {
    let root = Array::new(vec![1i64, 2]);
    let mut visitor = ExternalCounter::default();
    assert!(structural_walk(&root, &mut visitor, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(visitor.objects, 1);
}
