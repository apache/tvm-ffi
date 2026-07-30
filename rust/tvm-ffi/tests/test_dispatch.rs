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
    dispatch, structural_visit, Array, DefRegionKind, Object, VisitDispatch, WalkResult,
};

#[derive(Default)]
struct ExternalCounter {
    def_region: DefRegionKind,
    objects: usize,
}

#[dispatch(visit, def_region = def_region)]
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
struct CfgAttrCounter {
    def_region: DefRegionKind,
}

#[dispatch(visit, def_region = def_region)]
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

#[dispatch(visit, def_region = def_region)]
#[cfg(any())]
impl DisabledCounter {
    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        WalkResult::Advance
    }
}

struct CfgAttrDisabledCounter;
const _: usize = std::mem::size_of::<CfgAttrDisabledCounter>();

#[dispatch(visit, def_region = def_region)]
#[cfg_attr(all(), cfg(any()))]
impl CfgAttrDisabledCounter {
    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        WalkResult::Advance
    }
}

#[test]
fn generated_dispatch_uses_public_downstream_paths() {
    let root = Array::new(vec![1i64, 2]);
    let mut visitor = ExternalCounter::default();
    assert!(structural_visit(&root, &mut visitor).unwrap().is_continue());
    assert_eq!(visitor.objects, 1);
}
