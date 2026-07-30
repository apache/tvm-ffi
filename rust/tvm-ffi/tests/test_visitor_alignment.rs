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

//! Rust mirror of the C++ visitor example.
//!
//! `RecordingVisitor` lines up member-for-member with the C++
//! `TestVisitorObj` (tests/cpp/extra/test_structural_visit.cc), and its
//! `visit_array` handler plays the role of the C++ `TFuncObj::StructuralVisit`
//! hook (tests/cpp/testing_object.h): the first element is visited as a
//! recursive definition region, the rest inherit the surrounding state.

use std::ops::ControlFlow;

use tvm_ffi::{
    dispatch, structural_visit, Array, DefRegionKind, Result, String as FfiString, VisitDispatch,
    VisitValue, WalkResult,
};

/// C++: class TestVisitorObj : public StructuralVisitorObj
#[derive(Default)]
struct RecordingVisitor {
    /// C++: `def_region_mode_` (base-class member maintained by the walker).
    def_region: DefRegionKind,
    /// C++: `std::vector<ObjectRef> visited;`
    visited: Vec<String>,
    /// C++: `std::vector<TVMFFIDefRegionKind> modes;`
    modes: Vec<DefRegionKind>,
    /// C++: `ObjectRef interrupt_on;`
    interrupt_on: Option<i64>,
}

#[dispatch(visit, def_region = def_region)]
impl RecordingVisitor {
    /// C++ analog: `TFuncObj::StructuralVisit` — visit "params" (element 0)
    /// under a recursive definition region, then the "body" (element 1)
    /// under the inherited state, and skip the default recursion.
    fn visit_array(&mut self, array: Array<i64>) -> Result<WalkResult> {
        self.visited.push("array".to_string());
        self.modes.push(self.def_region_kind());

        // C++: visitor->WithDefRegionKind(kTVMFFIDefRegionKindRecursive,
        //          [&] { return visitor->VisitExpected(self->params); })
        let params = array.get(0).unwrap();
        if let ControlFlow::Break(payload) =
            self.subvisit_with_def_region(&params, DefRegionKind::Recursive)?
        {
            return Ok(WalkResult::InterruptWith(payload));
        }

        // C++: visitor->VisitExpected(self->body)  (inherits the current state)
        let body = array.get(1).unwrap();
        if let ControlFlow::Break(payload) = self.subvisit(&body)? {
            return Ok(WalkResult::InterruptWith(payload));
        }
        Ok(WalkResult::Skip)
    }

    /// C++ analog: `TestVisitorObj::VisitImpl` — record every value together
    /// with the active def-region state, optionally interrupt with a payload,
    /// otherwise delegate the default recursion.
    fn visit_any(&mut self, value: &VisitValue) -> Result<WalkResult> {
        let label = match value.cast::<i64>() {
            Some(integer) => integer.to_string(),
            None => "obj".to_string(),
        };
        // C++: visited.push_back(value_ref);
        //      modes.push_back(def_region_mode_);
        self.visited.push(label);
        self.modes.push(self.def_region_kind());

        // C++: if (value_ref.same_as(interrupt_on))
        //          return VisitInterrupt(String("stop"));
        if self.interrupt_on.is_some() && value.cast::<i64>() == self.interrupt_on {
            return Ok(WalkResult::interrupt_with(FfiString::from("stop")));
        }

        // C++: return DefaultVisitExpected(value);
        if let ControlFlow::Break(payload) = self.subvisit_children(value)? {
            return Ok(WalkResult::InterruptWith(payload));
        }
        Ok(WalkResult::Skip)
    }
}

/// C++ analog: TEST(StructuralVisitor, TraversesFunction) — the def-region
/// state flips to Recursive under "params" and back to None for the "body".
#[test]
fn records_values_and_def_region_modes() {
    let root = Array::new(vec![10i64, 20]);
    let mut visitor = RecordingVisitor::default();

    let outcome = structural_visit(&root, &mut visitor).unwrap();

    assert!(outcome.is_continue());
    assert_eq!(visitor.visited, vec!["array", "10", "20"]);
    assert_eq!(
        visitor.modes,
        vec![
            DefRegionKind::None,      // the array itself
            DefRegionKind::Recursive, // element 0: the "params" position
            DefRegionKind::None,      // element 1: the "body" position
        ]
    );
}

/// C++ analog: TEST(StructuralVisitor, StopsOnInterrupt) — the traversal
/// halts at the marked value and the payload reaches the caller.
#[test]
fn stops_on_interrupt_with_payload() {
    let root = Array::new(vec![10i64, 20]);
    let mut visitor = RecordingVisitor {
        interrupt_on: Some(20),
        ..RecordingVisitor::default()
    };

    let outcome = structural_visit(&root, &mut visitor).unwrap();

    let ControlFlow::Break(payload) = outcome else {
        panic!("traversal unexpectedly completed");
    };
    assert_eq!(FfiString::try_from(payload).unwrap().as_str(), "stop");
    assert_eq!(visitor.visited, vec!["array", "10", "20"]);
}
