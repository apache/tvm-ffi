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

use std::any::TypeId;

use tvm_ffi::match_any_internal::{ArmId, LeafLookupTable, LeafPatternMetadata, LeafPatternProbe};
use tvm_ffi::{match_any, Any, AnyView, Array, Function, Map, Module, Shape, Tensor, TypeIndex};

#[test]
fn matches_concrete_object_containers_in_source_order() {
    fn classify(expr: Any) -> (&'static str, usize) {
        match_any! {
            expr {
                Tensor(tensor)
                    if tensor.shape().len() == 2 => ("matrix", tensor.shape().len()),
                Tensor(tensor) => ("tensor", tensor.shape().len()),
                Shape(shape) => ("shape", shape.len()),
                Array::<i64>(array) => ("array", array.len()),
                _ => ("unsupported", 0),
            }
        }
    }

    let matrix = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let volume = Tensor::from_slice(&[0_f32; 24], &[2, 3, 4]).unwrap();
    let shape = Shape::from([2_i64, 3, 4, 5]);
    let array = [1_i64, 2, 3].into_iter().collect::<Array<i64>>();

    assert_eq!(classify(Any::from(matrix)), ("matrix", 2));
    assert_eq!(classify(Any::from(volume)), ("tensor", 3));
    assert_eq!(classify(Any::from(shape)), ("shape", 4));
    assert_eq!(classify(Any::from(array)), ("array", 3));
    assert_eq!(
        classify(Any::from(Map::<i64, i64>::default())),
        ("unsupported", 0)
    );
    assert_eq!(classify(Any::from(1_i64)), ("unsupported", 0));

    let tensor = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let view = AnyView::from(&tensor);
    let matched_view = match_any! {
        view {
            Tensor(tensor) => ("tensor", tensor.shape().len()),
            _ => ("unsupported", 0),
        }
    };
    assert_eq!(matched_view, ("tensor", 2));
}

#[test]
fn parameterized_containers_keep_ordered_conversion() {
    let array = [1.5_f64, 2.5].into_iter().collect::<Array<f64>>();
    let selected = match_any! {
        Any::from(array) {
            Array::<i64>(_) => "integer array",
            Tensor(_) => "tensor",
            Shape(_) => "shape",
            Array::<f64>(_) => "float array",
            _ => "unsupported",
        }
    };

    assert_eq!(selected, "float array");
}

#[test]
fn leaf_lookup_keeps_the_first_arm() {
    fn classify(value: Any) -> usize {
        match_any! {
            value {
                Module(_) => 0,
                Module(_) => 1,
                _ => 2,
            }
        }
    }

    let module: Module = Function::get_global("ffi.SystemLib")
        .unwrap()
        .call_tuple_with_len::<0, _>(())
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(classify(Any::from(module)), 0);
    assert_eq!(classify(Any::from(Array::<i64>::default())), 2);
}

#[test]
fn lookup_table_maps_runtime_indices_to_local_arm_ids() {
    const ARM_0: ArmId = 0;
    const ARM_1: ArmId = 1;
    const ARM_2: ArmId = 2;
    let pattern_list_id = TypeId::of::<(i32, i64, f32)>();
    let table = LeafLookupTable::build(pattern_list_id, &[(73, ARM_0), (73, ARM_1), (75, ARM_2)]);

    assert_eq!(table.lookup(pattern_list_id, 73), Ok(Some(ARM_0)));
    assert_eq!(table.lookup(pattern_list_id, 72), Ok(None));
    assert_eq!(table.lookup(pattern_list_id, 74), Ok(None));
    assert_eq!(table.lookup(pattern_list_id, 75), Ok(Some(ARM_2)));
    assert_eq!(table.lookup(pattern_list_id, 76), Ok(None));
}

#[test]
fn a_generic_pattern_list_cannot_reuse_another_lists_table() {
    let pattern_list_id = TypeId::of::<(i32, i64)>();
    let table = LeafLookupTable::build(pattern_list_id, &[(73, 0), (75, 1)]);

    assert_eq!(table.lookup(TypeId::of::<(u8, u16)>(), 73), Err(()));
}

#[test]
fn metadata_only_accepts_exact_leaf_patterns() {
    type Leaf = (Module, ());
    let leaf = LeafPatternProbe::<Leaf>::new();
    let mut type_indices = [0; 1];
    assert!((&leaf).leaf_pattern_list_id().is_some());
    (&leaf).fill_leaf_type_indices(&mut type_indices);
    assert!(type_indices[0] >= TypeIndex::kTVMFFIStaticObjectBegin as i32);

    type Parameterized = (Array<i64>, ());
    let parameterized = LeafPatternProbe::<Parameterized>::new();
    assert!((&parameterized).leaf_pattern_list_id().is_none());

    type NonFinal = (Tensor, ());
    let non_final = LeafPatternProbe::<NonFinal>::new();
    assert!((&non_final).leaf_pattern_list_id().is_none());

    struct NoAnyCompatibleMetadata;
    type Custom = (NoAnyCompatibleMetadata, ());
    let custom = LeafPatternProbe::<Custom>::new();
    assert!((&custom).leaf_pattern_list_id().is_none());
}
