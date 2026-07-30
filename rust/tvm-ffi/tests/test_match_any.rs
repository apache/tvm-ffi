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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::match_any_internal::{ArmId, LeafLookupTable, LeafPatternMetadata, LeafPatternProbe};
use tvm_ffi::object::{Object as ObjectBase, ObjectArc};
use tvm_ffi::{
    match_any, Any, AnyCompatible, AnyView, Array, Function, Map, Module, Shape, Tensor, TypeIndex,
};

// Mirrors the C++ `testing.TestIntPair` layout registered by the test library.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestIntPair"]
#[type_final]
struct TestIntPairObj {
    base: ObjectBase,
    a: i64,
    b: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct TestIntPair {
    data: ObjectArc<TestIntPairObj>,
}

impl TestIntPair {
    fn new(a: i64, b: i64) -> Self {
        Self {
            data: ObjectArc::new(TestIntPairObj {
                base: ObjectBase::new(),
                a,
                b,
            }),
        }
    }
}

struct CustomModuleMatcher;

impl<'a> TryFrom<AnyView<'a>> for CustomModuleMatcher {
    type Error = &'static str;

    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        value.try_as::<Module>().map(|_| Self).ok_or("not a Module")
    }
}

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
fn custom_try_into_matcher_keeps_ordered_compatibility() {
    fn matches(value: AnyView<'_>) -> bool {
        match_any! {
            value {
                CustomModuleMatcher(_) => true,
                _ => false,
            }
        }
    }

    let module: Module = Function::get_global("ffi.SystemLib")
        .unwrap()
        .call_tuple_with_len::<0, _>(())
        .unwrap()
        .try_into()
        .unwrap();
    assert!(matches(AnyView::from(&module)));
    assert!(!matches(AnyView::from(&Shape::from([1_i64, 2]))));
}

#[test]
fn parameterized_containers_keep_ordered_conversion() {
    let array = [1.5_f64, 2.5].into_iter().collect::<Array<f64>>();
    // Twenty arms make the macro consider lookup dispatch, but Array<T> is
    // content-dependent and therefore keeps the complete ordered conversion.
    let selected = match_any! {
        Any::from(array) {
            Array::<i64>(_) => "integer array",
            Tensor(_) => "tensor",
            Shape(_) => "shape",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Module(_) => "module",
            Array::<f64>(_) => "float array",
            _ => "unsupported",
        }
    };

    assert_eq!(selected, "float array");
}

#[test]
fn lookup_selects_later_arms_and_isolates_generic_pattern_lists() {
    fn classify<T>(value: AnyView<'_>) -> usize
    where
        T: AnyCompatible + 'static,
        for<'a> AnyView<'a>: TryInto<T>,
    {
        match_any! {
            value {
                T(ref _value) => 0,
                T(mut _value) => 1,
                T(_) => 2,
                T(_) => 3,
                T(_) => 4,
                T(_) => 5,
                T(_) => 6,
                T(_) => 7,
                T(_) => 8,
                T(_) => 9,
                T(_) => 10,
                T(_) => 11,
                T(_) => 12,
                T(_) => 13,
                T(_) => 14,
                T(_) => 15,
                T(_) => 16,
                T(_) => 17,
                T(_) => 18,
                Module(mut module) => {
                    let _ = &mut module;
                    19
                },
                _ => 20,
            }
        }
    }

    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let pair = TestIntPair::new(1, 2);
    let module: Module = Function::get_global("ffi.SystemLib")
        .unwrap()
        .call_tuple_with_len::<0, _>(())
        .unwrap()
        .try_into()
        .unwrap();

    // The first monomorphization initializes the table and exercises Arm0,
    // a later ArmId with a `mut` binding, and both fallback kinds.
    assert_eq!(classify::<TestIntPair>(AnyView::from(&pair)), 0);
    assert_eq!(classify::<TestIntPair>(AnyView::from(&module)), 19);
    assert_eq!(
        classify::<TestIntPair>(AnyView::from(&Shape::from([1_i64, 2]))),
        20
    );
    assert_eq!(classify::<TestIntPair>(AnyView::from(&1_i64)), 20);

    // Function-local statics are shared by generic monomorphizations. This
    // pattern list must not reuse TestIntPair's ArmId mapping.
    assert_eq!(classify::<Module>(AnyView::from(&module)), 0);
}

#[test]
fn dense_lookup_table_maps_runtime_indices_to_local_arm_ids() {
    const ARM_0: ArmId = 0;
    const ARM_2: ArmId = 2;
    let pattern_list_id = TypeId::of::<(i32, i64, f32)>();
    let object_begin = TypeIndex::kTVMFFIStaticObjectBegin as i32;
    let table = LeafLookupTable::build(
        pattern_list_id,
        &[object_begin + 4, object_begin + 4, object_begin + 7],
    );

    assert_eq!(table.lookup(pattern_list_id, object_begin + 3), Ok(None));
    assert_eq!(
        table.lookup(pattern_list_id, object_begin + 4),
        Ok(Some(ARM_0))
    );
    assert_eq!(table.lookup(pattern_list_id, object_begin + 5), Ok(None));
    assert_eq!(table.lookup(pattern_list_id, object_begin + 6), Ok(None));
    assert_eq!(
        table.lookup(pattern_list_id, object_begin + 7),
        Ok(Some(ARM_2))
    );
    assert_eq!(table.lookup(pattern_list_id, object_begin + 8), Ok(None));
    assert_eq!(
        table.lookup(TypeId::of::<(u8, u16)>(), object_begin + 4),
        Err(())
    );
}

#[test]
fn lookup_table_handles_the_u8_arm_id_boundary() {
    const LAST_ARM: ArmId = 254;
    let pattern_list_id = TypeId::of::<[i32; 255]>();
    let object_begin = TypeIndex::kTVMFFIStaticObjectBegin as i32;
    let type_indices: [i32; 255] = std::array::from_fn(|arm_id| object_begin + arm_id as i32);
    let table = LeafLookupTable::build(pattern_list_id, &type_indices);

    assert_eq!(
        table.lookup(pattern_list_id, object_begin + 254),
        Ok(Some(LAST_ARM))
    );

    const NEXT_ARM: ArmId = 255;
    let next_pattern_list_id = TypeId::of::<[i32; 256]>();
    let next_type_indices: [i32; 256] = std::array::from_fn(|arm_id| object_begin + arm_id as i32);
    let next_table = LeafLookupTable::build(next_pattern_list_id, &next_type_indices);

    assert_eq!(
        next_table.lookup(next_pattern_list_id, object_begin + 255),
        Ok(Some(NEXT_ARM))
    );
}

#[test]
fn sparse_lookup_table_preserves_source_order() {
    const ARM_0: ArmId = 0;
    const ARM_1: ArmId = 1;
    let pattern_list_id = TypeId::of::<(i32, i64)>();
    let object_begin = TypeIndex::kTVMFFIStaticObjectBegin as i32;
    // This span is one entry larger than the direct-table budget.
    let table = LeafLookupTable::build(
        pattern_list_id,
        &[object_begin + 4 * 1024, object_begin, object_begin],
    );

    assert_eq!(table.lookup(pattern_list_id, object_begin), Ok(Some(ARM_1)));
    assert_eq!(
        table.lookup(pattern_list_id, object_begin + 4 * 1024),
        Ok(Some(ARM_0))
    );
    assert_eq!(table.lookup(pattern_list_id, object_begin + 1), Ok(None));
}

#[test]
fn large_sparse_lookup_table_maps_runtime_indices() {
    const LAST_ARM: ArmId = 95;
    let pattern_list_id = TypeId::of::<[i32; 96]>();
    let object_begin = TypeIndex::kTVMFFIStaticObjectBegin as i32;
    let type_indices: [i32; 96] = std::array::from_fn(|arm_id| {
        object_begin + if arm_id == 1 { 0 } else { arm_id as i32 * 128 }
    });
    let table = LeafLookupTable::build(pattern_list_id, &type_indices);

    assert_eq!(table.lookup(pattern_list_id, object_begin), Ok(Some(0)));
    assert_eq!(
        table.lookup(pattern_list_id, object_begin + 95 * 128),
        Ok(Some(LAST_ARM))
    );
    assert_eq!(table.lookup(pattern_list_id, object_begin + 1), Ok(None));
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
