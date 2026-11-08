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

use tvm_ffi::match_any_internal::{ArmId, JumpTable, JumpTableMetadata, JumpTableProbe};
use tvm_ffi::{
    Any, AnyView, Array, Function, Map, Module, Shape, Tensor, TypeIndex, VisitInterrupt, match_any,
};

type CustomModuleMatcher = Module;

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

    use tvm_ffi::collections::array::ArrayObj;
    use tvm_ffi::{ObjectArc, ObjectRefCore};
    let source = Array::from_vec(vec![1_i64, 2, 3]);
    let count = || ObjectArc::strong_count(<Array<i64> as ObjectRefCore>::data(&source));
    let size = match_any! {
        &source {
            Array::<i64>(ref handle) if {
                assert_eq!(count(), 1);
                assert_eq!(handle.len(), 3);
                false
            } => unreachable!(),
            ArrayObj(ref node) if {
                let _: &ArrayObj = node;
                assert_eq!(count(), 1);
                assert_eq!(node.size, 3);
                false
            } => unreachable!(),
            ArrayObj(ref node) => {
                let _: &ArrayObj = node;
                assert_eq!(count(), 1);
                node.size
            },
            _ => 0,
        }
    };
    assert_eq!(size, 3);
    assert_eq!(count(), 1);
    let size = match_any! {
        &source {
            Array::<i64>(ref handle) => {
                let _: &Array<i64> = handle;
                assert_eq!(count(), 1);
                handle.len()
            },
            _ => 0,
        }
    };
    assert_eq!(size, 3);
    assert_eq!(count(), 1);
    let mut saved = None;
    assert_eq!(
        match_any! {
            &source {
                Array::<i64>(ref handle) if { saved = Some(handle); true } => saved.unwrap().len(),
                _ => 0,
            }
        },
        3
    );
    let node = 7;
    assert_eq!(
        match_any! {
            &source {
                ArrayObj(ref node) if node.size == 0 => 0,
                _ => node,
            }
        },
        7
    );

    // Matching distinguishes runtime categories and rejects numeric truncation.
    let mut evaluations = 0;
    let mut guards = 0;
    assert_eq!(
        match_any!({ evaluations += 1; Any::from(true) } {
            i64(_) if { guards += 1; true } => 0,
            f64(_) => 0,
            bool(n) if { guards += 1; !n } => 0,
            bool(n) => i32::from(n),
            _ => 0,
        }),
        1
    );
    assert_eq!((evaluations, guards), (1, 1));
    assert_eq!(
        match_any!(Any::from(1.25_f64) { f32(n) => n, _ => 0.0 }),
        1.25
    );
    for (stored, unsigned, signed) in [
        (-129_i64, None, None),
        (-128, None, Some(-128)),
        (127, Some(127), Some(127)),
        (128, Some(128), None),
        (255, Some(255), None),
        (256, None, None),
    ] {
        assert_eq!(
            match_any!(Any::from(stored) { u8(n) => Some(n), _ => None }),
            unsigned
        );
        assert_eq!(
            match_any!(Any::from(stored) { i8(n) => Some(n), _ => None }),
            signed
        );
    }
    assert_eq!(
        match_any!(Any::from(i64::MAX) { u64(n) => Some(n), _ => None }),
        Some(i64::MAX as u64)
    );
    assert_eq!(
        match_any!(Any::from(u64::MAX) { u64(_) => unreachable!(), i64(n) => n, _ => 0 }),
        -1
    );
    for (stored, expected) in [
        (-0.0, Some(0x8000_0000)),
        (f64::INFINITY, Some(f32::INFINITY.to_bits())),
        (f64::from_bits(0x7ff8_0000_0000_0001), None),
        (16_777_217.0, None),
        (1e40, None),
    ] {
        assert_eq!(
            match_any!(Any::from(stored) {
                f32(n) => Some(n.to_bits()),
                f64(n) => { assert_eq!(n.to_bits(), stored.to_bits()); None },
                _ => unreachable!(),
            }),
            expected
        );
    }
    // Rust permits float casts to change a NaN's sign and payload. A matched
    // f32 must remain NaN; a fallback f64 must retain the original stored bits.
    for bits in [0x7ff8_0000_0000_0000, 0xfff8_0000_2000_0000] {
        match_any!(Any::from(f64::from_bits(bits)) {
            f32(n) => assert!(n.is_nan()),
            f64(n) => assert_eq!(n.to_bits(), bits),
            _ => unreachable!(),
        });
    }
    assert_eq!(
        match_any!(Any::from(257_i64) { Option<u8>(n) => n, _ => None }),
        None
    );
    assert_eq!(
        match_any!(Any::from(()) { Option<u8>(n) => Some(n), _ => None }),
        Some(None)
    );

    assert!(match_any!(Any::from(true) { bool(ref n) => *n, _ => false }));
    assert!(match_any!(Any::from(()) { ()(()) => true, _ => false }));
    let pointer = core::ptr::null_mut::<core::ffi::c_void>();
    assert!(
        match_any!(Any::from(pointer) { *mut core::ffi::c_void(p) => p.is_null(), _ => false })
    );
    trait Payload {
        type Item;
    }
    struct Integer;
    impl Payload for Integer {
        type Item = i64;
    }
    assert_eq!(
        match_any!(Any::from(7_i64) { <Integer as Payload>::Item(n) => n, _ => 0 }),
        7
    );
    fn identity<'a>(view: AnyView<'a>) -> AnyView<'a> {
        match_any!(view { AnyView::<'_>(rest) => rest, _ => unreachable!() })
    }
    assert_eq!(
        identity(AnyView::from(&source)).debug_strong_count(),
        Some(1)
    );

    use tvm_ffi::MutationView;
    use tvm_ffi::extra::structural_mutate::InplaceMode;
    let mut capability_evaluations = 0;
    let capability = match_any!({
        capability_evaluations += 1;
        MutationView::borrowed(AnyView::from(&source))
    } {
        tvm_ffi::MutationView<Array<f64>>(_) => unreachable!(),
        tvm_ffi::MutationView<Array<i64>>(matched) if {
            assert_eq!(matched.inplace_mode(), InplaceMode::Disallow);
            assert_eq!(count(), 1);
            false
        } => unreachable!(),
        Array<i64>(owned) if {
            assert_eq!(owned.len(), 3);
            assert_eq!(count(), 2);
            false
        } => unreachable!(),
        ArrayObj(ref node) if { assert_eq!(count(), 1); node.size == 0 } => unreachable!(),
        AnyView::<'_>(view) if { assert_eq!(view.debug_strong_count(), Some(1)); false } => unreachable!(),
        tvm_ffi::MutationView<Any>(rest) => rest,
        _ => unreachable!(),
    });
    assert_eq!(capability_evaluations, 1);
    assert_eq!(count(), 1);
    assert_eq!(capability.inplace_mode(), InplaceMode::Disallow);
    assert_eq!(
        match_any!(&capability { ArrayObj(ref node) => node.size, _ => 0 }),
        3
    );

    let typed = match_any!(capability {
        tvm_ffi::MutationView<Array<i64>>(array) => array,
        _ => unreachable!(),
    });
    assert_eq!(typed.as_any_view().debug_strong_count(), Some(1));
    // Already narrowed inputs are normalized before attempts and guard recovery.
    assert_eq!(
        match_any!(typed {
            tvm_ffi::MutationView<Array<f64>>(_) => 0,
            ArrayObj(ref node) => node.size,
            _ => 0,
        }),
        3
    );
    assert!(match_any!(MutationView::borrowed(AnyView::from(&())) {
        tvm_ffi::MutationView<()>(_unit) => true, _ => false,
    }));

    fn native_control_flow(value: AnyView<'_>) -> Result<i64, ()> {
        for _ in 0..1 {
            match_any!(value {
                i64(n) if { if n < 0 { return Err(()); } true } => return Ok(n),
                bool(_) => continue,
                _ => break,
            });
        }
        let n = match_any!(value { f64(_) => Err(())?, _ => 0 });
        Ok(n)
    }
    assert_eq!(native_control_flow(AnyView::from(&7_i64)), Ok(7));
    assert_eq!(native_control_flow(AnyView::from(&-1_i64)), Err(()));
    assert_eq!(native_control_flow(AnyView::from(&true)), Ok(0));
    assert_eq!(native_control_flow(AnyView::from(&1.0_f64)), Err(()));
}

#[test]
fn type_alias_matcher_keeps_strict_compatibility() {
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
        .call_tuple(())
        .unwrap()
        .try_into()
        .unwrap();
    assert!(matches(AnyView::from(&module)));
    assert!(!matches(AnyView::from(&Shape::from([1_i64, 2]))));
    assert!(match_any! {
        module {
            CustomModuleMatcher(ref matcher) => {
                let _: &CustomModuleMatcher = matcher;
                true
            },
            _ => false,
        }
    });
}

#[test]
fn parameterized_containers_use_complete_conversion_semantics() {
    let array = [1.5_f64, 2.5].into_iter().collect::<Array<f64>>();
    // Both patterns have the same runtime Array TypeIndex, so matching must
    // inspect the element types in source order.
    let selected = match_any! {
        Any::from(array) {
            Array::<i64>(_) => "integer array",
            Array::<f64>(_) => "float array",
            _ => "unsupported",
        }
    };

    assert_eq!(selected, "float array");
    use tvm_ffi::{List, MutationView};
    let values = Array::from_vec(vec![257_i64]);
    assert_eq!(
        match_any!(&values { Array<u8>(_) => 0, Array<i64>(_) => 1, _ => 2 }),
        1
    );
    assert!(!match_any!(Array::from_vec(vec![true]) { Array<i64>(_) => true, _ => false }));
    assert!(!match_any!(&values { List<i64>(_) => true, _ => false }));
    let list = [257_i64].into_iter().collect::<List<i64>>();
    assert!(!match_any!(&list { Array<i64>(_) => true, List<u8>(_) => true, _ => false }));
    let nested = Array::from_vec(vec![Array::from_vec(vec![Some(257_i64)])]);
    assert_eq!(
        match_any!(&nested {
            Array<Array<Option<u8>>>(_) => 0,
            Array<Array<Option<i64>>>(_) => 1,
            _ => 2,
        }),
        1
    );
    let map = [(1_i64, values)]
        .into_iter()
        .collect::<Map<i64, Array<i64>>>();
    assert_eq!(
        match_any!(&map { Map<u8, Array<u8>>(_) => 0, Map<u8, Array<i64>>(_) => 1, _ => 2 }),
        1
    );
    // Transparent result carriers retain the same recursive numeric checks.
    use tvm_ffi::{Unchanged, UnchangedOr};
    assert!(!match_any!(Any::from(257_i64) { UnchangedOr<u8>(_) => true, _ => false }));
    assert!(match_any!(Any::from(Unchanged) { UnchangedOr<u8>(_) => true, _ => false }));
    assert!(!match_any!(MutationView::borrowed(AnyView::from(&nested)) {
        MutationView<UnchangedOr<Array<Array<Option<u8>>>>>(_) => true,
        _ => false,
    }));
    type Cap<'a, T> = MutationView<'a, T>;
    let cap = MutationView::borrowed(AnyView::from(&nested));
    let recovered = match_any!(cap {
        Cap<Array<Array<Option<u8>>>>(_) => unreachable!(),
        Cap<Array<Array<Option<i64>>>>(value) if {
            assert_eq!(value.inplace_mode(), tvm_ffi::extra::structural_mutate::InplaceMode::Disallow);
            false
        } => unreachable!(),
        Cap<Any>(value) => value,
        _ => unreachable!(),
    });
    assert_eq!(recovered.as_any_view().debug_strong_count(), Some(1));
}

#[test]
fn direct_lookup_table_maps_runtime_indices_to_local_arm_ids() {
    const ARM_0: ArmId = 0;
    const ARM_2: ArmId = 2;
    let type_list_id = TypeId::of::<(i32, i64, f32)>();
    let object_begin = TypeIndex::kTVMFFIStaticObjectBegin as i32;
    let table = JumpTable::build(
        type_list_id,
        &[object_begin + 4, object_begin + 4, object_begin + 7],
    );

    assert_eq!(table.lookup(type_list_id, object_begin + 3), Ok(None));
    assert_eq!(
        table.lookup(type_list_id, object_begin + 4),
        Ok(Some(ARM_0))
    );
    assert_eq!(table.lookup(type_list_id, object_begin + 5), Ok(None));
    assert_eq!(
        table.lookup(type_list_id, object_begin + 7),
        Ok(Some(ARM_2))
    );
    assert_eq!(table.lookup(type_list_id, object_begin + 8), Ok(None));
    assert_eq!(
        table.lookup(TypeId::of::<(u8, u16)>(), object_begin + 4),
        Err(())
    );
}

#[test]
// `(&probe)` mirrors the receiver `match_any!` generates for autoref specialization.
#[allow(clippy::needless_borrow)]
fn metadata_only_accepts_exact_leaf_patterns() {
    type Leaf = (VisitInterrupt, ());
    let leaf = JumpTableProbe::<Leaf>::new();
    let mut type_indices = [0; 1];
    assert!((&leaf).type_list_id().is_some());
    (&leaf).fill_type_indices(&mut type_indices);
    assert!(type_indices[0] >= TypeIndex::kTVMFFIStaticObjectBegin as i32);

    type Parameterized = (Array<i64>, ());
    let parameterized = JumpTableProbe::<Parameterized>::new();
    assert!((&parameterized).type_list_id().is_none());

    type NonFinal = (Tensor, ());
    let non_final = JumpTableProbe::<NonFinal>::new();
    assert!((&non_final).type_list_id().is_none());

    struct NoAnyCompatibleMetadata;
    type Custom = (NoAnyCompatibleMetadata, ());
    let custom = JumpTableProbe::<Custom>::new();
    assert!((&custom).type_list_id().is_none());

    // Large generic and lifetime-bearing matches must compile without a static
    // bound; their metadata does not justify a final-object table.
    fn generic<T: tvm_ffi::AnyCompatible>(view: AnyView<'_>) -> Option<T> {
        match_any!(view {
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            T(value) => Some(value),
            _ => None,
        })
    }
    fn generic_cap<T: tvm_ffi::AnyCompatible>(value: tvm_ffi::MutationView<'_>) -> bool {
        match_any!(value {
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            tvm_ffi::MutationView<T>(_) => true,
            _ => false,
        })
    }
    fn identity<'a>(view: AnyView<'a>) -> AnyView<'a> {
        match_any!(view {
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            AnyView::<'_>(rest) => rest,
            _ => unreachable!(),
        })
    }
    assert_eq!(generic::<i64>(identity(AnyView::from(&7_i64))), Some(7));
    assert_eq!(generic::<bool>(AnyView::from(&7_i64)), None);
    assert!(generic_cap::<i64>(tvm_ffi::MutationView::borrowed(
        AnyView::from(&7_i64)
    )));
}
