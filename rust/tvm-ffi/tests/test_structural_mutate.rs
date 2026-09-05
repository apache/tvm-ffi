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
use tvm_ffi::collections::map::MapObj;
use tvm_ffi::function::FunctionObj;
use tvm_ffi::object::ObjectRef;
use tvm_ffi::{
    dispatch, structural_map, structural_mutate, Any, AnyView, Array, CallbackMutator,
    DefRegionKind, Error, FieldGetter, Function, InplaceValue, Map, MapDispatch, MapValue,
    MutateCallbacks, Mutator, Object, ObjectArc, ObjectRefCore, Result, String as FfiString,
    StructuralMutator, StructuralVarRemap, TypeIndex, WalkOrder, RUNTIME_ERROR,
};

struct IncrementIntegers;

impl MapDispatch for IncrementIntegers {
    fn dispatch_map(
        &mut self,
        value: &MapValue,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .cast::<i64>()
            .map(|integer| Ok(Any::from(integer + 1)))
    }
}

#[derive(Default)]
struct ManualIncrement {
    remap: StructuralVarRemap,
}

impl StructuralMutator for ManualIncrement {
    fn dispatch_mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        if let Some(integer) = value.cast::<i64>() {
            Ok(Any::from(integer + 1))
        } else {
            self.default_mutate(value, def_region_kind)
        }
    }

    fn dispatch_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.default_maybe_inplace_mutate(value, def_region_kind)
    }

    fn var_remap_get(&mut self, var: &MapValue) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &MapValue, mutated_value: &Any) -> Result<()> {
        self.remap.set(var, mutated_value)
    }
}

#[derive(Default)]
struct ReplaceNone {
    remap: StructuralVarRemap,
    calls: usize,
}

impl StructuralMutator for ReplaceNone {
    fn dispatch_mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        if value.type_index() == TypeIndex::kTVMFFINone as i32 {
            self.calls += 1;
            Ok(Any::from(8i64))
        } else {
            self.default_mutate(value, def_region_kind)
        }
    }

    fn dispatch_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.default_maybe_inplace_mutate(value, def_region_kind)
    }

    fn var_remap_get(&mut self, var: &MapValue) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &MapValue, mutated_value: &Any) -> Result<()> {
        self.remap.set(var, mutated_value)
    }
}

struct RecursiveEntryMutator {
    remap: StructuralVarRemap,
    use_owned_value: bool,
    owned_value_pointer: Option<usize>,
}

impl StructuralMutator for RecursiveEntryMutator {
    fn dispatch_mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        if value.type_index() == TypeIndex::kTVMFFINone as i32 {
            if self.use_owned_value {
                let value = Array::new(vec![1i64]);
                self.owned_value_pointer = Some(array_pointer(&value) as usize);
                self.maybe_inplace_mutate(value, def_region_kind)
            } else {
                self.mutate(&1i64, def_region_kind)
            }
        } else if let Some(integer) = value.cast::<i64>() {
            Ok(Any::from(integer + 1))
        } else {
            self.default_mutate(value, def_region_kind)
        }
    }

    fn dispatch_maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.default_maybe_inplace_mutate(value, def_region_kind)
    }

    fn var_remap_get(&mut self, var: &MapValue) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &MapValue, mutated_value: &Any) -> Result<()> {
        self.remap.set(var, mutated_value)
    }
}

fn reflected_object() -> Any {
    // Reference the existing test library so its C++ startup registrations are linked.
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    Function::get_global("ffi.MakeObjectFromPackedArgs")
        .unwrap()
        .call_tuple((
            FfiString::from("testing.TestObjectBase"),
            FfiString::from("v_i64"),
            1i64,
            FfiString::from("v_f64"),
            2.5f64,
            FfiString::from("v_str"),
            FfiString::from("a reflected string"),
        ))
        .unwrap()
}

fn reflected_field<T: TryFrom<Any, Error = Error>>(value: &Any, name: &str) -> T {
    let object = ObjectRef::try_from(value.clone()).unwrap();
    FieldGetter::new(value.type_index(), name)
        .unwrap()
        .get::<_, T>(&**ObjectRef::data(&object))
        .unwrap()
}

fn array_pointer<T>(array: &Array<T>) -> *const tvm_ffi::collections::array::ArrayObj
where
    T: tvm_ffi::AnyCompatible + Clone,
{
    unsafe { ObjectArc::as_raw(<Array<T> as ObjectRefCore>::data(array)) }
}

fn map_pointer<K, V>(map: &Map<K, V>) -> *const tvm_ffi::collections::map::MapObj {
    unsafe { ObjectArc::as_raw(<Map<K, V> as ObjectRefCore>::data(map)) }
}

fn any_object_pointer(value: &Any) -> *const Object {
    let object = ObjectRef::try_from(value.clone()).unwrap();
    unsafe { ObjectArc::as_raw(<ObjectRef as ObjectRefCore>::data(&object)) }
}

fn call_global(name: &str, args: &[Any]) -> Any {
    let views: Vec<AnyView<'_>> = args.iter().map(AnyView::from).collect();
    Function::get_global(name)
        .unwrap()
        .call_packed(&views)
        .unwrap()
}

fn list_item(list: &Any, index: i64) -> i64 {
    Function::get_global("ffi.ListGetItem")
        .unwrap()
        .call_packed(&[AnyView::from(list), AnyView::from(&index)])
        .and_then(i64::try_from)
        .unwrap()
}

fn array_item(array: &Any, index: i64) -> Any {
    Function::get_global("ffi.ArrayGetItem")
        .unwrap()
        .call_packed(&[AnyView::from(array), AnyView::from(&index)])
        .unwrap()
}

fn dict_item(dict: &Any, key: i64) -> i64 {
    Function::get_global("ffi.DictGetItem")
        .unwrap()
        .call_packed(&[AnyView::from(dict), AnyView::from(&key)])
        .and_then(i64::try_from)
        .unwrap()
}

#[test]
fn unique_array_is_reused_while_shared_array_uses_copy_on_write() {
    let unique = Array::new(vec![1i64, 2, 3]);
    let unique_pointer = array_pointer(&unique);
    let mapped = structural_map(unique, &mut IncrementIntegers, WalkOrder::PostOrder)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(array_pointer(&mapped), unique_pointer);
    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![2, 3, 4]);

    let source = Array::new(vec![4i64, 5]);
    let source_pointer = array_pointer(&source);
    let mapped = structural_map(source.clone(), &mut IncrementIntegers, WalkOrder::PostOrder)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_ne!(array_pointer(&mapped), source_pointer);
    assert_eq!(source.iter().collect::<Vec<_>>(), vec![4, 5]);
    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![5, 6]);
}

#[test]
fn user_driven_mutator_controls_default_recursion_and_in_place_opt_in() {
    let unique = Array::new(vec![1i64, 2]);
    let unique_pointer = array_pointer(&unique);
    let mutated =
        structural_mutate::<Array<i64>, ManualIncrement>(unique, &mut ManualIncrement::default())
            .and_then(Array::<i64>::try_from)
            .unwrap();
    assert_eq!(array_pointer(&mutated), unique_pointer);
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);

    let source = Array::new(vec![3i64]);
    let source_pointer = array_pointer(&source);
    let mutated = structural_mutate(source.clone(), &mut ManualIncrement::default())
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_ne!(array_pointer(&mutated), source_pointer);
    assert_eq!(source.get(0).unwrap(), 3);
    assert_eq!(mutated.get(0).unwrap(), 4);

    let map: Map<i64, i64> = [(1, 10)].into_iter().collect();
    let source_map_pointer = map_pointer(&map);
    let mutated_map = structural_mutate(map, &mut ManualIncrement::default())
        .and_then(Map::<i64, i64>::try_from)
        .unwrap();
    assert_eq!(map_pointer(&mutated_map), source_map_pointer);
    assert_eq!(mutated_map.get(&1).unwrap(), Some(11));

    let dict = call_global("ffi.Dict", &[Any::from(1i64), Any::from(10i64)]);
    let dict_pointer = any_object_pointer(&dict);
    let mutated_dict = structural_mutate(dict, &mut ManualIncrement::default()).unwrap();
    assert_eq!(any_object_pointer(&mutated_dict), dict_pointer);
    assert_eq!(dict_item(&mutated_dict, 1), 11);
}

#[test]
fn none_values_are_dispatched_to_map_callbacks_and_user_mutators() {
    let mut map_calls = 0;
    let mapped = structural_map(
        Any::new(),
        |value: &MapValue| {
            map_calls += 1;
            assert_eq!(value.type_index(), TypeIndex::kTVMFFINone as i32);
            Any::from(7i64)
        },
        WalkOrder::PostOrder,
    )
    .and_then(i64::try_from)
    .unwrap();
    assert_eq!(mapped, 7);
    assert_eq!(map_calls, 1);

    let mut mutator = ReplaceNone::default();
    let mutated = structural_mutate(Any::new(), &mut mutator)
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(mutated, 8);
    assert_eq!(mutator.calls, 1);
}

#[test]
fn user_mutator_recursive_entries_reenter_the_same_mutator() {
    let mut borrowed = RecursiveEntryMutator {
        remap: StructuralVarRemap::default(),
        use_owned_value: false,
        owned_value_pointer: None,
    };
    let mutated = structural_mutate(Any::new(), &mut borrowed)
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(mutated, 2);

    let mut owned = RecursiveEntryMutator {
        remap: StructuralVarRemap::default(),
        use_owned_value: true,
        owned_value_pointer: None,
    };
    let mutated = structural_mutate(Any::new(), &mut owned)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(
        array_pointer(&mutated) as usize,
        owned.owned_value_pointer.unwrap()
    );
    assert_eq!(mutated.get(0).unwrap(), 2);
}

#[test]
fn reflected_fields_use_shallow_copy_and_setters() {
    let source = reflected_object();
    let mut regions = Vec::new();
    let mapped = structural_map(
        source.clone(),
        |integer: i64, kind: DefRegionKind| {
            regions.push(kind);
            Any::from(integer + 1)
        },
        WalkOrder::PostOrder,
    )
    .unwrap();

    assert_ne!(any_object_pointer(&mapped), any_object_pointer(&source));
    assert_eq!(reflected_field::<i64>(&source, "v_i64"), 1);
    assert_eq!(reflected_field::<i64>(&mapped, "v_i64"), 2);
    assert_eq!(reflected_field::<f64>(&mapped, "v_f64"), 2.5);
    assert_eq!(
        reflected_field::<FfiString>(&mapped, "v_str").as_str(),
        "a reflected string"
    );
    assert_eq!(regions, vec![DefRegionKind::None]);
}

#[test]
fn reflected_no_change_returns_original() {
    let source = reflected_object();
    let mapped = structural_map(
        source.clone(),
        |string: FfiString| Any::from(string),
        WalkOrder::PostOrder,
    )
    .unwrap();

    assert_eq!(any_object_pointer(&mapped), any_object_pointer(&source));
}

#[test]
fn reflected_object_without_shallow_copy_is_rejected_even_when_unchanged() {
    // Keep the C++ test library linked for its startup registrations.
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    // This existing C++ test type deletes its copy constructor.
    let source = Function::from_type_key_method("testing.TestNonCopyable", "__ffi_init__")
        .unwrap()
        .call_tuple((1i64,))
        .unwrap();
    // Leave its integer field unmatched to test the unchanged-object path.
    let error = match structural_map(
        source,
        |string: FfiString| Any::from(string),
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("reflected object without a shallow-copy hook unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.message().contains("__ffi_shallow_copy__"));
}

#[test]
fn callback_errors_preserve_message_and_add_object_context() {
    let error = match structural_map(
        Array::new(vec![1i64]),
        |_integer: i64| -> Result<i64> {
            Err(Error::new(RUNTIME_ERROR, "mapper failed", "origin"))
        },
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("fallible structural mapper unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_eq!(error.message(), "mapper failed");
    assert!(error.backtrace().contains("origin"));
    assert!(error.backtrace().contains("object `ffi.Array`"));

    let error = match structural_mutate(
        Array::new(vec![1i64]),
        |_integer: i64, _mutator: &mut CallbackMutator| -> Result<i64> {
            Err(Error::new(
                RUNTIME_ERROR,
                "callback mutator failed",
                "callback origin",
            ))
        },
    ) {
        Ok(_) => panic!("fallible callback mutator unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.message(), "callback mutator failed");
    assert!(error.backtrace().contains("callback origin"));
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn callback_panics_resume_after_the_registered_hook_returns() {
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        structural_map(
            Array::new(vec![1i64]),
            |_integer: i64| -> Any { panic!("mapper panic") },
            WalkOrder::PostOrder,
        )
    })) {
        Err(panic) => panic,
        Ok(_) => panic!("panicking mapper unexpectedly returned"),
    };

    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some("mapper panic"));

    let mapped = structural_map(
        Array::new(vec![1i64]),
        |integer: i64| Any::from(integer + 1),
        WalkOrder::PostOrder,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mapped.get(0).unwrap(), 2);
}

#[test]
fn unique_map_reuses_nested_unique_value_storage() {
    let child = Array::new(vec![1i64, 2]);
    let child_pointer = array_pointer(&child);
    let source: Map<i64, Array<i64>> = [(1, child)].into_iter().collect();
    let source_pointer = map_pointer(&source);

    let mapped = structural_map(source, &mut IncrementIntegers, WalkOrder::PostOrder)
        .and_then(Map::<i64, Array<i64>>::try_from)
        .unwrap();
    let mapped_child = mapped.get(&1).unwrap().unwrap();

    assert_eq!(map_pointer(&mapped), source_pointer);
    assert_eq!(array_pointer(&mapped_child), child_pointer);
    assert_eq!(mapped_child.iter().collect::<Vec<_>>(), vec![2, 3]);
}

#[test]
fn shared_map_and_dict_copy_only_when_a_value_changes() {
    let source: Map<i64, i64> = [(1, 10)].into_iter().collect();
    let source_pointer = map_pointer(&source);
    let unchanged = structural_map(
        source.clone(),
        |value: FfiString| Any::from(value),
        WalkOrder::PostOrder,
    )
    .and_then(Map::<i64, i64>::try_from)
    .unwrap();
    assert_eq!(map_pointer(&unchanged), source_pointer);

    let mapped = structural_map(source.clone(), &mut IncrementIntegers, WalkOrder::PostOrder)
        .and_then(Map::<i64, i64>::try_from)
        .unwrap();
    assert_ne!(map_pointer(&mapped), source_pointer);
    assert_eq!(source.get(&1).unwrap(), Some(10));
    assert_eq!(mapped.get(&1).unwrap(), Some(11));

    let dict = call_global("ffi.Dict", &[Any::from(1i64), Any::from(10i64)]);
    let dict_pointer = any_object_pointer(&dict);
    let unchanged_dict = structural_map(
        dict.clone(),
        |value: FfiString| Any::from(value),
        WalkOrder::PostOrder,
    )
    .unwrap();
    assert_eq!(any_object_pointer(&unchanged_dict), dict_pointer);

    let mapped_dict =
        structural_map(dict.clone(), &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    assert_ne!(any_object_pointer(&mapped_dict), dict_pointer);
    assert_eq!(dict_item(&dict, 1), 10);
    assert_eq!(dict_item(&mapped_dict, 1), 11);
}

#[test]
fn shared_map_callback_error_preserves_source_and_reports_object_context() {
    let source: Map<i64, i64> = [(1, 10), (2, 20)].into_iter().collect();
    let error = match structural_map(
        source.clone(),
        |_integer: i64| -> Result<i64> {
            Err(Error::new(RUNTIME_ERROR, "map mapper failed", "origin"))
        },
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("fallible map mapper unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_eq!(source.get(&1).unwrap(), Some(10));
    assert_eq!(source.get(&2).unwrap(), Some(20));
    assert_eq!(error.message(), "map mapper failed");
    assert!(error.backtrace().contains("origin"));
    assert!(error.backtrace().contains("object `ffi.Map`"));
}

#[test]
fn shared_outer_container_does_not_mutate_its_nested_child() {
    let nested = call_global("ffi.List", &[Any::from(1i64)]);
    let nested_pointer = any_object_pointer(&nested);
    let outer = call_global("ffi.Array", &[nested]);
    // The temporary argument array is dropped above, leaving the parent cell
    // as the nested List's only owning reference.
    let outer_alias = outer.clone();

    let mapped = structural_map(outer, &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    let source_nested = array_item(&outer_alias, 0);
    let mapped_nested = array_item(&mapped, 0);
    assert_eq!(any_object_pointer(&source_nested), nested_pointer);
    assert_ne!(any_object_pointer(&mapped_nested), nested_pointer);
    assert_eq!(list_item(&source_nested, 0), 1);
    assert_eq!(list_item(&mapped_nested, 0), 2);
}

#[test]
fn shared_list_uses_copy_on_write() {
    let source = call_global("ffi.List", &[Any::from(1i64), Any::from(2i64)]);
    let source_pointer = any_object_pointer(&source);
    let mapped =
        structural_map(source.clone(), &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();

    assert_ne!(any_object_pointer(&mapped), source_pointer);
    assert_eq!((list_item(&mapped, 0), list_item(&mapped, 1)), (2, 3));
    assert_eq!((list_item(&source, 0), list_item(&source, 1)), (1, 2));
}

#[derive(Default)]
struct GeneratedMapper {
    integers: Vec<(i64, DefRegionKind)>,
    catch_all: usize,
}

#[dispatch(map)]
impl GeneratedMapper {
    fn map_integer(&mut self, value: i64, kind: DefRegionKind) -> Any {
        self.integers.push((value, kind));
        Any::from(value + 1)
    }

    fn map_any(&mut self, value: &MapValue) -> Result<Any> {
        self.catch_all += 1;
        Ok(value.to_owned())
    }
}

#[test]
fn generated_map_dispatch_supports_kind_and_ordered_catch_all() {
    let mut mapper = GeneratedMapper::default();
    let mapped = structural_map(Array::new(vec![1i64, 2]), &mut mapper, WalkOrder::PostOrder)
        .and_then(Array::<i64>::try_from)
        .unwrap();

    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(
        mapper.integers,
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None),]
    );
    assert_eq!(mapper.catch_all, 1);
}

#[derive(Default)]
struct GeneratedLeafDispatch {
    integers: Vec<(i64, DefRegionKind)>,
}

#[dispatch(mutate)]
impl GeneratedLeafDispatch {
    fn mutate_integer(&mut self, value: i64, mutator: &mut Mutator) -> Any {
        let region = mutator.region();
        self.integers.push((value, region));
        Any::from(value + 1)
    }
}

struct GeneratedStatelessDispatch;

#[dispatch(mutate)]
impl GeneratedStatelessDispatch {
    fn mutate_integer(&mut self, value: i64) -> i64 {
        value + 1
    }
}

#[test]
fn generated_stateless_mutate_dispatch_is_a_direct_callback() {
    assert_eq!(
        structural_mutate(1i64, GeneratedStatelessDispatch)
            .and_then(i64::try_from)
            .unwrap(),
        2
    );
}

#[test]
fn generated_mutate_dispatch_defaults_unmatched_values_and_preserves_inplace_permit() {
    let root = Array::new(vec![1i64, 2]);
    let root_pointer = array_pointer(&root);
    let mut mutator = GeneratedLeafDispatch::default();
    let mutated = structural_mutate(root, &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();

    assert_eq!(array_pointer(&mutated), root_pointer);
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(
        mutator.integers,
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None)]
    );
}

#[derive(Default)]
struct GeneratedRecursiveDispatch {
    arrays: Vec<DefRegionKind>,
    integers: Vec<(i64, DefRegionKind)>,
}

#[dispatch(mutate)]
impl GeneratedRecursiveDispatch {
    fn mutate_array(&mut self, array: Array<i64>, mutator: &mut Mutator) -> Result<Array<i64>> {
        let region = mutator.region();
        self.arrays.push(region);
        let mut mutated = Vec::with_capacity(array.len());
        for value in array.iter() {
            mutated.push(i64::try_from(mutator.mutate(self, &value)?)?);
        }
        Ok(Array::new(mutated))
    }

    fn mutate_integer(&mut self, value: i64, mutator: &mut Mutator) -> Any {
        let region = mutator.region();
        self.integers.push((value, region));
        Any::from(value + 10)
    }
}

#[test]
fn generated_mutate_dispatch_recurses_through_context() {
    let mut mutator = GeneratedRecursiveDispatch::default();
    let mutated = structural_mutate(Array::new(vec![1i64, 2]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();

    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![11, 12]);
    assert_eq!(mutator.arrays, vec![DefRegionKind::None]);
    assert_eq!(
        mutator.integers,
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None)]
    );
}

#[derive(Default)]
struct GeneratedDefaultingDispatch {
    arrays: usize,
    integers: Vec<i64>,
}

#[dispatch(mutate)]
impl GeneratedDefaultingDispatch {
    fn mutate_array(&mut self, _array: Array<i64>, mutator: &mut Mutator) -> Result<Any> {
        self.arrays += 1;
        mutator.default_mutate(self)
    }

    fn mutate_integer(&mut self, value: i64) -> Any {
        self.integers.push(value);
        Any::from(value + 1)
    }
}

#[test]
fn generated_mutate_dispatch_can_default_recurse_from_a_typed_handler() {
    let mut mutator = GeneratedDefaultingDispatch::default();
    let mutated = structural_mutate(Array::new(vec![1i64, 2]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();

    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(mutator.arrays, 1);
    assert_eq!(mutator.integers, vec![1, 2]);
}

#[test]
fn pre_order_retained_alias_disables_in_place_mutation() {
    let root = call_global("ffi.List", &[Any::from(1i64)]);
    let root_pointer = any_object_pointer(&root);
    let mut retained = None;
    let mapped = structural_map(
        root,
        |value: &MapValue| {
            if value.type_index() == TypeIndex::kTVMFFIList as i32 {
                retained = Some(value.to_owned());
                value.to_owned()
            } else if let Some(integer) = value.cast::<i64>() {
                Any::from(integer + 1)
            } else {
                value.to_owned()
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let retained = retained.unwrap();

    assert_eq!(any_object_pointer(&retained), root_pointer);
    assert_ne!(any_object_pointer(&mapped), root_pointer);
    assert_eq!(list_item(&retained, 0), 1);
    assert_eq!(list_item(&mapped, 0), 2);
}

#[test]
fn closures_and_tuples_use_ordered_first_match() {
    let root = Array::new(vec![1i64, 2]);
    let mapped = structural_map(
        root,
        |integer: i64| Any::from(integer + 10),
        WalkOrder::PostOrder,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![11, 12]);

    let mut first_calls = 0;
    let mut later_calls = 0;
    let mapped = structural_map(
        Array::new(vec![3i64]),
        (
            |integer: i64| {
                first_calls += 1;
                Any::from(integer + 1)
            },
            |integer: i64| {
                later_calls += 1;
                Any::from(integer + 100)
            },
        ),
        WalkOrder::PostOrder,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mapped.get(0).unwrap(), 4);
    assert_eq!(first_calls, 1);
    assert_eq!(later_calls, 0);
}

#[test]
fn callbacks_return_values_convertible_into_any() {
    let mapped = structural_map(
        Array::new(vec![1i64, 2]),
        |integer: i64| integer + 10,
        WalkOrder::PostOrder,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![11, 12]);

    let mapped = structural_map(
        Array::new(vec![1i64, 2]),
        |integer: i64| -> Result<i64> { Ok(integer + 20) },
        WalkOrder::PostOrder,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![21, 22]);

    let mutated = structural_mutate(
        Array::new(vec![1i64, 2]),
        |integer: i64, _mutator: &mut CallbackMutator| integer * 2,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 4]);
}

#[test]
fn twelve_link_tuple_reaches_final_map_dispatch() {
    let mut final_dispatch = IncrementIntegers;
    let mapped = structural_map(
        1i64,
        (
            |_value: bool| Any::from(false),
            |_value: f64| Any::from(0.0f64),
            |value: FfiString| Any::from(value),
            |value: Function| Any::from(value),
            |_node: &MapObj| Any::new(),
            |_node: &FunctionObj| Any::new(),
            |value: Array<i64>| Any::from(value),
            |value: Array<f64>| Any::from(value),
            |value: Array<bool>| Any::from(value),
            |value: Array<FfiString>, _kind: DefRegionKind| Any::from(value),
            |value: Map<FfiString, i64>| Any::from(value),
            &mut final_dispatch,
        ),
        WalkOrder::PostOrder,
    )
    .and_then(i64::try_from)
    .unwrap();

    assert_eq!(mapped, 2);
}

#[test]
fn nested_tuple_chain_exceeds_flat_arity() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut catch_all = 0;
    let mapped = structural_map(
        root,
        (
            (
                |_value: bool| Any::from(false),
                |_value: f64| Any::from(0.0f64),
                |value: FfiString| Any::from(value),
                |value: Function| Any::from(value),
                |_node: &MapObj| Any::new(),
                |_node: &FunctionObj| Any::new(),
                |value: Array<f64>| Any::from(value),
                |value: Array<bool>| Any::from(value),
                |value: Array<Array<i64>>| Any::from(value),
                |value: Map<FfiString, i64>| Any::from(value),
                |value: Array<Function>, _kind: DefRegionKind| Any::from(value),
                |value: Map<i64, i64>| Any::from(value),
            ),
            (
                |value: i64| Any::from(value * 10),
                (
                    |value: Array<FfiString>| Any::from(value),
                    (|value: &MapValue| {
                        catch_all += 1;
                        value.to_owned()
                    },),
                ),
            ),
        ),
        WalkOrder::PostOrder,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![10, 20, 30]);
    assert_eq!(catch_all, 1);
}

#[test]
fn callbacks_run_in_the_configured_order() {
    let root = Array::new(vec![1i64, 2]);
    let mut pre = Vec::new();
    structural_map(
        root.clone(),
        |value: &MapValue| {
            pre.push(value.cast::<i64>());
            value.to_owned()
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert_eq!(pre, vec![None, Some(1), Some(2)]);

    let mut post = Vec::new();
    structural_map(
        root,
        |value: &MapValue| {
            post.push(value.cast::<i64>());
            value.to_owned()
        },
        WalkOrder::PostOrder,
    )
    .unwrap();
    assert_eq!(post, vec![Some(1), Some(2), None]);
}

#[test]
fn map_keys_are_anchors_and_object_leaves_are_preserved() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
        .into_iter()
        .collect();
    let mut key_callbacks = 0;
    let mapped = structural_map(
        root,
        (
            |_key: FfiString| {
                key_callbacks += 1;
                Any::from(FfiString::from("changed"))
            },
            |value: i64| Any::from(value + 1),
        ),
        WalkOrder::PostOrder,
    )
    .and_then(Map::<FfiString, i64>::try_from)
    .unwrap();
    assert_eq!(key_callbacks, 0);
    assert_eq!(mapped.get(&FfiString::from("a")).unwrap(), Some(2));
    assert_eq!(mapped.get(&FfiString::from("b")).unwrap(), Some(3));

    let string = FfiString::from("leaf");
    let heterogeneous = Function::get_global("ffi.Array")
        .unwrap()
        .call_packed(&[AnyView::from(&1i64), AnyView::from(&string)])
        .unwrap();
    let mapped = structural_map(
        heterogeneous,
        |value: i64| Any::from(value + 1),
        WalkOrder::PostOrder,
    )
    .unwrap();
    let get = Function::get_global("ffi.ArrayGetItem").unwrap();
    assert_eq!(
        get.call_packed(&[AnyView::from(&mapped), AnyView::from(&0i64)])
            .and_then(i64::try_from)
            .unwrap(),
        2
    );
    assert_eq!(
        get.call_packed(&[AnyView::from(&mapped), AnyView::from(&1i64)])
            .and_then(FfiString::try_from)
            .unwrap(),
        "leaf"
    );
}

#[test]
fn callback_mutate_defaults_unmatched_values_and_preserves_root_permit() {
    let root = Array::new(vec![1i64, 2]);
    let root_pointer = array_pointer(&root);
    let mutated = structural_mutate(root, |value: i64, _mutator: &mut CallbackMutator| {
        Any::from(value + 1)
    })
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(array_pointer(&mutated), root_pointer);
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
}

#[derive(Default)]
struct CallbackMutateStats {
    integers: Vec<i64>,
    defaults: usize,
}

fn stateful_mutate_integer(value: i64, mutator: &mut CallbackMutator<CallbackMutateStats>) -> Any {
    mutator.state_mut().integers.push(value);
    Any::from(value + 1)
}

fn stateful_mutate_default(
    _value: &MapValue,
    mutator: &mut CallbackMutator<CallbackMutateStats>,
) -> Result<Any> {
    mutator.state_mut().defaults += 1;
    mutator.default_mutate()
}

#[test]
fn callback_mutator_carries_reusable_mutable_state() {
    let mut mutator = MutateCallbacks::new(
        CallbackMutateStats::default(),
        (stateful_mutate_integer, stateful_mutate_default),
    );

    let first = structural_mutate(Array::new(vec![1i64, 2]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(first.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(mutator.state().integers, vec![1, 2]);
    assert_eq!(mutator.state().defaults, 1);

    let second = structural_mutate(Array::new(vec![3i64]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(second.iter().collect::<Vec<_>>(), vec![4]);

    let state = mutator.into_state();
    assert_eq!(state.integers, vec![1, 2, 3]);
    assert_eq!(state.defaults, 2);
}

#[derive(Default)]
struct CallbackMutateDepth {
    current: usize,
    maximum: usize,
    exits: usize,
}

fn stateful_mutate_recursive(
    value: &MapValue,
    mutator: &mut CallbackMutator<CallbackMutateDepth>,
) -> Result<Any> {
    assert_eq!(mutator.current().type_index(), value.type_index());
    {
        let state = mutator.state_mut();
        state.current += 1;
        state.maximum = state.maximum.max(state.current);
    }
    let mutated = mutator.default_mutate()?;
    {
        let state = mutator.state_mut();
        state.current -= 1;
        state.exits += 1;
    }
    Ok(mutated)
}

#[test]
fn callback_mutator_state_can_change_around_recursive_reborrows() {
    let root = Array::new(vec![Array::new(vec![1i64, 2])]);
    let mut mutator =
        MutateCallbacks::new(CallbackMutateDepth::default(), stateful_mutate_recursive);
    let mutated = structural_mutate(root, &mut mutator)
        .and_then(Array::<Array<i64>>::try_from)
        .unwrap();

    assert_eq!(
        mutated.get(0).unwrap().iter().collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(mutator.state().current, 0);
    assert_eq!(mutator.state().maximum, 3);
    assert_eq!(mutator.state().exits, 4);
}

#[test]
fn callback_mutate_current_default_is_repeatable_copy_path() {
    let root = Array::new(vec![1i64, 2]);
    let root_pointer = array_pointer(&root);
    let defaults = Cell::new(0);
    let mutated = structural_mutate(
        root,
        (
            |value: i64, _mutator: &mut CallbackMutator| Any::from(value + 1),
            |_value: &MapValue, mutator: &mut CallbackMutator| -> Result<Any> {
                defaults.set(defaults.get() + 1);
                let first = mutator.default_mutate()?;
                let second = mutator.default_mutate()?;
                assert_ne!(any_object_pointer(&first), any_object_pointer(&second));
                Ok(first)
            },
        ),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_ne!(array_pointer(&mutated), root_pointer);
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(defaults.get(), 1);
}

#[test]
fn callback_mutate_match_is_final_and_same_fn_can_reenter() {
    let integer_calls = Cell::new(0);
    let mutated = structural_mutate(
        Array::new(vec![1i64]),
        (
            |_array: Array<i64>, _mutator: &mut CallbackMutator| Any::from(Array::new(vec![10i64])),
            |value: i64, _mutator: &mut CallbackMutator| {
                integer_calls.set(integer_calls.get() + 1);
                Any::from(value + 1)
            },
        ),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![10]);
    assert_eq!(integer_calls.get(), 0);

    let calls = Cell::new(0);
    let mutated = structural_mutate(
        Array::new(vec![1i64, 2]),
        |_value: &MapValue, mutator: &mut CallbackMutator| {
            calls.set(calls.get() + 1);
            mutator.default_mutate()
        },
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(calls.get(), 3);
}

#[test]
fn callback_mutate_supports_node_links_nested_tuples_and_reflection() {
    let root = call_global(
        "ffi.Array",
        &[
            Any::from(Function::from_packed(|_| Ok(Any::new()))),
            reflected_object(),
        ],
    );
    let regions = RefCell::new(Vec::new());
    let mutated = structural_mutate(
        root,
        (
            (
                |_value: bool, _mutator: &mut CallbackMutator| Any::new(),
                |_node: &FunctionObj, _mutator: &mut CallbackMutator| Any::from(7i64),
            ),
            |value: i64, mutator: &mut CallbackMutator| {
                regions.borrow_mut().push(mutator.def_region_kind());
                Any::from(value + 1)
            },
        ),
    )
    .unwrap();

    assert_eq!(i64::try_from(array_item(&mutated, 0)).unwrap(), 7);
    let object = array_item(&mutated, 1);
    assert_eq!(reflected_field::<i64>(&object, "v_i64"), 2);
    assert_eq!(reflected_field::<f64>(&object, "v_f64"), 2.5);
    assert_eq!(
        reflected_field::<FfiString>(&object, "v_str").as_str(),
        "a reflected string"
    );
    assert_eq!(*regions.borrow(), vec![DefRegionKind::None]);
}

#[test]
fn callback_mutate_distinguishes_borrowed_and_owned_children() {
    let borrowed_child = Array::new(vec![1i64]);
    let borrowed_pointer = array_pointer(&borrowed_child);
    let mutated = structural_mutate(
        true,
        (
            |_value: bool, mutator: &mut CallbackMutator| mutator.mutate(&borrowed_child),
            |value: i64, _mutator: &mut CallbackMutator| Any::from(value + 1),
        ),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_ne!(array_pointer(&mutated), borrowed_pointer);
    assert_eq!(borrowed_child.get(0).unwrap(), 1);
    assert_eq!(mutated.get(0).unwrap(), 2);

    let owned_pointer = Cell::new(0usize);
    let mutated = structural_mutate(
        true,
        (
            |_value: bool, mutator: &mut CallbackMutator| {
                let child = Array::new(vec![1i64]);
                owned_pointer.set(array_pointer(&child) as usize);
                mutator.maybe_inplace_mutate(child)
            },
            |value: i64, _mutator: &mut CallbackMutator| Any::from(value + 1),
        ),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(array_pointer(&mutated) as usize, owned_pointer.get());
    assert_eq!(mutated.get(0).unwrap(), 2);
}

#[test]
fn nested_callback_mutate_restores_the_outer_active_mutator() {
    let mutated = structural_mutate(
        1i64,
        |value: i64, mutator: &mut CallbackMutator| -> Result<Any> {
            if value != 1 {
                return Ok(Any::from(value + 1));
            }
            let inner = structural_mutate(2i64, |value: i64, _mutator: &mut CallbackMutator| {
                Any::from(value + 10)
            })?;
            assert_eq!(i64::try_from(inner).unwrap(), 12);
            mutator.mutate(&3i64)
        },
    )
    .and_then(i64::try_from)
    .unwrap();
    assert_eq!(mutated, 4);
}

#[test]
fn callback_mutate_panics_resume_and_leave_the_next_run_usable() {
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        structural_mutate(
            Array::new(vec![1i64]),
            |_value: i64, _mutator: &mut CallbackMutator| -> Any {
                panic!("callback mutator panic")
            },
        )
    })) {
        Err(panic) => panic,
        Ok(_) => panic!("panicking callback mutator unexpectedly returned"),
    };
    assert_eq!(
        panic.downcast_ref::<&str>().copied(),
        Some("callback mutator panic")
    );

    let mutated = structural_mutate(
        Array::new(vec![1i64]),
        |value: i64, _mutator: &mut CallbackMutator| Any::from(value + 1),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.get(0).unwrap(), 2);
}
