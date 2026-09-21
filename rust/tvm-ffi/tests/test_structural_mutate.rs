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
    dispatch, structural_map, structural_mutate, structural_visit, structural_walk, Any, AnyView,
    Array, DefRegionKind, DefaultContextPolicy, DefaultMutContextPolicy, Error, FieldGetter,
    Function, InplaceMode, InplaceValue, IntoMapper, Map, MapDispatch, MapWithContextPolicy,
    MutContextPolicy, MutateCallbacks, MutateContext, MutateValue, Mutator, Object, ObjectArc,
    ObjectRefCore, Result, String as FfiString, StructuralMutator, StructuralVarRemap,
    StructuralView, StructuralVisitor, TypeIndex, Unchanged, UnchangedOr, VisitCallbacks,
    VisitContext, VisitInterrupt, WalkOrder, WalkResult, RUNTIME_ERROR,
};

struct IncrementIntegers;

impl MapDispatch for IncrementIntegers {
    fn dispatch_map(
        &mut self,
        value: &StructuralView,
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
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
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

    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
        self.remap.set(var, mutated_value)
    }
}

#[derive(Default)]
struct ReplaceNone {
    remap: StructuralVarRemap,
    calls: usize,
}

impl StructuralMutator for ReplaceNone {
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
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

    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
        self.remap.set(var, mutated_value)
    }
}

struct RecursiveEntryMutator {
    remap: StructuralVarRemap,
    use_owned_value: bool,
    owned_value_pointer: Option<usize>,
}

impl StructuralMutator for RecursiveEntryMutator {
    fn dispatch_mutate(
        &mut self,
        value: &StructuralView,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
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

    fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
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
        |value: &StructuralView| {
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
fn default_mutation_mode_preserves_ownership_and_unchanged_results() {
    struct Controlled {
        mode: InplaceMode,
        increment: bool,
    }
    impl StructuralMutator for Controlled {
        fn dispatch_mutate(&mut self, value: &StructuralView, _: DefRegionKind) -> Result<Any> {
            let integer = value.cast::<i64>().unwrap();
            Ok(if self.increment {
                Any::from(integer + 1)
            } else {
                Unchanged.into()
            })
        }
        fn dispatch_maybe_inplace_mutate(
            &mut self,
            value: InplaceValue<'_>,
            kind: DefRegionKind,
        ) -> Result<Any> {
            if self.increment {
                self.default_maybe_inplace_mutate_with_mode(value, kind, self.mode)
            } else {
                let result =
                    self.default_maybe_inplace_mutate_with_mode_result(value, kind, self.mode)?;
                assert!(result.is_unchanged());
                Ok(result.into())
            }
        }
    }
    for mode in [InplaceMode::Disallow, InplaceMode::Allow] {
        for increment in [false, true] {
            let root = Array::new(vec![1_i64]);
            let pointer = array_pointer(&root);
            let result = structural_mutate(root, &mut Controlled { mode, increment })
                .and_then(Array::<i64>::try_from)
                .unwrap();
            assert_eq!(
                array_pointer(&result) == pointer,
                mode == InplaceMode::Allow || !increment
            );
            assert_eq!(result.get(0).unwrap(), if increment { 2 } else { 1 });
        }
    }
}

#[test]
fn owned_entry_mode_is_forwarded_by_generated_and_closure_callbacks() {
    struct Entry {
        mode: InplaceMode,
        pointer: usize,
    }
    #[dispatch(mutate)]
    impl Entry {
        fn mutate_bool(&mut self, _: bool, ctx: &mut Mutator) -> Result<Any> {
            let child = Array::new(vec![1_i64]);
            self.pointer = array_pointer(&child) as usize;
            ctx.maybe_inplace_mutate_with_mode(self, child, DefRegionKind::Pattern, self.mode)
        }
        fn mutate_integer(&mut self, value: i64, ctx: &mut Mutator) -> i64 {
            assert_eq!(ctx.def_region_kind(), DefRegionKind::Pattern);
            value + 1
        }
    }
    assert_eq!(InplaceMode::default(), InplaceMode::Disallow);
    for mode in [InplaceMode::Disallow, InplaceMode::Allow] {
        let mut entry = Entry { mode, pointer: 0 };
        let result = structural_mutate(true, &mut entry)
            .and_then(Array::<i64>::try_from)
            .unwrap();
        assert_eq!(
            array_pointer(&result) as usize == entry.pointer,
            mode == InplaceMode::Allow
        );
        assert_eq!(result.get(0).unwrap(), 2);

        let pointer = Cell::new(0usize);
        let result = structural_mutate(
            true,
            (
                |_: bool, ctx: &mut MutateContext<'_>| {
                    let child = Array::new(vec![1_i64]);
                    pointer.set(array_pointer(&child) as usize);
                    ctx.maybe_inplace_mutate_with_mode(child, DefRegionKind::Pattern, mode)
                },
                |value: i64, ctx: &mut MutateContext<'_>| {
                    assert_eq!(ctx.def_region_kind(), DefRegionKind::Pattern);
                    value + 1
                },
            ),
        )
        .and_then(Array::<i64>::try_from)
        .unwrap();
        assert_eq!(
            array_pointer(&result) as usize == pointer.get(),
            mode == InplaceMode::Allow
        );
        assert_eq!(result.get(0).unwrap(), 2);
    }
}

#[test]
fn consuming_callbacks_forward_permissions_without_temporary_owners() {
    struct Forward {
        requested: InplaceMode,
        retain: bool,
        alias: Option<Any>,
        modes: Vec<InplaceMode>,
    }
    impl Forward {
        fn observe(&mut self, value: &MutateValue<'_, Array<Any>>) {
            self.modes.push(value.inplace_mode());
            let node = value
                .as_node::<tvm_ffi::collections::array::ArrayObj>()
                .unwrap();
            assert_eq!(node.size, 1);
            // A temporary typed handle must not permanently revoke permission.
            let temporary = value.cast::<Array<Any>>().unwrap();
            if self.retain && self.alias.is_none() {
                self.alias = Some(temporary.clone().into());
            }
            drop(temporary);
        }
    }
    #[dispatch(mutate)]
    impl Forward {
        fn mutate_string(&mut self, _: MutateValue<'_, FfiString>) -> Any {
            panic!("typed miss must preserve the capability for the next handler")
        }
        fn mutate_array(
            &mut self,
            value: MutateValue<'_, Array<Any>>,
            ctx: &mut Mutator,
        ) -> Result<UnchangedOr<Any>> {
            assert_eq!(ctx.inplace_mode(), value.inplace_mode());
            self.observe(&value);
            ctx.default_mutate_with_mode_result(self, value, self.requested)
        }
        fn mutate_integer(&mut self, value: i64, ctx: &mut Mutator) -> i64 {
            assert_eq!(ctx.inplace_mode(), InplaceMode::Disallow);
            value + 1
        }
    }
    use InplaceMode::{Allow, Disallow};
    for (requested, shared, retain) in [
        (Allow, false, false),
        (Disallow, false, false),
        (Allow, true, false),
        (Allow, false, true),
    ] {
        for generated in [true, false] {
            let child = Array::new(vec![1_i64]);
            let child_ptr = array_pointer(&child);
            let root = Array::new(vec![child]);
            let root_ptr = array_pointer(&root);
            let original = shared.then(|| root.clone());
            let mut state = Forward {
                requested,
                retain,
                alias: None,
                modes: vec![],
            };
            let result = if generated {
                structural_mutate(root, &mut state).unwrap()
            } else {
                let miss = |_: MutateValue<'_, FfiString>,
                            _: &mut MutateContext<'_, Forward>|
                 -> Any { panic!("typed miss must continue") };
                let increment = |value: i64, _: &mut MutateContext<'_, Forward>| value + 1;
                let descend = |value: MutateValue<'_, Array<Any>>,
                               ctx: &mut MutateContext<'_, Forward>| {
                    assert_eq!(value.inplace_mode(), ctx.inplace_mode());
                    ctx.state_mut().observe(&value);
                    let requested = ctx.state().requested;
                    ctx.default_mutate_with_mode(value, requested)
                };
                let mut callbacks = MutateCallbacks::new(state, (miss, (increment, descend)));
                let result = structural_mutate(root, &mut callbacks).unwrap();
                state = callbacks.into_state();
                result
            };
            let result = Array::<Array<i64>>::try_from(result).unwrap();
            let child = result.get(0).unwrap();
            let reuse = requested == Allow && !shared && !retain;
            assert_eq!(
                state.modes,
                vec![
                    if shared { Disallow } else { Allow },
                    if reuse { Allow } else { Disallow }
                ]
            );
            assert_eq!(array_pointer(&result) == root_ptr, reuse);
            assert_eq!(array_pointer(&child) == child_ptr, reuse);
            assert_eq!(child.get(0).unwrap(), 2);
            for alias in original.map(Any::from).into_iter().chain(state.alias) {
                let alias = Array::<Array<i64>>::try_from(alias).unwrap();
                assert_eq!(alias.get(0).unwrap().get(0).unwrap(), 1);
            }
        }
    }
}

#[test]
fn consuming_default_descent_transfers_between_contexts_without_node_borrows() {
    struct Inner<'a> {
        value: Option<MutateValue<'a>>,
        preserve_unchanged: bool,
        result: Any,
    }
    #[dispatch(mutate)]
    impl Inner<'_> {
        fn mutate_bool(&mut self, _: bool, ctx: &mut Mutator) -> Result<Any> {
            let value = self.value.take().unwrap();
            self.result = if self.preserve_unchanged {
                ctx.default_maybe_inplace_mutate_result(self, value)?.into()
            } else {
                ctx.default_maybe_inplace_mutate(self, value)?
            };
            Ok(Any::new())
        }
        fn mutate_integer(&mut self, value: i64) -> i64 {
            value + 1
        }
    }
    fn transfer(
        value: MutateValue<'_>,
        generated: bool,
        preserve: bool,
        retain: bool,
    ) -> Result<Any> {
        let alias = retain.then(|| value.cast::<Array<i64>>().unwrap());
        assert_eq!(value.inplace_mode(), InplaceMode::Allow);
        // Carry the result back to the array callback: Unchanged belongs to
        // that input, not to the nested traversal's boolean root.
        let mut inner = Inner {
            value: Some(value),
            preserve_unchanged: preserve,
            result: Any::new(),
        };
        let result = if generated {
            structural_mutate(true, &mut inner)?;
            inner.result
        } else {
            let mut callbacks = MutateCallbacks::new(
                inner,
                (
                    |_: bool, ctx: &mut MutateContext<'_, Inner<'_>>| -> Result<Any> {
                        let value = ctx.state_mut().value.take().unwrap();
                        let result = if ctx.state().preserve_unchanged {
                            ctx.default_maybe_inplace_mutate_result(value)?.into()
                        } else {
                            ctx.default_maybe_inplace_mutate(value)?
                        };
                        ctx.state_mut().result = result;
                        Ok(Any::new())
                    },
                    |value: i64, _: &mut MutateContext<'_, Inner<'_>>| value + 1,
                ),
            );
            structural_mutate(true, &mut callbacks)?;
            callbacks.into_state().result
        };
        if let Some(alias) = alias {
            assert_eq!(alias.get(0)?, 1);
        }
        Ok(result)
    }
    struct Outer {
        preserve: bool,
        retain: bool,
    }
    #[dispatch(mutate)]
    impl Outer {
        fn mutate_any(&mut self, value: MutateValue<'_>) -> Result<Any> {
            transfer(value, true, self.preserve, self.retain)
        }
    }
    for preserve in [false, true] {
        for retain in [false, true] {
            for generated in [false, true] {
                let root = Array::new(vec![1_i64]);
                let pointer = array_pointer(&root);
                let result = if generated {
                    structural_mutate(root, &mut Outer { preserve, retain })
                } else {
                    structural_mutate(root, |value: MutateValue<'_>, _: &mut MutateContext<'_>| {
                        transfer(value, false, preserve, retain)
                    })
                }
                .and_then(Array::<i64>::try_from)
                .unwrap();
                assert_eq!(array_pointer(&result) == pointer, !retain);
                assert_eq!(result.get(0).unwrap(), 2);
            }
        }
    }
}

#[test]
fn consuming_default_descent_preserves_unchanged_and_propagates_errors() {
    for fail in [false, true] {
        let root = Array::new(vec![1_i64]);
        let pointer = array_pointer(&root);
        let result = structural_mutate(
            root,
            |value: MutateValue<'_>, ctx: &mut MutateContext<'_>| -> Result<UnchangedOr<Any>> {
                if value.cast::<i64>().is_some() {
                    return if fail {
                        Err(Error::new(RUNTIME_ERROR, "child failed", ""))
                    } else {
                        Ok(UnchangedOr::unchanged())
                    };
                }
                let result = ctx.default_maybe_inplace_mutate_result(value)?;
                assert!(result.is_unchanged());
                Ok(result)
            },
        );
        if fail {
            assert!(result.err().unwrap().to_string().contains("child failed"));
        } else {
            let result = Array::<i64>::try_from(result.unwrap()).unwrap();
            assert_eq!(array_pointer(&result), pointer);
        }
    }
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
fn native_unchanged_hook_returns_original_without_exposing_marker() {
    let source = Any::from(FfiString::from("unchanged"));
    let source_pointer = any_object_pointer(&source);
    let mutated = structural_mutate(source, &mut ManualIncrement::default()).unwrap();

    assert_eq!(any_object_pointer(&mutated), source_pointer);
    assert_ne!(mutated.type_index(), TypeIndex::kTVMFFIUnchanged as i32);
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
    struct Delegate(Error);
    #[dispatch(visit, policy = (DefaultContextPolicy, DefaultContextPolicy))]
    impl Delegate {
        fn visit_any(
            &mut self,
            value: &StructuralView,
            kind: DefRegionKind,
        ) -> Result<Option<VisitInterrupt>> {
            if value.cast::<i64>().is_some() {
                return Err(self.0.clone());
            }
            self.default_visit_children(value, kind)
        }
    }
    #[dispatch(mutate, policy = (DefaultMutContextPolicy, DefaultMutContextPolicy))]
    impl Delegate {
        fn mutate_any(&mut self, value: MutateValue<'_>, ctx: &mut Mutator) -> Result<Any> {
            if value.cast::<i64>().is_some() {
                return Err(self.0.clone());
            }
            ctx.default_maybe_inplace_mutate(self, value)
        }
    }
    let child = reflected_object();
    let root = call_global(
        "ffi.MakeObjectFromPackedArgs",
        &[
            FfiString::from("testing.TestObjectPtrHolder").into(),
            FfiString::from("value").into(),
            child.clone(),
        ],
    );
    let payload = ObjectRef::try_from(Any::from(Array::new(vec![7i64]))).unwrap();
    let records = call_global("ffi.List", &[Any::new()]);
    let context = call_global(
        "ffi.MakeObjectFromPackedArgs",
        &[
            FfiString::from("ffi.VisitErrorContext").into(),
            FfiString::from("reverse_visit_pattern").into(),
            records,
            FfiString::from("prev_error_context").into(),
            Any::from(payload.clone()),
        ],
    );
    let cause = Error::new(RUNTIME_ERROR, "cause", "");
    let source = Error::new_with_cause_and_extra_context(
        RUNTIME_ERROR,
        "callback failed",
        "origin",
        Some(&cause),
        Some(&ObjectRef::try_from(context.clone()).unwrap()),
    );
    let mut errors = Vec::new();
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let leaf_error = structural_map(
            1i64,
            |_value: i64| -> Result<i64> { Err(source.clone()) },
            order,
        )
        .err()
        .unwrap();
        assert!(leaf_error.same_as(&source));
        errors.push(
            structural_map(
                root.clone(),
                |_value: i64| -> Result<i64> { Err(source.clone()) },
                order,
            )
            .err()
            .unwrap(),
        );
        errors.push(
            structural_walk(
                &root,
                |_value: i64| -> Result<WalkResult> { Err(source.clone()) },
                order,
            )
            .err()
            .unwrap(),
        );
        let mut mapper = MapWithContextPolicy::new(
            (|_: i64| -> Result<i64> { Err(source.clone()) }).into_mapper(),
            (DefaultMutContextPolicy, DefaultMutContextPolicy),
        );
        errors.push(
            structural_map(root.clone(), &mut mapper, order)
                .err()
                .unwrap(),
        );
    }
    errors.push(
        structural_mutate(
            root.clone(),
            |_value: i64, _: &mut MutateContext| -> Result<i64> { Err(source.clone()) },
        )
        .err()
        .unwrap(),
    );
    errors.push(
        structural_visit(
            &root,
            |_value: i64, _: &mut VisitContext<'_, ()>| -> Result<()> { Err(source.clone()) },
        )
        .err()
        .unwrap(),
    );
    let mut visitor = VisitCallbacks::new(
        (),
        |value: &StructuralView, ctx: &mut VisitContext<'_, ()>| {
            if value.cast::<i64>().is_some() {
                return Err(source.clone());
            }
            ctx.visit_children()
        },
    )
    .with_policy((DefaultContextPolicy, DefaultContextPolicy));
    errors.push(structural_visit(&root, &mut visitor).err().unwrap());
    let mut mutator =
        MutateCallbacks::new((), |value: MutateValue<'_>, ctx: &mut MutateContext| {
            if value.cast::<i64>().is_some() {
                return Err(source.clone());
            }
            ctx.default_maybe_inplace_mutate(value)
        })
        .with_policy((DefaultMutContextPolicy, DefaultMutContextPolicy));
    errors.push(structural_mutate(root.clone(), &mut mutator).err().unwrap());
    let mut delegate = Delegate(source.clone());
    errors.push(structural_visit(&root, &mut delegate).err().unwrap());
    errors.push(
        structural_mutate(root.clone(), &mut delegate)
            .err()
            .unwrap(),
    );
    let seeded = errors[0].clone();
    for error in errors {
        assert_eq!(error.message(), "callback failed");
        assert!(error.backtrace().contains("origin"));
        assert!(error
            .backtrace()
            .contains("object `testing.TestObjectBase`"));
        assert!(error.cause_chain().unwrap().same_as(&cause));
        let context = Any::from(error.extra_context().unwrap());
        assert!(reflected_field::<ObjectRef>(&context, "prev_error_context").same_as(&payload));
        let records = Any::from(reflected_field::<ObjectRef>(
            &context,
            "reverse_visit_pattern",
        ));
        let size = i64::try_from(call_global("ffi.ListSize", &[records.clone()])).unwrap();
        assert_eq!(size, 3);
        assert_eq!(
            call_global("ffi.ListGetItem", &[records.clone(), 0_i64.into()]).try_as::<()>(),
            Some(())
        );
        let innermost = call_global("ffi.ListGetItem", &[records.clone(), 1_i64.into()]);
        assert_eq!(any_object_pointer(&innermost), any_object_pointer(&child));
        let outermost = call_global("ffi.ListGetItem", &[records, (size - 1).into()]);
        assert_eq!(any_object_pointer(&outermost), any_object_pointer(&root));
        let paths = call_global(
            "ffi.VisitErrorContext.FindAccessPaths",
            &[root.clone(), context, false.into()],
        );
        let paths = paths.try_as::<Array<Any>>().unwrap();
        assert_eq!(paths.len(), 1);
        assert_eq!(
            call_global("ffi.ReprPrint", &[paths.get(0).unwrap()])
                .try_as::<FfiString>()
                .unwrap()
                .as_str(),
            "<root>.value"
        );
    }
    let records = Any::from(reflected_field::<ObjectRef>(
        &context,
        "reverse_visit_pattern",
    ));
    assert_eq!(
        i64::try_from(call_global("ffi.ListSize", &[records])).unwrap(),
        1
    );
    assert_eq!(source.backtrace(), "origin");

    // Keep nonconsecutive occurrences: child -> root -> child is a distinct path.
    let error = structural_map(
        child.clone(),
        |_: i64| -> Result<i64> { Err(seeded.clone()) },
        WalkOrder::PreOrder,
    )
    .err()
    .unwrap();
    let context = Any::from(error.extra_context().unwrap());
    let records = Any::from(reflected_field::<ObjectRef>(
        &context,
        "reverse_visit_pattern",
    ));
    assert_eq!(
        i64::try_from(call_global("ffi.ListSize", &[records.clone()])).unwrap(),
        4
    );
    for (i, expected) in [&child, &root, &child].into_iter().enumerate() {
        let node = call_global("ffi.ListGetItem", &[records.clone(), (i as i64 + 1).into()]);
        assert_eq!(any_object_pointer(&node), any_object_pointer(expected));
    }

    // The failing node may be a pre-order replacement or a post-order rebuilt parent.
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let root = Array::new(vec![0i64]);
        let mut failed_node = None;
        let error = structural_map(
            root.clone(),
            (
                |value: Array<i64>| -> Result<Any> {
                    let value = if order == WalkOrder::PreOrder {
                        Array::new(vec![1i64])
                    } else {
                        value
                    };
                    failed_node = Some(value.clone());
                    if order == WalkOrder::PreOrder {
                        Ok(value.into())
                    } else {
                        Err(Error::new(RUNTIME_ERROR, "failed", ""))
                    }
                },
                |value: i64| -> Result<i64> {
                    if order == WalkOrder::PreOrder {
                        Err(Error::new(RUNTIME_ERROR, "failed", ""))
                    } else {
                        Ok(value + 1)
                    }
                },
            ),
            order,
        )
        .err()
        .unwrap();
        let context = Any::from(error.extra_context().unwrap());
        let records = Any::from(reflected_field::<ObjectRef>(
            &context,
            "reverse_visit_pattern",
        ));
        let node = call_global("ffi.ListGetItem", &[records.clone(), 0i64.into()]);
        assert!(node
            .try_as::<ObjectRef>()
            .unwrap()
            .same_as(&failed_node.unwrap()));
        assert_eq!(
            i64::try_from(call_global("ffi.ListSize", &[records])).unwrap(),
            1
        );
        assert_eq!(root.get(0).unwrap(), 0);
    }
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

    fn map_any(&mut self, value: &tvm_ffi::StructuralView) -> Result<Any> {
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
        let region = mutator.def_region_kind();
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
        let region = mutator.def_region_kind();
        self.arrays.push(region);
        let mut mutated = Vec::with_capacity(array.len());
        for value in array.iter() {
            mutated.push(i64::try_from(mutator.mutate(self, &value)?)?);
        }
        Ok(Array::new(mutated))
    }

    fn mutate_integer(&mut self, value: i64, mutator: &mut Mutator) -> Any {
        let region = mutator.def_region_kind();
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
    preserve_unchanged: bool,
    arrays: usize,
    integers: Vec<i64>,
}

#[dispatch(mutate)]
impl GeneratedDefaultingDispatch {
    fn mutate_array(&mut self, array: Array<i64>, mutator: &mut Mutator) -> Result<Any> {
        self.arrays += 1;
        if self.preserve_unchanged {
            mutator.default_mutate_result(self, &array).map(Into::into)
        } else {
            mutator.default_mutate(self, &array)
        }
    }

    fn mutate_integer(&mut self, value: i64) -> Any {
        self.integers.push(value);
        Any::from(value + 1)
    }
}

#[test]
fn generated_mutate_dispatch_can_default_recurse_from_a_typed_handler() {
    for preserve_unchanged in [false, true] {
        let mut mutator = GeneratedDefaultingDispatch {
            preserve_unchanged,
            ..Default::default()
        };
        let mutated = structural_mutate(Array::new(vec![1i64, 2]), &mut mutator)
            .and_then(Array::<i64>::try_from)
            .unwrap();

        assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(mutator.arrays, 1);
        assert_eq!(mutator.integers, vec![1, 2]);
    }
}

#[test]
fn pre_order_mapping_preserves_inplace_permission() {
    struct Observe<'a>(&'a Cell<Option<InplaceMode>>);
    impl<State> MutContextPolicy<State> for Observe<'_> {
        fn default_mutate(
            &self,
            value: MutateValue<'_>,
            ctx: &mut tvm_ffi::MutateContext<'_, State>,
        ) -> Result<UnchangedOr<Any>> {
            if value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                self.0.set(Some(value.inplace_mode()));
            }
            ctx.default_maybe_inplace_mutate_result(value)
        }
    }
    for with_policy in [false, true] {
        for case in ["inline", "unique", "shared", "retained"] {
            let root = match case {
                "inline" => Any::from(true),
                "retained" => call_global("ffi.List", &[Any::from(1_i64)]),
                _ => Any::from(Array::new(vec![true])),
            };
            let root_pointer = (case == "retained").then(|| any_object_pointer(&root));
            let alias = (case == "shared").then(|| root.clone());
            let mut retained = None;
            let mut pointer = std::ptr::null();
            let mode = Cell::new(None);
            let mut mapper = (|value: &StructuralView| {
                if let Some(integer) = value.cast::<i64>() {
                    return Any::from(integer + 1);
                }
                let mapped = if case == "retained" {
                    retained = Some(value.to_owned());
                    value.to_owned()
                } else {
                    Array::new(vec![1_i64]).into()
                };
                pointer = any_object_pointer(&mapped);
                mapped
            })
            .into_mapper();
            let output = if with_policy {
                structural_map(
                    root,
                    MapWithContextPolicy::new(&mut mapper, Observe(&mode)),
                    WalkOrder::PreOrder,
                )
                .unwrap()
            } else {
                structural_map(root, &mut mapper, WalkOrder::PreOrder).unwrap()
            };
            let reuse = case == "unique";
            assert_eq!(
                any_object_pointer(&output) == pointer,
                reuse,
                "{case}, policy={with_policy}"
            );
            if with_policy {
                assert_eq!(
                    mode.get(),
                    Some(if reuse {
                        InplaceMode::Allow
                    } else {
                        InplaceMode::Disallow
                    })
                );
            }
            if case == "retained" {
                let retained = retained.unwrap();
                assert_eq!(any_object_pointer(&retained), root_pointer.unwrap());
                assert_eq!(list_item(&retained, 0), 1);
                assert_eq!(list_item(&output, 0), 2);
            } else {
                assert_eq!(Array::<i64>::try_from(output).unwrap().get(0).unwrap(), 2);
            }
            if let Some(alias) = alias {
                assert!(Array::<bool>::try_from(alias).unwrap().get(0).unwrap());
            }
        }
    }
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
        |integer: i64, _mutator: &mut MutateContext<'_>| integer * 2,
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 4]);
}

#[test]
fn recursive_mutate_returns_unchanged_or_a_replacement() {
    fn clamp_negative_integers(
        value: &StructuralView,
        mutator: &mut MutateContext<'_>,
    ) -> Result<UnchangedOr<Any>> {
        if let Some(integer) = value.cast::<i64>() {
            if integer >= 0 {
                // Keep this input without constructing an owning return value.
                return Ok(UnchangedOr::unchanged());
            }
            return Ok(UnchangedOr::changed(Any::from(0i64)));
        }

        // Recurse into containers. Keep the unchanged marker if no child changed.
        mutator.default_mutate_result(value)
    }

    // No rewrite: the public entry resolves unchanged to the original array.
    let source = Array::new(vec![1i64, 2]);
    let result = structural_mutate(source.clone(), clamp_negative_integers)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(result.iter().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(array_pointer(&result), array_pointer(&source));

    // One rewrite: replace only the negative integer and keep the source intact.
    let source = Array::new(vec![-1i64, 2]);
    let result = structural_mutate(source.clone(), clamp_negative_integers)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(result.iter().collect::<Vec<_>>(), vec![0, 2]);
    assert_ne!(array_pointer(&result), array_pointer(&source));
    assert_eq!(source.iter().collect::<Vec<_>>(), vec![-1, 2]);
}

#[test]
fn pre_order_unchanged_reuses_unmodified_subtrees() {
    let unchanged = Array::new(vec![1i64, 2]);
    let changed = Array::new(vec![-1i64, 2]);
    let source = Array::new(vec![unchanged.clone(), changed.clone()]);

    let mapped = structural_map(
        source.clone(),
        (
            |integer: i64| -> UnchangedOr<i64> {
                if integer < 0 {
                    UnchangedOr::changed(0)
                } else {
                    UnchangedOr::unchanged()
                }
            },
            // Keeping an array still lets pre-order map transform its children.
            |_value: &StructuralView| Unchanged,
        ),
        WalkOrder::PreOrder,
    )
    .and_then(Array::<Array<i64>>::try_from)
    .unwrap();

    let mapped_unchanged = mapped.get(0).unwrap();
    let mapped_changed = mapped.get(1).unwrap();
    assert_eq!(mapped_unchanged.iter().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(mapped_changed.iter().collect::<Vec<_>>(), vec![0, 2]);

    // Reuse the untouched subtree and copy the shared containers that changed.
    assert_eq!(array_pointer(&mapped_unchanged), array_pointer(&unchanged));
    assert_ne!(array_pointer(&mapped_changed), array_pointer(&changed));
    assert_ne!(array_pointer(&mapped), array_pointer(&source));
    assert_eq!(
        source.get(1).unwrap().iter().collect::<Vec<_>>(),
        vec![-1, 2]
    );
}

#[test]
fn post_order_unchanged_preserves_descendant_rewrites() {
    for with_policy in [false, true] {
        for shared in [false, true] {
            let root = Array::new(vec![1_i64]);
            let pointer = array_pointer(&root);
            let alias = shared.then(|| root.clone());
            let callbacks = (
                |integer: i64| integer + 1,
                |value: &StructuralView| {
                    assert_eq!(value.cast::<Array<i64>>().unwrap().get(0).unwrap(), 2);
                    Unchanged
                },
            );
            let mapped = if with_policy {
                structural_map(
                    root,
                    MapWithContextPolicy::new(callbacks.into_mapper(), DefaultMutContextPolicy),
                    WalkOrder::PostOrder,
                )
            } else {
                structural_map(root, callbacks, WalkOrder::PostOrder)
            }
            .and_then(Array::<i64>::try_from)
            .unwrap();
            assert_eq!(mapped.get(0).unwrap(), 2);
            if let Some(alias) = alias {
                assert_eq!(alias.get(0).unwrap(), 1);
                assert_ne!(array_pointer(&mapped), pointer);
            } else {
                assert_eq!(array_pointer(&mapped), pointer);
            }
        }
    }
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
                    (|value: &StructuralView| {
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
        |value: &StructuralView| {
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
        |value: &StructuralView| {
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
    let mutated = structural_mutate(root, |value: i64, _mutator: &mut MutateContext<'_>| {
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

fn stateful_mutate_integer(
    value: i64,
    mutator: &mut MutateContext<'_, CallbackMutateStats>,
) -> Any {
    mutator.state_mut().integers.push(value);
    Any::from(value + 1)
}

fn stateful_mutate_default(
    value: &tvm_ffi::StructuralView,
    mutator: &mut MutateContext<'_, CallbackMutateStats>,
) -> Result<Any> {
    mutator.state_mut().defaults += 1;
    mutator.default_mutate(value)
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
    value: &StructuralView,
    mutator: &mut MutateContext<'_, CallbackMutateDepth>,
) -> Result<Any> {
    {
        let state = mutator.state_mut();
        state.current += 1;
        state.maximum = state.maximum.max(state.current);
    }
    let mutated = mutator.default_mutate(value)?;
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
fn callback_mutate_explicit_default_is_repeatable_copy_path() {
    let root = Array::new(vec![1i64, 2]);
    let root_pointer = array_pointer(&root);
    let defaults = Cell::new(0);
    let mutated = structural_mutate(
        root,
        (
            |value: i64, _mutator: &mut MutateContext<'_>| Any::from(value + 1),
            |value: &StructuralView, mutator: &mut MutateContext<'_>| -> Result<Any> {
                defaults.set(defaults.get() + 1);
                let first = mutator.default_mutate(value)?;
                let second = mutator.default_mutate(value)?;
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
            |_array: Array<i64>, _mutator: &mut MutateContext<'_>| {
                Any::from(Array::new(vec![10i64]))
            },
            |value: i64, _mutator: &mut MutateContext<'_>| {
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
        |value: &StructuralView, mutator: &mut MutateContext<'_>| {
            calls.set(calls.get() + 1);
            mutator.default_mutate(value)
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
                |_value: bool, _mutator: &mut MutateContext<'_>| Any::new(),
                |_node: &FunctionObj, _mutator: &mut MutateContext<'_>| Any::from(7i64),
            ),
            |value: i64, mutator: &mut MutateContext<'_>| {
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
            |_value: bool, mutator: &mut MutateContext<'_>| mutator.mutate(&borrowed_child),
            |value: i64, _mutator: &mut MutateContext<'_>| Any::from(value + 1),
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
            |_value: bool, mutator: &mut MutateContext<'_>| {
                let child = Array::new(vec![1i64]);
                owned_pointer.set(array_pointer(&child) as usize);
                mutator.maybe_inplace_mutate(child)
            },
            |value: i64, _mutator: &mut MutateContext<'_>| Any::from(value + 1),
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
        |value: i64, mutator: &mut MutateContext<'_>| -> Result<Any> {
            if value != 1 {
                return Ok(Any::from(value + 1));
            }
            let inner = structural_mutate(2i64, |value: i64, _mutator: &mut MutateContext<'_>| {
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
            |_value: i64, _mutator: &mut MutateContext<'_>| -> Any {
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
        |value: i64, _mutator: &mut MutateContext<'_>| Any::from(value + 1),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.get(0).unwrap(), 2);
}

#[derive(Default)]
struct PolicyState {
    depth: usize,
    events: Vec<(&'static str, usize)>,
}
#[dispatch(map)]
impl PolicyState {
    fn map_integer(&mut self, value: i64) -> i64 {
        self.events.push(("integer", self.depth));
        value + 1
    }
    fn map_array(&mut self, value: Array<Any>) -> Any {
        self.events.push(("callback", self.depth));
        value.into()
    }
}

struct ArrayPolicy;
impl MutContextPolicy<PolicyState> for ArrayPolicy {
    fn default_mutate(
        &self,
        value: MutateValue<'_>,
        ctx: &mut MutateContext<'_, PolicyState>,
    ) -> Result<UnchangedOr<Any>> {
        if value
            .as_node::<tvm_ffi::collections::array::ArrayObj>()
            .is_none()
        {
            return ctx.default_maybe_inplace_mutate_result(value);
        }
        ctx.state_mut().depth += 1;
        let depth = ctx.state().depth;
        ctx.state_mut().events.push(("enter", depth));
        let result = ctx.default_maybe_inplace_mutate_result(value);
        ctx.state_mut().depth -= 1;
        let depth = ctx.state().depth;
        ctx.state_mut().events.push(("exit", depth));
        result
    }
}
struct RecordPolicy;
impl MutContextPolicy<PolicyState> for RecordPolicy {
    fn default_mutate(
        &self,
        value: MutateValue<'_>,
        ctx: &mut MutateContext<'_, PolicyState>,
    ) -> Result<UnchangedOr<Any>> {
        if value
            .as_node::<tvm_ffi::collections::array::ArrayObj>()
            .is_some()
        {
            let depth = ctx.state().depth;
            ctx.state_mut().events.push(("next", depth));
        }
        ctx.default_maybe_inplace_mutate_result(value)
    }
}

#[dispatch(mutate, policy = (ArrayPolicy, RecordPolicy))]
impl PolicyState {
    fn mutate_integer(&mut self, value: i64) -> i64 {
        self.map_integer(value)
    }
}

#[test]
fn mutation_policies_share_state_and_preserve_callback_order() {
    let root = || Array::new(vec![Any::from(Array::new(vec![1_i64])), Any::from(2_i64)]);
    let pre = vec![
        ("callback", 0),
        ("enter", 1),
        ("next", 1),
        ("callback", 1),
        ("enter", 2),
        ("next", 2),
        ("integer", 2),
        ("exit", 1),
        ("integer", 1),
        ("exit", 0),
    ];
    let post = vec![
        ("enter", 1),
        ("next", 1),
        ("enter", 2),
        ("next", 2),
        ("integer", 2),
        ("exit", 1),
        ("callback", 1),
        ("integer", 1),
        ("exit", 0),
        ("callback", 0),
    ];
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut mapper = MapWithContextPolicy::new(
            PolicyState::default(),
            (ArrayPolicy, (RecordPolicy, DefaultMutContextPolicy)),
        );
        let output = structural_map(root(), &mut mapper, order).unwrap();
        assert_eq!(i64::try_from(array_item(&output, 1)).unwrap(), 3);
        assert_eq!(
            i64::try_from(array_item(&array_item(&output, 0), 0)).unwrap(),
            2
        );
        assert_eq!(mapper.state().depth, 0);
        assert_eq!(
            &mapper.state().events,
            if order == WalkOrder::PreOrder {
                &pre
            } else {
                &post
            }
        );
    }
    let mut mutator = MutateCallbacks::new(
        PolicyState::default(),
        (
            |x: i64, ctx: &mut MutateContext<'_, PolicyState>| {
                let depth = ctx.state().depth;
                ctx.state_mut().events.push(("integer", depth));
                x + 1
            },
            |value: MutateValue<'_>, ctx: &mut MutateContext<'_, PolicyState>| {
                let depth = ctx.state().depth;
                ctx.state_mut().events.push(("callback", depth));
                ctx.default_maybe_inplace_mutate_result(value)
            },
        ),
    )
    .with_policy((ArrayPolicy, RecordPolicy));
    let output = structural_mutate(root(), &mut mutator).unwrap();
    assert_eq!(i64::try_from(array_item(&output, 1)).unwrap(), 3);
    assert_eq!(mutator.state().events, pre);
    assert_eq!(mutator.state().depth, 0);

    let mut dispatch = PolicyState::default();
    let output = structural_mutate(root(), &mut dispatch).unwrap();
    assert_eq!(i64::try_from(array_item(&output, 1)).unwrap(), 3);
    assert_eq!(
        i64::try_from(array_item(&array_item(&output, 0), 0)).unwrap(),
        2
    );
    assert_eq!(dispatch.depth, 0);
    assert_eq!(
        dispatch.events,
        pre.into_iter()
            .filter(|(tag, _)| *tag != "callback")
            .collect::<Vec<_>>()
    );
}

#[test]
fn map_policy_entries_preserve_descent_and_callback_composition() {
    struct Stop;
    impl<State> MutContextPolicy<State> for Stop {
        fn default_mutate(
            &self,
            _: MutateValue<'_>,
            _: &mut MutateContext<'_, State>,
        ) -> Result<UnchangedOr<Any>> {
            Ok(UnchangedOr::unchanged())
        }
    }

    let root = || Array::new(vec![1_i64]);
    let first = |value: Any| Array::<i64>::try_from(value).unwrap().get(0).unwrap();
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        for entry in 0..2 {
            let mut state = PolicyState::default();
            let output = {
                let mut mapper =
                    MapWithContextPolicy::new(&mut state, (DefaultMutContextPolicy, Stop));
                match entry {
                    0 => structural_map(root(), mapper, order),
                    _ => structural_map(root(), &mut mapper, order),
                }
            }
            .unwrap();
            assert_eq!(first(output), 1);
            assert_eq!(state.events, vec![("callback", 0)]);
        }

        let mut dispatch = IncrementIntegers;
        let callbacks = (|s: FfiString| s, (&mut dispatch,));
        assert_eq!(first(structural_map(root(), callbacks, order).unwrap()), 2);

        let callbacks = (|x: i64| x + 1, (|s: FfiString| s,));
        assert_eq!(first(structural_map(root(), callbacks, order).unwrap()), 2);
        let mapper = MapWithContextPolicy::new(callbacks.into_mapper(), Stop);
        assert_eq!(first(structural_map(root(), mapper, order).unwrap()), 1);
    }
}

#[test]
fn mutation_policy_continuations_preserve_ownership_and_markers() {
    struct Ownership {
        increment: bool,
        retained: Option<Any>,
        mode: InplaceMode,
        retain: bool,
    }
    #[dispatch(map)]
    impl Ownership {
        fn map_integer(&mut self, x: i64) -> UnchangedOr<i64> {
            if self.increment {
                UnchangedOr::changed(x + 1)
            } else {
                UnchangedOr::unchanged()
            }
        }
    }
    struct Control {
        mode: InplaceMode,
        retain: bool,
    }
    impl MutContextPolicy<Ownership> for Control {
        fn default_mutate(
            &self,
            value: MutateValue<'_>,
            ctx: &mut MutateContext<'_, Ownership>,
        ) -> Result<UnchangedOr<Any>> {
            let array = value
                .as_node::<tvm_ffi::collections::array::ArrayObj>()
                .is_some();
            if array && self.retain {
                ctx.state_mut().retained = Some(value.to_owned());
            }
            let result = ctx.default_mutate_with_mode_result(value, self.mode)?;
            if array && !ctx.state().increment {
                assert!(result.is_unchanged());
            }
            Ok(result)
        }
    }
    #[dispatch(mutate, policy = (
        Control { mode: self.mode, retain: self.retain },
        DefaultMutContextPolicy,
    ))]
    impl Ownership {
        fn mutate_integer(&mut self, value: i64) -> UnchangedOr<i64> {
            self.map_integer(value)
        }
        fn mutate_array(
            &mut self,
            value: MutateValue<'_, Array<i64>>,
            mutator: &mut Mutator,
        ) -> Result<UnchangedOr<Any>> {
            mutator.default_maybe_inplace_mutate_result(self, value)
        }
    }
    for entry in 0..4 {
        // pre-order map, post-order map, closure mutation, generated mutation
        for case in 0..4 {
            // unique, shared, alias retained by policy, forced copy
            for increment in [false, true] {
                let root = Array::new(vec![1_i64]);
                let pointer = array_pointer(&root);
                let alias = (case == 1).then(|| root.clone());
                let policy = (
                    Control {
                        mode: if case == 3 {
                            InplaceMode::Disallow
                        } else {
                            InplaceMode::Allow
                        },
                        retain: case == 2,
                    },
                    DefaultMutContextPolicy,
                );
                let state = Ownership {
                    increment,
                    retained: None,
                    mode: policy.0.mode,
                    retain: policy.0.retain,
                };
                let (output, state) = if entry < 2 {
                    let mut mapper = MapWithContextPolicy::new(state, policy);
                    let output = structural_map(
                        root,
                        &mut mapper,
                        if entry == 0 {
                            WalkOrder::PreOrder
                        } else {
                            WalkOrder::PostOrder
                        },
                    )
                    .unwrap();
                    (output, mapper.into_state())
                } else if entry == 3 {
                    let mut dispatch = state;
                    let output = structural_mutate(root, &mut dispatch).unwrap();
                    (output, dispatch)
                } else {
                    let mut mutator = MutateCallbacks::new(
                        state,
                        (
                            |x: i64, ctx: &mut MutateContext<'_, Ownership>| {
                                ctx.state_mut().map_integer(x)
                            },
                            |value: MutateValue<'_>, ctx: &mut MutateContext<'_, Ownership>| {
                                ctx.default_maybe_inplace_mutate_result(value)
                            },
                        ),
                    )
                    .with_policy(policy);
                    let output = structural_mutate(root, &mut mutator).unwrap();
                    (output, mutator.into_state())
                };
                let output = Array::<i64>::try_from(output).unwrap();
                assert_eq!(output.get(0).unwrap(), if increment { 2 } else { 1 });
                assert_eq!(array_pointer(&output) == pointer, !increment || case == 0);
                if let Some(alias) = alias {
                    assert_eq!(alias.get(0).unwrap(), 1);
                }
                if let Some(alias) = state.retained {
                    assert_eq!(Array::<i64>::try_from(alias).unwrap().get(0).unwrap(), 1);
                }
            }
        }
    }
}

#[test]
fn mutation_policy_regions_retargeting_and_error_restore() {
    use DefRegionKind::{None as Use, Pattern, Simple};
    #[derive(Default)]
    struct Regions(Vec<(i64, DefRegionKind)>, DefRegionKind);
    #[dispatch(map)]
    impl Regions {
        fn map_integer(&mut self, x: i64, kind: DefRegionKind) -> i64 {
            self.0.push((x, kind));
            x
        }
        fn map_array(&mut self, value: Array<i64>, kind: DefRegionKind) -> Any {
            self.0.push((200, kind));
            value.into()
        }
    }
    struct Redirect(DefRegionKind);
    impl MutContextPolicy<Regions> for Redirect {
        fn default_mutate(
            &self,
            value: MutateValue<'_>,
            ctx: &mut MutateContext<'_, Regions>,
        ) -> Result<UnchangedOr<Any>> {
            if value.cast::<bool>() == Some(false) {
                // Bypass this container's callback, but enter the next policy and redispatch its children.
                return ctx.with_def_region_kind(Pattern, |ctx| {
                    ctx.default_mutate(&Array::new(vec![1_i64]))
                        .map(UnchangedOr::changed)
                });
            }
            if value.cast::<i64>() == Some(2) {
                assert_eq!(ctx.def_region_kind(), Pattern);
                ctx.mutate_with(&99_i64, self.0)?;
                let error: Result<()> = ctx.with_def_region_kind(Use, |ctx| {
                    assert_eq!(ctx.def_region_kind(), Pattern);
                    Err(Error::new(RUNTIME_ERROR, "scoped error", ""))
                });
                assert!(error.is_err());
                assert_eq!(ctx.def_region_kind(), Pattern);
            }
            ctx.default_maybe_inplace_mutate_result(value)
        }
    }
    struct Observe;
    impl MutContextPolicy<Regions> for Observe {
        fn default_mutate(
            &self,
            value: MutateValue<'_>,
            ctx: &mut MutateContext<'_, Regions>,
        ) -> Result<UnchangedOr<Any>> {
            if value
                .as_node::<tvm_ffi::collections::array::ArrayObj>()
                .is_some()
            {
                assert_eq!(ctx.def_region_kind(), Pattern);
                let kind = ctx.def_region_kind();
                ctx.state_mut().0.push((100, kind));
            }
            ctx.default_maybe_inplace_mutate_result(value)
        }
    }
    #[dispatch(mutate, policy = (Redirect(self.1), Observe))]
    impl Regions {
        fn mutate_any(
            &mut self,
            value: MutateValue<'_>,
            mutator: &mut Mutator,
        ) -> Result<UnchangedOr<Any>> {
            if let Some(x) = value.cast::<i64>() {
                self.map_integer(x, mutator.def_region_kind());
            }
            if value
                .as_node::<tvm_ffi::collections::array::ArrayObj>()
                .is_some()
            {
                self.0.push((200, mutator.def_region_kind()));
            }
            mutator.default_maybe_inplace_mutate_result(self, value)
        }
    }
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    for requested in [Use, Simple] {
        for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
            let mut mapper =
                MapWithContextPolicy::new(Regions::default(), (Redirect(requested), Observe));
            let output = structural_map(false, &mut mapper, order).unwrap();
            assert_eq!(i64::try_from(array_item(&output, 0)).unwrap(), 1);
            let mut expected = vec![(100, Pattern), (1, Pattern)];
            if order == WalkOrder::PostOrder {
                expected.push((200, Use)); // The root callback sees the result after descent.
            }
            assert_eq!(mapper.state().0, expected);
            mapper.state_mut().0.clear();
            let graph = Function::get_global("testing.make_visit_region_graph")
                .unwrap()
                .call_tuple((false,))
                .unwrap();
            structural_map(graph, &mut mapper, order).unwrap();
            let expected = if order == WalkOrder::PreOrder {
                vec![(1, Simple), (2, Pattern), (99, Pattern), (3, Use)]
            } else {
                vec![(1, Simple), (99, Pattern), (2, Pattern), (3, Use)]
            };
            assert_eq!(mapper.state().0, expected);
        }
        let mut mutator = MutateCallbacks::new(
            Regions::default(),
            |value: MutateValue<'_>, ctx: &mut MutateContext<'_, Regions>| {
                if let Some(x) = value.cast::<i64>() {
                    let kind = ctx.def_region_kind();
                    ctx.state_mut().0.push((x, kind));
                }
                if value
                    .as_node::<tvm_ffi::collections::array::ArrayObj>()
                    .is_some()
                {
                    let kind = ctx.def_region_kind();
                    ctx.state_mut().0.push((200, kind));
                }
                ctx.default_maybe_inplace_mutate_result(value)
            },
        )
        .with_policy((Redirect(requested), Observe));
        let output = structural_mutate(false, &mut mutator).unwrap();
        assert_eq!(i64::try_from(array_item(&output, 0)).unwrap(), 1);
        assert_eq!(mutator.state().0, vec![(100, Pattern), (1, Pattern)]);
        mutator.state_mut().0.clear();
        let graph = Function::get_global("testing.make_visit_region_graph")
            .unwrap()
            .call_tuple((false,))
            .unwrap();
        structural_mutate(graph, &mut mutator).unwrap();
        assert_eq!(
            mutator.state().0,
            vec![(1, Simple), (2, Pattern), (99, Pattern), (3, Use)]
        );
        let mut dispatch = Regions(vec![], requested);
        let output = structural_mutate(false, &mut dispatch).unwrap();
        assert_eq!(i64::try_from(array_item(&output, 0)).unwrap(), 1);
        assert_eq!(dispatch.0, vec![(100, Pattern), (1, Pattern)]);
        dispatch.0.clear();
        let graph = Function::get_global("testing.make_visit_region_graph")
            .unwrap()
            .call_tuple((false,))
            .unwrap();
        structural_mutate(graph, &mut dispatch).unwrap();
        assert_eq!(
            dispatch.0,
            vec![(1, Simple), (2, Pattern), (99, Pattern), (3, Use)]
        );
    }
}

#[test]
fn mutation_policy_halts_restore_state_and_skip_later_policies() {
    struct Halt(bool);
    impl MutContextPolicy<PolicyState> for Halt {
        fn default_mutate(
            &self,
            _: MutateValue<'_>,
            _: &mut MutateContext<'_, PolicyState>,
        ) -> Result<UnchangedOr<Any>> {
            if self.0 {
                Err(Error::new(RUNTIME_ERROR, "stop descent", ""))
            } else {
                Ok(UnchangedOr::unchanged())
            }
        }
    }
    for fail in [false, true] {
        for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
            let mut mapper = MapWithContextPolicy::new(
                PolicyState::default(),
                (ArrayPolicy, (Halt(fail), RecordPolicy)),
            );
            let result = structural_map(Array::new(vec![1_i64]), &mut mapper, order);
            assert_eq!(result.is_err(), fail);
            assert_eq!(mapper.state().depth, 0);
            assert!(!mapper
                .state()
                .events
                .iter()
                .any(|(tag, _)| matches!(*tag, "integer" | "next")));
            if let Err(error) = result {
                assert!(error.message().contains("stop descent"));
            }
        }
        let mut mutator = MutateCallbacks::new(
            PolicyState::default(),
            |x: i64, _: &mut MutateContext<'_, PolicyState>| x + 1,
        )
        .with_policy((ArrayPolicy, (Halt(fail), RecordPolicy)));
        assert_eq!(
            structural_mutate(Array::new(vec![1_i64]), &mut mutator).is_err(),
            fail
        );
        assert_eq!(mutator.state().events, vec![("enter", 1), ("exit", 0)]);
        assert_eq!(mutator.state().depth, 0);
    }
}
