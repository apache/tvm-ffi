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
use tvm_ffi::extra::structural_mutate::{
    IntoMutateResult, StructuralMutator, StructuralMutatorBase, StructuralMutatorObj,
};
use tvm_ffi::function::FunctionObj;
use tvm_ffi::object::ObjectRef;
use tvm_ffi::{
    dispatch, structural_map, structural_mutate, structural_visit, structural_walk, Any, AnyView,
    Array, DefRegionKind, Dispatcher, Error, FieldGetter, Function, InplaceMode, IntoMapper, Map,
    MutationView, Object, ObjectArc, ObjectRefCore, Result, String as FfiString,
    StructuralMutatorView, StructuralVarRemap, StructuralVisitorView, TypeIndex, Unchanged,
    UnchangedOr, VisitInterrupt, WalkOrder, WalkResult, RUNTIME_ERROR,
};

struct IncrementIntegers;

impl<'receiver, 'value> Dispatcher<(AnyView<'value>, DefRegionKind), Option<Result<Any>>>
    for &'receiver mut IncrementIntegers
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, _def_region_kind): (AnyView<'value>, DefRegionKind),
    ) -> Option<Result<Any>> {
        value
            .try_as::<i64>()
            .map(|integer| Ok(Any::from(integer + 1)))
    }
}

#[repr(C, align(64))]
struct ScopedMutateBase<'a> {
    base: StructuralMutatorObj,
    depth: &'a Cell<usize>,
    calls: &'a Cell<usize>,
}
impl<'a> ScopedMutateBase<'a> {
    fn new(depth: &'a Cell<usize>, calls: &'a Cell<usize>) -> Self {
        Self {
            base: StructuralMutatorObj::default(),
            depth,
            calls,
        }
    }
}
// SAFETY: the ABI prefix is first; only copied external Cell references span
// recursion, and the guard restores scoped state on every exit.
unsafe impl StructuralMutatorBase for ScopedMutateBase<'_> {
    unsafe fn default_mutate<D>(
        this: *mut Self,
        dispatch: &mut D,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, D>,
    ) -> Result<UnchangedOr<Any>> {
        assert_eq!(this.addr() % 64, 0);
        let depth = (*this).depth;
        let previous = depth.get();
        depth.set(previous + 1);
        (*this).calls.set((*this).calls.get() + 1);
        struct Restore<'a>(&'a Cell<usize>, usize);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _restore = Restore(depth, previous);
        StructuralMutatorObj::default_mutate(dispatch, value, mutator)
    }
}

#[derive(Default)]
struct ManualIncrement {
    remap: StructuralVarRemap,
    remap_probe: Option<Any>,
}

#[derive(Default)]
struct ReplaceNone {
    calls: usize,
}

struct RecursiveEntryMutator {
    use_owned_value: bool,
    owned_value_pointer: Option<usize>,
}

impl<'s, 'v, 'c, 'r>
    Dispatcher<
        (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, ManualIncrement>,
        ),
        Option<Result<Any>>,
    > for &'s mut ManualIncrement
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, mutator): (
            MutationView<'v>,
            &'c mut StructuralMutatorView<'r, ManualIncrement>,
        ),
    ) -> Option<Result<Any>> {
        Some(if value.as_any_view().try_as::<bool>() == Some(true) {
            let key = self.remap_probe.as_ref().unwrap().clone();
            (|| {
                assert!(mutator.var_remap_get(self, AnyView::from(&key))?.is_none());
                assert!(mutator.var_remap_get(self, AnyView::from(&1_i64)).is_err());
                assert!(mutator
                    .var_remap_set(self, AnyView::from(&1_i64), &Any::new())
                    .is_err());
                let initial = mutator.def_region_kind();
                mutator.with_def_region_kind(DefRegionKind::Pattern, |mutator| {
                    assert_eq!(mutator.def_region_kind(), DefRegionKind::Pattern);
                    mutator.var_remap_set(self, AnyView::from(&key), &Any::new())?;
                    assert_eq!(
                        mutator
                            .var_remap_get(self, AnyView::from(&key))?
                            .unwrap()
                            .type_index(),
                        TypeIndex::kTVMFFINone as i32
                    );
                    // A None replacement does not suppress reflected descent.
                    let result =
                        mutator.mutate(self, MutationView::borrowed(AnyView::from(&key)))?;
                    assert_eq!(
                        RemapNode::try_from(result.value_or_else(|| key.clone()))?
                            .data
                            .value,
                        8
                    );
                    mutator.var_remap_set(self, AnyView::from(&key), &Any::from(41i64))
                })?;
                assert_eq!(mutator.def_region_kind(), initial);
                let replacement = mutator.var_remap_get(self, AnyView::from(&key))?.unwrap();
                self.remap.set(AnyView::from(&key), &replacement)?;
                Ok(replacement)
            })()
        } else if let Some(integer) = value.as_any_view().try_as::<i64>() {
            Ok(Any::from(integer + 1))
        } else {
            mutator.default_mutate(self, value).map(Into::into)
        })
    }
}
#[dispatch(map)]
impl ManualIncrement {
    fn map_integer(&mut self, value: i64, _: DefRegionKind) -> Result<Any> {
        Ok(Any::from(value + 1))
    }
}
#[dispatch(mutate, MutationView)]
impl ReplaceNone {
    fn mutate_any(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        if value.as_any_view().type_index() == TypeIndex::kTVMFFINone as i32 {
            self.calls += 1;
            Ok(Any::from(8i64))
        } else {
            mutator.default_mutate(self, value).map(Into::into)
        }
    }
}
#[dispatch(mutate, MutationView)]
impl RecursiveEntryMutator {
    fn mutate_any(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        if value.as_any_view().type_index() == TypeIndex::kTVMFFINone as i32 {
            if self.use_owned_value {
                let value = Array::new(vec![1i64]);
                self.owned_value_pointer = Some(array_pointer(&value) as usize);
                let mut owner: Any = value.into();
                let change = mutator.mutate(self, &mut owner)?;
                Ok(change.value_or(owner))
            } else {
                mutator
                    .mutate(self, MutationView::borrowed(AnyView::from(&1i64)))
                    .map(Into::into)
            }
        } else if let Some(integer) = value.as_any_view().try_as::<i64>() {
            Ok(Any::from(integer + 1))
        } else {
            mutator.default_mutate(self, value).map(Into::into)
        }
    }
}

#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "testing.rust.MutationRemapNode"]
#[type_register]
#[type_s_eq_hash_kind = "DAGNode"]
struct RemapNodeObj {
    base: Object,
    #[reflect]
    value: i64,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
struct RemapNode {
    data: ObjectArc<RemapNodeObj>,
}

#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "testing.rust.MutationFieldNode"]
#[type_register]
struct FieldNodeObj {
    base: Object,
    a: Array<i64>,
    b: Any,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
struct FieldNode {
    data: ObjectArc<FieldNodeObj>,
}

tvm_ffi::static_init_block! {
    // Register before the parallel tests read the runtime type table.
    <RemapNodeObj as tvm_ffi::object::ObjectCore>::type_index();
    <FieldNodeObj as tvm_ffi::object::ObjectCore>::type_index();
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
    assert_eq!(
        std::mem::size_of::<StructuralMutatorView<'_, ManualIncrement>>(),
        std::mem::size_of::<*mut ()>()
    );
    let unique = Array::new(vec![1i64, 2]);
    let unique_pointer = array_pointer(&unique);
    let mutated = structural_mutate::<Array<i64>, _>(unique, ManualIncrement::default())
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(array_pointer(&mutated), unique_pointer);
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);

    let source = Array::new(vec![3i64]);
    let source_pointer = array_pointer(&source);
    let mutated = structural_mutate(source.clone(), ManualIncrement::default())
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_ne!(array_pointer(&mutated), source_pointer);
    assert_eq!(source.get(0).unwrap(), 3);
    assert_eq!(mutated.get(0).unwrap(), 4);

    let map: Map<i64, i64> = [(1, 10)].into_iter().collect();
    let source_map_pointer = map_pointer(&map);
    let mutated_map = structural_mutate(map, ManualIncrement::default())
        .and_then(Map::<i64, i64>::try_from)
        .unwrap();
    assert_eq!(map_pointer(&mutated_map), source_map_pointer);
    assert_eq!(mutated_map.get(&1).unwrap(), Some(11));

    let dict = call_global("ffi.Dict", &[Any::from(1i64), Any::from(10i64)]);
    let dict_pointer = any_object_pointer(&dict);
    let mutated_dict = structural_mutate(dict, ManualIncrement::default()).unwrap();
    assert_eq!(any_object_pointer(&mutated_dict), dict_pointer);
    assert_eq!(dict_item(&mutated_dict, 1), 11);

    let key = Any::from(RemapNode {
        data: ObjectArc::new(RemapNodeObj {
            base: Object::new(),
            value: 7,
        }),
    });
    let mut dispatch = ManualIncrement::default();
    let first = structural_mutate(Array::new(vec![key.clone(), key.clone()]), &mut dispatch)
        .and_then(Array::<Any>::try_from)
        .unwrap();
    let first_node = first.get(0).unwrap();
    assert_eq!(
        any_object_pointer(&first_node),
        any_object_pointer(&first.get(1).unwrap())
    );
    assert_eq!(
        RemapNode::try_from(first_node.clone()).unwrap().data.value,
        8
    );
    let second = structural_mutate(key.clone(), &mut dispatch).unwrap();
    assert_ne!(any_object_pointer(&first_node), any_object_pointer(&second));
    assert_eq!(RemapNode::try_from(second).unwrap().data.value, 8);
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let first = structural_map(
            Array::new(vec![key.clone(), key.clone()]),
            &mut dispatch,
            order,
        )
        .and_then(Array::<Any>::try_from)
        .unwrap();
        let first_node = first.get(0).unwrap();
        assert_eq!(
            any_object_pointer(&first_node),
            any_object_pointer(&first.get(1).unwrap())
        );
        assert_eq!(
            RemapNode::try_from(first_node.clone()).unwrap().data.value,
            8
        );
        let second = structural_map(key.clone(), &mut dispatch, order).unwrap();
        assert_ne!(any_object_pointer(&first_node), any_object_pointer(&second));
    }
    dispatch.remap_probe = Some(key.clone());
    for _ in 0..2 {
        assert_eq!(
            i64::try_from(structural_mutate(true, &mut dispatch).unwrap()).unwrap(),
            41
        );
        assert_eq!(
            i64::try_from(dispatch.remap.get(AnyView::from(&key)).unwrap().unwrap()).unwrap(),
            41
        );
    }
    dispatch.remap.clear();
    assert!(dispatch.remap.get(AnyView::from(&key)).unwrap().is_none());
}

#[test]
fn none_values_are_dispatched_to_map_callbacks_and_user_mutators() {
    let mut map_calls = 0;
    let mapped = structural_map(
        Any::new(),
        |value: AnyView<'_>| {
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
        use_owned_value: false,
        owned_value_pointer: None,
    };
    let mutated = structural_mutate(Any::new(), &mut borrowed)
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(mutated, 2);

    let mut owned = RecursiveEntryMutator {
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

    #[repr(align(128))]
    #[derive(Default)]
    struct BoundDispatcher {
        visits: usize,
    }
    #[dispatch(mutate, MutationView)]
    impl BoundDispatcher {
        fn mutate_bool(
            &mut self,
            _: bool,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            for operation in 0..4 {
                let mut other = Self::default();
                let key = Any::new();
                let rejected =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match operation {
                        0 => mutator
                            .mutate(&mut other, MutationView::borrowed(AnyView::from(&1i64)))
                            .map(Into::into),
                        1 => mutator
                            .default_mutate(
                                &mut other,
                                MutationView::borrowed(AnyView::from(&1i64)),
                            )
                            .map(Into::into),
                        2 => mutator
                            .var_remap_get(&mut other, AnyView::from(&key))
                            .map(|_| Any::new()),
                        _ => mutator
                            .var_remap_set(&mut other, AnyView::from(&key), &key)
                            .map(|_| Any::new()),
                    }));
                assert!(rejected.is_err());
                assert_eq!(other.visits, 0);
            }
            mutator
                .mutate(self, MutationView::borrowed(AnyView::from(&1i64)))
                .map(Into::into)
        }
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            self.visits += 1;
            Ok(Any::from(value + 1))
        }
    }
    assert_eq!(
        i64::try_from(structural_mutate(true, BoundDispatcher::default()).unwrap()).unwrap(),
        2
    );
    let mut borrowed = BoundDispatcher::default();
    for _ in 0..2 {
        assert_eq!(
            i64::try_from(structural_mutate(true, &mut borrowed).unwrap()).unwrap(),
            2
        );
    }
    assert_eq!(borrowed.visits, 2);
}

#[test]
fn default_mutation_mode_preserves_ownership_and_unchanged_results() {
    struct Controlled {
        mode: InplaceMode,
        increment: bool,
    }

    #[dispatch(mutate, MutationView)]
    impl Controlled {
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            Ok(if self.increment {
                Any::from(value + 1)
            } else {
                Unchanged.into()
            })
        }
        fn mutate_any(
            &mut self,
            value: MutationView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            let mode = self.mode;
            if self.increment {
                mutator
                    .default_mutate(self, value.with_inplace_mode(mode))
                    .map(Into::into)
            } else {
                let result = mutator.default_mutate(self, value.with_inplace_mode(mode))?;
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
fn borrowed_capabilities_cannot_enable_low_level_default_reuse() {
    struct BorrowedEntry;
    #[dispatch(mutate, MutationView)]
    impl BorrowedEntry {
        fn mutate_any(
            &mut self,
            value: MutationView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            if let Some(integer) = value.as_any_view().try_as::<i64>() {
                return Ok(Any::from(integer + 1));
            }
            assert_eq!(value.as_any_view().try_as::<bool>(), Some(true));
            let source = Array::new(vec![1_i64]);
            let pointer = array_pointer(&source);
            let value = MutationView::borrowed(AnyView::from(&source));
            assert_eq!(value.inplace_mode(), InplaceMode::Disallow);
            assert_eq!(
                ObjectArc::strong_count(<Array<i64> as ObjectRefCore>::data(&source)),
                1
            );
            let value = value.with_inplace_mode(InplaceMode::Allow);
            assert_eq!(value.inplace_mode(), InplaceMode::Disallow);
            let result = mutator.default_mutate(self, value)?;
            let result = Array::<i64>::try_from(result.value_or_else(|| source.clone().into()))?;
            assert_ne!(array_pointer(&result), pointer);
            assert_eq!(source.get(0).unwrap(), 1);
            assert_eq!(result.get(0).unwrap(), 2);
            assert_eq!(
                ObjectArc::strong_count(<Array<i64> as ObjectRefCore>::data(&source)),
                1
            );
            Ok(result.into())
        }
    }

    structural_mutate(true, &mut BorrowedEntry).unwrap();
}

#[test]
fn borrowed_capabilities_keep_their_mode_in_dispatch_adapters() {
    struct Inspect<'a> {
        source: &'a Array<i64>,
        result: Option<Any>,
    }
    #[dispatch(mutate, MutationView)]
    impl<'a> Inspect<'a> {
        fn mutate_bool(
            &mut self,
            _: bool,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            let source = self.source;
            let value = MutationView::borrowed(AnyView::from(source))
                .try_cast::<Array<i64>>()
                .unwrap_or_else(|_| unreachable!());
            self.result = Some(mutator.mutate(self, value)?.into());
            Ok(Any::new())
        }
        fn mutate_array(
            &mut self,
            value: MutationView<'_, Array<i64>>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            assert_eq!(value.inplace_mode(), InplaceMode::Disallow);
            Unchanged.into_mutate_result()
        }
    }
    let source = Array::new(vec![1_i64]);
    let mut inspect = Inspect {
        source: &source,
        result: None,
    };
    structural_mutate(true, &mut inspect).unwrap();
    let result = inspect.result.take().unwrap();
    assert!(UnchangedOr::<Any>::try_from(result).unwrap().is_unchanged());

    assert_eq!(source.get(0).unwrap(), 1);
    assert_eq!(
        ObjectArc::strong_count(<Array<i64> as ObjectRefCore>::data(&source)),
        1
    );
}

#[test]
fn owned_entry_mode_is_forwarded_by_generated_callbacks() {
    struct Entry {
        mode: InplaceMode,
        pointer: usize,
        increment: bool,
        shared: bool,
    }
    #[dispatch(mutate, MutationView)]
    impl Entry {
        fn mutate_bool(
            &mut self,
            _: bool,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            let child = Array::new(vec![1_i64]);
            self.pointer = array_pointer(&child) as usize;
            mutator.with_def_region_kind(DefRegionKind::Pattern, |mutator| {
                let mut owner: Any = child.into();
                let alias = self.shared.then(|| owner.clone());
                let count = if self.shared { 2 } else { 1 };
                let capability = MutationView::from(&mut owner).with_inplace_mode(self.mode);
                assert_eq!(capability.inplace_mode(), self.mode);
                assert_eq!(capability.as_any_view().debug_strong_count(), Some(count));
                assert_eq!(
                    capability
                        .downcast_ref::<tvm_ffi::collections::array::ArrayObj>()
                        .unwrap()
                        .size,
                    1
                );
                assert!(capability.downcast_ref::<FunctionObj>().is_none());
                assert_eq!(capability.as_any_view().debug_strong_count(), Some(count));
                assert_eq!(capability.inplace_mode(), self.mode);
                let change = mutator.mutate(self, capability)?;
                if !self.increment {
                    assert!(change.is_unchanged());
                }
                assert_eq!(owner.debug_strong_count(), Some(count));
                let result = change.value_or(owner);
                if let Some(alias) = alias {
                    assert_eq!(Array::<i64>::try_from(alias)?.get(0)?, 1);
                }
                Ok(result)
            })
        }
        fn mutate_integer(
            &mut self,
            value: i64,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            assert_eq!(mutator.def_region_kind(), DefRegionKind::Pattern);
            Ok(if self.increment {
                Any::from(value + 1)
            } else {
                Unchanged.into()
            })
        }
    }

    assert_eq!(InplaceMode::default(), InplaceMode::Disallow);
    for (mode, increment, shared) in [
        (InplaceMode::Disallow, true, false),
        (InplaceMode::Allow, true, false),
        (InplaceMode::Allow, false, false),
        (InplaceMode::Allow, true, true),
    ] {
        let mut entry = Entry {
            mode,
            pointer: 0,
            increment,
            shared,
        };
        let result = structural_mutate(true, &mut entry)
            .and_then(Array::<i64>::try_from)
            .unwrap();
        assert_eq!(
            array_pointer(&result) as usize == entry.pointer,
            mode == InplaceMode::Allow && !shared
        );
        assert_eq!(result.get(0).unwrap(), if increment { 2 } else { 1 });
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
        fn observe(&mut self, value: &MutationView<'_, Array<Any>>) {
            self.modes.push(value.inplace_mode());
            let count = value.as_any_view().debug_strong_count();
            let node = value
                .downcast_ref::<tvm_ffi::collections::array::ArrayObj>()
                .unwrap();
            assert_eq!(node.size, 1);
            let base = value.downcast_ref::<Object>().unwrap();
            assert_eq!(base as *const Object as usize, node as *const _ as usize);
            assert!(value.downcast_ref::<FunctionObj>().is_none());
            assert_eq!(value.as_any_view().debug_strong_count(), count);
            // A temporary typed handle must not permanently revoke permission.
            let temporary = value.as_any_view().try_as::<Array<Any>>().unwrap();
            if self.retain && self.alias.is_none() {
                self.alias = Some(temporary.clone().into());
            }
            drop(temporary);
        }
    }
    #[dispatch(mutate, MutationView)]
    impl Forward {
        fn mutate_string(&mut self, _: MutationView<'_, FfiString>) -> Result<Any> {
            panic!("typed miss must preserve the capability for the next handler")
        }
        fn mutate_array(
            &mut self,
            value: MutationView<'_, Array<Any>>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            let mode = value.inplace_mode();
            let count = value.as_any_view().debug_strong_count();
            let value = tvm_ffi::match_any!(value {
                MutationView<FfiString>(_) => unreachable!(),
                MutationView<Array<Any>>(array) if {
                    assert_eq!(array.inplace_mode(), mode);
                    assert_eq!(array.as_any_view().debug_strong_count(), count);
                    false
                } => unreachable!(),
                MutationView<Array<Any>>(array) => array,
                _ => unreachable!(),
            });
            self.observe(&value);
            mutator
                .default_mutate(self, value.with_inplace_mode(self.requested))
                .into_mutate_result()
        }
        fn mutate_integer(
            &mut self,
            value: i64,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            Ok(Any::from(value + 1))
        }
    }

    use InplaceMode::{Allow, Disallow};
    for (requested, shared, retain) in [
        (Allow, false, false),
        (Disallow, false, false),
        (Allow, true, false),
        (Allow, false, true),
    ] {
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
        let result = structural_mutate(root, &mut state).unwrap();
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
    let mut owner: Any = FieldNode {
        data: ObjectArc::new(FieldNodeObj {
            base: Object::new(),
            a: Array::new(vec![1_i64]),
            b: Array::new(vec![2_i64]).into(),
        }),
    }
    .into();
    let mut value = MutationView::from(&mut owner)
        .try_cast::<FieldNode>()
        .ok()
        .unwrap();
    assert_eq!(value.a.get(0).unwrap(), 1);
    assert_eq!(value.as_any_view().debug_strong_count(), Some(1));
    let a = value.field(|node| &node.a);
    assert_eq!(a.size, 1);
    assert_eq!(a.inplace_mode(), Allow);
    assert_eq!(a.as_any_view().debug_strong_count(), Some(1));
    drop(a);
    let mut value = value.with_inplace_mode(Disallow);
    let b = value.field(|node| &node.b);
    assert_eq!(b.inplace_mode(), Disallow);
    assert_eq!(b.as_any_view().debug_strong_count(), Some(1));
    drop(b);
    assert_eq!(value.as_any_view().debug_strong_count(), Some(1));
}

#[test]
fn consuming_default_descent_transfers_between_contexts_without_node_borrows() {
    struct Inner<'a> {
        value: Option<MutationView<'a>>,
        result: Any,
    }
    #[dispatch(mutate, MutationView)]
    impl<'a> Inner<'a> {
        fn mutate_bool(
            &mut self,
            _: bool,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            let value = self.value.take().unwrap();
            self.result = mutator.default_mutate(self, value)?.into();
            Ok(Any::new())
        }
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            Ok(Any::from(value + 1))
        }
    }

    fn transfer(value: MutationView<'_>, retain: bool) -> Result<Any> {
        let alias = retain.then(|| value.as_any_view().try_as::<Array<i64>>().unwrap());
        assert_eq!(value.inplace_mode(), InplaceMode::Allow);
        // Carry the result back to the array callback: Unchanged belongs to
        // that input, not to the nested traversal's boolean root.
        let mut inner = Inner {
            value: Some(value),
            result: Any::new(),
        };
        structural_mutate(true, &mut inner)?;
        let result = inner.result;
        if let Some(alias) = alias {
            assert_eq!(alias.get(0)?, 1);
        }
        Ok(result)
    }
    struct Outer {
        retain: bool,
    }
    #[dispatch(mutate, MutationView)]
    impl Outer {
        fn mutate_any(
            &mut self,
            value: MutationView<'_>,
            _mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            transfer(value, self.retain)
        }
    }

    for retain in [false, true] {
        for borrowed in [false, true] {
            let root = Array::new(vec![1_i64]);
            let pointer = array_pointer(&root);
            let result = if borrowed {
                structural_mutate(root, &mut Outer { retain })
            } else {
                structural_mutate(root, Outer { retain })
            }
            .and_then(Array::<i64>::try_from)
            .unwrap();
            assert_eq!(array_pointer(&result) == pointer, !retain);
            assert_eq!(result.get(0).unwrap(), 2);
        }
    }
}

#[test]
fn consuming_default_descent_preserves_unchanged_and_propagates_errors() {
    struct Preserve(bool);
    #[dispatch(mutate, MutationView)]
    impl Preserve {
        fn mutate_any(
            &mut self,
            value: MutationView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            if value.as_any_view().try_as::<i64>().is_some() {
                return if self.0 {
                    Err(Error::new(RUNTIME_ERROR, "child failed", ""))
                } else {
                    Ok(UnchangedOr::<Any>::unchanged().into())
                };
            }
            let result = mutator.default_mutate(self, value)?;
            assert!(result.is_unchanged());
            Ok(result.into())
        }
    }
    for fail in [false, true] {
        let root = Array::new(vec![1_i64]);
        let pointer = array_pointer(&root);
        let result = structural_mutate(root, Preserve(fail));
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
    let mutated = structural_mutate(source, ManualIncrement::default()).unwrap();

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
    struct VisitDelegate(Error);
    #[dispatch(visit)]
    impl VisitDelegate {
        fn visit_any(
            &mut self,
            value: AnyView<'_>,
            visitor: &mut StructuralVisitorView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            if value.try_as::<i64>().is_some() {
                return Err(self.0.clone());
            }
            visitor.default_visit(self, &value)
        }
    }

    #[dispatch(mutate, MutationView)]
    impl Delegate {
        fn mutate_any(
            &mut self,
            value: MutationView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            if value.as_any_view().try_as::<i64>().is_some() {
                return Err(self.0.clone());
            }
            mutator.default_mutate(self, value).map(Into::into)
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
    }
    let mut leaf = Delegate(source.clone());
    errors.push(structural_mutate(root.clone(), &mut leaf).err().unwrap());
    errors.push(
        structural_visit(&root, &mut VisitDelegate(leaf.0.clone()))
            .err()
            .unwrap(),
    );
    let mut delegate = Delegate(source.clone());
    errors.push(
        structural_visit(&root, &mut VisitDelegate(delegate.0.clone()))
            .err()
            .unwrap(),
    );
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

    #[repr(C)]
    struct RejectBase(StructuralMutatorObj);
    // SAFETY: the exact prefix is first; no Base borrow survives the callback.
    unsafe impl StructuralMutatorBase for RejectBase {
        unsafe fn default_mutate<D>(
            _this: *mut Self,
            _dispatch: &mut D,
            _value: MutationView<'_>,
            _mutator: &mut StructuralMutatorView<'_, D>,
        ) -> Result<UnchangedOr<Any>> {
            Err(Error::new(RUNTIME_ERROR, "base failed", "base origin"))
        }
    }
    let error = structural_map(
        source.clone(),
        StructuralMutator::new(RejectBase(StructuralMutatorObj::default()), |x: i64| x),
        WalkOrder::PreOrder,
    )
    .err()
    .unwrap();
    assert_eq!(error.message(), "base failed");
    assert!(error.backtrace().contains("base origin"));
    assert!(error.backtrace().contains("object `ffi.Map`"));
    assert_eq!(source.get(&1).unwrap(), Some(10));
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
    fn map_integer(&mut self, value: i64, kind: DefRegionKind) -> Result<Any> {
        self.integers.push((value, kind));
        Ok(Any::from(value + 1))
    }

    fn map_any(&mut self, value: tvm_ffi::AnyView<'_>) -> Result<Any> {
        self.catch_all += 1;
        Ok(Any::from(value))
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

#[dispatch(mutate, MutationView)]
impl GeneratedLeafDispatch {
    fn mutate_integer(
        &mut self,
        value: i64,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        let region = mutator.def_region_kind();
        self.integers.push((value, region));
        Ok(Any::from(value + 1))
    }
}

struct GeneratedStatelessDispatch;

#[dispatch(mutate, MutationView)]
impl GeneratedStatelessDispatch {
    fn mutate_integer(
        &mut self,
        value: i64,
        _mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        Ok(Any::from(value + 1))
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

#[dispatch(mutate, MutationView)]
impl GeneratedRecursiveDispatch {
    fn mutate_array(
        &mut self,
        array: Array<i64>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        let region = mutator.def_region_kind();
        self.arrays.push(region);
        let mut mutated = Vec::with_capacity(array.len());
        for value in array.iter() {
            mutated.push(i64::try_from(
                mutator
                    .mutate(self, MutationView::borrowed(AnyView::from(&value)))?
                    .value_or_else(|| value.into()),
            )?);
        }
        Ok(Array::new(mutated).into())
    }

    fn mutate_integer(
        &mut self,
        value: i64,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        let region = mutator.def_region_kind();
        self.integers.push((value, region));
        Ok(Any::from(value + 10))
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

#[dispatch(mutate, MutationView)]
impl GeneratedDefaultingDispatch {
    fn mutate_array(
        &mut self,
        array: Array<i64>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        self.arrays += 1;
        mutator
            .default_mutate(self, MutationView::borrowed(AnyView::from(&array)))
            .map(Into::into)
    }

    fn mutate_integer(&mut self, value: i64) -> Result<Any> {
        self.integers.push(value);
        Ok(Any::from(value + 1))
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
fn pre_order_mapping_preserves_inplace_permission() {
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
        let mut mapper = (|value: AnyView<'_>| {
            if let Some(integer) = value.try_as::<i64>() {
                return Any::from(integer + 1);
            }
            let mapped = if case == "retained" {
                retained = Some(Any::from(value));
                Any::from(value)
            } else {
                Array::new(vec![1_i64]).into()
            };
            pointer = any_object_pointer(&mapped);
            mapped
        })
        .into_mapper();
        let depth = Cell::new(0);
        let calls = Cell::new(0);
        let output = structural_map(
            root,
            StructuralMutator::new(ScopedMutateBase::new(&depth, &calls), &mut mapper),
            WalkOrder::PreOrder,
        )
        .unwrap();
        assert_eq!(depth.get(), 0);
        assert_eq!(calls.get(), 2);
        let reuse = case == "unique";
        assert_eq!(any_object_pointer(&output) == pointer, reuse, "{case}");
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

    struct Double;
    #[dispatch(mutate, MutationView)]
    impl Double {
        fn mutate_integer(
            &mut self,
            integer: i64,
            _: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            (integer * 2).into_mutate_result()
        }
    }
    let mutated = structural_mutate(Array::new(vec![1i64, 2]), Double)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 4]);
}

#[test]
fn recursive_mutate_returns_unchanged_or_a_replacement() {
    struct ClampNegativeIntegers;
    #[dispatch(mutate, MutationView)]
    impl ClampNegativeIntegers {
        fn mutate_any(
            &mut self,
            value: MutationView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            if let Some(integer) = value.as_any_view().try_as::<i64>() {
                if integer >= 0 {
                    // Keep this input without constructing an owning return value.
                    return Ok(UnchangedOr::<Any>::unchanged().into());
                }
                return Ok(UnchangedOr::changed(Any::from(0i64)).into());
            }
            // Keep the unchanged marker if no child changed.
            mutator.default_mutate(self, value).into_mutate_result()
        }
    }

    // No rewrite: the public entry resolves unchanged to the original array.
    let source = Array::new(vec![1i64, 2]);
    let result = structural_mutate(source.clone(), ClampNegativeIntegers)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(result.iter().collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(array_pointer(&result), array_pointer(&source));

    // One rewrite: replace only the negative integer and keep the source intact.
    let source = Array::new(vec![-1i64, 2]);
    let result = structural_mutate(source.clone(), ClampNegativeIntegers)
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
            |_value: AnyView<'_>| Unchanged,
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
    for borrowed in [false, true] {
        for shared in [false, true] {
            let root = Array::new(vec![1_i64]);
            let pointer = array_pointer(&root);
            let alias = shared.then(|| root.clone());
            let callbacks = (
                |integer: i64| integer + 1,
                |value: AnyView<'_>| {
                    assert_eq!(value.try_as::<Array<i64>>().unwrap().get(0).unwrap(), 2);
                    Unchanged
                },
            );
            let mapped = if borrowed {
                let mut mapper = callbacks.into_mapper();
                structural_map(root, &mut mapper, WalkOrder::PostOrder)
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
                    (|value: AnyView<'_>| {
                        catch_all += 1;
                        Any::from(value)
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
        |value: AnyView<'_>| {
            pre.push(value.try_as::<i64>());
            Any::from(value)
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert_eq!(pre, vec![None, Some(1), Some(2)]);

    let mut post = Vec::new();
    structural_map(
        root,
        |value: AnyView<'_>| {
            post.push(value.try_as::<i64>());
            Any::from(value)
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
    let mutated = structural_mutate(root, GeneratedStatelessDispatch)
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

#[dispatch(mutate, MutationView)]
impl CallbackMutateStats {
    fn mutate_integer(&mut self, value: i64) -> Result<Any> {
        self.integers.push(value);
        Ok(Any::from(value + 1))
    }
    fn mutate_default(
        &mut self,
        value: AnyView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        self.defaults += 1;
        mutator
            .default_mutate(self, MutationView::borrowed(value))
            .map(Into::into)
    }
}

#[test]
fn callback_mutator_carries_reusable_mutable_state() {
    let mut mutator = CallbackMutateStats::default();

    let first = structural_mutate(Array::new(vec![1i64, 2]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(first.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(mutator.integers, vec![1, 2]);
    assert_eq!(mutator.defaults, 1);

    let second = structural_mutate(Array::new(vec![3i64]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(second.iter().collect::<Vec<_>>(), vec![4]);

    let state = mutator;
    assert_eq!(state.integers, vec![1, 2, 3]);
    assert_eq!(state.defaults, 2);
}

#[derive(Default)]
struct CallbackMutateDepth {
    current: usize,
    maximum: usize,
    exits: usize,
}

#[dispatch(mutate, MutationView)]
impl CallbackMutateDepth {
    fn mutate_any(
        &mut self,
        value: AnyView<'_>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<Any> {
        self.current += 1;
        self.maximum = self.maximum.max(self.current);
        let mutated = mutator.default_mutate(self, MutationView::borrowed(value))?;
        self.current -= 1;
        self.exits += 1;
        Ok(mutated.into())
    }
}

#[test]
fn callback_mutator_state_can_change_around_recursive_reborrows() {
    let root = Array::new(vec![Array::new(vec![1i64, 2])]);
    let mut mutator = CallbackMutateDepth::default();
    let mutated = structural_mutate(root, &mut mutator)
        .and_then(Array::<Array<i64>>::try_from)
        .unwrap();

    assert_eq!(
        mutated.get(0).unwrap().iter().collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(mutator.current, 0);
    assert_eq!(mutator.maximum, 3);
    assert_eq!(mutator.exits, 4);
}

#[test]
fn callback_mutate_explicit_default_is_repeatable_copy_path() {
    let root = Array::new(vec![1i64, 2]);
    let root_pointer = array_pointer(&root);
    let defaults = Cell::new(0);
    struct Repeat<'a>(&'a Cell<i32>);
    #[dispatch(mutate, MutationView)]
    impl<'a> Repeat<'a> {
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            Ok(Any::from(value + 1))
        }
        fn mutate_any(
            &mut self,
            value: AnyView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            self.0.set(self.0.get() + 1);
            let first = mutator
                .default_mutate(self, MutationView::borrowed(value))?
                .value_or_else(|| value.into());
            let second = mutator
                .default_mutate(self, MutationView::borrowed(value))?
                .value_or_else(|| value.into());
            assert_ne!(any_object_pointer(&first), any_object_pointer(&second));
            Ok(first)
        }
    }
    let mutated = structural_mutate(root, Repeat(&defaults))
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_ne!(array_pointer(&mutated), root_pointer);
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(defaults.get(), 1);
}

#[test]
fn callback_mutate_match_is_final_and_same_fn_can_reenter() {
    let integer_calls = Cell::new(0);
    struct First<'a>(&'a Cell<i32>);
    #[dispatch(mutate, MutationView)]
    impl<'a> First<'a> {
        fn mutate_array(
            &mut self,
            _: Array<i64>,
            _: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            Ok(Any::from(Array::new(vec![10i64])))
        }
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            self.0.set(self.0.get() + 1);
            Ok(Any::from(value + 1))
        }
    }
    let mutated = structural_mutate(Array::new(vec![1i64]), First(&integer_calls))
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![10]);
    assert_eq!(integer_calls.get(), 0);

    let calls = Cell::new(0);
    struct Reenter<'a>(&'a Cell<i32>);
    #[dispatch(mutate, MutationView)]
    impl<'a> Reenter<'a> {
        fn mutate_any(
            &mut self,
            value: AnyView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            self.0.set(self.0.get() + 1);
            mutator
                .default_mutate(self, MutationView::borrowed(value))
                .map(Into::into)
        }
    }
    let mutated = structural_mutate(Array::new(vec![1i64, 2]), Reenter(&calls))
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
    struct Nodes<'a>(&'a RefCell<Vec<DefRegionKind>>);
    #[dispatch(mutate, MutationView)]
    impl<'a> Nodes<'a> {
        fn mutate_bool(&mut self, _: bool) -> Result<Any> {
            Ok(Any::new())
        }
        fn mutate_function(&mut self, _: &FunctionObj) -> Result<Any> {
            Ok(Any::from(7i64))
        }
        fn mutate_integer(
            &mut self,
            value: i64,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            self.0.borrow_mut().push(mutator.def_region_kind());
            Ok(Any::from(value + 1))
        }
    }
    let mutated = structural_mutate(root, Nodes(&regions)).unwrap();

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
    struct Borrowed<'a>(&'a Array<i64>);
    #[dispatch(mutate, MutationView)]
    impl<'a> Borrowed<'a> {
        fn mutate_bool(
            &mut self,
            _: bool,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            let child = self.0;
            mutator
                .mutate(self, MutationView::borrowed(AnyView::from(child)))
                .map(Into::into)
        }
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            Ok(Any::from(value + 1))
        }
    }
    let mutated = structural_mutate(true, Borrowed(&borrowed_child))
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_ne!(array_pointer(&mutated), borrowed_pointer);
    assert_eq!(borrowed_child.get(0).unwrap(), 1);
    assert_eq!(mutated.get(0).unwrap(), 2);

    let owned_pointer = Cell::new(0usize);
    struct Owned<'a>(&'a Cell<usize>);
    #[dispatch(mutate, MutationView)]
    impl<'a> Owned<'a> {
        fn mutate_bool(
            &mut self,
            _: bool,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            let child = Array::new(vec![1i64]);
            self.0.set(array_pointer(&child) as usize);
            let mut owner: Any = child.into();
            let change = mutator.mutate(self, MutationView::from(&mut owner))?;
            Ok(change.value_or(owner))
        }
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            Ok(Any::from(value + 1))
        }
    }
    let mutated = structural_mutate(true, Owned(&owned_pointer))
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(array_pointer(&mutated) as usize, owned_pointer.get());
    assert_eq!(mutated.get(0).unwrap(), 2);
}

#[test]
fn nested_callback_mutate_restores_the_outer_active_mutator() {
    struct Inner;
    #[dispatch(mutate, MutationView)]
    impl Inner {
        fn mutate_integer(
            &mut self,
            value: i64,
            _: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            Ok(Any::from(value + 10))
        }
    }
    #[repr(align(128))]
    struct Outer<'a>(&'a Cell<usize>);
    #[dispatch(mutate, MutationView)]
    impl<'a> Outer<'a> {
        fn mutate_integer(
            &mut self,
            value: i64,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            assert_eq!(std::ptr::from_mut(self).addr() % 128, 0);
            assert_eq!(self.0.get(), 1);
            if value != 1 {
                return Ok(Any::from(value + 1));
            }
            let inner = structural_mutate(2i64, Inner)?;
            assert_eq!(i64::try_from(inner).unwrap(), 12);
            mutator
                .mutate(self, MutationView::borrowed(AnyView::from(&3i64)))
                .map(Into::into)
        }
    }
    let depth = Cell::new(0);
    let calls = Cell::new(0);
    let mutated = structural_mutate(
        Array::new(vec![1i64]),
        StructuralMutator::new(ScopedMutateBase::new(&depth, &calls), Outer(&depth)),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.get(0).unwrap(), 4);
    assert_eq!(depth.get(), 0);
    assert_eq!(calls.get(), 1);
}

#[test]
fn callback_mutate_panics_resume_and_leave_the_next_run_usable() {
    struct Panic;
    #[dispatch(mutate, MutationView)]
    impl Panic {
        fn mutate_integer(
            &mut self,
            _: i64,
            _: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            panic!("callback mutator panic")
        }
    }
    let depth = Cell::new(0);
    let calls = Cell::new(0);
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        structural_mutate(
            Array::new(vec![1i64]),
            StructuralMutator::new(ScopedMutateBase::new(&depth, &calls), Panic),
        )
    })) {
        Err(panic) => panic,
        Ok(_) => panic!("panicking callback mutator unexpectedly returned"),
    };
    assert_eq!(
        panic.downcast_ref::<&str>().copied(),
        Some("callback mutator panic")
    );

    assert_eq!(depth.get(), 0);
    assert_eq!(calls.get(), 1);
    let mutated = structural_mutate(Array::new(vec![1i64]), GeneratedStatelessDispatch)
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(mutated.get(0).unwrap(), 2);
}

#[test]
fn map_entries_preserve_descent_and_callback_composition() {
    let root = || Array::new(vec![1_i64]);
    let first = |value: Any| Array::<i64>::try_from(value).unwrap().get(0).unwrap();
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let depth = Cell::new(0);
        let calls = Cell::new(0);
        let mut dispatch = IncrementIntegers;
        let callbacks = (|s: FfiString| s, (&mut dispatch,));
        assert_eq!(
            first(
                structural_map(
                    root(),
                    StructuralMutator::new(ScopedMutateBase::new(&depth, &calls), callbacks),
                    order
                )
                .unwrap()
            ),
            2
        );

        let callbacks = (|x: i64| x + 1, (|s: FfiString| s,));
        assert_eq!(
            first(
                structural_map(
                    root(),
                    StructuralMutator::new(ScopedMutateBase::new(&depth, &calls), callbacks),
                    order
                )
                .unwrap()
            ),
            2
        );
        assert_eq!(depth.get(), 0);
        assert_eq!(calls.get(), 4);
    }
}

#[test]
fn mutation_descent_preserves_ownership_and_markers() {
    struct Ownership {
        increment: bool,
        retained: Option<Any>,
        retain: bool,
        mode: InplaceMode,
    }
    impl Ownership {
        fn map_integer(&mut self, x: i64, _kind: DefRegionKind) -> Result<Any> {
            (if self.increment {
                UnchangedOr::changed(x + 1)
            } else {
                UnchangedOr::unchanged()
            })
            .into_mutate_result()
        }
    }
    struct MapOwnership(Ownership);
    #[dispatch(map)]
    impl MapOwnership {
        fn map_integer(&mut self, value: i64, kind: DefRegionKind) -> Result<Any> {
            self.0.map_integer(value, kind)
        }
    }
    #[dispatch(mutate, MutationView)]
    impl Ownership {
        fn mutate_integer(&mut self, value: i64) -> Result<Any> {
            self.map_integer(value, DefRegionKind::None)
        }
        fn mutate_array(
            &mut self,
            value: MutationView<'_, Array<i64>>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            if self.retain {
                self.retained = Some(Any::from(value.as_any_view()));
            }
            let result = mutator.default_mutate(self, value.with_inplace_mode(self.mode))?;
            if !self.increment {
                assert!(result.is_unchanged());
            }
            result.into_mutate_result()
        }
    }

    for entry in 0..3 {
        // Pre-order map, post-order map, handler-controlled mutation.
        for case in 0..4 {
            // Unique, shared, retained alias, explicit copy descent.
            for increment in [false, true] {
                let root = Array::new(vec![1_i64]);
                let pointer = array_pointer(&root);
                let alias = (case == 1).then(|| root.clone());
                let state = Ownership {
                    increment,
                    retained: (entry < 2 && case >= 2).then(|| Any::from(root.clone())),
                    retain: case == 2,
                    mode: if case == 3 {
                        InplaceMode::Disallow
                    } else {
                        InplaceMode::Allow
                    },
                };
                let depth = Cell::new(0);
                let calls = Cell::new(0);
                let (output, state) = if entry < 2 {
                    let mut mapper = MapOwnership(state);
                    let output = structural_map(
                        root,
                        StructuralMutator::new(ScopedMutateBase::new(&depth, &calls), &mut mapper),
                        if entry == 0 {
                            WalkOrder::PreOrder
                        } else {
                            WalkOrder::PostOrder
                        },
                    )
                    .unwrap();
                    (output, mapper.0)
                } else {
                    let mut mutator = state;
                    let output = structural_mutate(
                        root,
                        StructuralMutator::new(ScopedMutateBase::new(&depth, &calls), &mut mutator),
                    )
                    .unwrap();
                    (output, mutator)
                };
                assert_eq!(depth.get(), 0);
                assert_eq!(calls.get(), if entry < 2 { 2 } else { 1 });
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
fn mutation_regions_retargeting_and_error_restore() {
    use DefRegionKind::{None as Use, Pattern, Simple};
    #[derive(Default)]
    struct Regions(Vec<(i64, DefRegionKind)>, DefRegionKind);
    impl Regions {
        fn map_integer(&mut self, x: i64, kind: DefRegionKind) -> Result<Any> {
            self.0.push((x, kind));
            Ok(Any::from(x))
        }
        fn map_array(&mut self, value: Array<i64>, kind: DefRegionKind) -> Result<Any> {
            self.0.push((200, kind));
            Ok(value.into())
        }
    }
    #[derive(Default)]
    struct MapRegions(Regions);
    #[dispatch(map)]
    impl MapRegions {
        fn map_integer(&mut self, value: i64, kind: DefRegionKind) -> Result<Any> {
            self.0.map_integer(value, kind)
        }
        fn map_array(&mut self, value: Array<i64>, kind: DefRegionKind) -> Result<Any> {
            self.0.map_array(value, kind)
        }
    }
    #[dispatch(mutate, MutationView)]
    impl Regions {
        fn mutate_any(
            &mut self,
            value: MutationView<'_>,
            mutator: &mut StructuralMutatorView<'_, Self>,
        ) -> Result<Any> {
            if value.as_any_view().try_as::<bool>() == Some(true) {
                assert_eq!(mutator.def_region_kind(), Use);
                assert_eq!(
                    mutator.with_def_region_kind(Pattern, |mutator| mutator.def_region_kind()),
                    Pattern
                );
                assert_eq!(mutator.def_region_kind(), Use);
                let result = mutator.with_def_region_kind(Pattern, |mutator| {
                    assert_eq!(mutator.def_region_kind(), Pattern);
                    mutator.with_def_region_kind(Use, |mutator| {
                        assert_eq!(mutator.def_region_kind(), Pattern);
                        self.0.push((100, mutator.def_region_kind()));
                        {
                            let array = Array::new(vec![5_i64]);
                            mutator
                                .default_mutate(self, MutationView::borrowed(AnyView::from(&array)))
                                .map(|result| result.value_or_else(|| array.into()))
                        }
                    })
                })?;
                assert_eq!(mutator.def_region_kind(), Use);
                let error: Result<()> = mutator.with_def_region_kind(Pattern, |mutator| {
                    assert_eq!(mutator.def_region_kind(), Pattern);
                    Err(Error::new(RUNTIME_ERROR, "handler region error", ""))
                });
                assert!(error
                    .unwrap_err()
                    .message()
                    .contains("handler region error"));
                assert_eq!(mutator.def_region_kind(), Use);
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    mutator.with_def_region_kind(Pattern, |mutator| -> Result<()> {
                        assert_eq!(mutator.def_region_kind(), Pattern);
                        panic!("handler region panic")
                    })
                }))
                .err()
                .unwrap();
                assert_eq!(
                    panic.downcast_ref::<&str>().copied(),
                    Some("handler region panic")
                );
                assert_eq!(mutator.def_region_kind(), Use);
                // The next callback also observes the restored ABI region.
                mutator.mutate(self, MutationView::borrowed(AnyView::from(&6_i64)))?;
                return Ok(result.into());
            }
            if value.as_any_view().try_as::<bool>() == Some(false) {
                return mutator.with_def_region_kind(Pattern, |mutator| {
                    self.0.push((100, mutator.def_region_kind()));
                    let array = Array::new(vec![1_i64]);
                    mutator
                        .default_mutate(self, MutationView::borrowed(AnyView::from(&array)))
                        .map(|result| result.value_or_else(|| array.into()))
                });
            }
            if let Some(x) = value.as_any_view().try_as::<i64>() {
                assert!(value.downcast_ref::<Object>().is_none());
                self.map_integer(x, mutator.def_region_kind())?;
                if x == 2 {
                    assert_eq!(mutator.def_region_kind(), Pattern);
                    mutator.with_def_region_kind(self.1, |mutator| {
                        mutator.mutate(self, MutationView::borrowed(AnyView::from(&99_i64)))
                    })?;
                    let error: Result<()> = mutator.with_def_region_kind(Use, |mutator| {
                        assert_eq!(mutator.def_region_kind(), Pattern);
                        Err(Error::new(RUNTIME_ERROR, "scoped error", ""))
                    });
                    assert!(error.is_err());
                    assert_eq!(mutator.def_region_kind(), Pattern);
                }
            }
            if value
                .downcast_ref::<tvm_ffi::collections::array::ArrayObj>()
                .is_some()
            {
                self.0.push((200, mutator.def_region_kind()));
            }
            mutator.default_mutate(self, value).into_mutate_result()
        }
    }

    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    for requested in [Use, Simple] {
        for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
            let mut mapper = MapRegions::default();
            let graph = Function::get_global("testing.make_visit_region_graph")
                .unwrap()
                .call_tuple((false,))
                .unwrap();
            structural_map(graph, &mut mapper, order).unwrap();
            let expected = vec![(1, Simple), (2, Pattern), (3, Use)];
            assert_eq!(mapper.0 .0, expected);
        }
        let mut scoped = Regions(vec![], requested);
        let output = structural_mutate(true, &mut scoped).unwrap();
        assert_eq!(i64::try_from(array_item(&output, 0)).unwrap(), 5);
        assert_eq!(scoped.0, vec![(100, Pattern), (5, Pattern), (6, Use)]);
        let mut mutator = Regions(vec![], requested);
        let output = structural_mutate(false, &mut mutator).unwrap();
        assert_eq!(i64::try_from(array_item(&output, 0)).unwrap(), 1);
        assert_eq!(mutator.0, vec![(100, Pattern), (1, Pattern)]);
        mutator.0.clear();
        let graph = Function::get_global("testing.make_visit_region_graph")
            .unwrap()
            .call_tuple((false,))
            .unwrap();
        structural_mutate(graph, &mut mutator).unwrap();
        assert_eq!(
            mutator.0,
            vec![(1, Simple), (2, Pattern), (99, Pattern), (3, Use)]
        );
    }
}
