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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use tvm_ffi::derive::{Object as DeriveObject, ObjectRef as DeriveObjectRef};
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyViewToOwnedAny, TVMFFIByteArray, TVMFFIFieldFlagBitMask, TVMFFIFieldInfo,
    TVMFFISEqHashKind, TVMFFITypeMetadata, TVMFFITypeRegisterAttr,
};
use tvm_ffi::{
    dispatch, structural_map, structural_mutate, Any, AnyView, Array, DefRegionKind, Error,
    Function, InplaceValue, Map, MapDispatch, MapValue, Object, ObjectArc, ObjectCore,
    ObjectRefCore, Result, String as FfiString, StructuralMutator, StructuralVarRemap, TypeIndex,
    WalkOrder, RUNTIME_ERROR,
};

// These registration entry points are needed only to build reflected test
// types. Keep them local instead of expanding tvm-ffi-sys's public API.
unsafe extern "C" {
    fn TVMFFITypeGetOrAllocIndex(
        type_key: *const TVMFFIByteArray,
        static_type_index: i32,
        type_depth: i32,
        num_child_slots: i32,
        child_slots_can_overflow: i32,
        parent_type_index: i32,
    ) -> i32;
    fn TVMFFITypeRegisterField(type_index: i32, info: *const TVMFFIFieldInfo) -> i32;
    fn TVMFFITypeRegisterMetadata(type_index: i32, metadata: *const TVMFFITypeMetadata) -> i32;
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralDagNode"]
#[type_final]
struct RustDagNodeObj {
    base: Object,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustDagNode {
    data: ObjectArc<RustDagNodeObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralFreeVar"]
#[type_final]
struct RustFreeVarObj {
    base: Object,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustFreeVar {
    data: ObjectArc<RustFreeVarObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralPair"]
#[type_final]
struct RustPairObj {
    base: Object,
    first: Any,
    ignored: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustPair {
    data: ObjectArc<RustPairObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralNoCopy"]
#[type_final]
struct RustNoCopyObj {
    base: Object,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustNoCopy {
    data: ObjectArc<RustNoCopyObj>,
}

static SHALLOW_COPY_CALLS: AtomicUsize = AtomicUsize::new(0);
static REFLECTED_TEST_LOCK: Mutex<()> = Mutex::new(());

unsafe extern "C" fn any_field_getter(field: *mut std::ffi::c_void, result: *mut TVMFFIAny) -> i32 {
    TVMFFIAnyViewToOwnedAny(field.cast(), result)
}

unsafe extern "C" fn any_field_setter(
    field: *mut std::ffi::c_void,
    value: *const TVMFFIAny,
) -> i32 {
    let mut replacement = TVMFFIAny::new();
    let code = TVMFFIAnyViewToOwnedAny(value, &mut replacement);
    if code != 0 {
        return code;
    }
    let field = &mut *field.cast::<Any>();
    *field = Any::from_raw_ffi_any(replacement);
    0
}

fn register_any_field(type_index: i32, name: &'static str, offset: usize, flags: i64) {
    let field = TVMFFIFieldInfo {
        name: unsafe { TVMFFIByteArray::from_str(name) },
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-mutation test field") },
        metadata: unsafe { TVMFFIByteArray::from_str("") },
        flags,
        size: std::mem::size_of::<Any>() as i64,
        alignment: std::mem::align_of::<Any>() as i64,
        offset: offset as i64,
        getter: Some(any_field_getter),
        setter: any_field_setter as *mut std::ffi::c_void,
        default_value_or_factory: TVMFFIAny::new(),
        field_static_type_index: -1,
    };
    assert_eq!(unsafe { TVMFFITypeRegisterField(type_index, &field) }, 0);
}

fn register_identity_type(type_key: &'static str, total_size: usize, kind: TVMFFISEqHashKind) {
    let type_key = unsafe { TVMFFIByteArray::from_str(type_key) };
    let type_index = unsafe {
        TVMFFITypeGetOrAllocIndex(
            &type_key,
            -1,
            Object::TYPE_DEPTH + 1,
            0,
            1,
            Object::type_index(),
        )
    };
    assert!(type_index >= TypeIndex::kTVMFFIDynObjectBegin as i32);
    let metadata = TVMFFITypeMetadata {
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-mutation test identity") },
        creator: None,
        total_size: i32::try_from(total_size).unwrap(),
        structural_eq_hash_kind: kind as i32,
    };
    assert_eq!(
        unsafe { TVMFFITypeRegisterMetadata(type_index, &metadata) },
        0
    );
}

static REGISTER_IDENTITY_TYPES: LazyLock<()> = LazyLock::new(|| {
    register_identity_type(
        RustDagNodeObj::TYPE_KEY,
        std::mem::size_of::<RustDagNodeObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode,
    );
    register_identity_type(
        RustFreeVarObj::TYPE_KEY,
        std::mem::size_of::<RustFreeVarObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar,
    );
    register_identity_type(
        RustNoCopyObj::TYPE_KEY,
        std::mem::size_of::<RustNoCopyObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
    );

    let type_key = unsafe { TVMFFIByteArray::from_str(RustPairObj::TYPE_KEY) };
    let type_index = unsafe {
        TVMFFITypeGetOrAllocIndex(
            &type_key,
            -1,
            Object::TYPE_DEPTH + 1,
            0,
            1,
            Object::type_index(),
        )
    };
    assert!(type_index >= TypeIndex::kTVMFFIDynObjectBegin as i32);
    register_any_field(
        type_index,
        "first",
        std::mem::offset_of!(RustPairObj, first),
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64,
    );
    register_any_field(
        type_index,
        "ignored",
        std::mem::offset_of!(RustPairObj, ignored),
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64,
    );
    let metadata = TVMFFITypeMetadata {
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-mutation reflected pair") },
        creator: None,
        total_size: i32::try_from(std::mem::size_of::<RustPairObj>()).unwrap(),
        structural_eq_hash_kind: TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode as i32,
    };
    assert_eq!(
        unsafe { TVMFFITypeRegisterMetadata(type_index, &metadata) },
        0
    );

    let shallow_copy = Function::from_packed(|args| {
        SHALLOW_COPY_CALLS.fetch_add(1, Ordering::Relaxed);
        let source = RustPair::try_from(args[0])?;
        Ok(Any::from(RustPair {
            data: ObjectArc::new(RustPairObj {
                base: Object::new(),
                first: source.data.first.clone(),
                ignored: source.data.ignored.clone(),
            }),
        }))
    });
    let attr_name = unsafe { TVMFFIByteArray::from_str("__ffi_shallow_copy__") };
    let mut attr_value = Any::from(shallow_copy);
    assert_eq!(
        unsafe {
            TVMFFITypeRegisterAttr(type_index, &attr_name, Any::as_data_ptr(&mut attr_value))
        },
        0
    );

    // The native Rust mapper deliberately refuses to invoke foreign mutator
    // hooks because their ABI requires a C++ StructuralMutatorObj. A custom
    // Rust StructuralMutator remains able to take this type over explicitly.
    let foreign_mutate = Function::from_packed(|args| Ok(Any::from(args[1])));
    let attr_name = unsafe { TVMFFIByteArray::from_str("__s_mutate__") };
    let mut attr_value = Any::from(foreign_mutate);
    assert_eq!(
        unsafe {
            TVMFFITypeRegisterAttr(
                RustFreeVarObj::type_index(),
                &attr_name,
                Any::as_data_ptr(&mut attr_value),
            )
        },
        0
    );
});

fn ensure_test_types_registered() {
    LazyLock::force(&REGISTER_IDENTITY_TYPES);
}

fn rust_dag_node() -> RustDagNode {
    LazyLock::force(&REGISTER_IDENTITY_TYPES);
    RustDagNode {
        data: ObjectArc::new(RustDagNodeObj {
            base: Object::new(),
        }),
    }
}

fn rust_free_var() -> RustFreeVar {
    LazyLock::force(&REGISTER_IDENTITY_TYPES);
    RustFreeVar {
        data: ObjectArc::new(RustFreeVarObj {
            base: Object::new(),
        }),
    }
}

fn rust_pair(first: impl Into<Any>, ignored: impl Into<Any>) -> RustPair {
    LazyLock::force(&REGISTER_IDENTITY_TYPES);
    RustPair {
        data: ObjectArc::new(RustPairObj {
            base: Object::new(),
            first: first.into(),
            ignored: ignored.into(),
        }),
    }
}

fn rust_no_copy() -> RustNoCopy {
    ensure_test_types_registered();
    RustNoCopy {
        data: ObjectArc::new(RustNoCopyObj {
            base: Object::new(),
        }),
    }
}

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
    fn mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        if let Some(integer) = value.cast::<i64>() {
            Ok(Any::from(integer + 1))
        } else {
            self.default_mutate(value, def_region_kind)
        }
    }

    fn maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.default_maybe_inplace_mutate(value, def_region_kind)
    }

    fn var_remap_get(&mut self, var: &MapValue) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &MapValue, mapped_value: &Any) -> Result<()> {
        self.remap.set(var, mapped_value)
    }
}

struct RemappingFreeVar {
    remap: StructuralVarRemap,
    type_index: i32,
    calls: usize,
}

impl StructuralMutator for RemappingFreeVar {
    fn mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        if value.type_index() == self.type_index {
            if let Some(mapped) = self.remap.get(value)? {
                return Ok(mapped);
            }
            self.calls += 1;
            let mapped = Any::from(41i64);
            self.remap.set(value, &mapped)?;
            Ok(mapped)
        } else {
            self.default_mutate(value, def_region_kind)
        }
    }

    fn maybe_inplace_mutate(
        &mut self,
        value: InplaceValue<'_>,
        def_region_kind: DefRegionKind,
    ) -> Result<Any> {
        self.default_maybe_inplace_mutate(value, def_region_kind)
    }

    fn var_remap_get(&mut self, var: &MapValue) -> Result<Option<Any>> {
        self.remap.get(var)
    }

    fn var_remap_set(&mut self, var: &MapValue, mapped_value: &Any) -> Result<()> {
        self.remap.set(var, mapped_value)
    }
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

fn dict_size(dict: &Any) -> i64 {
    Function::get_global("ffi.DictSize")
        .unwrap()
        .call_packed(&[AnyView::from(dict)])
        .and_then(i64::try_from)
        .unwrap()
}

#[test]
fn unique_array_is_reused_while_shared_array_uses_copy_on_write() {
    ensure_test_types_registered();
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
    ensure_test_types_registered();
    let unique = Array::new(vec![1i64, 2]);
    let unique_pointer = array_pointer(&unique);
    let mapped = structural_mutate(unique, &mut ManualIncrement::default())
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_eq!(array_pointer(&mapped), unique_pointer);
    assert_eq!(mapped.iter().collect::<Vec<_>>(), vec![2, 3]);

    let source = Array::new(vec![3i64]);
    let source_pointer = array_pointer(&source);
    let mapped = structural_mutate(source.clone(), &mut ManualIncrement::default())
        .and_then(Array::<i64>::try_from)
        .unwrap();
    assert_ne!(array_pointer(&mapped), source_pointer);
    assert_eq!(source.get(0).unwrap(), 3);
    assert_eq!(mapped.get(0).unwrap(), 4);

    let map: Map<i64, i64> = [(1, 10)].into_iter().collect();
    let source_map_pointer = map_pointer(&map);
    let mapped_map = structural_mutate(map, &mut ManualIncrement::default())
        .and_then(Map::<i64, i64>::try_from)
        .unwrap();
    assert_ne!(map_pointer(&mapped_map), source_map_pointer);
    assert_eq!(mapped_map.get(&1).unwrap(), Some(11));

    let dict = call_global("ffi.Dict", &[Any::from(1i64), Any::from(10i64)]);
    let dict_pointer = any_object_pointer(&dict);
    let mapped_dict = structural_mutate(dict, &mut ManualIncrement::default()).unwrap();
    assert_ne!(any_object_pointer(&mapped_dict), dict_pointer);
    assert_eq!(dict_item(&mapped_dict, 1), 11);
}

#[test]
fn user_mutator_can_store_a_changed_free_var_result() {
    ensure_test_types_registered();
    let var = rust_free_var();
    let type_index = RustFreeVarObj::type_index();
    let root = call_global("ffi.Array", &[Any::from(var.clone()), Any::from(var)]);
    let mut mutator = RemappingFreeVar {
        remap: StructuralVarRemap::default(),
        type_index,
        calls: 0,
    };

    let mapped = structural_mutate(root, &mut mutator).unwrap();
    assert_eq!(mutator.calls, 1);
    assert_eq!(i64::try_from(array_item(&mapped, 0)).unwrap(), 41);
    assert_eq!(i64::try_from(array_item(&mapped, 1)).unwrap(), 41);
}

#[test]
fn dag_identity_caches_the_final_pre_order_replacement() {
    ensure_test_types_registered();
    let node = rust_dag_node();
    let root = call_global("ffi.Array", &[Any::from(node.clone()), Any::from(node)]);
    let mut identity_calls = 0;
    let mapped = structural_map(
        root,
        (
            |_node: &RustDagNodeObj| {
                identity_calls += 1;
                Any::from(Array::new(vec![1i64]))
            },
            |integer: i64| Any::from(integer + 1),
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();

    let first = array_item(&mapped, 0);
    let second = array_item(&mapped, 1);
    assert_eq!(identity_calls, 1);
    assert_eq!(any_object_pointer(&first), any_object_pointer(&second));
    assert_eq!(i64::try_from(array_item(&first, 0)).unwrap(), 2);
    assert_eq!(i64::try_from(array_item(&second, 0)).unwrap(), 2);
}

#[test]
fn free_var_identity_caches_the_final_pre_order_replacement() {
    ensure_test_types_registered();
    let node = rust_free_var();
    let type_index = RustFreeVarObj::type_index();
    let root = call_global("ffi.Array", &[Any::from(node.clone()), Any::from(node)]);
    let mut identity_calls = 0;
    let mapped = structural_map(
        root,
        |value: &MapValue| {
            if value.type_index() == type_index {
                identity_calls += 1;
                Any::from(Array::new(vec![1i64]))
            } else if let Some(integer) = value.cast::<i64>() {
                Any::from(integer + 1)
            } else {
                value.to_owned()
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap();

    let first = array_item(&mapped, 0);
    let second = array_item(&mapped, 1);
    assert_eq!(identity_calls, 1);
    assert_eq!(any_object_pointer(&first), any_object_pointer(&second));
    assert_eq!(i64::try_from(array_item(&first, 0)).unwrap(), 2);
    assert_eq!(i64::try_from(array_item(&second, 0)).unwrap(), 2);
}

#[test]
fn reflected_fields_use_shallow_copy_setters_and_field_flags() {
    ensure_test_types_registered();
    let _guard = REFLECTED_TEST_LOCK.lock().unwrap();
    let source = rust_pair(1i64, 9i64);
    let source_pointer = unsafe { ObjectArc::as_raw(&source.data) };
    let mut regions = Vec::new();
    let mapped = structural_map(
        source.clone(),
        |integer: i64, kind: DefRegionKind| {
            regions.push(kind);
            Any::from(integer + 1)
        },
        WalkOrder::PostOrder,
    )
    .and_then(RustPair::try_from)
    .unwrap();

    assert_ne!(unsafe { ObjectArc::as_raw(&mapped.data) }, source_pointer);
    assert_eq!(i64::try_from(source.data.first.clone()).unwrap(), 1);
    assert_eq!(i64::try_from(mapped.data.first.clone()).unwrap(), 2);
    assert_eq!(i64::try_from(mapped.data.ignored.clone()).unwrap(), 9);
    assert_eq!(regions, vec![DefRegionKind::Recursive]);
}

#[test]
fn reflected_no_change_still_validates_copy_and_returns_original() {
    ensure_test_types_registered();
    let _guard = REFLECTED_TEST_LOCK.lock().unwrap();
    let source = rust_pair(1i64, 9i64);
    let source_pointer = unsafe { ObjectArc::as_raw(&source.data) };
    let calls_before = SHALLOW_COPY_CALLS.load(Ordering::Relaxed);
    let mapped = structural_map(
        source.clone(),
        |string: FfiString| Any::from(string),
        WalkOrder::PostOrder,
    )
    .and_then(RustPair::try_from)
    .unwrap();

    assert_eq!(unsafe { ObjectArc::as_raw(&mapped.data) }, source_pointer);
    assert_eq!(SHALLOW_COPY_CALLS.load(Ordering::Relaxed), calls_before + 1);
}

#[test]
fn reflected_object_without_shallow_copy_is_rejected_even_when_unchanged() {
    ensure_test_types_registered();
    let error = match structural_map(
        rust_no_copy(),
        |_integer: i64| Any::from(0i64),
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("reflected object without a shallow-copy hook unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(error.message().contains("__ffi_shallow_copy__"));
}

#[test]
fn callback_errors_preserve_message_and_add_structural_path() {
    ensure_test_types_registered();
    let error = match structural_map(
        Array::new(vec![1i64]),
        |_integer: i64| -> Result<Any> {
            Err(Error::new(RUNTIME_ERROR, "mapper failed", "origin"))
        },
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("fallible structural mapper unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_eq!(error.message(), "mapper failed");
    assert!(error.backtrace().contains("origin"));
    assert!(error.backtrace().contains("sequence item [0]"));
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn foreign_mutation_hook_requires_explicit_rust_takeover() {
    ensure_test_types_registered();
    let error = match structural_map(
        rust_free_var(),
        |_integer: i64| Any::from(0i64),
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("foreign structural mutation unexpectedly used reflection"),
        Err(error) => error,
    };
    assert!(error.message().contains("registers foreign `__s_mutate__`"));
    assert!(error
        .message()
        .contains("implement its mutation explicitly in Rust"));
}

#[test]
fn unique_list_reuses_storage_while_map_and_dict_are_rebuilt() {
    ensure_test_types_registered();
    let list = call_global("ffi.List", &[Any::from(1i64), Any::from(2i64)]);
    let list_pointer = any_object_pointer(&list);
    let mapped_list = structural_map(list, &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    assert_eq!(any_object_pointer(&mapped_list), list_pointer);
    assert_eq!(
        (list_item(&mapped_list, 0), list_item(&mapped_list, 1)),
        (2, 3)
    );

    let small: Map<i64, i64> = [(1, 10), (2, 20)].into_iter().collect();
    let small_pointer = map_pointer(&small);
    let mapped_small = structural_map(small, &mut IncrementIntegers, WalkOrder::PostOrder)
        .and_then(Map::<i64, i64>::try_from)
        .unwrap();
    assert_ne!(map_pointer(&mapped_small), small_pointer);
    assert_eq!(mapped_small.get(&1).unwrap(), Some(11));
    assert_eq!(mapped_small.get(&2).unwrap(), Some(21));

    // More than four entries selects the dense map layout.
    let dense: Map<i64, i64> = (0..9).map(|value| (value, value * 10)).collect();
    let dense_pointer = map_pointer(&dense);
    let mapped_dense = structural_map(dense, &mut IncrementIntegers, WalkOrder::PostOrder)
        .and_then(Map::<i64, i64>::try_from)
        .unwrap();
    assert_ne!(map_pointer(&mapped_dense), dense_pointer);
    for value in 0..9 {
        assert_eq!(mapped_dense.get(&value).unwrap(), Some(value * 10 + 1));
    }

    let dict = call_global(
        "ffi.Dict",
        &[
            Any::from(1i64),
            Any::from(10i64),
            Any::from(2i64),
            Any::from(20i64),
        ],
    );
    let dict_pointer = any_object_pointer(&dict);
    let mapped_dict = structural_map(dict, &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    assert_ne!(any_object_pointer(&mapped_dict), dict_pointer);
    assert_eq!(
        (dict_item(&mapped_dict, 1), dict_item(&mapped_dict, 2)),
        (11, 21)
    );

    let mut dense_dict_args = Vec::new();
    for value in 0..9i64 {
        dense_dict_args.push(Any::from(value));
        dense_dict_args.push(Any::from(value * 10));
    }
    let dense_dict = call_global("ffi.Dict", &dense_dict_args);
    let dense_dict_pointer = any_object_pointer(&dense_dict);
    let mapped_dense_dict =
        structural_map(dense_dict, &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    assert_ne!(any_object_pointer(&mapped_dense_dict), dense_dict_pointer);
    for value in 0..9i64 {
        assert_eq!(dict_item(&mapped_dense_dict, value), value * 10 + 1);
    }
}

#[test]
fn shared_map_and_dict_copy_only_when_a_value_changes() {
    ensure_test_types_registered();
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
fn dict_entries_are_snapshotted_before_callbacks() {
    ensure_test_types_registered();
    let dict = call_global(
        "ffi.Dict",
        &[
            Any::from(1i64),
            Any::from(10i64),
            Any::from(2i64),
            Any::from(20i64),
        ],
    );
    let captured = dict.clone();
    let set_item = Function::get_global("ffi.DictSetItem").unwrap();
    let mut inserted = false;
    let mapped = structural_map(
        dict,
        |value: &MapValue| {
            if let Some(integer) = value.cast::<i64>() {
                if !inserted {
                    set_item
                        .call_packed(&[
                            AnyView::from(&captured),
                            AnyView::from(&3i64),
                            AnyView::from(&30i64),
                        ])
                        .unwrap();
                    inserted = true;
                }
                Any::from(integer + 1)
            } else {
                value.to_owned()
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap();

    assert_eq!(dict_size(&mapped), 2);
    assert_eq!((dict_item(&mapped, 1), dict_item(&mapped, 2)), (11, 21));
    assert_eq!(dict_size(&captured), 3);
    assert_eq!(dict_item(&captured, 3), 30);
}

#[test]
fn shared_outer_container_does_not_mutate_its_nested_child() {
    ensure_test_types_registered();
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
fn shared_list_uses_snapshot_and_copy_on_write() {
    ensure_test_types_registered();
    let source = call_global("ffi.List", &[Any::from(1i64), Any::from(2i64)]);
    let source_pointer = any_object_pointer(&source);
    let captured = source.clone();
    let append = Function::get_global("ffi.ListAppend").unwrap();
    let mut appended = false;

    let mapped = structural_map(
        source,
        |value: &MapValue| {
            if let Some(integer) = value.cast::<i64>() {
                if !appended {
                    append
                        .call_packed(&[AnyView::from(&captured), AnyView::from(&3i64)])
                        .unwrap();
                    appended = true;
                }
                Any::from(integer + 1)
            } else {
                value.to_owned()
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap();

    assert_ne!(any_object_pointer(&mapped), source_pointer);
    assert_eq!((list_item(&mapped, 0), list_item(&mapped, 1)), (2, 3));
    assert_eq!(
        (
            list_item(&captured, 0),
            list_item(&captured, 1),
            list_item(&captured, 2),
        ),
        (1, 2, 3)
    );
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
    ensure_test_types_registered();
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

#[test]
fn pre_order_retained_alias_disables_in_place_mutation() {
    ensure_test_types_registered();
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
    ensure_test_types_registered();
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
fn callbacks_run_in_the_configured_order() {
    ensure_test_types_registered();
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
    ensure_test_types_registered();
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
