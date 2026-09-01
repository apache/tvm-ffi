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
    Function, InplaceValue, Map, MapDispatch, MapValue, MutateCallbacks, Mutator, Object,
    ObjectArc, ObjectCore, ObjectRefCore, Result, String as FfiString, StructuralMutator,
    StructuralVarRemap, TypeIndex, WalkOrder, RUNTIME_ERROR,
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

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralHookNode"]
#[type_final]
struct RustHookNodeObj {
    base: Object,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustHookNode {
    data: ObjectArc<RustHookNodeObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralFailingGetter"]
#[type_final]
struct RustFailingGetterObj {
    base: Object,
    value: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustFailingGetter {
    data: ObjectArc<RustFailingGetterObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralFailingSetter"]
#[type_final]
struct RustFailingSetterObj {
    base: Object,
    value: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustFailingSetter {
    data: ObjectArc<RustFailingSetterObj>,
}

static SHALLOW_COPY_CALLS: AtomicUsize = AtomicUsize::new(0);
static REGISTERED_MUTATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static REGISTERED_MAYBE_INPLACE_MUTATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static REFLECTED_TEST_LOCK: Mutex<()> = Mutex::new(());
static REGISTERED_HOOK_TEST_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static RETAINED_MUTATOR: RefCell<Option<Any>> = const { RefCell::new(None) };
    static PROBE_FOREIGN_THREAD_MUTATOR: Cell<bool> = const { Cell::new(false) };
    static FOREIGN_THREAD_MUTATOR_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn call_mutator_from_foreign_thread(mutator: AnyView<'_>) -> String {
    // Keep an owning reference on this thread while the worker constructs a
    // borrowed ABI view from the raw object address.
    let mut owner = Any::from(mutator);
    let raw = unsafe { *Any::as_data_ptr(&mut owner) };
    let type_index = raw.type_index;
    let object = unsafe { raw.data_union.v_obj } as usize;
    std::thread::spawn(move || {
        let mut raw = TVMFFIAny::new();
        raw.type_index = type_index;
        raw.data_union.v_obj = object as *mut _;
        let borrowed = std::mem::ManuallyDrop::new(unsafe { Any::from_raw_ffi_any(raw) });
        match Function::get_global("ffi.StructuralMutatorMutate")
            .unwrap()
            .call_packed(&[AnyView::from(&*borrowed), AnyView::from(&1i64)])
        {
            Err(error) => error.message().to_string(),
            Ok(_) => "foreign-thread mutator call unexpectedly succeeded".to_string(),
        }
    })
    .join()
    .unwrap()
}

fn run_registered_mutate_hook(args: &[AnyView<'_>], calls: &AtomicUsize) -> Result<Any> {
    assert_eq!(args.len(), 2);
    if PROBE_FOREIGN_THREAD_MUTATOR.with(Cell::get) {
        let message = call_mutator_from_foreign_thread(args[0]);
        FOREIGN_THREAD_MUTATOR_ERROR.with(|error| error.replace(Some(message)));
        return Ok(Any::from(args[1]));
    }
    Function::get_global("ffi.StructuralMutatorDefRegionKind")
        .unwrap()
        .call_packed(&[args[0]])?;

    // Keep one reference so the test can verify that a type hook cannot use
    // the mutator after the structural-mutation call has ended.
    RETAINED_MUTATOR.with(|retained| {
        retained.replace(Some(Any::from(args[0])));
    });

    let cached = Function::get_global("ffi.StructuralMutatorVarRemapGet")
        .unwrap()
        .call_packed(&[args[0], args[1]])?;
    if cached.type_index() != TypeIndex::kTVMFFINone as i32 {
        return Ok(cached);
    }

    let child = Any::from(1i64);
    let mutated = Function::get_global("ffi.StructuralMutatorMutate")
        .unwrap()
        .call_packed(&[args[0], AnyView::from(&child)])?;
    Function::get_global("ffi.StructuralMutatorVarRemapSet")
        .unwrap()
        .call_packed(&[args[0], args[1], AnyView::from(&mutated)])?;
    calls.fetch_add(1, Ordering::Relaxed);
    Ok(mutated)
}

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

unsafe extern "C" fn clone_any_then_fail(
    source: *mut std::ffi::c_void,
    result: *mut TVMFFIAny,
) -> i32 {
    let code = TVMFFIAnyViewToOwnedAny(source.cast(), result);
    if code != 0 {
        return code;
    }
    Error::set_raised(&Error::new(
        RUNTIME_ERROR,
        "callback failed after writing an owning result",
        "",
    ));
    -1
}

unsafe extern "C" fn setter_safe_call(
    _handle: *mut std::ffi::c_void,
    args: *const TVMFFIAny,
    _num_args: i32,
    result: *mut TVMFFIAny,
) -> i32 {
    clone_any_then_fail(args.add(1).cast_mut().cast(), result)
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

fn register_test_type(type_key: &'static str, total_size: usize, kind: TVMFFISEqHashKind) -> i32 {
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
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-mutation test type") },
        creator: None,
        total_size: i32::try_from(total_size).unwrap(),
        structural_eq_hash_kind: kind as i32,
    };
    assert_eq!(
        unsafe { TVMFFITypeRegisterMetadata(type_index, &metadata) },
        0
    );
    type_index
}

fn register_function_attr(type_index: i32, name: &'static str, function: Function) {
    let name = unsafe { TVMFFIByteArray::from_str(name) };
    let mut value = Any::from(function);
    assert_eq!(
        unsafe { TVMFFITypeRegisterAttr(type_index, &name, Any::as_data_ptr(&mut value)) },
        0
    );
}

static REGISTER_TEST_TYPES: LazyLock<()> = LazyLock::new(|| {
    register_test_type(
        RustDagNodeObj::TYPE_KEY,
        std::mem::size_of::<RustDagNodeObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode,
    );
    register_test_type(
        RustFreeVarObj::TYPE_KEY,
        std::mem::size_of::<RustFreeVarObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar,
    );
    register_test_type(
        RustNoCopyObj::TYPE_KEY,
        std::mem::size_of::<RustNoCopyObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
    );
    register_test_type(
        RustHookNodeObj::TYPE_KEY,
        std::mem::size_of::<RustHookNodeObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode,
    );

    let pair_type_index = register_test_type(
        RustPairObj::TYPE_KEY,
        std::mem::size_of::<RustPairObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
    );
    register_any_field(
        pair_type_index,
        "first",
        std::mem::offset_of!(RustPairObj, first),
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64,
    );
    register_any_field(
        pair_type_index,
        "ignored",
        std::mem::offset_of!(RustPairObj, ignored),
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64,
    );

    let getter_type_index = register_test_type(
        RustFailingGetterObj::TYPE_KEY,
        std::mem::size_of::<RustFailingGetterObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
    );
    let field = TVMFFIFieldInfo {
        name: unsafe { TVMFFIByteArray::from_str("value") },
        doc: unsafe { TVMFFIByteArray::from_str("Fail after producing an owning field value") },
        metadata: unsafe { TVMFFIByteArray::from_str("") },
        flags: 0,
        size: std::mem::size_of::<Any>() as i64,
        alignment: std::mem::align_of::<Any>() as i64,
        offset: std::mem::offset_of!(RustFailingGetterObj, value) as i64,
        getter: Some(clone_any_then_fail),
        setter: any_field_setter as *mut std::ffi::c_void,
        default_value_or_factory: TVMFFIAny::new(),
        field_static_type_index: -1,
    };
    assert_eq!(
        unsafe { TVMFFITypeRegisterField(getter_type_index, &field) },
        0
    );
    let shallow_copy = Function::from_packed(|args| {
        let source = RustFailingGetter::try_from(args[0])?;
        Ok(Any::from(RustFailingGetter {
            data: ObjectArc::new(RustFailingGetterObj {
                base: Object::new(),
                value: source.data.value.clone(),
            }),
        }))
    });
    register_function_attr(getter_type_index, "__ffi_shallow_copy__", shallow_copy);

    let setter_type_index = register_test_type(
        RustFailingSetterObj::TYPE_KEY,
        std::mem::size_of::<RustFailingSetterObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
    );
    let setter = unsafe { Function::from_extern_c(std::ptr::null_mut(), setter_safe_call, None) };
    let field = TVMFFIFieldInfo {
        name: unsafe { TVMFFIByteArray::from_str("value") },
        doc: unsafe { TVMFFIByteArray::from_str("Fail after producing an owning setter result") },
        metadata: unsafe { TVMFFIByteArray::from_str("") },
        flags: TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitSetterIsFunctionObj as i64,
        size: std::mem::size_of::<Any>() as i64,
        alignment: std::mem::align_of::<Any>() as i64,
        offset: std::mem::offset_of!(RustFailingSetterObj, value) as i64,
        getter: Some(any_field_getter),
        setter: unsafe {
            ObjectArc::as_raw(<Function as ObjectRefCore>::data(&setter))
                .cast_mut()
                .cast()
        },
        default_value_or_factory: TVMFFIAny::new(),
        field_static_type_index: -1,
    };
    assert_eq!(
        unsafe { TVMFFITypeRegisterField(setter_type_index, &field) },
        0
    );
    let shallow_copy = Function::from_packed(|args| {
        let source = RustFailingSetter::try_from(args[0])?;
        Ok(Any::from(RustFailingSetter {
            data: ObjectArc::new(RustFailingSetterObj {
                base: Object::new(),
                value: source.data.value.clone(),
            }),
        }))
    });
    register_function_attr(setter_type_index, "__ffi_shallow_copy__", shallow_copy);

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
    register_function_attr(pair_type_index, "__ffi_shallow_copy__", shallow_copy);

    let registered_mutate =
        Function::from_packed(|args| run_registered_mutate_hook(args, &REGISTERED_MUTATE_CALLS));
    register_function_attr(
        RustHookNodeObj::type_index(),
        "__s_mutate__",
        registered_mutate,
    );

    let registered_maybe_inplace_mutate = Function::from_packed(|args| {
        run_registered_mutate_hook(args, &REGISTERED_MAYBE_INPLACE_MUTATE_CALLS)
    });
    register_function_attr(
        RustHookNodeObj::type_index(),
        "__s_maybe_inplace_mutate__",
        registered_maybe_inplace_mutate,
    );
});

fn ensure_test_types_registered() {
    LazyLock::force(&REGISTER_TEST_TYPES);
}

fn rust_dag_node() -> RustDagNode {
    LazyLock::force(&REGISTER_TEST_TYPES);
    RustDagNode {
        data: ObjectArc::new(RustDagNodeObj {
            base: Object::new(),
        }),
    }
}

fn rust_free_var() -> RustFreeVar {
    LazyLock::force(&REGISTER_TEST_TYPES);
    RustFreeVar {
        data: ObjectArc::new(RustFreeVarObj {
            base: Object::new(),
        }),
    }
}

fn rust_pair(first: impl Into<Any>, ignored: impl Into<Any>) -> RustPair {
    LazyLock::force(&REGISTER_TEST_TYPES);
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

fn rust_hook_node() -> RustHookNode {
    ensure_test_types_registered();
    RustHookNode {
        data: ObjectArc::new(RustHookNodeObj {
            base: Object::new(),
        }),
    }
}

fn rust_failing_getter(value: impl Into<Any>) -> RustFailingGetter {
    ensure_test_types_registered();
    RustFailingGetter {
        data: ObjectArc::new(RustFailingGetterObj {
            base: Object::new(),
            value: value.into(),
        }),
    }
}

fn rust_failing_setter(value: impl Into<Any>) -> RustFailingSetter {
    ensure_test_types_registered();
    RustFailingSetter {
        data: ObjectArc::new(RustFailingSetterObj {
            base: Object::new(),
            value: value.into(),
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

struct RemappingFreeVar {
    remap: StructuralVarRemap,
    type_index: i32,
    calls: usize,
}

impl StructuralMutator for RemappingFreeVar {
    fn dispatch_mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        if value.type_index() == self.type_index {
            if let Some(mutated) = self.remap.get(value)? {
                return Ok(mutated);
            }
            self.calls += 1;
            let mutated = Any::from(41i64);
            self.remap.set(value, &mutated)?;
            Ok(mutated)
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

#[derive(Default)]
struct RejectAliasedReentry {
    remap: StructuralVarRemap,
    rejected: bool,
}

impl StructuralMutator for RejectAliasedReentry {
    fn dispatch_mutate(&mut self, value: &MapValue, def_region_kind: DefRegionKind) -> Result<Any> {
        if let Some(integer) = value.cast::<i64>() {
            let retained = RETAINED_MUTATOR.with(|slot| slot.borrow().as_ref().unwrap().clone());
            let error = match Function::get_global("ffi.StructuralMutatorMutate")
                .unwrap()
                .call_packed(&[AnyView::from(&retained), AnyView::from(&integer)])
            {
                Ok(_) => panic!("aliased structural-mutator reentry unexpectedly succeeded"),
                Err(error) => error,
            };
            assert!(error
                .message()
                .contains("may only be called by its active registered hook"));
            self.rejected = true;
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

    let mutated = structural_mutate(root, &mut mutator).unwrap();
    assert_eq!(mutator.calls, 1);
    assert_eq!(i64::try_from(array_item(&mutated, 0)).unwrap(), 41);
    assert_eq!(i64::try_from(array_item(&mutated, 1)).unwrap(), 41);
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
fn reflected_getter_releases_partial_result_on_error() {
    let tracked = FfiString::from("a reference-counted reflected field value");
    let source = rust_failing_getter(tracked.clone());
    let count_before = AnyView::from(&tracked).debug_strong_count();

    let error = match structural_map(
        source.clone(),
        |_integer: i64| Any::from(0i64),
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("failing getter unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_eq!(
        error.message(),
        "callback failed after writing an owning result"
    );
    assert_eq!(AnyView::from(&tracked).debug_strong_count(), count_before);
}

#[test]
fn function_setter_releases_partial_result_on_error() {
    let replacement = FfiString::from("a reference-counted setter result");
    let source = rust_failing_setter(1i64);
    let count_before = AnyView::from(&replacement).debug_strong_count();

    let error = match structural_map(
        source,
        |_value: i64| Any::from(replacement.clone()),
        WalkOrder::PostOrder,
    ) {
        Ok(_) => panic!("failing Function setter unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_eq!(
        error.message(),
        "callback failed after writing an owning result"
    );
    assert_eq!(
        AnyView::from(&replacement).debug_strong_count(),
        count_before
    );
}

#[test]
fn callback_errors_preserve_message_and_add_object_context() {
    ensure_test_types_registered();
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
        |_integer: i64, _mutator: &mut Mutator| -> Result<i64> {
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
fn registered_mutation_hooks_receive_the_rust_mutator() {
    ensure_test_types_registered();
    let _guard = REGISTERED_HOOK_TEST_LOCK.lock().unwrap();
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });

    let source = rust_hook_node();
    let mutate_calls_before = REGISTERED_MUTATE_CALLS.load(Ordering::Relaxed);
    let mutated = structural_mutate(source.clone(), &mut ManualIncrement::default())
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(mutated, 2);
    assert_eq!(
        REGISTERED_MUTATE_CALLS.load(Ordering::Relaxed),
        mutate_calls_before + 1
    );

    let retained = RETAINED_MUTATOR.with(|retained| retained.take().unwrap());
    let error = match Function::get_global("ffi.StructuralMutatorMutate")
        .unwrap()
        .call_packed(&[AnyView::from(&retained), AnyView::from(&1i64)])
    {
        Ok(_) => panic!("retained structural mutator unexpectedly remained active"),
        Err(error) => error,
    };
    assert!(error.message().contains("retained after its active call"));

    let mutate_calls_before = REGISTERED_MUTATE_CALLS.load(Ordering::Relaxed);
    let mutated = structural_mutate(source.clone(), |value: i64, _mutator: &mut Mutator| {
        Any::from(value + 1)
    })
    .and_then(i64::try_from)
    .unwrap();
    assert_eq!(mutated, 2);
    assert_eq!(
        REGISTERED_MUTATE_CALLS.load(Ordering::Relaxed),
        mutate_calls_before + 1
    );
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });

    let maybe_inplace_calls_before = REGISTERED_MAYBE_INPLACE_MUTATE_CALLS.load(Ordering::Relaxed);
    let mutated = structural_mutate(rust_hook_node(), &mut ManualIncrement::default())
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(mutated, 2);
    assert_eq!(
        REGISTERED_MAYBE_INPLACE_MUTATE_CALLS.load(Ordering::Relaxed),
        maybe_inplace_calls_before + 1
    );
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });
}

#[test]
fn registered_hook_cannot_reenter_through_an_aliased_mutator_handle() {
    ensure_test_types_registered();
    let _guard = REGISTERED_HOOK_TEST_LOCK.lock().unwrap();
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });
    let mut mutator = RejectAliasedReentry::default();

    let mutated = structural_mutate(rust_hook_node(), &mut mutator)
        .and_then(i64::try_from)
        .unwrap();

    assert_eq!(mutated, 2);
    assert!(mutator.rejected);
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });
}

#[test]
fn registered_hook_rejects_foreign_thread_mutator_callback() {
    ensure_test_types_registered();
    let _guard = REGISTERED_HOOK_TEST_LOCK.lock().unwrap();
    FOREIGN_THREAD_MUTATOR_ERROR.with(|error| {
        error.take();
    });

    PROBE_FOREIGN_THREAD_MUTATOR.with(|enabled| enabled.set(true));
    let result = structural_mutate(rust_hook_node(), &mut ManualIncrement::default());
    PROBE_FOREIGN_THREAD_MUTATOR.with(|enabled| enabled.set(false));
    result.unwrap();

    let message = FOREIGN_THREAD_MUTATOR_ERROR.with(|error| error.take().unwrap());
    assert!(message.contains("invoked from a different thread"));
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });
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
    ensure_test_types_registered();
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
fn shared_map_callback_error_preserves_source_and_reports_object_context() {
    ensure_test_types_registered();
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
fn shared_list_uses_copy_on_write() {
    ensure_test_types_registered();
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

#[derive(Default)]
struct GeneratedLeafState {
    integers: Vec<(i64, DefRegionKind)>,
}

struct GeneratedLeafDispatch;

#[dispatch(mutate)]
impl GeneratedLeafDispatch {
    fn mutate_integer(&self, value: i64, mutator: &mut Mutator<GeneratedLeafState>) -> Any {
        let region = mutator.region();
        mutator.state_mut().integers.push((value, region));
        Any::from(value + 1)
    }
}

struct GeneratedStatelessDispatch;

#[dispatch(mutate)]
impl GeneratedStatelessDispatch {
    fn mutate_integer(&self, value: i64, _mutator: &mut Mutator) -> i64 {
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
    let mut mutator = MutateCallbacks::new(GeneratedLeafState::default(), GeneratedLeafDispatch);
    let mutated = structural_mutate(root, &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();

    assert_eq!(array_pointer(&mutated), root_pointer);
    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(
        mutator.state().integers,
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None)]
    );
}

#[test]
fn generated_mutate_dispatch_default_remap_crosses_registered_hooks() {
    ensure_test_types_registered();
    let _guard = REGISTERED_HOOK_TEST_LOCK.lock().unwrap();
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });

    let mut mutator = MutateCallbacks::new(GeneratedLeafState::default(), GeneratedLeafDispatch);
    let mutated = structural_mutate(rust_hook_node(), &mut mutator)
        .and_then(i64::try_from)
        .unwrap();
    assert_eq!(mutated, 2);
    assert_eq!(mutator.state().integers, vec![(1, DefRegionKind::None)]);
    RETAINED_MUTATOR.with(|retained| {
        retained.take();
    });
}

#[derive(Default)]
struct GeneratedRecursiveState {
    arrays: Vec<DefRegionKind>,
    integers: Vec<(i64, DefRegionKind)>,
}

struct GeneratedRecursiveDispatch;

#[dispatch(mutate)]
impl GeneratedRecursiveDispatch {
    fn mutate_array(
        &self,
        array: Array<i64>,
        mutator: &mut Mutator<GeneratedRecursiveState>,
    ) -> Result<Array<i64>> {
        let region = mutator.region();
        mutator.state_mut().arrays.push(region);
        let mut mutated = Vec::with_capacity(array.len());
        for value in array.iter() {
            mutated.push(i64::try_from(mutator.mutate(&value)?)?);
        }
        Ok(Array::new(mutated))
    }

    fn mutate_integer(&self, value: i64, mutator: &mut Mutator<GeneratedRecursiveState>) -> Any {
        let region = mutator.region();
        mutator.state_mut().integers.push((value, region));
        Any::from(value + 10)
    }
}

#[test]
fn generated_mutate_dispatch_recurses_through_context() {
    let mut mutator = MutateCallbacks::new(
        GeneratedRecursiveState::default(),
        GeneratedRecursiveDispatch,
    );
    let mutated = structural_mutate(Array::new(vec![1i64, 2]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();

    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![11, 12]);
    assert_eq!(mutator.state().arrays, vec![DefRegionKind::None]);
    assert_eq!(
        mutator.state().integers,
        vec![(1, DefRegionKind::None), (2, DefRegionKind::None)]
    );
}

#[test]
fn generated_mutate_dispatch_inherits_region_during_explicit_recursion() {
    ensure_test_types_registered();
    let _guard = REFLECTED_TEST_LOCK.lock().unwrap();
    let root = rust_pair(Array::new(vec![1i64]), Any::new());
    let mut mutator = MutateCallbacks::new(
        GeneratedRecursiveState::default(),
        GeneratedRecursiveDispatch,
    );

    let mutated = structural_mutate(root, &mut mutator)
        .and_then(RustPair::try_from)
        .unwrap();
    let first = Array::<i64>::try_from(mutated.data.first.clone()).unwrap();

    assert_eq!(first.iter().collect::<Vec<_>>(), vec![11]);
    assert_eq!(mutator.state().arrays, vec![DefRegionKind::Recursive]);
    assert_eq!(
        mutator.state().integers,
        vec![(1, DefRegionKind::Recursive)]
    );
}

#[derive(Default)]
struct GeneratedDefaultingState {
    arrays: usize,
    integers: Vec<i64>,
}

struct GeneratedDefaultingDispatch;

#[dispatch(mutate)]
impl GeneratedDefaultingDispatch {
    fn mutate_array(
        &self,
        _array: Array<i64>,
        mutator: &mut Mutator<GeneratedDefaultingState>,
    ) -> Result<Any> {
        mutator.state_mut().arrays += 1;
        mutator.default_mutate()
    }

    fn mutate_integer(&self, value: i64, mutator: &mut Mutator<GeneratedDefaultingState>) -> Any {
        mutator.state_mut().integers.push(value);
        Any::from(value + 1)
    }
}

#[test]
fn generated_mutate_dispatch_can_default_recurse_from_a_typed_handler() {
    let mut mutator = MutateCallbacks::new(
        GeneratedDefaultingState::default(),
        GeneratedDefaultingDispatch,
    );
    let mutated = structural_mutate(Array::new(vec![1i64, 2]), &mut mutator)
        .and_then(Array::<i64>::try_from)
        .unwrap();

    assert_eq!(mutated.iter().collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(mutator.state().arrays, 1);
    assert_eq!(mutator.state().integers, vec![1, 2]);
}

struct GeneratedRemappingState {
    type_index: i32,
    calls: usize,
}

struct GeneratedRemappingDispatch;

#[dispatch(mutate)]
impl GeneratedRemappingDispatch {
    fn mutate_dag_node(
        &self,
        _value: &RustDagNodeObj,
        _mutator: &mut Mutator<GeneratedRemappingState>,
    ) -> Any {
        Any::from(42i64)
    }

    fn mutate_any(
        &self,
        value: &MapValue,
        mutator: &mut Mutator<GeneratedRemappingState>,
    ) -> Result<Any> {
        if value.type_index() != mutator.state().type_index {
            return mutator.default_mutate();
        }
        if let Some(mutated) = mutator.var_remap_get(value)? {
            return Ok(mutated);
        }
        mutator.state_mut().calls += 1;
        let mutated = Any::from(41i64);
        mutator.var_remap_set(value, &mutated)?;
        Ok(mutated)
    }
}

#[test]
fn generated_mutate_dispatch_uses_fresh_invocation_local_var_remap() {
    ensure_test_types_registered();
    let var = rust_free_var();
    let mut mutator = MutateCallbacks::new(
        GeneratedRemappingState {
            type_index: RustFreeVarObj::type_index(),
            calls: 0,
        },
        GeneratedRemappingDispatch,
    );

    for expected_calls in [1, 2] {
        let root = call_global(
            "ffi.Array",
            &[Any::from(var.clone()), Any::from(var.clone())],
        );
        let mutated = structural_mutate(root, &mut mutator).unwrap();
        assert_eq!(mutator.state().calls, expected_calls);
        assert_eq!(i64::try_from(array_item(&mutated, 0)).unwrap(), 41);
        assert_eq!(i64::try_from(array_item(&mutated, 1)).unwrap(), 41);
    }
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
fn callbacks_return_values_convertible_into_any() {
    ensure_test_types_registered();
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
        |integer: i64, _mutator: &mut Mutator| integer * 2,
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
            |_node: &RustDagNodeObj| Any::new(),
            |_node: &RustFreeVarObj| Any::new(),
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
    ensure_test_types_registered();
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
                |_node: &RustDagNodeObj| Any::new(),
                |_node: &RustFreeVarObj| Any::new(),
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

#[test]
fn callback_mutate_defaults_unmatched_values_and_preserves_root_permit() {
    ensure_test_types_registered();
    let root = Array::new(vec![1i64, 2]);
    let root_pointer = array_pointer(&root);
    let mutated = structural_mutate(root, |value: i64, _mutator: &mut Mutator| {
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

fn stateful_mutate_integer(value: i64, mutator: &mut Mutator<CallbackMutateStats>) -> Any {
    mutator.state_mut().integers.push(value);
    Any::from(value + 1)
}

fn stateful_mutate_default(
    _value: &MapValue,
    mutator: &mut Mutator<CallbackMutateStats>,
) -> Result<Any> {
    mutator.state_mut().defaults += 1;
    mutator.default_mutate()
}

#[test]
fn callback_mutator_carries_reusable_mutable_state() {
    ensure_test_types_registered();
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
    mutator: &mut Mutator<CallbackMutateDepth>,
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
    ensure_test_types_registered();
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
    ensure_test_types_registered();
    let root = Array::new(vec![1i64, 2]);
    let root_pointer = array_pointer(&root);
    let defaults = Cell::new(0);
    let mutated = structural_mutate(
        root,
        (
            |value: i64, _mutator: &mut Mutator| Any::from(value + 1),
            |_value: &MapValue, mutator: &mut Mutator| -> Result<Any> {
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
    ensure_test_types_registered();
    let integer_calls = Cell::new(0);
    let mutated = structural_mutate(
        Array::new(vec![1i64]),
        (
            |_array: Array<i64>, _mutator: &mut Mutator| Any::from(Array::new(vec![10i64])),
            |value: i64, _mutator: &mut Mutator| {
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
        |_value: &MapValue, mutator: &mut Mutator| {
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
    ensure_test_types_registered();
    let _guard = REFLECTED_TEST_LOCK.lock().unwrap();
    let root = call_global(
        "ffi.Array",
        &[Any::from(rust_dag_node()), Any::from(rust_pair(1i64, 9i64))],
    );
    let regions = RefCell::new(Vec::new());
    let mutated = structural_mutate(
        root,
        (
            (
                |_value: f64, _mutator: &mut Mutator| Any::new(),
                |_node: &RustDagNodeObj, _mutator: &mut Mutator| Any::from(7i64),
            ),
            |value: i64, mutator: &mut Mutator| {
                regions.borrow_mut().push(mutator.def_region_kind());
                Any::from(value + 1)
            },
        ),
    )
    .unwrap();

    assert_eq!(i64::try_from(array_item(&mutated, 0)).unwrap(), 7);
    let pair = RustPair::try_from(array_item(&mutated, 1)).unwrap();
    assert_eq!(i64::try_from(pair.data.first.clone()).unwrap(), 2);
    assert_eq!(i64::try_from(pair.data.ignored.clone()).unwrap(), 9);
    assert_eq!(*regions.borrow(), vec![DefRegionKind::Recursive]);
}

#[test]
fn callback_mutate_distinguishes_borrowed_and_owned_children() {
    ensure_test_types_registered();
    let borrowed_child = Array::new(vec![1i64]);
    let borrowed_pointer = array_pointer(&borrowed_child);
    let mutated = structural_mutate(
        true,
        (
            |_value: bool, mutator: &mut Mutator| mutator.mutate(&borrowed_child),
            |value: i64, _mutator: &mut Mutator| Any::from(value + 1),
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
            |_value: bool, mutator: &mut Mutator| {
                let child = Array::new(vec![1i64]);
                owned_pointer.set(array_pointer(&child) as usize);
                mutator.maybe_inplace_mutate(child)
            },
            |value: i64, _mutator: &mut Mutator| Any::from(value + 1),
        ),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(array_pointer(&mutated) as usize, owned_pointer.get());
    assert_eq!(mutated.get(0).unwrap(), 2);
}

#[test]
fn callback_mutate_can_use_its_invocation_local_var_remap() {
    ensure_test_types_registered();
    let var = rust_free_var();
    let calls = Cell::new(0);
    let type_index = RustFreeVarObj::type_index();
    let mut mutator = MutateCallbacks::new(
        (),
        |value: &MapValue, mutator: &mut Mutator| -> Result<Any> {
            if value.type_index() != type_index {
                return mutator.default_mutate();
            }
            if let Some(mutated) = mutator.var_remap_get(value)? {
                return Ok(mutated);
            }
            calls.set(calls.get() + 1);
            let mutated = Any::from(41i64);
            mutator.var_remap_set(value, &mutated)?;
            Ok(mutated)
        },
    );

    for expected_calls in 1..=2 {
        let root = call_global(
            "ffi.Array",
            &[Any::from(var.clone()), Any::from(var.clone())],
        );
        let mutated = structural_mutate(root, &mut mutator).unwrap();
        assert_eq!(calls.get(), expected_calls);
        assert_eq!(i64::try_from(array_item(&mutated, 0)).unwrap(), 41);
        assert_eq!(i64::try_from(array_item(&mutated, 1)).unwrap(), 41);
    }
}

#[test]
fn nested_callback_mutate_restores_the_outer_active_mutator() {
    let mutated = structural_mutate(1i64, |value: i64, mutator: &mut Mutator| -> Result<Any> {
        if value != 1 {
            return Ok(Any::from(value + 1));
        }
        let inner = structural_mutate(2i64, |value: i64, _mutator: &mut Mutator| {
            Any::from(value + 10)
        })?;
        assert_eq!(i64::try_from(inner).unwrap(), 12);
        mutator.mutate(&3i64)
    })
    .and_then(i64::try_from)
    .unwrap();
    assert_eq!(mutated, 4);
}

#[test]
fn callback_mutate_panics_resume_and_leave_the_next_run_usable() {
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        structural_mutate(
            Array::new(vec![1i64]),
            |_value: i64, _mutator: &mut Mutator| -> Any { panic!("callback mutator panic") },
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
        |value: i64, _mutator: &mut Mutator| Any::from(value + 1),
    )
    .and_then(Array::<i64>::try_from)
    .unwrap();
    assert_eq!(mutated.get(0).unwrap(), 2);
}
