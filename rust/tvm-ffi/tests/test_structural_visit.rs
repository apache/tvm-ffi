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
use std::sync::LazyLock;

use tvm_ffi::derive::{Object as DeriveObject, ObjectRef as DeriveObjectRef};
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyViewToOwnedAny, TVMFFIByteArray, TVMFFIFieldFlagBitMask, TVMFFIFieldInfo,
    TVMFFISEqHashKind, TVMFFITypeMetadata, TVMFFITypeRegisterAttr,
};
use tvm_ffi::{
    dispatch, structural_visit, structural_walk, Any, AnyView, Array, DLDataType, DLDataTypeCode,
    DefRegionKind, Error, Function, Map, Object, ObjectArc, ObjectCore, ObjectRefCast, Result,
    String as FfiString, StructuralVisitor, TypeIndex, VisitInterrupt, VisitValue, WalkOrder,
    WalkResult, RUNTIME_ERROR,
};

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
#[type_key = "testing.RustStructuralVisitDefRegion"]
#[type_final]
struct RustVisitDefRegionObj {
    base: Object,
    recursive: Any,
    plain: Any,
    non_recursive: Any,
    both: Any,
    ignored: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustVisitDefRegion {
    data: ObjectArc<RustVisitDefRegionObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralVisitHook"]
#[type_final]
struct RustVisitHookObj {
    base: Object,
    selected: Any,
    ignored: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustVisitHook {
    data: ObjectArc<RustVisitHookObj>,
}

#[repr(C)]
#[derive(DeriveObject)]
#[type_key = "testing.RustStructuralVisitFailingGetter"]
#[type_final]
struct RustVisitFailingGetterObj {
    base: Object,
    value: Any,
}

#[repr(C)]
#[derive(DeriveObjectRef, Clone)]
struct RustVisitFailingGetter {
    data: ObjectArc<RustVisitFailingGetterObj>,
}

thread_local! {
    static RETAINED_VISITOR: RefCell<Option<Any>> = const { RefCell::new(None) };
    static REGISTERED_HOOK_REGIONS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
    static PROBE_FOREIGN_THREAD_VISITOR: Cell<bool> = const { Cell::new(false) };
    static FOREIGN_THREAD_VISITOR_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

unsafe extern "C" fn clone_any_field(field: *mut std::ffi::c_void, result: *mut TVMFFIAny) -> i32 {
    TVMFFIAnyViewToOwnedAny(field.cast(), result)
}

unsafe extern "C" fn clone_any_field_then_fail(
    field: *mut std::ffi::c_void,
    result: *mut TVMFFIAny,
) -> i32 {
    let code = TVMFFIAnyViewToOwnedAny(field.cast(), result);
    if code != 0 {
        return code;
    }
    Error::set_raised(&runtime_error(
        "visit getter failed after writing an owning result",
    ));
    -1
}

fn register_any_field(type_index: i32, name: &'static str, offset: usize, flags: i64) {
    let field = TVMFFIFieldInfo {
        name: unsafe { TVMFFIByteArray::from_str(name) },
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-visit test field") },
        metadata: unsafe { TVMFFIByteArray::from_str("") },
        flags,
        size: std::mem::size_of::<Any>() as i64,
        alignment: std::mem::align_of::<Any>() as i64,
        offset: offset as i64,
        getter: Some(clone_any_field),
        setter: std::ptr::null_mut(),
        default_value_or_factory: TVMFFIAny::new(),
        field_static_type_index: -1,
    };
    assert_eq!(unsafe { TVMFFITypeRegisterField(type_index, &field) }, 0);
}

fn register_visit_type(type_key: &'static str, total_size: usize, kind: TVMFFISEqHashKind) -> i32 {
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
        doc: unsafe { TVMFFIByteArray::from_str("Rust structural-visit test object") },
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

fn call_visitor_from_foreign_thread(visitor: AnyView<'_>) -> String {
    // Keep the object alive on this thread. The worker uses a non-owning view,
    // so it neither transfers nor releases the visitor's reference count.
    let mut owner = Any::from(visitor);
    let raw = unsafe { *Any::as_data_ptr(&mut owner) };
    let type_index = raw.type_index;
    let object = unsafe { raw.data_union.v_obj } as usize;
    std::thread::spawn(move || {
        let mut raw = TVMFFIAny::new();
        raw.type_index = type_index;
        raw.data_union.v_obj = object as *mut _;
        let borrowed = std::mem::ManuallyDrop::new(unsafe { Any::from_raw_ffi_any(raw) });
        match Function::get_global("ffi.StructuralVisitorVisit")
            .unwrap()
            .call_packed(&[AnyView::from(&*borrowed), AnyView::from(&1i64)])
        {
            Err(error) => error.message().to_string(),
            Ok(_) => "foreign-thread visitor call unexpectedly succeeded".to_string(),
        }
    })
    .join()
    .unwrap()
}

fn registered_visit_hook(args: &[AnyView<'_>]) -> Result<Any> {
    assert_eq!(args.len(), 2);
    if PROBE_FOREIGN_THREAD_VISITOR.with(Cell::get) {
        let message = call_visitor_from_foreign_thread(args[0]);
        FOREIGN_THREAD_VISITOR_ERROR.with(|error| error.replace(Some(message)));
    }
    RETAINED_VISITOR.with(|retained| {
        retained.replace(Some(Any::from(args[0])));
    });
    let def_region_kind = Function::get_global("ffi.StructuralVisitorDefRegionKind")?
        .call_packed(&[args[0]])
        .and_then(i64::try_from)?;
    REGISTERED_HOOK_REGIONS.with(|regions| regions.borrow_mut().push(def_region_kind));

    let node = RustVisitHook::try_from(args[1])?;
    Function::get_global("ffi.StructuralVisitorVisit")?
        .call_packed(&[args[0], AnyView::from(&node.data.selected)])
}

static REGISTER_HOOK_TYPE: LazyLock<()> = LazyLock::new(|| {
    let type_index = register_visit_type(
        RustVisitHookObj::TYPE_KEY,
        std::mem::size_of::<RustVisitHookObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
    );
    register_any_field(
        type_index,
        "selected",
        std::mem::offset_of!(RustVisitHookObj, selected),
        0,
    );
    register_any_field(
        type_index,
        "ignored",
        std::mem::offset_of!(RustVisitHookObj, ignored),
        0,
    );
    register_function_attr(
        type_index,
        "__s_visit__",
        Function::from_packed(registered_visit_hook),
    );
});

static REGISTER_FAILING_GETTER_TYPE: LazyLock<()> = LazyLock::new(|| {
    let type_index = register_visit_type(
        RustVisitFailingGetterObj::TYPE_KEY,
        std::mem::size_of::<RustVisitFailingGetterObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
    );
    let field = TVMFFIFieldInfo {
        name: unsafe { TVMFFIByteArray::from_str("value") },
        doc: unsafe { TVMFFIByteArray::from_str("Fail after producing an owning field value") },
        metadata: unsafe { TVMFFIByteArray::from_str("") },
        flags: 0,
        size: std::mem::size_of::<Any>() as i64,
        alignment: std::mem::align_of::<Any>() as i64,
        offset: std::mem::offset_of!(RustVisitFailingGetterObj, value) as i64,
        getter: Some(clone_any_field_then_fail),
        setter: std::ptr::null_mut(),
        default_value_or_factory: TVMFFIAny::new(),
        field_static_type_index: -1,
    };
    assert_eq!(unsafe { TVMFFITypeRegisterField(type_index, &field) }, 0);
});

fn registered_primitive_visit_hook(args: &[AnyView<'_>]) -> Result<Any> {
    assert_eq!(args.len(), 2);
    Function::get_global("ffi.StructuralVisitorVisit")?
        .call_packed(&[args[0], AnyView::from(&7i64)])
}

static REGISTER_PRIMITIVE_HOOK: LazyLock<()> = LazyLock::new(|| {
    register_function_attr(
        TypeIndex::kTVMFFIDataType as i32,
        "__s_visit__",
        Function::from_packed(registered_primitive_visit_hook),
    );
});

static REGISTER_REGION_TYPES: LazyLock<()> = LazyLock::new(|| {
    let type_index = register_visit_type(
        RustVisitDefRegionObj::TYPE_KEY,
        std::mem::size_of::<RustVisitDefRegionObj>(),
        TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar,
    );
    for (name, offset, flags) in [
        (
            "recursive",
            std::mem::offset_of!(RustVisitDefRegionObj, recursive),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64,
        ),
        (
            "plain",
            std::mem::offset_of!(RustVisitDefRegionObj, plain),
            0,
        ),
        (
            "non_recursive",
            std::mem::offset_of!(RustVisitDefRegionObj, non_recursive),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive as i64,
        ),
        (
            "both",
            std::mem::offset_of!(RustVisitDefRegionObj, both),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64
                | TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive as i64,
        ),
        (
            "ignored",
            std::mem::offset_of!(RustVisitDefRegionObj, ignored),
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64,
        ),
    ] {
        register_any_field(type_index, name, offset, flags);
    }
});

fn ensure_region_types_registered() {
    LazyLock::force(&REGISTER_REGION_TYPES);
}

fn rust_visit_hook(selected: impl Into<Any>, ignored: impl Into<Any>) -> RustVisitHook {
    LazyLock::force(&REGISTER_HOOK_TYPE);
    RustVisitHook {
        data: ObjectArc::new(RustVisitHookObj {
            base: Object::new(),
            selected: selected.into(),
            ignored: ignored.into(),
        }),
    }
}

fn rust_visit_failing_getter(value: impl Into<Any>) -> RustVisitFailingGetter {
    LazyLock::force(&REGISTER_FAILING_GETTER_TYPE);
    RustVisitFailingGetter {
        data: ObjectArc::new(RustVisitFailingGetterObj {
            base: Object::new(),
            value: value.into(),
        }),
    }
}

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

#[test]
fn plain_walk_uses_registered_array_hook() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                integers += 1;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, 3);
}

#[test]
fn reflected_getter_releases_partial_result_on_error() {
    let tracked = FfiString::from("a reference-counted reflected visit field");
    let root = rust_visit_failing_getter(tracked.clone());
    let count_before = AnyView::from(&tracked).debug_strong_count();

    let error = match structural_walk(
        &root,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    ) {
        Ok(_) => panic!("failing getter unexpectedly succeeded"),
        Err(error) => error,
    };

    assert_eq!(
        error.message(),
        "visit getter failed after writing an owning result"
    );
    assert_eq!(AnyView::from(&tracked).debug_strong_count(), count_before);
}

#[test]
fn plain_walk_visits_map_values_without_visiting_keys() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
        .into_iter()
        .collect();
    let mut integers = 0;
    let mut strings = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                integers += 1;
            } else if value.cast::<FfiString>().is_some() {
                strings += 1;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, 2);
    assert_eq!(strings, 0);
}

#[test]
fn registered_function_hook_controls_children_interrupts_and_lifetime() {
    RETAINED_VISITOR.with(|retained| {
        retained.take();
    });
    REGISTERED_HOOK_REGIONS.with(|regions| regions.borrow_mut().clear());
    let root = rust_visit_hook(11i64, 99i64);

    let mut integers = Vec::new();
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if let Some(integer) = value.cast::<i64>() {
                integers.push(integer);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    // Reflection would visit both fields. The registered hook deliberately
    // visits only `selected`.
    assert_eq!(integers, vec![11]);
    REGISTERED_HOOK_REGIONS.with(|regions| {
        assert_eq!(regions.borrow().as_slice(), &[DefRegionKind::None as i64]);
    });

    #[derive(Default)]
    struct RecordingVisitor {
        integers: Vec<i64>,
    }
    impl StructuralVisitor for RecordingVisitor {
        fn visit(
            &mut self,
            value: &VisitValue,
            def_region_kind: DefRegionKind,
        ) -> Result<Option<VisitInterrupt>> {
            if let Some(integer) = value.cast::<i64>() {
                self.integers.push(integer);
            }
            self.default_visit_children(value, def_region_kind)
        }
    }
    let mut visitor = RecordingVisitor::default();
    assert!(structural_visit(&root, &mut visitor).unwrap().is_none());
    assert_eq!(visitor.integers, vec![11]);

    ensure_region_types_registered();
    let wrapped = RustVisitDefRegion {
        data: ObjectArc::new(RustVisitDefRegionObj {
            base: Object::new(),
            recursive: Any::from(root.clone()),
            plain: Any::new(),
            non_recursive: Any::new(),
            both: Any::new(),
            ignored: Any::new(),
        }),
    };
    assert!(structural_walk(
        &wrapped,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    REGISTERED_HOOK_REGIONS.with(|regions| {
        assert_eq!(
            regions.borrow().last().copied(),
            Some(DefRegionKind::Recursive as i64)
        );
    });

    let outcome = structural_walk(
        &root,
        |value: &VisitValue| match value.cast::<i64>() {
            Some(11) => WalkResult::interrupt_with(FfiString::from("stop")),
            _ => WalkResult::Advance,
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .unwrap();
    assert_eq!(FfiString::try_from(outcome.value).unwrap().as_str(), "stop");

    let retained = RETAINED_VISITOR.with(|retained| retained.take().unwrap());
    let error = match Function::get_global("ffi.StructuralVisitorVisit")
        .unwrap()
        .call_packed(&[AnyView::from(&retained), AnyView::from(&1i64)])
    {
        Err(error) => error,
        Ok(_) => panic!("retained structural visitor unexpectedly remained active"),
    };
    assert!(error.message().contains("retained after its active call"));
}

#[test]
fn registered_hook_rejects_foreign_thread_visitor_callback() {
    LazyLock::force(&REGISTER_HOOK_TYPE);
    RETAINED_VISITOR.with(|retained| {
        retained.take();
    });
    FOREIGN_THREAD_VISITOR_ERROR.with(|error| {
        error.take();
    });

    let root = rust_visit_hook(11i64, 99i64);
    PROBE_FOREIGN_THREAD_VISITOR.with(|enabled| enabled.set(true));
    let result = structural_walk(
        &root,
        |_value: &VisitValue| WalkResult::Advance,
        WalkOrder::PreOrder,
    );
    PROBE_FOREIGN_THREAD_VISITOR.with(|enabled| enabled.set(false));
    assert!(result.unwrap().is_none());

    let message = FOREIGN_THREAD_VISITOR_ERROR.with(|error| error.take().unwrap());
    assert!(message.contains("invoked from a different thread"));
    RETAINED_VISITOR.with(|retained| {
        retained.take();
    });
}

#[test]
fn primitive_hook_fast_path_preserves_pre_and_post_order() {
    LazyLock::force(&REGISTER_PRIMITIVE_HOOK);
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);

    let mut pre = Vec::new();
    assert!(structural_walk(
        &dtype,
        |value: &VisitValue| {
            if value.cast::<DLDataType>().is_some() {
                pre.push("dtype");
            } else if value.cast::<i64>().is_some() {
                pre.push("child");
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(pre, ["dtype", "child"]);

    let mut skipped = Vec::new();
    assert!(structural_walk(
        &dtype,
        |value: &VisitValue| {
            if value.cast::<DLDataType>().is_some() {
                skipped.push("dtype");
                WalkResult::Skip
            } else {
                skipped.push("child");
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(skipped, ["dtype"]);

    let mut post = Vec::new();
    assert!(structural_walk(
        &dtype,
        |value: &VisitValue| {
            if value.cast::<DLDataType>().is_some() {
                post.push("dtype");
            } else if value.cast::<i64>().is_some() {
                post.push("child");
            }
            WalkResult::Advance
        },
        WalkOrder::PostOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(post, ["child", "dtype"]);
}

#[test]
fn primitive_fast_path_preserves_none_interrupt_and_error() {
    let mut none_calls = 0;
    assert!(structural_walk(
        &Any::new(),
        |_value: &VisitValue| {
            none_calls += 1;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(none_calls, 0);

    let interrupt = structural_walk(
        &1i64,
        |_value: i64| WalkResult::interrupt_with(9i64),
        WalkOrder::PostOrder,
    )
    .unwrap()
    .unwrap();
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 9);

    let error = match structural_walk(
        &1i64,
        |_value: i64| -> Result<WalkResult> { Err(runtime_error("primitive failed")) },
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("failing primitive callback unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "primitive failed");
}

#[test]
fn registered_map_hook_visits_all_values_without_visiting_keys() {
    // More than 4 entries forces the dense (block + iteration list) layout.
    let root: Map<FfiString, i64> = (0..9)
        .map(|i| (FfiString::from(format!("k{i}")), i as i64))
        .collect();
    let mut sum = 0;
    let mut strings = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            if let Some(integer) = value.cast::<i64>() {
                sum += integer;
            } else if value.cast::<FfiString>().is_some() {
                strings += 1;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(sum, (0..9).sum::<i64>());
    assert_eq!(strings, 0);
}

#[test]
fn interrupt_payload_crosses_map_traversal() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64), (FfiString::from("b"), 2i64)]
        .into_iter()
        .collect();
    let outcome = structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                return WalkResult::interrupt_with(99i64);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("map walk unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 99);
}

#[test]
fn handler_error_crosses_map_traversal() {
    let root: Map<FfiString, i64> = [(FfiString::from("a"), 1i64)].into_iter().collect();
    let error = match structural_walk(
        &root,
        |value: &VisitValue| -> Result<WalkResult> {
            if value.cast::<i64>().is_some() {
                Err(runtime_error("map handler failed"))
            } else {
                Ok(WalkResult::Advance)
            }
        },
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("map handler unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "map handler failed");
    assert!(error.backtrace().contains("object `ffi.Map`"));
}

#[test]
fn interrupt_stops_without_running_remaining_callbacks() {
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = 0;
    let outcome = structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>().is_some() {
                integers += 1;
                return WalkResult::Interrupt;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert!(outcome.is_some());
    assert_eq!(integers, 1);
}

/// Visitor-layer traversal that overrides the def-region for one child and
/// inherits it for the next, mirroring a C++ visitor using
/// `WithDefRegionKind`.
#[derive(Default)]
struct ManualRegionVisitor {
    seen: Vec<DefRegionKind>,
}

impl StructuralVisitor for ManualRegionVisitor {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(array) = value.cast::<Array<i64>>() {
            // Override the state for exactly this child's subtree...
            let overridden = array.get(0).unwrap();
            if let Some(interrupt) = self.visit_child(&overridden, DefRegionKind::NonRecursive)? {
                return Ok(Some(interrupt));
            }
            // ...and forward the received state to inherit it.
            let inherited = array.get(1).unwrap();
            return self.visit_child(&inherited, def_region_kind);
        }
        if value.cast::<i64>().is_some() {
            self.seen.push(def_region_kind);
        }
        Ok(None)
    }
}

#[test]
fn manual_child_visit_can_override_def_region() {
    let root = Array::new(vec![7i64, 8]);
    let mut probe = ManualRegionVisitor::default();
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(
        probe.seen,
        vec![DefRegionKind::NonRecursive, DefRegionKind::None]
    );
}

#[derive(Default)]
struct GenericDispatchProbe {
    integers: Vec<i64>,
    objects: usize,
    catch_all: usize,
}

#[dispatch(visit)]
impl GenericDispatchProbe {
    fn visit_integer(&mut self, value: i64) -> WalkResult {
        self.integers.push(value);
        WalkResult::Advance
    }

    // Trailing DefRegionKind: handlers may mix arities within one impl.
    fn visit_object(&mut self, _value: &tvm_ffi::Object, kind: DefRegionKind) -> WalkResult {
        assert_eq!(kind, DefRegionKind::None);
        self.objects += 1;
        WalkResult::Advance
    }

    fn visit_any(&mut self, _value: &VisitValue) -> WalkResult {
        self.catch_all += 1;
        WalkResult::Advance
    }
}

#[test]
fn generated_dispatch_supports_pod_and_ordered_catch_all() {
    let root = Array::new(vec![1i64, 2]);
    let mut probe = GenericDispatchProbe::default();
    assert!(structural_walk(&root, &mut probe, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.integers, vec![1, 2]);
    assert_eq!(probe.objects, 1);

    let floats = Array::new(vec![1.0f64, 2.0]);
    assert!(structural_walk(&floats, &mut probe, WalkOrder::PreOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.objects, 2);
    assert_eq!(probe.catch_all, 2);
}

/// Visitor-layer enter/exit straddling: run enter logic, delegate the
/// default child recursion, then run exit logic with the same locals in
/// scope — the C++ `DefaultVisitExpected` pattern.
#[derive(Default)]
struct StraddleVisitor {
    events: Vec<String>,
}

impl StructuralVisitor for StraddleVisitor {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        let label = match value.cast::<i64>() {
            Some(integer) => format!("int:{integer}"),
            None => "node".to_string(),
        };
        self.events.push(format!("enter:{label}"));
        if let Some(interrupt) = self.default_visit_children(value, def_region_kind)? {
            return Ok(Some(interrupt));
        }
        self.events.push(format!("exit:{label}"));
        Ok(None)
    }
}

#[test]
fn visitor_can_straddle_default_children() {
    let root = Array::new(vec![1i64, 2]);
    let mut probe = StraddleVisitor::default();
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(
        probe.events,
        vec![
            "enter:node",
            "enter:int:1",
            "exit:int:1",
            "enter:int:2",
            "exit:int:2",
            "exit:node",
        ]
    );
}

#[derive(Default)]
struct OrderProbe {
    events: Vec<String>,
}

#[dispatch(visit)]
impl OrderProbe {
    fn visit_array(&mut self, _array: Array<i64>) -> WalkResult {
        self.events.push("array".to_string());
        WalkResult::Advance
    }

    fn visit_integer(&mut self, value: i64) -> WalkResult {
        self.events.push(format!("int:{value}"));
        WalkResult::Advance
    }
}

#[test]
fn stateful_structural_walk_supports_post_order() {
    let root = Array::new(vec![1i64, 2]);
    let mut probe = OrderProbe::default();
    assert!(structural_walk(&root, &mut probe, WalkOrder::PostOrder)
        .unwrap()
        .is_none());
    assert_eq!(probe.events, vec!["int:1", "int:2", "array"]);
}

#[test]
fn nested_walk_restores_the_outer_active_visitor() {
    let outer = Array::new(vec![10i64, 20]);
    let inner = Array::new(vec![1i64, 2]);
    let mut entered_inner = false;
    let mut outer_values = Vec::new();
    let mut inner_values = Vec::new();

    assert!(structural_walk(
        &outer,
        |value: &VisitValue| -> Result<WalkResult> {
            if let Some(value) = value.cast::<i64>() {
                outer_values.push(value);
            }
            if !entered_inner {
                entered_inner = true;
                structural_walk(
                    &inner,
                    |value: &VisitValue| {
                        if let Some(value) = value.cast::<i64>() {
                            inner_values.push(value);
                        }
                        WalkResult::Advance
                    },
                    WalkOrder::PreOrder,
                )?;
            }
            Ok(WalkResult::Advance)
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(outer_values, vec![10, 20]);
    assert_eq!(inner_values, vec![1, 2]);
}

#[test]
fn interrupt_payload_is_returned_to_the_caller() {
    let root = Array::new(vec![1i64, 2]);
    let outcome = structural_walk(
        &root,
        |value: &VisitValue| {
            if value.cast::<i64>() == Some(1) {
                return WalkResult::interrupt_with(42i64);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    let Some(interrupt) = outcome else {
        panic!("walk unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 42);
}

#[test]
fn handler_errors_include_native_visit_path() {
    let root = Array::new(vec![1i64]);
    let error = match structural_walk(
        &root,
        |value: &VisitValue| -> Result<WalkResult> {
            if value.cast::<i64>().is_some() {
                Err(runtime_error("handler failed"))
            } else {
                Ok(WalkResult::Advance)
            }
        },
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("handler unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "handler failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn visitor_errors_include_native_visit_path() {
    struct FailingVisitor;

    impl StructuralVisitor for FailingVisitor {
        fn visit(
            &mut self,
            value: &VisitValue,
            def_region_kind: DefRegionKind,
        ) -> Result<Option<VisitInterrupt>> {
            if value.cast::<i64>().is_some() {
                return Err(runtime_error("visitor failed"));
            }
            self.default_visit_children(value, def_region_kind)
        }
    }

    let root = Array::new(vec![1i64]);
    let error = match structural_visit(&root, &mut FailingVisitor) {
        Err(error) => error,
        Ok(_) => panic!("visitor unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "visitor failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn callback_panics_resume_after_the_registered_hook_returns() {
    let root = Array::new(vec![1i64]);
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        structural_walk(
            &root,
            |_value: i64| -> WalkResult { panic!("visitor panic") },
            WalkOrder::PreOrder,
        )
    })) {
        Err(panic) => panic,
        Ok(_) => panic!("panicking visitor unexpectedly returned"),
    };
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some("visitor panic"));
}

#[test]
fn visitor_interrupt_propagates_through_default_children() {
    struct InterruptingVisitor;

    impl StructuralVisitor for InterruptingVisitor {
        fn visit(
            &mut self,
            value: &VisitValue,
            def_region_kind: DefRegionKind,
        ) -> Result<Option<VisitInterrupt>> {
            if value.cast::<i64>() == Some(2) {
                return Ok(Some(VisitInterrupt::with(7i64)));
            }
            self.default_visit_children(value, def_region_kind)
        }
    }

    let root = Array::new(vec![1i64, 2, 3]);
    let outcome = structural_visit(&root, &mut InterruptingVisitor).unwrap();
    let Some(interrupt) = outcome else {
        panic!("visitor traversal unexpectedly completed");
    };
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 7);
}

#[test]
fn closure_walk_receives_def_region_kind() {
    // C++: StructuralWalk<kPreOrder>(root,
    //          [&](const TVarObj* var, TVMFFIDefRegionKind kind) { ... })
    let root = Array::new(vec![1i64, 2]);
    let mut kinds = Vec::new();
    assert!(structural_walk(
        &root,
        |value: &VisitValue, kind: DefRegionKind| {
            if value.cast::<i64>().is_some() {
                kinds.push(kind);
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(kinds, vec![DefRegionKind::None; 2]);
}

#[test]
fn closure_walk_supports_post_order_and_skip() {
    let root = Array::new(vec![1i64, 2]);
    let mut order_probe = Vec::new();
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            order_probe.push(value.cast::<i64>());
            WalkResult::Advance
        },
        WalkOrder::PostOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(order_probe, vec![Some(1), Some(2), None]);

    let mut visited = 0;
    assert!(structural_walk(
        &root,
        |value: &VisitValue| {
            visited += 1;
            if value.cast::<i64>().is_none() {
                WalkResult::Skip
            } else {
                WalkResult::Advance
            }
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(visited, 1);
}

// ---------------------------------------------------------------------------
// Tuple walkers: structural_walk(root, (link1, link2, ...), order) — links
// are tried in order and the first whose argument type matches the value
// runs, the Rust analog of the variadic C++ StructuralWalk callback chain.
// ---------------------------------------------------------------------------
#[test]
fn chain_accepts_owned_object_ref_links() {
    let root = Array::new(vec![Array::new(vec![1i64]), Array::new(vec![2i64, 3])]);
    let mut lengths = Vec::new();
    assert!(structural_walk(
        &root,
        (
            |array: Array<i64>| {
                lengths.push(array.len());
                WalkResult::Advance
            },
            |_value: i64| WalkResult::Advance,
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    // The outer Array<Array<i64>> fails the strict element check and falls
    // through the chain; only the inner arrays match the typed link.
    assert_eq!(lengths, vec![1, 2]);
}

#[test]
fn chain_links_may_mix_def_region_arity() {
    // Like #[dispatch(visit)] handlers, each link independently opts into
    // the trailing DefRegionKind argument.
    let root = Array::new(vec![1i64, 2]);
    let mut kinds = Vec::new();
    let mut objects = 0;
    assert!(structural_walk(
        &root,
        (
            |_value: i64, kind: DefRegionKind| {
                kinds.push(kind);
                WalkResult::Advance
            },
            |_value: &VisitValue, kind: DefRegionKind| {
                assert_eq!(kind, DefRegionKind::None);
                objects += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(kinds, vec![DefRegionKind::None; 2]);
    assert_eq!(objects, 1);
}

#[test]
fn chain_links_can_skip_children() {
    let root = Array::new(vec![Array::new(vec![1i64]), Array::new(vec![2i64])]);
    let mut arrays = 0;
    let mut integers = 0;
    assert!(structural_walk(
        &root,
        (
            |_array: Array<i64>| {
                arrays += 1;
                WalkResult::Skip
            },
            |_value: i64| {
                integers += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(arrays, 2);
    assert_eq!(integers, 0); // both inner arrays were skipped
}

#[test]
fn chain_link_errors_include_native_visit_path() {
    let root = Array::new(vec![1i64]);
    let error = match structural_walk(
        &root,
        (
            |_value: i64| -> Result<WalkResult> { Err(runtime_error("link failed")) },
            |_value: &VisitValue| WalkResult::Advance,
        ),
        WalkOrder::PreOrder,
    ) {
        Err(error) => error,
        Ok(_) => panic!("link unexpectedly succeeded"),
    };
    assert_eq!(error.message(), "link failed");
    assert!(error.backtrace().contains("object `ffi.Array`"));
}

#[test]
fn chain_supports_post_order() {
    // Rust borrow rules apply per link: state shared across links goes
    // through a RefCell (or a single #[dispatch(visit)] visitor).
    let root = Array::new(vec![1i64, 2]);
    let events = std::cell::RefCell::new(Vec::new());
    assert!(structural_walk(
        &root,
        (
            |value: i64| {
                events.borrow_mut().push(format!("int:{value}"));
                WalkResult::Advance
            },
            |_object: &Object| {
                events.borrow_mut().push("array".to_string());
                WalkResult::Advance
            },
        ),
        WalkOrder::PostOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(events.into_inner(), vec!["int:1", "int:2", "array"]);
}

#[derive(Default)]
struct ObjectCounter {
    objects: usize,
}

#[dispatch(visit)]
impl ObjectCounter {
    fn visit_object(&mut self, _value: &Object) -> WalkResult {
        self.objects += 1;
        WalkResult::Advance
    }
}

#[test]
fn chain_splices_dispatch_visitors_between_closures() {
    // A `&mut` typed visitor participates in the chain like any other link,
    // keeping its own no-match fall-through semantics.
    let root = Array::new(vec![1i64, 2]);
    let mut counter = ObjectCounter::default();
    let mut integers = 0;
    assert!(structural_walk(
        &root,
        (&mut counter, |_value: i64| {
            integers += 1;
            WalkResult::Advance
        },),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(counter.objects, 1);
    assert_eq!(integers, 2);
}

#[test]
fn chain_supports_full_arity() {
    // Doubles as the first-match ordering probe: earlier misses fall
    // through, the first matching link claims the value, later links
    // never run.
    let root = Array::new(vec![1i64, 2, 3]);
    let mut integers = Vec::new();
    let mut objects = 0;
    let mut others = 0;
    assert!(structural_walk(
        &root,
        (
            |_value: f64| WalkResult::Advance,
            |_value: bool| WalkResult::Advance,
            |_value: tvm_ffi::String| WalkResult::Advance,
            |_value: Array<f64>| WalkResult::Advance,
            |value: i64| {
                integers.push(value);
                WalkResult::Advance
            },
            |_object: &Object, _kind: DefRegionKind| {
                objects += 1;
                WalkResult::Advance
            },
            |_value: &VisitValue, _kind: DefRegionKind| {
                others += 1;
                WalkResult::Advance
            },
            |_value: &VisitValue| WalkResult::Advance,
        ),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(integers, vec![1, 2, 3]);
    assert_eq!(objects, 1); // the array itself; integers matched earlier
    assert_eq!(others, 0); // every value matched an earlier link
}

#[test]
fn typed_lambda_walks_bare_and_as_single_link_tuple() {
    // A lone typed handler needs no tuple: unmatched values (the array
    // itself) advance normally. The 1-tuple spelling routes through the
    // chain impls instead and must agree.
    let root = Array::new(vec![1i64, 2, 3]);
    let mut bare = 0;
    assert!(structural_walk(
        &root,
        |value: i64| {
            bare += value;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    let mut tupled = 0;
    assert!(structural_walk(
        &root,
        (|value: i64| {
            tupled += value;
            WalkResult::Advance
        },),
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!((bare, tupled), (6, 6));
}

#[test]
fn bare_node_lambda_takes_def_region_kind() {
    let root = Array::new(vec![1i64, 2]);
    let mut objects = 0;
    assert!(structural_walk(
        &root,
        |_object: &Object, kind: DefRegionKind| {
            assert_eq!(kind, DefRegionKind::None);
            objects += 1;
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(objects, 1);
}

struct InheritedRegionProbe {
    at_root: bool,
    seen: Vec<DefRegionKind>,
}

impl StructuralVisitor for InheritedRegionProbe {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if self.at_root {
            self.at_root = false;
            let outer = value.cast::<Array<Array<i64>>>().unwrap();
            let inner = outer.get(0).unwrap();
            return self.visit_child(&inner, DefRegionKind::Recursive);
        }
        self.seen.push(def_region_kind);
        self.default_visit_children(value, def_region_kind)
    }
}

#[test]
fn def_region_is_inherited_through_containers() {
    let root = Array::new(vec![Array::new(vec![1i64, 2])]);
    let mut probe = InheritedRegionProbe {
        at_root: true,
        seen: Vec::new(),
    };
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(probe.seen, vec![DefRegionKind::Recursive; 3]);
}

#[test]
fn reflected_field_def_region_reaches_typed_handler() {
    ensure_region_types_registered();
    let root = RustVisitDefRegion {
        data: ObjectArc::new(RustVisitDefRegionObj {
            base: Object::new(),
            recursive: Any::from(1i64),
            plain: Any::from(2i64),
            non_recursive: Any::from(3i64),
            both: Any::from(4i64),
            ignored: Any::from(5i64),
        }),
    };
    let mut seen = Vec::new();
    assert!(structural_walk(
        &root,
        |_value: i64, kind: DefRegionKind| {
            seen.push(kind);
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap()
    .is_none());
    assert_eq!(
        seen,
        vec![
            DefRegionKind::Recursive,
            DefRegionKind::None,
            DefRegionKind::NonRecursive,
            DefRegionKind::NonRecursive,
        ]
    );
}

struct FreeVarClampProbe {
    at_root: bool,
    seen: Vec<(&'static str, DefRegionKind)>,
}

impl StructuralVisitor for FreeVarClampProbe {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if self.at_root {
            self.at_root = false;
            let root = value.cast::<Array<ObjectRef>>().unwrap();
            for index in 0..root.len() {
                let child = root.get(index).unwrap();
                if let Some(interrupt) = self.visit_child(&child, DefRegionKind::NonRecursive)? {
                    return Ok(Some(interrupt));
                }
            }
            return Ok(None);
        }

        if value.as_node::<RustVisitDefRegionObj>().is_some() {
            self.seen.push(("free_var", def_region_kind));
        } else if value.cast::<Array<i64>>().is_some() {
            self.seen.push(("array", def_region_kind));
        } else if let Some(integer) = value.cast::<i64>() {
            self.seen.push((
                if integer == 6 {
                    "free_child"
                } else {
                    "array_child"
                },
                def_region_kind,
            ));
        }
        self.default_visit_children(value, def_region_kind)
    }
}

#[test]
fn non_recursive_region_is_clamped_for_free_var_children_only() {
    ensure_region_types_registered();
    let free_var = RustVisitDefRegion {
        data: ObjectArc::new(RustVisitDefRegionObj {
            base: Object::new(),
            recursive: Any::new(),
            plain: Any::from(6i64),
            non_recursive: Any::new(),
            both: Any::new(),
            ignored: Any::new(),
        }),
    };
    let free_var: ObjectRef = free_var.try_cast().unwrap();
    let array: ObjectRef = Array::new(vec![7i64]).try_cast().unwrap();
    let root = Array::new(vec![free_var, array]);
    let mut probe = FreeVarClampProbe {
        at_root: true,
        seen: Vec::new(),
    };
    assert!(structural_visit(&root, &mut probe).unwrap().is_none());
    assert_eq!(
        probe.seen,
        vec![
            ("free_var", DefRegionKind::NonRecursive),
            ("free_child", DefRegionKind::None),
            ("array", DefRegionKind::NonRecursive),
            ("array_child", DefRegionKind::NonRecursive),
        ]
    );
}
