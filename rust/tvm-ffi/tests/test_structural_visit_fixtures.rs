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

//! Tests for custom Rust hooks and reflected fields. This separate integration
//! test executable isolates fixture registration from ordinary traversal tests.

use std::cell::{Cell, RefCell};
use std::sync::LazyLock;

use tvm_ffi::derive::{Object as DeriveObject, ObjectRef as DeriveObjectRef};
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIAnyViewToOwnedAny, TVMFFIByteArray, TVMFFIFieldFlagBitMask, TVMFFIFieldInfo,
    TVMFFISEqHashKind, TVMFFITypeMetadata, TVMFFITypeRegisterAttr,
};
use tvm_ffi::{
    get_type_attr, structural_visit, structural_walk, Any, AnyView, Array, DefRegionKind, Error,
    FieldGetter, Function, Object, ObjectArc, ObjectCore, ObjectRefCast, Result,
    String as FfiString, StructuralVisitor, TypeIndex, VisitContext, VisitInterrupt, VisitValue,
    WalkOrder, WalkResult, RUNTIME_ERROR,
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

// The runtime type table leaves registration synchronization to its callers.
// Only this test executable registers custom fixtures. Each test in this
// executable waits for registration before reading reflection data. The ordinary
// traversal tests run in a separate executable and do not share this registry.
static REGISTER_TEST_TYPES: LazyLock<()> = LazyLock::new(|| {
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

fn test_prelude() {
    LazyLock::force(&REGISTER_TEST_TYPES);
}

fn rust_visit_hook(selected: impl Into<Any>, ignored: impl Into<Any>) -> RustVisitHook {
    test_prelude();
    RustVisitHook {
        data: ObjectArc::new(RustVisitHookObj {
            base: Object::new(),
            selected: selected.into(),
            ignored: ignored.into(),
        }),
    }
}

fn rust_visit_failing_getter(value: impl Into<Any>) -> RustVisitFailingGetter {
    test_prelude();
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
fn public_reflection_access_uses_registered_field_and_type_attr() {
    test_prelude();
    let root = rust_visit_hook(FfiString::from("owned field"), 99i64);
    let type_index = RustVisitHookObj::type_index();

    let getter = FieldGetter::new(type_index, "selected").unwrap();
    let selected = getter.get::<_, FfiString>(&*root.data).unwrap();
    drop(root);
    assert_eq!(selected.as_str(), "owned field");

    let wrong_type = rust_visit_failing_getter(0i64);
    assert!(getter.get_any(&*wrong_type.data).is_err());
    assert!(FieldGetter::new(type_index, "missing").is_err());

    assert!(Function::try_from(get_type_attr(type_index, "__s_visit__").unwrap()).is_ok());
    assert!(Function::from_type_attr(type_index, "__s_visit__").is_ok());
    assert!(get_type_attr(type_index, "missing").is_none());
}

#[test]
fn reflected_getter_releases_partial_result_on_error() {
    test_prelude();
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
fn registered_function_hook_controls_children_interrupts_and_lifetime() {
    test_prelude();
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

    let callback_integers = RefCell::new(Vec::new());
    assert!(
        structural_visit(&root, |value: i64, _visitor: &mut VisitContext<'_, ()>| {
            callback_integers.borrow_mut().push(value)
        },)
        .unwrap()
        .is_none()
    );
    assert_eq!(*callback_integers.borrow(), vec![11]);

    test_prelude();
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
    test_prelude();
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
fn reflected_field_def_region_reaches_typed_handler() {
    test_prelude();
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

#[test]
fn non_recursive_region_is_clamped_for_free_var_children_only() {
    test_prelude();
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

#[test]
fn callback_visit_supports_node_links_nested_tuples_and_def_regions() {
    test_prelude();
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
    let seen = RefCell::new(Vec::new());

    assert!(structural_visit(
        &root,
        (
            (
                |_value: f64, _visitor: &mut VisitContext<'_, ()>| {},
                |_node: &RustVisitDefRegionObj, visitor: &mut VisitContext<'_, ()>| {
                    assert_eq!(visitor.def_region_kind(), DefRegionKind::None);
                    visitor.visit_children()
                },
            ),
            |value: i64, visitor: &mut VisitContext<'_, ()>| {
                seen.borrow_mut().push((value, visitor.def_region_kind()));
            },
        ),
    )
    .unwrap()
    .is_none());
    assert_eq!(
        *seen.borrow(),
        vec![
            (1, DefRegionKind::Recursive),
            (2, DefRegionKind::None),
            (3, DefRegionKind::NonRecursive),
            (4, DefRegionKind::NonRecursive),
        ]
    );
}
