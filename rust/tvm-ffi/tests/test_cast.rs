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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::object::{is_instance_of, ObjectRef};
use tvm_ffi::*;

// The hierarchy Object <- testing.rust.CastBase <- testing.rust.CastDerived;
// the casts only consult the type index stored in the object header.

// must have repr(C) for the object header to stay in the same position
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.CastBase"]
#[type_register]
struct TestBaseObj {
    base_: Object,
    value: i64,
    // counter for recording the number of times the object is deleted
    delete_counter: Arc<AtomicU32>,
}

impl Drop for TestBaseObj {
    fn drop(&mut self) {
        self.delete_counter.fetch_add(1, Ordering::Relaxed);
    }
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct TestBase {
    data: ObjectArc<TestBaseObj>,
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.CastDerived"]
#[type_register]
#[type_final]
struct TestDerivedObj {
    base_: TestBaseObj,
    extra: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct TestDerived {
    data: ObjectArc<TestDerivedObj>,
}

tvm_ffi::impl_object_ref_upcast!(TestDerived => TestBase);

// unwrap_err() requires the Ok type to implement Debug, which ObjectRef types do not
fn expect_err<T>(res: Result<T>) -> Error {
    match res {
        Ok(_) => panic!("expected the cast to fail"),
        Err(err) => err,
    }
}

fn new_base(value: i64, delete_counter: Arc<AtomicU32>) -> TestBase {
    TestBase {
        data: ObjectArc::new(TestBaseObj {
            base_: Object::new(),
            value,
            delete_counter,
        }),
    }
}

fn new_derived(value: i64, extra: i64, delete_counter: Arc<AtomicU32>) -> TestDerived {
    TestDerived {
        data: ObjectArc::new(TestDerivedObj {
            base_: TestBaseObj {
                base_: Object::new(),
                value,
                delete_counter,
            },
            extra,
        }),
    }
}

#[test]
fn test_is_instance_of() {
    // Keep the testing library linked so its static type registrations run.
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);

    let object_index = tvm_ffi::Object::type_index();
    let base_index = TestBaseObj::type_index();
    let derived_index = TestDerivedObj::type_index();
    assert_eq!(Object::TYPE_DEPTH, 0);
    assert_eq!(TestBaseObj::TYPE_DEPTH, 1);
    assert_eq!(TestDerivedObj::TYPE_DEPTH, 2);
    // reflexive
    assert!(is_instance_of::<TestBaseObj>(base_index));
    // child -> ancestors at every depth
    assert!(is_instance_of::<TestBaseObj>(derived_index));
    assert!(is_instance_of::<Object>(derived_index));
    assert!(is_instance_of::<Object>(base_index));
    // the reverse direction does not hold
    assert!(!is_instance_of::<TestDerivedObj>(base_index));
    assert!(!is_instance_of::<TestBaseObj>(object_index));
    // non-object type indices never match an object type
    assert!(!is_instance_of::<Object>(TypeIndex::kTVMFFIInt as i32));
}

// Bindings of the C++ testing.TestObjectBase and testing.TestObjectDerived.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestObjectBase"]
struct CxxTestBaseObj {
    base_: Object,
    v_i64: i64,
    v_f64: f64,
    v_str: String,
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestObjectDerived"]
#[type_final]
struct CxxTestDerivedObj {
    base_: CxxTestBaseObj,
    v_map: Map<Any, Any>,
    v_array: Array<Any>,
}

#[test]
fn test_field_default() {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let v_i64 = reflection::field_default::<CxxTestBaseObj>("v_i64").unwrap();
    assert_eq!(i64::try_from(v_i64).unwrap(), 10);
    let v_str = reflection::field_default::<CxxTestBaseObj>("v_str").unwrap();
    assert_eq!(String::try_from(v_str).unwrap().as_str(), "hello");
    // no default, unknown, and inherited fields
    assert!(reflection::field_default::<CxxTestDerivedObj>("v_map").is_none());
    assert!(reflection::field_default::<CxxTestBaseObj>("missing").is_none());
    assert!(reflection::field_default::<CxxTestDerivedObj>("v_i64").is_none());
}

#[test]
fn test_upcast_downcast_roundtrip() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let derived = new_derived(7, 8, delete_counter.clone());
    // upcast to the direct parent
    let base: TestBase = derived.into();
    assert_eq!(base.data.value, 7);
    // upcast further to the root ObjectRef
    let obj: ObjectRef = base.into();
    // downcast all the way back
    let derived2: TestDerived = obj.downcast().unwrap();
    assert_eq!(derived2.data.base_.value, 7);
    assert_eq!(derived2.data.extra, 8);
    // every step moved ownership; no extra references were created
    assert_eq!(ObjectArc::strong_count(&derived2.data), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    drop(derived2);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_borrowed_node_cast_preserves_reference_count() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let base: TestBase = new_derived(7, 8, delete_counter.clone()).into();
    let strong_count = ObjectArc::strong_count(TestBase::data(&base));

    let derived = base.downcast_ref::<TestDerivedObj>().unwrap();
    assert_eq!(derived.base_.value, 7);
    assert_eq!(derived.extra, 8);
    let base_node = base.downcast_ref::<TestBaseObj>().unwrap();
    assert_eq!(base_node.value, 7);
    assert_eq!(ObjectArc::strong_count(TestBase::data(&base)), strong_count);

    let base_only = new_base(1, delete_counter.clone());
    assert!(base_only.downcast_ref::<TestDerivedObj>().is_none());
    assert_eq!(ObjectArc::strong_count(TestBase::data(&base_only)), 1);

    let any = Any::from(base);
    let view = AnyView::from(&any);
    let derived = any.downcast_ref::<TestDerivedObj>().unwrap();
    assert!(std::ptr::eq(
        view.downcast_ref::<TestDerivedObj>().unwrap(),
        derived
    ));
    assert_eq!(view.downcast_ref::<TestBaseObj>().unwrap().value, 7);
    assert!(any.downcast_ref::<Object>().is_some());
    assert_eq!(any.debug_strong_count(), Some(1));
    assert_eq!(derived.extra, 8);

    let base_only = Any::from(base_only);
    assert!(base_only.downcast_ref::<TestDerivedObj>().is_none());
    assert!(AnyView::from(&base_only)
        .downcast_ref::<TestDerivedObj>()
        .is_none());
}

#[test]
fn test_generated_borrow_and_upcast_conversions() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let derived = new_derived(7, 8, delete_counter.clone());

    let borrowed_clone = TestDerived::from(&derived);
    assert!(borrowed_clone.same_as(&derived));

    let base = TestBase::from(&derived);
    assert!(base.same_as(&derived));
    assert_eq!(base.data.value, 7);
    assert_eq!(ObjectArc::strong_count(&derived.data), 3);

    drop(base);
    drop(borrowed_clone);
    drop(derived);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_cast_checks_parameterized_container_type() {
    assert!(Array::from_vec(vec![1_i64, 2_i64])
        .downcast::<Array<f32>>()
        .is_err());
}

#[test]
fn test_downcast_failure() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let base = new_base(1, delete_counter.clone());
    // A failed downcast hands the reference back unchanged.
    let base = base.downcast::<TestDerived>().unwrap_err();
    assert_eq!(base.data.value, 1);
    assert_eq!(ObjectArc::strong_count(&base.data), 1);
    let err = expect_err(TestDerived::try_from(base));
    assert!(err.message().contains("testing.rust.CastBase"));
    assert!(err.message().contains("testing.rust.CastDerived"));
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_try_cast_from_any_view_preserves_runtime_subtype() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let base: TestBase = new_derived(3, 4, delete_counter.clone()).into();
    let raw = Any::into_raw_ffi_any(Any::from(base));

    // Array::get calls this method directly, without check_any_strict first.
    let base = unsafe { TestBase::try_cast_from_any_view(&raw) }.unwrap();
    let derived: TestDerived = base.downcast().unwrap();
    assert_eq!(derived.data.base_.value, 3);
    assert_eq!(derived.data.extra, 4);
    assert_eq!(ObjectArc::strong_count(&derived.data), 2);
    drop(derived);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    drop(unsafe { Any::from_raw_ffi_any(raw) });
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_any_conversion_preserves_runtime_subtype() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let base: TestBase = new_derived(5, 6, delete_counter.clone()).into();
    let derived_from_view: TestDerived = AnyView::from(&base).try_into().unwrap();
    assert_eq!(derived_from_view.data.extra, 6);
    drop(derived_from_view);
    let derived_from_any: TestDerived = Any::from(base).try_into().unwrap();
    assert_eq!(derived_from_any.data.extra, 6);
    drop(derived_from_any);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

// `TestBaseObj` opts in; `TestDerivedObj` does not, so only `TestBase` crosses threads.
// SAFETY: test objects are never mutated after construction.
unsafe impl tvm_ffi::ImmutableObject for TestBaseObj {}

#[test]
fn test_thread_safe_object_handles() {
    fn thread_safe<T: Send + Sync>() {}
    thread_safe::<TestBase>();
    let base = new_base(3, Arc::new(AtomicU32::new(0)));
    let value = std::thread::spawn(move || base.data.value).join().unwrap();
    assert_eq!(value, 3);
}
