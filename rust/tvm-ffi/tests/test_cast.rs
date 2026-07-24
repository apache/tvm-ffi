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
use tvm_ffi::object::ObjectRef;
use tvm_ffi::*;

// Reference a symbol exported by libtvm_ffi_testing so the linker keeps the
// library dependency; its static initializers register the testing type keys.
#[test]
fn test_cast_dummy_c_api() {
    let ret = unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() };
    assert_eq!(ret, 0);
}

// The type keys below are registered by libtvm_ffi_testing with the hierarchy
// Object <- testing.TestObjectBase <- testing.TestObjectDerived. The Rust-side
// field layout does not need to match the C++ classes: the objects are created
// and destroyed purely on the Rust side, and the casts only consult the type
// index stored in the object header.

// must have repr(C) for the object header to stay in the same position
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestObjectBase"]
struct TestBaseObj {
    base: Object,
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
#[type_key = "testing.TestObjectDerived"]
struct TestDerivedObj {
    base: TestBaseObj,
    extra: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct TestDerived {
    data: ObjectArc<TestDerivedObj>,
}

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
            base: Object::new(),
            value,
            delete_counter,
        }),
    }
}

fn new_derived(value: i64, extra: i64, delete_counter: Arc<AtomicU32>) -> TestDerived {
    TestDerived {
        data: ObjectArc::new(TestDerivedObj {
            base: TestBaseObj {
                base: Object::new(),
                value,
                delete_counter,
            },
            extra,
        }),
    }
}

#[test]
fn test_is_instance_of() {
    let object_index = tvm_ffi::Object::type_index();
    let base_index = TestBaseObj::type_index();
    let derived_index = TestDerivedObj::type_index();
    // reflexive
    assert!(is_instance_of(base_index, base_index));
    // child -> ancestors at every depth
    assert!(is_instance_of(derived_index, base_index));
    assert!(is_instance_of(derived_index, object_index));
    assert!(is_instance_of(base_index, object_index));
    // the reverse direction does not hold
    assert!(!is_instance_of(base_index, derived_index));
    assert!(!is_instance_of(object_index, base_index));
    // non-object type indices never match an object type
    assert!(!is_instance_of(TypeIndex::kTVMFFIInt as i32, object_index));
    assert!(!is_instance_of(TypeIndex::kTVMFFINone as i32, object_index));
}

#[test]
fn test_upcast_downcast_roundtrip() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let derived = new_derived(7, 8, delete_counter.clone());
    // upcast to the direct parent
    let base: TestBase = derived.try_cast().unwrap();
    assert_eq!(base.data.value, 7);
    // upcast further to the root ObjectRef
    let obj: ObjectRef = base.try_cast().unwrap();
    // downcast all the way back
    let derived2: TestDerived = obj.try_cast().unwrap();
    assert_eq!(derived2.data.base.value, 7);
    assert_eq!(derived2.data.extra, 8);
    // every step moved ownership; no extra references were created
    assert_eq!(ObjectArc::strong_count(&derived2.data), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    drop(derived2);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_cast_checks_parameterized_container_type() {
    assert!(Array::new(vec![1_i64, 2_i64])
        .try_cast::<Array<f32>>()
        .is_err());
}

#[test]
fn test_downcast_failure() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let base = new_base(1, delete_counter.clone());
    let err = expect_err(base.try_cast::<TestDerived>());
    assert!(err.message().contains("testing.TestObjectBase"));
    assert!(err.message().contains("testing.TestObjectDerived"));
    // try_cast consumes the value even when the cast fails
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_cast_unrelated_type_failure() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let derived = new_derived(1, 2, delete_counter.clone());
    let err = expect_err(derived.try_cast::<Shape>());
    assert!(err.message().contains("testing.TestObjectDerived"));
    assert!(err.message().contains("ffi.Shape"));
}

#[test]
fn test_cast_clone_shares_ownership() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let derived = new_derived(3, 4, delete_counter.clone());
    let base: TestBase = derived.cast_clone().unwrap();
    assert_eq!(base.data.value, 3);
    assert_eq!(ObjectArc::strong_count(&derived.data), 2);
    // a failed cast_clone leaves the original untouched
    assert!(derived.cast_clone::<Shape>().is_err());
    assert_eq!(ObjectArc::strong_count(&derived.data), 2);
    drop(base);
    assert_eq!(ObjectArc::strong_count(&derived.data), 1);
    drop(derived);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_any_conversion_preserves_runtime_subtype() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let base: TestBase = new_derived(5, 6, delete_counter.clone())
        .try_cast()
        .unwrap();
    let derived_from_view: TestDerived = AnyView::from(&base).try_into().unwrap();
    assert_eq!(derived_from_view.data.extra, 6);
    drop(derived_from_view);
    let derived_from_any: TestDerived = Any::from(base).try_into().unwrap();
    assert_eq!(derived_from_any.data.extra, 6);
    drop(derived_from_any);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_any_conversion_rejects_supertype() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let base = new_base(9, delete_counter.clone());
    let any = Any::from(base);
    let res: Result<TestDerived> = any.try_into();
    let err = expect_err(res);
    assert!(err.message().contains("testing.TestObjectBase"));
    assert!(err.message().contains("testing.TestObjectDerived"));
}
