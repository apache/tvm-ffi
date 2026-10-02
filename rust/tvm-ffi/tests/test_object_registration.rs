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
use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::object::{is_instance_of, ObjectRef};
use tvm_ffi::*;
use tvm_ffi_sys::{TVMFFIByteArray, TVMFFIGetTypeInfo, TVMFFITypeKeyToIndex};

// Type keys that nothing else registers: deriving `Object` with
// `#[type_register]` registers them on first use, under their parents, as C++
// `TVM_FFI_DECLARE_OBJECT_INFO` does.

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.RegisteredBase"]
#[type_register]
#[type_child_slots = 2]
#[type_child_slots_can_overflow = false]
struct RegisteredBaseObj {
    base: Object,
    value: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct RegisteredBase {
    data: ObjectArc<RegisteredBaseObj>,
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.RegisteredLeaf"]
#[type_register]
#[type_final]
struct RegisteredLeafObj {
    base: RegisteredBaseObj,
    extra: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct RegisteredLeaf {
    data: ObjectArc<RegisteredLeafObj>,
}

tvm_ffi::impl_object_upcast!(RegisteredLeaf => RegisteredBase);

fn registered_index(type_key: &str) -> i32 {
    let key = unsafe { TVMFFIByteArray::from_str(type_key) };
    let mut index = -1;
    assert_eq!(unsafe { TVMFFITypeKeyToIndex(&key, &mut index) }, 0);
    index
}

#[test]
fn test_derived_object_types_are_registered_on_first_use() {
    // The leaf comes first: registering it registers its parent before it.
    let leaf_index = RegisteredLeafObj::type_index();
    let base_index = RegisteredBaseObj::type_index();
    assert!(base_index >= TypeIndex::kTVMFFIDynObjectBegin as i32);
    assert_eq!(registered_index("testing.rust.RegisteredBase"), base_index);
    assert_eq!(registered_index("testing.rust.RegisteredLeaf"), leaf_index);
    // The leaf takes the first of its parent's reserved child slots.
    assert_eq!(leaf_index, base_index + 1);
    // Registration happens once; the index is stable.
    assert_eq!(RegisteredLeafObj::type_index(), leaf_index);

    let info = unsafe { &*TVMFFIGetTypeInfo(leaf_index) };
    assert_eq!(info.type_depth, RegisteredLeafObj::TYPE_DEPTH);
    let parent = unsafe { &**info.type_acenstors.add(1) };
    assert_eq!(parent.type_index, base_index);

    assert!(is_instance_of::<RegisteredBaseObj>(leaf_index));
    assert!(is_instance_of::<Object>(base_index));
    assert!(!is_instance_of::<RegisteredLeafObj>(base_index));
}

#[test]
fn test_registered_object_types_round_trip_through_any() {
    let leaf = RegisteredLeaf {
        data: ObjectArc::new(RegisteredLeafObj {
            base: RegisteredBaseObj {
                base: Object::new(),
                value: 7,
            },
            extra: 8,
        }),
    };
    let any = Any::from(leaf);
    assert_eq!(any.type_index(), RegisteredLeafObj::type_index());
    let base = RegisteredBase::try_from(any).unwrap();
    assert_eq!(base.data.value, 7);
    let obj: ObjectRef = base.try_cast().unwrap();
    let leaf: RegisteredLeaf = obj.try_cast().unwrap();
    assert_eq!(leaf.data.extra, 8);
}

// Types defined here, registered from several threads at once.
macro_rules! concurrent_types {
    ($($name:ident = $key:literal),*) => {
        $(
            #[repr(C)]
            #[derive(Object)]
            #[type_key = $key]
            #[type_register]
            struct $name {
                base: Object,
            }
        )*
    };
}

concurrent_types!(
    Concurrent0 = "testing.rust.Concurrent0",
    Concurrent1 = "testing.rust.Concurrent1",
    Concurrent2 = "testing.rust.Concurrent2",
    Concurrent3 = "testing.rust.Concurrent3",
    Concurrent4 = "testing.rust.Concurrent4",
    Concurrent5 = "testing.rust.Concurrent5",
    Concurrent6 = "testing.rust.Concurrent6",
    Concurrent7 = "testing.rust.Concurrent7"
);

#[test]
fn test_registration_from_threads() {
    let registrations: [fn() -> i32; 8] = [
        <Concurrent0 as ObjectCore>::type_index,
        <Concurrent1 as ObjectCore>::type_index,
        <Concurrent2 as ObjectCore>::type_index,
        <Concurrent3 as ObjectCore>::type_index,
        <Concurrent4 as ObjectCore>::type_index,
        <Concurrent5 as ObjectCore>::type_index,
        <Concurrent6 as ObjectCore>::type_index,
        <Concurrent7 as ObjectCore>::type_index,
    ];
    let barrier = std::sync::Barrier::new(registrations.len());
    let indices: Vec<i32> = std::thread::scope(|scope| {
        let handles: Vec<_> = registrations
            .iter()
            .map(|register| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    register()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, &index) in indices.iter().enumerate() {
        assert_eq!(
            index,
            registered_index(&format!("testing.rust.Concurrent{i}"))
        );
        assert!(
            !indices[..i].contains(&index),
            "type indices must be distinct"
        );
    }
}

// A type that binds a key its defining library registers, without
// `#[type_register]`, does not register it.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.NeverRegistered"]
struct UnregisteredBindingObj {
    base: Object,
}

#[test]
#[should_panic(expected = "is not registered: load the library that defines it first")]
fn test_binding_does_not_register() {
    let _ = <UnregisteredBindingObj as ObjectCore>::type_index();
}
