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
// `#[object(register)]` registers them when the test binary loads, under their
// parents, as C++ `TVM_FFI_DECLARE_OBJECT_INFO` does.

#[repr(C)]
#[derive(Object)]
#[object(
    type_key = "testing.rust.RegisteredBase",
    register,
    child_slots = 2,
    child_slots_can_overflow = false
)]
struct RegisteredBaseObj {
    base_: Object,
    value: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct RegisteredBase {
    data: ObjectArc<RegisteredBaseObj>,
}

#[repr(C)]
#[derive(Object)]
#[object(type_key = "testing.rust.RegisteredLeaf", register, final)]
struct RegisteredLeafObj {
    base_: RegisteredBaseObj,
    extra: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct RegisteredLeaf {
    data: ObjectArc<RegisteredLeafObj>,
}

tvm_ffi::impl_object_ref_upcast!(RegisteredLeaf => RegisteredBase);

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
            base_: RegisteredBaseObj {
                base_: Object::new(),
                value: 7,
            },
            extra: 8,
        }),
    };
    let any = Any::from(leaf);
    assert_eq!(any.type_index(), RegisteredLeafObj::type_index());
    let base = RegisteredBase::try_from(any).unwrap();
    assert_eq!(base.data.value, 7);
    let obj: ObjectRef = base.into();
    let leaf: RegisteredLeaf = obj.downcast().unwrap();
    assert_eq!(leaf.data.extra, 8);
}

// A type that binds a key its defining library registers, without
// `#[object(register)]`, does not register it.
#[repr(C)]
#[derive(Object)]
#[object(type_key = "testing.rust.NeverRegistered")]
struct UnregisteredBindingObj {
    base_: Object,
}

#[test]
#[should_panic(expected = "is not registered: load the library that defines it first")]
fn test_binding_does_not_register() {
    let _ = <UnregisteredBindingObj as ObjectCore>::type_index();
}
