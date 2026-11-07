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

//! An enum registered in C++ (`testing.TestEnumVariant`, values `Alpha` and `Beta` with the
//! attribute `code`), extended from Rust.

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::enum_ref::EnumObj;
use tvm_ffi::reflection::EnumDef;
use tvm_ffi::{EnumRef, ObjectArc, ObjectRefCore, Result, VALUE_ERROR};

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestEnumVariant"]
#[type_final]
struct TestEnumVariantObj {
    base_: EnumObj,
}

#[derive(Clone, ObjectRef)]
struct TestEnumVariant {
    data: ObjectArc<TestEnumVariantObj>,
}

impl EnumRef for TestEnumVariant {}

#[test]
fn test_enum_lookup_register_and_attrs() -> Result<()> {
    // Keep the testing library linked so its static type registrations run.
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );

    let alpha = TestEnumVariant::get("Alpha")?;
    assert_eq!(alpha.name(), "Alpha");
    assert!(alpha.same_as(&TestEnumVariant::get("Alpha")?));
    assert_eq!(
        TestEnumVariant::get("Delta").unwrap_err().kind(),
        VALUE_ERROR
    );

    let code = TestEnumVariant::attr_map::<i64>("code")?.unwrap();
    assert_eq!(code.get(&alpha)?, Some(10));
    assert!(TestEnumVariant::attr_map::<i64>("no_such_attr")?.is_none());

    let gamma = EnumDef::<TestEnumVariant>::new("Gamma")?
        .set_attr("code", 30_i64)?
        .value();
    assert!(gamma.same_as(&TestEnumVariant::get("Gamma")?));
    // The map read before the registration sees the new value.
    assert_eq!(code.get(&gamma)?, Some(30));
    assert_eq!(
        EnumDef::<TestEnumVariant>::new("Beta")
            .err()
            .unwrap()
            .kind(),
        VALUE_ERROR
    );
    Ok(())
}
