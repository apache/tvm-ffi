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
//! Object types defined in Rust register their fields and methods as C++
//! `refl::ObjectDef` does: the reflection records match the ones C++ registers
//! for `testing.SchemaAllTypes`, and the runtime reads and sets the fields and
//! calls the methods through them.

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::reflection::ObjectDef;
use tvm_ffi::*;
use tvm_ffi_sys::TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskWritable;
use tvm_ffi_sys::{
    TVMFFIByteArray, TVMFFIFieldInfo, TVMFFIFieldSetter, TVMFFIGetTypeInfo, TVMFFIMethodInfo,
    TVMFFITypeInfo, TVMFFITypeKeyToIndex,
};

/// Fields of the types `testing.SchemaAllTypes` registers with `def_rw`, and
/// methods of the signatures it and `testing.schema_*` register.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.Reflected"]
#[type_register]
#[type_mutable]
#[type_reflection(ReflectedObj::register_reflection)]
struct ReflectedObj {
    object: Object,
    #[def_rw(doc = "bool field")]
    v_bool: bool,
    #[def_rw]
    v_int: i64,
    #[def_rw]
    v_float: f64,
    #[def_rw]
    v_device: DLDevice,
    #[def_rw]
    v_dtype: DLDataType,
    #[def_rw]
    v_string: String,
    #[def_rw]
    v_bytes: Bytes,
    #[def_rw]
    v_opt_int: Option<i64>,
    #[def_rw]
    v_opt_str: Option<String>,
    #[def_rw]
    v_arr_int: Array<i64>,
    #[def_rw]
    v_arr_str: Array<String>,
    #[def_rw]
    v_map_str_int: Map<String, i64>,
    #[def_rw]
    v_map_str_arr_int: Map<String, Array<i64>>,
    // Not reflected.
    hidden: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct Reflected {
    data: ObjectArc<ReflectedObj>,
}

impl ReflectedObj {
    fn register_reflection(def: &mut ObjectDef<Self>) {
        def.def(
            "add_int",
            |this: Reflected, x: i64| -> Result<i64> { Ok(this.data.v_int + x) },
            "add int method",
        )
        .def_static(
            "schema_arr_map_opt",
            |_: Array<Option<i64>>,
             mp: Map<String, Array<i64>>,
             _: Option<String>|
             -> Result<Map<String, Array<i64>>> { Ok(mp) },
            "",
        );
        // A registering function may register other types: each registration
        // takes the lock on its own.
        RegisteredInHookObj::type_index();
    }
}

fn reflected(v_int: i64, v_string: &str) -> Reflected {
    Reflected {
        data: ObjectArc::new(ReflectedObj {
            object: Object::new(),
            v_bool: true,
            v_int,
            v_float: 0.5,
            v_device: DLDevice::new(DLDeviceType::kDLCPU, 0),
            v_dtype: DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
            v_string: String::from(v_string),
            v_bytes: Bytes::from(&b"bytes"[..]),
            v_opt_int: None,
            v_opt_str: Some(String::from("opt")),
            v_arr_int: Array::new(vec![1, 2]),
            v_arr_str: Array::new(vec![String::from("a")]),
            v_map_str_int: Map::new(),
            v_map_str_arr_int: Map::new(),
            hidden: 0,
        }),
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.RegisteredInHook"]
#[type_register]
struct RegisteredInHookObj {
    object: Object,
}

/// A parent and a child that each declare fields; the child's type info finds
/// the parent's fields through its ancestor, as for C++.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.ReflectedParent"]
#[type_register]
#[type_child_slots = 1]
struct ReflectedParentObj {
    object: Object,
    #[def_ro(name = "label", doc = "the label")]
    r#label_: String,
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.ReflectedChild"]
#[type_register]
#[type_final]
struct ReflectedChildObj {
    parent: ReflectedParentObj,
    #[def_ro]
    count: i64,
    #[def_ro]
    any: Any,
    #[def_ro]
    object: tvm_ffi::object::ObjectRef,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct ReflectedChild {
    data: ObjectArc<ReflectedChildObj>,
}

/// A second definition of `testing.rust.ReflectedParent`, as another library
/// that links the crate defining it would have: it takes the registered index
/// and does not register the reflection again.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.ReflectedParent"]
#[type_register]
#[type_child_slots = 1]
struct ReflectedParentCopyObj {
    object: Object,
    #[def_ro(name = "label", doc = "the label")]
    label: String,
}

/// The dummy call keeps `libtvm_ffi_testing`, which registers the C++ types
/// and functions compared with, linked.
fn type_info(type_key: &str) -> &'static TVMFFITypeInfo {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let key = unsafe { TVMFFIByteArray::from_str(type_key) };
    let mut index = -1;
    assert_eq!(unsafe { TVMFFITypeKeyToIndex(&key, &mut index) }, 0);
    unsafe { &*TVMFFIGetTypeInfo(index) }
}

fn fields(info: &TVMFFITypeInfo) -> &[TVMFFIFieldInfo] {
    unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) }
}

fn methods(info: &TVMFFITypeInfo) -> &[TVMFFIMethodInfo] {
    unsafe { std::slice::from_raw_parts(info.methods, info.num_methods as usize) }
}

fn find_field<'a>(info: &'a TVMFFITypeInfo, name: &str) -> &'a TVMFFIFieldInfo {
    fields(info)
        .iter()
        .find(|field| field.name.as_str() == name)
        .unwrap_or_else(|| panic!("no field {name}"))
}

fn find_method<'a>(info: &'a TVMFFITypeInfo, name: &str) -> &'a TVMFFIMethodInfo {
    methods(info)
        .iter()
        .find(|method| method.name.as_str() == name)
        .unwrap_or_else(|| panic!("no method {name}"))
}

/// The `type_schema` entry of a metadata JSON string, parsed by the runtime.
fn type_schema(metadata: &TVMFFIByteArray) -> String {
    let parsed: Map<String, Any> = Function::get_global("ffi.json.Parse")
        .unwrap()
        .call_tuple_with_len::<1, _>((String::from(metadata.as_str()),))
        .unwrap()
        .try_into()
        .unwrap();
    parsed
        .get(&String::from("type_schema"))
        .unwrap()
        .unwrap()
        .try_into()
        .unwrap()
}

fn global_metadata(name: &str) -> String {
    Function::get_global("ffi.GetGlobalFuncMetadata")
        .unwrap()
        .call_tuple_with_len::<1, _>((String::from(name),))
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
fn test_fields_match_cpp() {
    ReflectedObj::type_index();
    let rust = type_info(ReflectedObj::TYPE_KEY);
    let cpp = type_info("testing.SchemaAllTypes");
    let names: Vec<&str> = fields(rust).iter().map(|f| f.name.as_str()).collect();
    assert!(!names.contains(&"hidden"));
    assert_eq!(names.len(), 13);
    for field in fields(rust) {
        let name = field.name.as_str();
        let cpp_field = find_field(cpp, name);
        assert_eq!(
            type_schema(&field.metadata).as_str(),
            type_schema(&cpp_field.metadata).as_str(),
            "{name}"
        );
        // Both are writable, and carry no other flags since neither has a
        // default value here.
        let writable = kTVMFFIFieldFlagBitMaskWritable as i64;
        assert_eq!(field.flags, writable, "{name}");
        assert_eq!(cpp_field.flags & writable, writable, "{name}");
        assert_eq!(
            field.field_static_type_index, cpp_field.field_static_type_index,
            "{name}"
        );
        assert!(field.getter.is_some() && !field.setter.is_null(), "{name}");
    }
    assert_eq!(find_field(rust, "v_bool").doc.as_str(), "bool field");
    assert_eq!(find_field(rust, "v_int").doc.size, 0);
}

#[test]
fn test_type_metadata_and_finality() {
    ReflectedChildObj::type_index();
    let info = type_info(ReflectedChildObj::TYPE_KEY);
    let metadata = unsafe { &*info.metadata };
    assert_eq!(
        metadata.total_size as usize,
        std::mem::size_of::<ReflectedChildObj>()
    );
    let is_final = |type_key: &str| -> bool {
        reflection::get_type_attr(type_info(type_key).type_index, "__ffi_type_final__")
            .unwrap()
            .try_into()
            .unwrap()
    };
    assert!(is_final(ReflectedChildObj::TYPE_KEY));
    assert!(!is_final(ReflectedParentObj::TYPE_KEY));
}

#[test]
fn test_field_getters_and_setters() {
    let object = reflected(7, "seven");
    let type_index = ReflectedObj::type_index();
    let v_int = FieldGetter::new(type_index, "v_int").unwrap();
    assert_eq!(v_int.get::<_, i64>(&*object.data).unwrap(), 7);
    let v_string = FieldGetter::new(type_index, "v_string").unwrap();
    assert_eq!(
        v_string.get::<_, String>(&*object.data).unwrap().as_str(),
        "seven"
    );
    let v_opt_str = FieldGetter::new(type_index, "v_opt_str").unwrap();
    assert_eq!(
        v_opt_str
            .get::<_, Option<String>>(&*object.data)
            .unwrap()
            .unwrap()
            .as_str(),
        "opt"
    );

    // Set fields as other languages do: the setter at the field's address.
    let info = type_info(ReflectedObj::TYPE_KEY);
    let set = |name: &str, value: Any| -> Result<()> {
        let field = find_field(info, name);
        let setter: TVMFFIFieldSetter = unsafe { std::mem::transmute(field.setter) };
        let address = unsafe {
            ObjectArc::as_raw(&object.data)
                .cast::<u8>()
                .add(field.offset as usize)
                .cast_mut()
                .cast()
        };
        let mut value = value;
        let ret = unsafe { setter(address, Any::as_data_ptr(&mut value)) };
        if ret == 0 {
            Ok(())
        } else {
            Err(Error::from_raised())
        }
    };
    set("v_int", Any::from(8i64)).unwrap();
    set("v_string", Any::from(String::from("eight"))).unwrap();
    set("v_opt_int", Any::from(3i64)).unwrap();
    assert_eq!(object.data.v_int, 8);
    assert_eq!(object.data.v_string.as_str(), "eight");
    assert_eq!(object.data.v_opt_int, Some(3));
    let error = set("v_int", Any::from(String::from("nine"))).unwrap_err();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert_eq!(object.data.v_int, 8);
}

#[test]
fn test_inherited_fields() {
    let child = ReflectedChild {
        data: ObjectArc::new(ReflectedChildObj {
            parent: ReflectedParentObj {
                object: Object::new(),
                r#label_: String::from("parent"),
            },
            count: 2,
            any: Any::from(1.5f64),
            object: Any::from(Function::from_typed(|| -> Result<()> { Ok(()) }))
                .try_into()
                .unwrap(),
        }),
    };
    let type_index = ReflectedChildObj::type_index();
    let label = FieldGetter::new(type_index, "label").unwrap();
    assert_eq!(
        label.get::<_, String>(&*child.data).unwrap().as_str(),
        "parent"
    );
    let any = FieldGetter::new(type_index, "any").unwrap();
    assert_eq!(any.get::<_, f64>(&*child.data).unwrap(), 1.5);
    let parent = type_info(ReflectedParentObj::TYPE_KEY);
    let label = find_field(parent, "label");
    assert_eq!(label.doc.as_str(), "the label");
    assert_eq!(label.flags, 0);
    // `Any` and object fields are recorded as C++ records them.
    let rust = type_info(ReflectedChildObj::TYPE_KEY);
    let cpp = type_info("testing.TestDeepCopyEdges");
    for (name, cpp_name) in [("any", "v_any"), ("object", "v_obj")] {
        let (field, cpp_field) = (find_field(rust, name), find_field(cpp, cpp_name));
        assert_eq!(
            type_schema(&field.metadata).as_str(),
            type_schema(&cpp_field.metadata).as_str()
        );
        assert_eq!(
            field.field_static_type_index,
            cpp_field.field_static_type_index
        );
    }
}

#[test]
fn test_methods() {
    let type_index = ReflectedObj::type_index();
    let info = type_info(ReflectedObj::TYPE_KEY);

    let add_int = find_method(info, "add_int");
    assert_eq!(add_int.flags, 0);
    assert_eq!(add_int.doc.as_str(), "add int method");
    let add_int = Function::from_type_method(type_index, "add_int").unwrap();
    let object = reflected(7, "seven");
    let sum = add_int
        .call_tuple_with_len::<2, _>((&object, 3i64))
        .unwrap();
    assert_eq!(i64::try_from(sum).unwrap(), 10);

    // A static method's metadata is the metadata C++ records for a function of
    // the same signature.
    let schema_arr_map_opt = find_method(info, "schema_arr_map_opt");
    assert_ne!(schema_arr_map_opt.flags, 0);
    assert_eq!(schema_arr_map_opt.doc.size, 0);
    assert_eq!(
        schema_arr_map_opt.metadata.as_str(),
        global_metadata("testing.schema_arr_map_opt").as_str()
    );
}

#[test]
fn test_runtime_compares_fields() {
    // `ffi.RecursiveEq` compares objects field by field through the getters.
    let eq = Function::get_global("ffi.RecursiveEq").unwrap();
    let equal = |a: &Reflected, b: &Reflected| -> bool {
        eq.call_tuple_with_len::<2, _>((a, b))
            .unwrap()
            .try_into()
            .unwrap()
    };
    assert!(equal(&reflected(1, "one"), &reflected(1, "one")));
    assert!(!equal(&reflected(1, "one"), &reflected(2, "one")));
    assert!(!equal(&reflected(1, "one"), &reflected(1, "two")));
}

#[test]
fn test_reflection_registers_once() {
    ReflectedParentObj::type_index();
    assert_eq!(
        ReflectedParentCopyObj::type_index(),
        ReflectedParentObj::type_index()
    );
    assert_eq!(fields(type_info(ReflectedParentObj::TYPE_KEY)).len(), 1);
}
