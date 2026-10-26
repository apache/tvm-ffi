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
use tvm_ffi::reflection::{FieldFlags, ObjectDef};
use tvm_ffi::*;
use tvm_ffi_sys::{
    TVMFFIFieldFlagBitMask, TVMFFIFieldInfo, TVMFFIFieldSetter, TVMFFIGetTypeInfo,
    TVMFFIMethodInfo, TVMFFISEqHashKind, TVMFFITypeInfo,
};

// Rust-defined types whose `#[reflect]` fields and `#[type_reflection]`
// methods reach the type table, so that the C++ structural engines, `repr`,
// deep copy and other languages see them.

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.ReflectedPoint"]
#[type_register]
#[type_s_eq_hash_kind = "TreeNode"]
#[type_reflection(PointObj::reflect_methods)]
struct PointObj {
    base: Object,
    /// The x coordinate.
    ///
    /// Signed.
    #[reflect]
    x: i64,
    /// Not the registered doc: `doc = ` wins.
    #[reflect(writable, doc = "The y coordinate.")]
    y: f64,
    #[reflect(seq_hash_ignore, repr_off)]
    label: tvm_ffi::String,
    // Not reflected: not in the type table, but still shallow-copied.
    hits: i64,
}

impl PointObj {
    fn reflect_methods(def: &mut ObjectDef<Self>) {
        def.def(
            "norm1",
            |p: Point| -> Result<f64> { Ok(p.data.x.abs() as f64 + p.data.y.abs()) },
            "The L1 norm.",
        )
        .def(
            "shifted",
            |p: Point, dx: i64| -> Result<Point> {
                Ok(point(p.data.x + dx, p.data.y, p.data.label.as_str()))
            },
            "",
        )
        .def_static(
            "origin",
            || -> Result<Point> { Ok(point(0, 0.0, "origin")) },
            "The origin.",
        );
    }
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct Point {
    data: ObjectArc<PointObj>,
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.ReflectedPoint3"]
#[type_register]
#[type_final]
struct Point3Obj {
    base: PointObj,
    #[reflect]
    z: i64,
    #[reflect(name = "labels")]
    tags: Array<tvm_ffi::String>,
    #[reflect]
    r#type: Option<Point>,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct Point3 {
    data: ObjectArc<Point3Obj>,
}

tvm_ffi::impl_object_upcast!(Point3 => Point);

fn point_obj(x: i64, y: f64, label: &str) -> PointObj {
    PointObj {
        base: Object::new(),
        x,
        y,
        label: label.into(),
        hits: 7,
    }
}

fn point(x: i64, y: f64, label: &str) -> Point {
    Point {
        data: ObjectArc::new(point_obj(x, y, label)),
    }
}

fn point3(x: i64, z: i64) -> Point3 {
    Point3 {
        data: ObjectArc::new(Point3Obj {
            base: point_obj(x, 0.5, "p3"),
            z,
            tags: Array::new(vec!["a".into(), "b".into()]),
            r#type: Some(point(1, 2.0, "inner")),
        }),
    }
}

fn type_info<N: ObjectCore>() -> &'static TVMFFITypeInfo {
    unsafe { &*TVMFFIGetTypeInfo(N::type_index()) }
}

fn fields(info: &TVMFFITypeInfo) -> &[TVMFFIFieldInfo] {
    unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) }
}

fn methods(info: &TVMFFITypeInfo) -> &[TVMFFIMethodInfo] {
    if info.num_methods == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(info.methods, info.num_methods as usize) }
}

/// The metadata that records `type_schema`, as C++ `Metadata::ToJSON` writes it.
fn metadata_of(type_schema: &str) -> std::string::String {
    format!(
        r#"{{"type_schema":"{}"}}"#,
        type_schema.replace('"', "\\\"")
    )
}

fn call(name: &str, args: &[AnyView]) -> Any {
    Function::get_global(name)
        .unwrap()
        .call_packed(args)
        .unwrap()
}

// A type registers on first use, and the runtime does not guard its type table
// against readers on other threads, so each test registers every type here
// before it reads the table.
fn register_types() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        PointObj::type_index();
        Point3Obj::type_index();
        CounterObj::type_index();
        TallyObj::type_index();
        no_imports::LeafObj::type_index();
    });
}

#[test]
fn test_reflected_fields_are_registered_with_the_type() {
    register_types();
    let info = type_info::<PointObj>();
    let names: Vec<&str> = fields(info).iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["x", "y", "label"]);

    let [x, y, label] = fields(info) else {
        unreachable!()
    };
    assert_eq!(x.flags, 0);
    assert_eq!(x.offset, std::mem::offset_of!(PointObj, x) as i64);
    assert_eq!(x.field_static_type_index, TypeIndex::kTVMFFIInt as i32);
    assert_eq!(
        x.metadata.as_str(),
        r#"{"type_schema":"{\"type\":\"int\"}"}"#
    );
    assert_eq!(
        y.flags,
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskWritable as i64
    );
    assert_eq!(
        label.flags,
        (FieldFlags::SEQ_HASH_IGNORE | FieldFlags::REPR_OFF).bits()
    );
    assert_eq!(label.field_static_type_index, -1);

    let metadata = unsafe { &*info.metadata };
    assert_eq!(
        metadata.total_size as usize,
        std::mem::size_of::<PointObj>()
    );
    assert_eq!(
        metadata.structural_eq_hash_kind,
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode as i32
    );

    // The subtype registers only its own fields, and inherits the kind.
    let info = type_info::<Point3Obj>();
    let names: Vec<&str> = fields(info).iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["z", "labels", "type"]);
    assert_eq!(
        fields(info)[1].metadata.as_str(),
        r#"{"type_schema":"{\"type\":\"ffi.Array\",\"args\":[{\"type\":\"ffi.String\"}]}"}"#
    );
    assert_eq!(
        unsafe { (*info.metadata).structural_eq_hash_kind },
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode as i32
    );
}

#[test]
fn test_field_docs_come_from_doc_comments_or_reflect() {
    register_types();
    let [x, y, label] = fields(type_info::<PointObj>()) else {
        unreachable!()
    };
    assert_eq!(x.doc.as_str(), "The x coordinate.\n\nSigned.");
    assert_eq!(y.doc.as_str(), "The y coordinate.");
    assert_eq!(label.doc.as_str(), "");
    // A renamed field keeps its Rust type and offset.
    let labels = &fields(type_info::<Point3Obj>())[1];
    assert_eq!(labels.name.as_str(), "labels");
    assert_eq!(labels.offset, std::mem::offset_of!(Point3Obj, tags) as i64);
    let p3 = point3(1, 2);
    let getter = FieldGetter::new(Point3Obj::type_index(), "labels").unwrap();
    let tags = getter.get::<_, Array<tvm_ffi::String>>(&*p3.data).unwrap();
    assert_eq!(tags.len(), 2);
}

#[test]
fn test_methods_are_registered_with_the_type() {
    register_types();
    let info = type_info::<PointObj>();
    let names: Vec<&str> = methods(info).iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["norm1", "shifted", "origin"]);
    let [norm1, shifted, origin] = methods(info) else {
        unreachable!()
    };
    assert_eq!(norm1.flags, 0);
    assert_eq!(norm1.doc.as_str(), "The L1 norm.");
    assert_eq!(shifted.doc.as_str(), "");
    assert_eq!(
        origin.flags,
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskIsStaticMethod as i64
    );
    assert_eq!(origin.doc.as_str(), "The origin.");
    // Each method records its type schema, as C++ `ObjectDef::def` does.
    let point = r#"{"type":"testing.rust.ReflectedPoint"}"#;
    assert_eq!(
        shifted.metadata.as_str(),
        metadata_of(&format!(
            r#"{{"type":"ffi.Function","named_args":{{"return":[{point}],"params":[{point},{{"type":"int"}}]}}}}"#
        ))
    );
    assert_eq!(
        origin.metadata.as_str(),
        metadata_of(&format!(
            r#"{{"type":"ffi.Function","named_args":{{"return":[{point}],"params":[]}}}}"#
        ))
    );
    // The subtype's table holds only its own methods.
    assert_eq!(methods(type_info::<Point3Obj>()).len(), 0);
}

#[test]
fn test_methods_are_called_through_the_type_table() {
    register_types();
    let norm1 = Function::from_type_key_method("testing.rust.ReflectedPoint", "norm1").unwrap();
    let norm = f64::try_from(norm1.call_tuple((point(3, -4.5, "p"),)).unwrap()).unwrap();
    assert_eq!(norm, 7.5);
    // An object of a subtype is an object of the type.
    let norm = f64::try_from(norm1.call_tuple((point3(2, 9),)).unwrap()).unwrap();
    assert_eq!(norm, 2.5);
    let shifted = Function::from_type_method(PointObj::type_index(), "shifted").unwrap();
    let moved = Point::try_from(shifted.call_tuple((point(3, 1.0, "p"), 4i64)).unwrap()).unwrap();
    assert_eq!(
        (moved.data.x, moved.data.y, moved.data.label.as_str()),
        (7, 1.0, "p")
    );
    let origin = Function::from_type_key_method("testing.rust.ReflectedPoint", "origin").unwrap();
    let origin = Point::try_from(origin.call_tuple(()).unwrap()).unwrap();
    assert_eq!((origin.data.x, origin.data.label.as_str()), (0, "origin"));
    // A wrong argument is a type error, not a panic.
    let error = norm1.call_tuple((1i64,)).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);
}

// A type with methods but no reflected fields registers its reflection too.

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.ReflectedTally"]
#[type_register]
#[type_final]
#[type_reflection(Self::reflect_methods)]
struct TallyObj {
    base: Object,
    count: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct Tally {
    data: ObjectArc<TallyObj>,
}

impl TallyObj {
    fn reflect_methods(def: &mut ObjectDef<Self>) {
        def.def("count", |t: Tally| -> Result<i64> { Ok(t.data.count) }, "");
    }
}

#[test]
fn test_types_with_only_methods_register_their_reflection() {
    register_types();
    let info = type_info::<TallyObj>();
    assert_eq!(info.num_fields, 0);
    assert!(!info.metadata.is_null());
    assert_eq!(methods(info).len(), 1);
    assert!(
        bool::try_from(get_type_attr(TallyObj::type_index(), "__ffi_type_final__").unwrap())
            .unwrap()
    );
    let count = Function::from_type_method(TallyObj::type_index(), "count").unwrap();
    let tally = Tally {
        data: ObjectArc::new(TallyObj {
            base: Object::new(),
            count: 11,
        }),
    };
    assert_eq!(
        i64::try_from(count.call_tuple((tally,)).unwrap()).unwrap(),
        11
    );
}

#[test]
fn test_reflected_fields_read_and_write_through_the_c_abi() {
    register_types();
    let p3 = point3(3, 4);
    let x = FieldGetter::new(Point3Obj::type_index(), "x").unwrap();
    let z = FieldGetter::new(Point3Obj::type_index(), "z").unwrap();
    assert_eq!(x.get::<_, i64>(&*p3.data).unwrap(), 3);
    assert_eq!(z.get::<_, i64>(&*p3.data).unwrap(), 4);

    let p = point(1, 2.0, "p");
    let y = &fields(type_info::<PointObj>())[1];
    let setter: TVMFFIFieldSetter = unsafe { std::mem::transmute(y.setter) };
    let field = unsafe {
        ObjectArc::as_raw(&p.data)
            .cast::<u8>()
            .add(y.offset as usize)
            .cast_mut()
    };
    let value = Any::from(8.5f64);
    assert_eq!(
        unsafe { setter(field.cast(), &Any::into_raw_ffi_any(value.clone())) },
        0
    );
    assert_eq!(p.data.y, 8.5);
    // A value of the wrong type is an error, not a write.
    let value = Any::from(tvm_ffi::String::from("no"));
    assert_ne!(
        unsafe { setter(field.cast(), &Any::into_raw_ffi_any(value.clone())) },
        0
    );
    let error = Error::from_raised();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert_eq!(p.data.y, 8.5);
}

#[test]
fn test_cpp_structural_equality_and_hash_see_reflected_fields() {
    register_types();
    let equal = |a: &Point, b: &Point| -> bool {
        let (a, b) = (Any::from(a.clone()), Any::from(b.clone()));
        let args = [
            AnyView::from(&a),
            AnyView::from(&b),
            AnyView::from(&false),
            AnyView::from(&false),
        ];
        bool::try_from(call("ffi.StructuralEqual", &args)).unwrap()
    };
    let hash = |a: &Point| -> i64 {
        let a = Any::from(a.clone());
        let args = [
            AnyView::from(&a),
            AnyView::from(&false),
            AnyView::from(&false),
        ];
        i64::try_from(call("ffi.StructuralHash", &args)).unwrap()
    };
    // `label` is ignored.
    assert!(equal(&point(1, 2.0, "a"), &point(1, 2.0, "b")));
    assert_eq!(hash(&point(1, 2.0, "a")), hash(&point(1, 2.0, "b")));
    assert!(!equal(&point(1, 2.0, "a"), &point(2, 2.0, "a")));
    assert!(!equal(&point(1, 2.0, "a"), &point(1, 3.0, "a")));
    // Inherited and own fields of the subtype.
    assert!(equal(&point3(1, 2).into(), &point3(1, 2).into()));
    assert!(!equal(&point3(1, 2).into(), &point3(1, 3).into()));
    assert!(!equal(&point3(1, 2).into(), &point3(0, 2).into()));
}

#[test]
fn test_repr_and_deep_copy_see_reflected_fields() {
    register_types();
    let p = Any::from(point(5, 1.5, "hidden"));
    let repr = tvm_ffi::String::try_from(call("ffi.ReprPrint", &[AnyView::from(&p)])).unwrap();
    assert!(repr.as_str().contains("x=5"), "{}", repr.as_str());
    assert!(!repr.as_str().contains("hidden"), "{}", repr.as_str());

    let original = point3(1, 2);
    let any = Any::from(original.clone());
    let copy = Point3::try_from(call("ffi.DeepCopy", &[AnyView::from(&any)])).unwrap();
    assert!(!copy.same_as(&original));
    assert_eq!((copy.data.base.x, copy.data.z), (1, 2));
    assert_eq!(copy.data.tags.len(), 2);
    // The nested object is copied too.
    let (inner, original_inner) = (
        copy.data.r#type.clone().unwrap(),
        original.data.r#type.clone().unwrap(),
    );
    assert!(!inner.same_as(&original_inner));
    assert_eq!(inner.data.x, 1);
}

#[test]
fn test_shallow_copy_and_finality_are_type_attributes() {
    register_types();
    let original = point(1, 2.0, "p");
    let shallow_copy =
        Function::try_from(get_type_attr(PointObj::type_index(), "__ffi_shallow_copy__").unwrap())
            .unwrap();
    let copy = Point::try_from(shallow_copy.call_tuple((original.clone(),)).unwrap()).unwrap();
    assert!(!copy.same_as(&original));
    assert_eq!(
        (
            copy.data.x,
            copy.data.y,
            copy.data.label.as_str(),
            copy.data.hits
        ),
        (1, 2.0, "p", 7)
    );
    // Applied to another type, it is a type error.
    let other = Counter {
        data: ObjectArc::new(CounterObj {
            base: Object::new(),
            count: 0,
        }),
    };
    let error = shallow_copy.call_tuple((other,)).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);

    let is_final = |type_index| {
        bool::try_from(get_type_attr(type_index, "__ffi_type_final__").unwrap()).unwrap()
    };
    assert!(!is_final(PointObj::type_index()));
    assert!(is_final(Point3Obj::type_index()));
}

// A registered type without `#[reflect]` has no reflection, as before.

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.Unreflected"]
#[type_register]
#[type_final]
struct CounterObj {
    base: Object,
    count: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct Counter {
    data: ObjectArc<CounterObj>,
}

#[test]
fn test_types_without_reflect_register_no_reflection() {
    register_types();
    let info = type_info::<CounterObj>();
    assert_eq!(info.num_fields, 0);
    assert!(info.metadata.is_null());
    assert!(get_type_attr(CounterObj::type_index(), "__ffi_shallow_copy__").is_none());
}

// The derives name every item by path, so they need nothing in scope.
mod no_imports {
    #[repr(C)]
    #[derive(tvm_ffi::derive::Object)]
    #[type_key = "testing.rust.ReflectedNoImports"]
    #[type_register]
    pub struct LeafObj {
        base: super::PointObj,
        #[reflect]
        pub w: i64,
    }

    #[repr(C)]
    #[derive(tvm_ffi::derive::ObjectRef, Clone)]
    pub struct Leaf {
        pub data: tvm_ffi::ObjectArc<LeafObj>,
    }

    pub fn leaf(w: i64) -> Leaf {
        Leaf {
            data: tvm_ffi::ObjectArc::new(LeafObj {
                base: super::point_obj(0, 0.0, ""),
                w,
            }),
        }
    }
}

#[test]
fn test_derives_need_nothing_in_scope() {
    register_types();
    let leaf = no_imports::leaf(9);
    let w = FieldGetter::new(no_imports::LeafObj::type_index(), "w").unwrap();
    assert_eq!(w.get::<_, i64>(&*leaf.data).unwrap(), 9);
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.DuplicateMethod"]
#[type_register]
#[type_reflection(Self::reflect_methods)]
struct DuplicateMethodObj {
    base: Object,
}

impl DuplicateMethodObj {
    fn reflect_methods(def: &mut ObjectDef<Self>) {
        def.def(
            "same",
            |_: tvm_ffi::object::ObjectRef| -> Result<i64> { Ok(1) },
            "",
        )
        .def(
            "same",
            |_: tvm_ffi::object::ObjectRef| -> Result<i64> { Ok(2) },
            "",
        );
    }
}

#[test]
#[should_panic(expected = "duplicate method `same` of `testing.rust.DuplicateMethod`")]
fn test_duplicate_method_names_fail_the_registration() {
    DuplicateMethodObj::type_index();
}
