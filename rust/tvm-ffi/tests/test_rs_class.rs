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
use tvm_ffi::derive::Object;
use tvm_ffi::object::ObjectRef;
use tvm_ffi::reflection::{FieldFlags, ObjectDef};
use tvm_ffi::*;
use tvm_ffi_sys::{TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFISEqHashKind, TVMFFITypeInfo};

// Object types defined in Rust by `#[rs_class]`, whose fields the C++
// structural engines, `repr` and deep copy see.

/// A root type, under `ffi.Object`.
#[rs_class(
    "testing.rust.RsPoint",
    s_eq_hash_kind = "TreeNode",
    reflection = RsPointObj::reflect_methods
)]
pub struct RsPointObj {
    base: Object,
    /// The x coordinate.
    pub x: i64,
    #[reflect(writable, seq_hash_ignore, doc = "A label.")]
    pub label: tvm_ffi::String,
}

def_object_ref!(pub RsPoint, ObjectRef, RsPointObj);

impl RsPointObj {
    fn reflect_methods(def: &mut ObjectDef<Self>) {
        def.def(
            "with_label",
            |p: RsPoint, label: tvm_ffi::String| -> Result<RsPoint> {
                Ok(point(p.x, label.as_str()))
            },
            "The point with another label.",
        );
    }
}

/// A subtype of a Rust-defined type, with a private field.
#[rs_class("testing.rust.RsPoint3", final)]
pub struct RsPoint3Obj {
    base: RsPointObj,
    pub z: i64,
    pub tags: Array<tvm_ffi::String>,
    tag: i64,
}

def_object_ref!(
    /// The reference with its own docs.
    pub RsPoint3,
    RsPoint,
    RsPoint3Obj,
);

/// A subtype without fields, which still registers its reflection.
#[rs_class("testing.rust.RsMarker", final)]
pub struct RsMarkerObj {
    base: RsPointObj,
}

def_object_ref!(pub RsMarker, RsPoint, RsMarkerObj);

/// Scalar fields, whose literal arguments to `new` infer as the field types.
#[rs_class("testing.rust.RsScalars", final)]
pub struct RsScalarsObj {
    base: Object,
    pub big: i64,
    pub count: u64,
    pub port: u16,
    pub ratio: f32,
}

def_object_ref!(pub RsScalars, ObjectRef, RsScalarsObj);

// `testing.TestCxxClassBase` from libtvm_ffi_testing, bound as tvm-ffi-stubgen
// binds a complete C++ type that can have children: with a public `new` and
// `ShallowCopy`, so that a Rust type can derive from it.

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.TestCxxClassBase"]
pub struct TestCxxClassBaseObj {
    base: Object,
    pub v_i64: i64,
    pub v_i32: i32,
}

impl TestCxxClassBaseObj {
    pub fn new(v_i64: i64, v_i32: i32) -> Self {
        let base = Object::new();
        Self { base, v_i64, v_i32 }
    }
}

impl ShallowCopy for TestCxxClassBaseObj {
    fn shallow_copy(&self) -> Self {
        Self {
            base: ShallowCopy::shallow_copy(&self.base),
            v_i64: Clone::clone(&self.v_i64),
            v_i32: Clone::clone(&self.v_i32),
        }
    }
}

// `def_object_ref!` takes any object type, not only an `rs_class` one.
def_object_ref!(pub TestCxxClassBase, ObjectRef, TestCxxClassBaseObj);

/// A subtype of a C++ type.
#[rs_class("testing.rust.RsCxxChild", final, s_eq_hash_kind = "TreeNode")]
pub struct RsCxxChildObj {
    base: TestCxxClassBaseObj,
    pub name: tvm_ffi::String,
}

def_object_ref!(pub RsCxxChild, TestCxxClassBase, RsCxxChildObj);

/// A root type with `placeholder`, whose fields have the field types tvm-ffi
/// gives placeholders.
#[rs_class("testing.rust.RsPlaceholder", s_eq_hash_kind = "TreeNode", placeholder)]
pub struct RsPlaceholderObj {
    base: Object,
    pub count: i64,
    pub ratio: f64,
    pub flag: bool,
    pub label: tvm_ffi::String,
    pub tags: Array<tvm_ffi::String>,
    pub table: Map<tvm_ffi::String, i64>,
    pub point: Option<RsPoint>,
}

def_object_ref!(pub RsPlaceholder, ObjectRef, RsPlaceholderObj);

/// A subtype with `placeholder`, whose placeholder starts from its parent's.
#[rs_class("testing.rust.RsPlaceholderChild", final, placeholder)]
pub struct RsPlaceholderChildObj {
    base: RsPlaceholderObj,
    pub z: i32,
}

def_object_ref!(pub RsPlaceholderChild, RsPlaceholder, RsPlaceholderChildObj);

/// A crate that reaches tvm-ffi only through a re-export names it with `crate`.
mod reexported {
    // The bare `tvm_ffi` here is not the crate, as in a crate without the dependency.
    #[allow(dead_code)]
    mod tvm_ffi {}

    #[::tvm_ffi::rs_class("testing.rust.RsReexported", crate = ::tvm_ffi)]
    pub struct ReexportedObj {
        base: ::tvm_ffi::Object,
        pub v: i64,
    }

    ::tvm_ffi::def_object_ref!(
        pub Reexported,
        ::tvm_ffi::object::ObjectRef,
        ReexportedObj,
        crate = ::tvm_ffi
    );
}

fn point(x: i64, label: &str) -> RsPoint {
    RsPointObj::new(Object::new(), x, label).into()
}

fn point3(x: i64, z: i64) -> RsPoint3 {
    let tags = Array::new(vec!["a".into(), "b".into()]);
    RsPoint3Obj::new(RsPointObj::new(Object::new(), x, "p3"), z, tags, 7).into()
}

/// The dummy call keeps libtvm_ffi_testing, which registers `testing.TestCxxClassBase`, linked.
fn cxx_child(v_i64: i64, v_i32: i32, name: &str) -> RsCxxChild {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    RsCxxChildObj::new(TestCxxClassBaseObj::new(v_i64, v_i32), name).into()
}

fn type_info<N: ObjectCore>() -> &'static TVMFFITypeInfo {
    unsafe { &*TVMFFIGetTypeInfo(N::type_index()) }
}

fn field_names<N: ObjectCore>() -> Vec<&'static str> {
    fields(type_info::<N>())
        .iter()
        .map(|f| f.name.as_str())
        .collect()
}

fn fields(info: &TVMFFITypeInfo) -> &[TVMFFIFieldInfo] {
    if info.num_fields == 0 {
        return &[]; // `fields` may be null
    }
    unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) }
}

fn call(name: &str, args: &[AnyView]) -> Any {
    Function::get_global(name)
        .unwrap()
        .call_packed(args)
        .unwrap()
}

fn structural_equal<T: AnyCompatible + Clone>(a: &T, b: &T) -> bool {
    let (a, b) = (Any::from(a.clone()), Any::from(b.clone()));
    let args = [
        AnyView::from(&a),
        AnyView::from(&b),
        AnyView::from(&false),
        AnyView::from(&false),
    ];
    bool::try_from(call("ffi.StructuralEqual", &args)).unwrap()
}

fn deep_copy<T: AnyCompatible + Clone + TryFrom<Any, Error = Error>>(value: &T) -> T {
    let any = Any::from(value.clone());
    T::try_from(call("ffi.DeepCopy", &[AnyView::from(&any)])).unwrap()
}

// A type registers on first use, and the runtime does not guard its type table
// against readers on other threads, so each test registers every type here
// before it reads the table.
fn register_types() {
    static REGISTERED: std::sync::Once = std::sync::Once::new();
    REGISTERED.call_once(|| {
        RsPointObj::type_index();
        RsPoint3Obj::type_index();
        RsMarkerObj::type_index();
        RsScalarsObj::type_index();
        RsCxxChildObj::type_index();
        RsPlaceholderObj::type_index();
        RsPlaceholderChildObj::type_index();
        reexported::ReexportedObj::type_index();
    });
}

#[test]
fn test_fields_are_registered_with_their_flags() {
    register_types();
    assert_eq!(field_names::<RsPointObj>(), ["x", "label"]);
    let [x, label] = fields(type_info::<RsPointObj>()) else {
        unreachable!()
    };
    assert_eq!(x.flags, FieldFlags::NONE.bits());
    assert_eq!(x.offset, std::mem::offset_of!(RsPointObj, x) as i64);
    assert_eq!(x.doc.as_str(), "The x coordinate.");
    assert_eq!(
        label.flags,
        (FieldFlags::WRITABLE | FieldFlags::SEQ_HASH_IGNORE).bits()
    );
    assert_eq!(label.doc.as_str(), "A label.");
    let metadata = unsafe { &*type_info::<RsPointObj>().metadata };
    assert_eq!(
        metadata.total_size as usize,
        std::mem::size_of::<RsPointObj>()
    );
    assert_eq!(
        metadata.structural_eq_hash_kind,
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode as i32
    );

    // A subtype registers its own fields, private ones too, and inherits the kind.
    assert_eq!(field_names::<RsPoint3Obj>(), ["z", "tags", "tag"]);
    let kind = |info: &TVMFFITypeInfo| unsafe { (*info.metadata).structural_eq_hash_kind };
    assert_eq!(
        kind(type_info::<RsPoint3Obj>()),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode as i32
    );
    // One without fields registers its metadata all the same.
    assert!(field_names::<RsMarkerObj>().is_empty());
    assert_eq!(
        kind(type_info::<RsMarkerObj>()),
        TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode as i32
    );
}

#[test]
fn test_reflection_registers_methods() {
    register_types();
    let info = type_info::<RsPointObj>();
    assert_eq!(info.num_methods, 1);
    let method = unsafe { &*info.methods };
    assert_eq!(method.name.as_str(), "with_label");
    assert_eq!(method.doc.as_str(), "The point with another label.");
    let with_label = Function::from_type_key_method("testing.rust.RsPoint", "with_label").unwrap();
    let relabeled = with_label
        .call_tuple((point(4, "a"), tvm_ffi::String::from("b")))
        .unwrap();
    let relabeled = RsPoint::try_from(relabeled).unwrap();
    assert_eq!((relabeled.x, relabeled.label.as_str()), (4, "b"));
    // The subtype has no methods of its own.
    assert_eq!(type_info::<RsPoint3Obj>().num_methods, 0);
}

#[test]
fn test_constructors_deref_and_upcast() {
    register_types();
    let p3 = point3(1, 3);
    // Through `RsPoint3Obj` to `RsPointObj`.
    assert_eq!((p3.x, p3.label.as_str()), (1, "p3"));
    assert_eq!((p3.z, p3.tags.len(), p3.tag), (3, 2, 7));

    let point3_obj: &RsPoint3Obj = &p3;
    let point_obj: &RsPointObj = point3_obj;
    assert_eq!(point_obj.x, 1);

    let upcast: RsPoint = p3.clone().into();
    assert!(upcast.same_as(&p3));
    assert_eq!(upcast.x, 1);
    let back = RsPoint3::try_from(Any::from(upcast)).unwrap();
    assert!(back.same_as(&p3));
    assert!(RsPoint3::try_from(Any::from(point(1, ""))).is_err());

    let marker = RsMarker::from(RsMarkerObj::new(RsPointObj::new(Object::new(), 4, "m")));
    assert_eq!(marker.x, 4);
    assert!(RsPoint::from(&marker).same_as(&marker));

    // A root type converts into `ObjectRef`, its parent type.
    let root = point(2, "r");
    assert!(ObjectRef::from(&root).same_as(&root));
}

#[test]
fn test_new_takes_literals_of_the_field_types() {
    register_types();
    let scalars = RsScalars::from(RsScalarsObj::new(Object::new(), 5_000_000_000, 1, 2, 0.5));
    assert_eq!(
        (scalars.big, scalars.count, scalars.port, scalars.ratio),
        (5_000_000_000, 1, 2, 0.5)
    );
    assert_eq!(
        field_names::<RsScalarsObj>(),
        ["big", "count", "port", "ratio"]
    );
}

#[test]
fn test_cpp_structural_equality_and_deep_copy_see_the_fields() {
    register_types();
    // `label` is ignored.
    assert!(structural_equal(&point(1, "a"), &point(1, "b")));
    assert!(!structural_equal(&point(1, "a"), &point(2, "a")));
    assert!(structural_equal(&point3(1, 2), &point3(1, 2)));
    assert!(!structural_equal(&point3(1, 2), &point3(0, 2)));

    let original = point3(5, 6);
    let copy = deep_copy(&original);
    assert!(!copy.same_as(&original));
    assert_eq!((copy.x, copy.label.as_str()), (5, "p3"));
    assert_eq!((copy.z, copy.tags.len(), copy.tag), (6, 2, 7));

    let repr = tvm_ffi::String::try_from(call(
        "ffi.ReprPrint",
        &[AnyView::from(&Any::from(original))],
    ))
    .unwrap();
    assert!(repr.as_str().contains("z=6"), "{}", repr.as_str());
}

#[test]
fn test_shallow_copy_and_finality_are_type_attributes() {
    register_types();
    let attr = |type_index, name| get_type_attr(type_index, name).unwrap();
    let is_final = |type_index| bool::try_from(attr(type_index, "__ffi_type_final__")).unwrap();
    assert!(!is_final(RsPointObj::type_index()));
    assert!(is_final(RsPoint3Obj::type_index()));
    assert!(is_final(RsMarkerObj::type_index()));

    let original = RsMarker::from(RsMarkerObj::new(RsPointObj::new(Object::new(), 3, "m")));
    let shallow_copy =
        Function::try_from(attr(RsMarkerObj::type_index(), "__ffi_shallow_copy__")).unwrap();
    let copy = RsMarker::try_from(shallow_copy.call_tuple((original.clone(),)).unwrap()).unwrap();
    assert!(!copy.same_as(&original));
    assert_eq!((copy.x, copy.label.as_str()), (3, "m"));

    // C++ cannot construct the type from its fields.
    assert!(Function::from_type_key_method("testing.rust.RsPoint", "__ffi_init__").is_err());
}

#[test]
fn test_subtype_of_a_cpp_type() {
    register_types();
    let child = cxx_child(7, 8, "c");
    assert_eq!((child.v_i64, child.v_i32, child.name.as_str()), (7, 8, "c"));
    // The C++ fields are reflected under the subtype.
    let v_i64 = FieldGetter::new(RsCxxChildObj::type_index(), "v_i64").unwrap();
    assert_eq!(v_i64.get::<_, i64>(&*child.base).unwrap(), 7);
    assert_eq!(field_names::<RsCxxChildObj>(), ["name"]);

    let base = TestCxxClassBase::from(child.clone());
    assert_eq!(base.v_i32, 8);

    assert!(structural_equal(&child, &cxx_child(7, 8, "c")));
    assert!(!structural_equal(&child, &cxx_child(7, 9, "c")));
    assert!(!structural_equal(&child, &cxx_child(7, 8, "d")));

    let copy = deep_copy(&child);
    assert!(!copy.same_as(&child));
    assert_eq!((copy.v_i64, copy.v_i32, copy.name.as_str()), (7, 8, "c"));
}

#[test]
fn test_crate_path_names_a_reexported_tvm_ffi() {
    register_types();
    let value = reexported::Reexported::from(reexported::ReexportedObj::new(Object::new(), 5));
    assert_eq!(value.v, 5);
    assert!(ObjectRef::from(&value).same_as(&value));
    assert_eq!(field_names::<reexported::ReexportedObj>(), ["v"]);
}

#[test]
fn test_placeholder_makes_the_object_from_its_parent_and_fields() {
    register_types();
    let child = RsPlaceholderChild::from(RsPlaceholderChildObj::placeholder());
    assert_eq!(
        (child.count, child.ratio, child.flag, child.z),
        (0, 0.0, false, 0)
    );
    assert_eq!(child.label.as_str(), "");
    assert!(child.tags.is_empty() && child.table.is_empty() && child.point.is_none());
    // A valid object of its type, which the reflection sees as any other.
    let other = RsPlaceholderChild::from(RsPlaceholderChildObj::placeholder());
    assert!(!other.same_as(&child));
    assert!(structural_equal(&child, &other));
    assert!(structural_equal(&deep_copy(&child), &child));
}
