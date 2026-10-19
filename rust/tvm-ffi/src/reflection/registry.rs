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

//! Registration of the reflection of an object type defined in Rust.
//!
//! [`ObjectDef`] fills the same type table entries as C++ `refl::ObjectDef<T>`
//! and Python `_register_py_class`: the type metadata, one record per field
//! with C ABI getter and setter, and the `__ffi_shallow_copy__` and
//! `__ffi_type_final__` type attributes. Through them the C++ structural
//! equality, hashing and mutation, `repr`, deep copy and Python's
//! generated classes reach the fields of a Rust type.
//!
//! Only `#[derive(Object)]` calls [`ObjectDef`], for a `#[type_register]`
//! type with `#[reflect]` fields; the derive guarantees that each registered
//! field's type and offset match.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::BitOr;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::object::{Object, ObjectArc, ObjectCore, ObjectRef, ObjectRefCore};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldFlagBitMask, TVMFFIFieldInfo, TVMFFIFieldSetter,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFISEqHashKind, TVMFFITypeIndex, TVMFFITypeMetadata,
    TVMFFITypeRegisterAttr, TVMFFITypeRegisterField, TVMFFITypeRegisterMetadata,
};
use crate::type_traits::AnyCompatible;
use crate::{Any, Error, Function, Result, INTERNAL_ERROR, TYPE_ERROR};

/// A copy of an object's own fields under a fresh object header (the C++ copy
/// constructor), which `__ffi_shallow_copy__` returns.
///
/// `#[derive(Object)]` implements it for a type with `#[reflect]` fields by
/// copying the parent with its `ShallowCopy` and cloning every other field. A
/// Rust type whose parent is bound from C++ needs that parent to implement it.
pub trait ShallowCopy {
    fn shallow_copy(&self) -> Self;
}

impl ShallowCopy for Object {
    fn shallow_copy(&self) -> Self {
        Object::new()
    }
}

/// How the structural engines treat instances of a type (C++ `_type_s_eq_hash_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SEqHashKind {
    /// Rejected by structural equality and hashing; the C++ default.
    Unsupported,
    /// Compared and hashed field by field.
    TreeNode,
    /// A free variable: bound by position inside a definition region, else by identity.
    FreeVar,
    /// Compared field by field, and once equal, mapped so later occurrences compare by mapping.
    DAGNode,
    /// A tree node whose identity is checked first, as a fast path.
    ConstTreeNode,
    /// Compared and hashed by identity only.
    UniqueInstance,
}

impl SEqHashKind {
    fn raw(self) -> i32 {
        let kind = match self {
            SEqHashKind::Unsupported => TVMFFISEqHashKind::kTVMFFISEqHashKindUnsupported,
            SEqHashKind::TreeNode => TVMFFISEqHashKind::kTVMFFISEqHashKindTreeNode,
            SEqHashKind::FreeVar => TVMFFISEqHashKind::kTVMFFISEqHashKindFreeVar,
            SEqHashKind::DAGNode => TVMFFISEqHashKind::kTVMFFISEqHashKindDAGNode,
            SEqHashKind::ConstTreeNode => TVMFFISEqHashKind::kTVMFFISEqHashKindConstTreeNode,
            SEqHashKind::UniqueInstance => TVMFFISEqHashKind::kTVMFFISEqHashKindUniqueInstance,
        };
        kind as i32
    }
}

/// The reflection flags of one field (C++ `refl::AttachFieldFlag` and friends), combined with `|`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldFlags(i64);

impl FieldFlags {
    /// A read-only field that takes part in structural equality and hashing (`def_ro`).
    pub const NONE: Self = Self(0);
    /// Assignable through reflection, as from Python (`def_rw`).
    pub const WRITABLE: Self = Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskWritable);
    /// Skipped by structural equality and hashing (`SEqHashIgnore`).
    pub const SEQ_HASH_IGNORE: Self =
        Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore);
    /// Entered as a pattern definition region (`SEqHashDefPattern`).
    pub const SEQ_HASH_DEF_PATTERN: Self =
        Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefPattern);
    /// Entered as a simple definition region (`SEqHashDefSimple`).
    pub const SEQ_HASH_DEF_SIMPLE: Self =
        Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefSimple);
    /// Left out of `repr`.
    pub const REPR_OFF: Self = Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskReprOff);
    /// Left out of `__eq__`.
    pub const COMPARE_OFF: Self =
        Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskCompareOff);
    /// Left out of `__hash__`.
    pub const HASH_OFF: Self = Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskHashOff);
    /// Not a constructor argument of the Python class.
    pub const INIT_OFF: Self = Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskInitOff);
    /// A keyword-only constructor argument of the Python class.
    pub const KW_ONLY: Self = Self::of(TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskKwOnly);

    const fn of(bit: TVMFFIFieldFlagBitMask) -> Self {
        Self(bit as i64)
    }

    /// The raw `TVMFFIFieldInfo::flags` value.
    pub fn bits(self) -> i64 {
        self.0
    }
}

impl BitOr for FieldFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Registers the reflection of the object type `N` (C++ `refl::ObjectDef<N>`)
/// for `#[derive(Object)]`: the metadata on construction, then each field in
/// call order, then the type attributes.
///
/// When `N` already has metadata in the type table, every call is a no-op.
#[doc(hidden)]
pub struct ObjectDef<N: ObjectCore> {
    type_index: i32,
    fresh: bool,
    _object: PhantomData<N>,
}

impl<N: ObjectCore> ObjectDef<N> {
    /// Registers `N`'s metadata, unless the type already has some, while the
    /// derive registers `N`'s type index, before `N::type_index()` returns.
    ///
    /// `kind` of `None` takes the nearest ancestor's kind, as a C++ subclass
    /// inherits `_type_s_eq_hash_kind`.
    ///
    /// # Safety
    ///
    /// The caller holds `TYPE_REGISTRATION` for the builder's lifetime, and
    /// `type_index` is `N`'s.
    pub unsafe fn registering(type_index: i32, kind: Option<SEqHashKind>) -> Result<Self> {
        let info = TVMFFIGetTypeInfo(type_index);
        crate::ensure!(
            !info.is_null(),
            INTERNAL_ERROR,
            "no type info for `{}` (type index {type_index})",
            N::TYPE_KEY
        );
        let fresh = (*info).metadata.is_null();
        if fresh {
            let kind = match kind {
                Some(kind) => kind.raw(),
                None => inherited_kind(info),
            };
            let metadata = TVMFFITypeMetadata {
                doc: TVMFFIByteArray::from_str(""),
                creator: None,
                total_size: size_of::<N>() as i32,
                structural_eq_hash_kind: kind,
            };
            crate::check_safe_call!(TVMFFITypeRegisterMetadata(type_index, &metadata))?;
        }
        Ok(Self {
            type_index,
            fresh,
            _object: PhantomData,
        })
    }

    /// Registers the field of type `F` at byte offset `offset` in `N` under
    /// `name`, which must be unique in `N`.
    ///
    /// # Safety
    ///
    /// `offset` is `std::mem::offset_of!` of a field of `N` whose type is `F`:
    /// the registered getter and setter read and write an `F` there.
    pub unsafe fn field<F: AnyCompatible>(
        self,
        name: &str,
        offset: usize,
        flags: FieldFlags,
    ) -> Result<Self> {
        if !self.fresh {
            return Ok(self);
        }
        let (static_type_index, schema) = field_type_info(&F::type_str());
        let metadata = format!(r#"{{"type_schema":{}}}"#, json_string(&schema));
        let info = TVMFFIFieldInfo {
            name: unsafe { TVMFFIByteArray::from_str(name) },
            doc: unsafe { TVMFFIByteArray::from_str("") },
            metadata: unsafe { TVMFFIByteArray::from_str(&metadata) },
            flags: flags.bits(),
            size: size_of::<F>() as i64,
            alignment: align_of::<F>() as i64,
            offset: offset as i64,
            getter: Some(field_getter::<F>),
            setter: field_setter::<F> as TVMFFIFieldSetter as *mut c_void,
            default_value_or_factory: TVMFFIAny::new(),
            field_static_type_index: static_type_index,
        };
        // The table copies the strings before returning.
        crate::check_safe_call!(unsafe { TVMFFITypeRegisterField(self.type_index, &info) })?;
        Ok(self)
    }

    /// Registers `__ffi_shallow_copy__`, which reflected mutation (copy on
    /// write through the field setters) needs; C++ `ObjectDef` registers the
    /// copy constructor.
    pub fn shallow_copy(self) -> Result<Self>
    where
        N: ShallowCopy,
    {
        if !self.fresh {
            return Ok(self);
        }
        let shallow_copy = Function::from_typed(|object: ObjectRef| shallow_copy_any::<N>(&object));
        register_type_attr(
            self.type_index,
            "__ffi_shallow_copy__",
            Any::from(shallow_copy),
        )?;
        Ok(self)
    }

    /// Registers `__ffi_type_final__`, which no other registered metadata
    /// exposes.
    pub fn finish(self) -> Result<()> {
        if !self.fresh {
            return Ok(());
        }
        register_type_attr(
            self.type_index,
            "__ffi_type_final__",
            Any::from(N::TYPE_FINAL),
        )
    }
}

/// The structural kind of the nearest ancestor of `info` with metadata.
unsafe fn inherited_kind(info: *const crate::tvm_ffi_sys::TVMFFITypeInfo) -> i32 {
    for depth in (0..(*info).type_depth).rev() {
        let ancestor = *(*info).type_acenstors.add(depth as usize);
        if !ancestor.is_null() && !(*ancestor).metadata.is_null() {
            return (*(*ancestor).metadata).structural_eq_hash_kind;
        }
    }
    SEqHashKind::Unsupported.raw()
}

/// `__ffi_shallow_copy__` of `N`: a copy of `object`, which must be an `N`, as a new object.
fn shallow_copy_any<N: ObjectCore + ShallowCopy>(object: &ObjectRef) -> Result<Any> {
    let Some(node) = object.as_node::<N>() else {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "__ffi_shallow_copy__ of `{}` applied to a `{}`",
                N::TYPE_KEY,
                object_type_key(object)
            ),
            "",
        ));
    };
    let copy = ObjectArc::new(node.shallow_copy());
    // SAFETY: `ObjectRef` is the view of any object; the allocation keeps its type index.
    let copy =
        unsafe { ObjectRef::from_data(ObjectArc::from_raw(ObjectArc::into_raw(copy).cast())) };
    Ok(Any::from(copy))
}

fn object_type_key(object: &ObjectRef) -> String {
    let type_index =
        unsafe { (*ObjectArc::as_raw(ObjectRef::data(object)).cast::<TVMFFIObject>()).type_index };
    let info = unsafe { TVMFFIGetTypeInfo(type_index) };
    if info.is_null() {
        return format!("type index {type_index}");
    }
    unsafe { (*info).type_key.as_str().to_string() }
}

fn register_type_attr(type_index: i32, name: &str, mut value: Any) -> Result<()> {
    let name = unsafe { TVMFFIByteArray::from_str(name) };
    // The table takes its own reference to the value.
    let value = unsafe { Any::as_data_ptr(&mut value) };
    crate::check_safe_call!(unsafe { TVMFFITypeRegisterAttr(type_index, &name, value) })
}

/// Reflection getter of a field of type `F`: an owned `Any` of the field's value, as the C++
/// `FieldGetter<T>` returns. `field` is the field's address inside the object.
unsafe extern "C" fn field_getter<F: AnyCompatible>(
    field: *mut c_void,
    result: *mut TVMFFIAny,
) -> i32 {
    let read = catch_unwind(AssertUnwindSafe(|| {
        let value: Any = (*field.cast::<F>()).to_any();
        std::ptr::write(result, Any::into_raw_ffi_any(value));
    }));
    match read {
        Ok(()) => 0,
        Err(_) => {
            let message = format!("panic while reading a field of type {}", F::type_str());
            Error::set_raised(&Error::new(INTERNAL_ERROR, &message, ""));
            -1
        }
    }
}

/// Reflection setter of a field of type `F`: replaces the field with `value` converted to `F`,
/// as the C++ `FieldSetter<T>` does. `field` is the field's address inside the object.
unsafe extern "C" fn field_setter<F: AnyCompatible>(
    field: *mut c_void,
    value: *const TVMFFIAny,
) -> i32 {
    let written = catch_unwind(AssertUnwindSafe(|| {
        match F::try_cast_from_any_view(&*value) {
            Ok(converted) => {
                *field.cast::<F>() = converted;
                None
            }
            Err(()) => Some(Error::new(
                TYPE_ERROR,
                &format!(
                    "expected a field value of type {}, got {}",
                    F::type_str(),
                    F::get_mismatch_type_info(&*value)
                ),
                "",
            )),
        }
    }));
    let error = match written {
        Ok(None) => return 0,
        Ok(Some(error)) => error,
        Err(_) => Error::new(
            INTERNAL_ERROR,
            &format!("panic while writing a field of type {}", F::type_str()),
            "",
        ),
    };
    Error::set_raised(&error);
    -1
}

/// The static type index and the JSON type schema of a field, from the Rust type's `type_str()`
/// (the names `AnyCompatible` reports, such as `int`, `ffi.String`, `Array<my.Node>`), as C++
/// `TypeTraits<T>::field_static_type_index` and `TypeTraits<T>::TypeSchema()` state them.
fn field_type_info(type_str: &str) -> (i32, String) {
    use TVMFFITypeIndex::*;
    /// C++ `TypeIndex::kTVMFFIAny`, the static index of a field holding any value.
    const ANY: i32 = -1;
    let (name, args) = split_generic(type_str);
    let args: Vec<String> = args.iter().map(|arg| field_type_info(arg).1).collect();
    let schema = |key: &str| match args.is_empty() {
        true => format!(r#"{{"type":"{key}"}}"#),
        false => format!(r#"{{"type":"{key}","args":[{}]}}"#, args.join(",")),
    };
    match name {
        "None" => (kTVMFFINone as i32, schema("None")),
        "int" => (kTVMFFIInt as i32, schema("int")),
        "bool" => (kTVMFFIBool as i32, schema("bool")),
        "float" => (kTVMFFIFloat as i32, schema("float")),
        "void*" => (kTVMFFIOpaquePtr as i32, schema("void*")),
        "DataType" => (kTVMFFIDataType as i32, schema("DataType")),
        "Device" => (kTVMFFIDevice as i32, schema("Device")),
        "DLTensor*" => (kTVMFFIDLTensorPtr as i32, schema("DLTensor*")),
        "Any" => (ANY, schema("Any")),
        // C++ `String`, `Bytes` and `BigInt` fields are typed `Any` statically (they have
        // inline forms), as is `Optional<T>`.
        "ffi.String" | "ffi.Bytes" => (ANY, schema(name)),
        "BigInt" => (ANY, schema("ffi.BigInt")),
        "Optional" => (ANY, schema("Optional")),
        "ffi.Function" => (kTVMFFIFunction as i32, schema(name)),
        "ffi.Shape" => (kTVMFFIShape as i32, schema(name)),
        "Array" => (kTVMFFIArray as i32, schema("ffi.Array")),
        "List" => (kTVMFFIList as i32, schema("ffi.List")),
        // Any other object reference, `Map` and `Dict` included, is typed `Object` statically.
        "Map" => (kTVMFFIStaticObjectBegin as i32, schema("ffi.Map")),
        "Dict" => (kTVMFFIStaticObjectBegin as i32, schema("ffi.Dict")),
        _ => (kTVMFFIStaticObjectBegin as i32, schema(name)),
    }
}

/// `Name<A, B<C>>` as (`Name`, [`A`, `B<C>`]); a plain name has no arguments.
fn split_generic(type_str: &str) -> (&str, Vec<&str>) {
    let Some(open) = type_str.find('<') else {
        return (type_str.trim(), Vec::new());
    };
    let Some(inner) = type_str[open + 1..].strip_suffix('>') else {
        return (type_str.trim(), Vec::new());
    };
    let mut args = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    for (index, byte) in inner.bytes().enumerate() {
        match byte {
            b'<' => depth += 1,
            b'>' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                args.push(inner[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    args.push(inner[start..].trim());
    (type_str[..open].trim(), args)
}

/// `text` as a JSON string literal.
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::{field_type_info, json_string, split_generic};

    #[test]
    fn generic_names_split_at_top_level_commas() {
        assert_eq!(split_generic("int"), ("int", vec![]));
        assert_eq!(split_generic("Array<my.Node>"), ("Array", vec!["my.Node"]));
        assert_eq!(
            split_generic("Map<ffi.String, Array<my.Node>>"),
            ("Map", vec!["ffi.String", "Array<my.Node>"])
        );
    }

    #[test]
    fn field_schemas_follow_the_cpp_type_traits() {
        assert_eq!(field_type_info("int"), (1, r#"{"type":"int"}"#.into()));
        assert_eq!(
            field_type_info("ffi.String"),
            (-1, r#"{"type":"ffi.String"}"#.into())
        );
        assert_eq!(
            field_type_info("my.Node"),
            (64, r#"{"type":"my.Node"}"#.into())
        );
        assert_eq!(
            field_type_info("Array<my.Node>"),
            (
                71,
                r#"{"type":"ffi.Array","args":[{"type":"my.Node"}]}"#.into()
            )
        );
        assert_eq!(
            field_type_info("Map<ffi.String, int>"),
            (
                64,
                r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"int"}]}"#.into()
            )
        );
        assert_eq!(
            field_type_info("Optional<my.Node>"),
            (
                -1,
                r#"{"type":"Optional","args":[{"type":"my.Node"}]}"#.into()
            )
        );
        assert_eq!(json_string(r#"{"type":"int"}"#), r#""{\"type\":\"int\"}""#);
    }
}
