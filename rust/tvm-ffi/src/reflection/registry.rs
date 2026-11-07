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
//! [`ObjectDef`] fills the type table entries that every language reads: the
//! type metadata, one record per field with C ABI getter and setter, one
//! record per method, and the `__ffi_shallow_copy__` and `__ffi_type_final__`
//! type attributes; for an `#[rs_object]` type also the creator, and from it
//! the `__ffi_new__` and `__ffi_init__` type attributes. Through them
//! structural equality, hashing and mutation, `repr`, deep copy, JSON graphs
//! and Python's generated classes reach the fields and methods of a Rust type,
//! and construct it.
//!
//! Only `#[derive(Object)]` creates an [`ObjectDef`], for a `#[type_register]`
//! type with `#[reflect]` fields or a `#[type_reflection]` function; the
//! derive guarantees that each registered field's type and offset match. The
//! `#[type_reflection]` function receives the builder to register methods
//! with [`ObjectDef::def`] and [`ObjectDef::def_static`].

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::BitOr;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::NativeFunction;
use crate::function_internal::AsPackedCallable;
use crate::object::{
    lock_type_registration, Object, ObjectArc, ObjectCore, ObjectRef, ObjectRefCore,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldFlagBitMask, TVMFFIFieldInfo, TVMFFIFieldSetter,
    TVMFFIGetTypeInfo, TVMFFIMethodInfo, TVMFFIObject, TVMFFIObjectCreator, TVMFFIObjectHandle,
    TVMFFISEqHashKind, TVMFFITypeIndex, TVMFFITypeMetadata, TVMFFITypeRegisterAttr,
    TVMFFITypeRegisterField, TVMFFITypeRegisterMetadata, TVMFFITypeRegisterMethod,
};
use crate::type_traits::{type_schema_metadata, AnyCompatible};
use crate::{Any, Error, Function, Result, INTERNAL_ERROR, TYPE_ERROR, VALUE_ERROR};

/// A copy of an object's own fields under a fresh object header, which
/// `__ffi_shallow_copy__` returns; fields that are references share their
/// objects, as `Clone` of an `Arc` does.
///
/// `#[derive(Object)]` implements it for a type with `#[reflect]` fields by
/// copying the parent with its `ShallowCopy` and cloning every other field. A
/// Rust type whose parent is defined in another language needs that parent to
/// implement it.
pub trait ShallowCopy {
    /// The copy, under a fresh object header.
    fn shallow_copy(&self) -> Self;
}

impl ShallowCopy for Object {
    fn shallow_copy(&self) -> Self {
        Object::new()
    }
}

/// How the structural engines treat instances of a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SEqHashKind {
    /// Rejected by structural equality and hashing; the default.
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

/// The reflection flags of one field, combined with `|`.
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

/// Registers the reflection of the object type `N`: the metadata, each
/// `#[reflect]` field, the methods and the type attributes.
///
/// `#[derive(Object)]` creates one while it registers a `#[type_register]`
/// type and passes it to the function that `#[type_reflection(path)]`, or the
/// `reflection = path` option of [`rs_object`](crate::rs_object), names, which
/// registers methods with [`ObjectDef::def`] and [`ObjectDef::def_static`].
///
/// That function runs under the registration lock, before `N::type_index()`
/// returns, so it must not call `type_index()` of `N` or of another Rust type
/// that is not registered yet, nor register attributes with [`TypeAttrDef`]
/// or [`EnumDef`](super::EnumDef). When `N` already has metadata in the type
/// table, every call is a no-op; registration fails if a previous one stopped
/// partway.
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::ObjectRef;
/// use tvm_ffi::reflection::{self, ObjectDef};
/// use tvm_ffi::{rs_object, Object, ObjectArc, Result};
///
/// #[rs_object("my_ext.MyPair", final, reflection = MyPairObj::reflect_methods)]
/// pub struct MyPairObj {
///     base_: Object,
///     pub first: i64,
///     pub second: i64,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// pub struct MyPair {
///     data: ObjectArc<MyPairObj>,
/// }
///
/// impl MyPairObj {
///     fn reflect_methods(def: &mut ObjectDef<Self>) {
///         def.def("sum", |p: MyPair| -> Result<i64> { Ok(p.first + p.second) }, "The sum.");
///     }
/// }
///
/// let pair = MyPair::from_object(MyPairObj::new(Object::new(), 1, 2));
/// let sum = reflection::get_method("my_ext.MyPair", "sum")?;
/// assert_eq!(i64::try_from(sum.call_tuple((&pair,))?)?, 3);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub struct ObjectDef<N: ObjectCore> {
    type_index: i32,
    fresh: bool,
    /// The creator registered in the metadata, from which `finish` registers
    /// `__ffi_new__` and `__ffi_init__`.
    creator: Option<TVMFFIObjectCreator>,
    /// The first failed method registration, which `finish` reports.
    error: Option<Error>,
    method_names: Vec<String>,
    _object: PhantomData<N>,
}

impl<N: ObjectCore> ObjectDef<N> {
    /// Registers `N`'s metadata, unless the type already has some, while the
    /// derive registers `N`'s type index, before `N::type_index()` returns.
    ///
    /// `kind` of `None` takes the nearest ancestor's kind. `creator` makes an
    /// object of `N` for reflection to fill (`unsafe_init_creator::<N>` for an
    /// `#[rs_object]`).
    ///
    /// # Safety
    ///
    /// The caller holds `TYPE_REGISTRATION` for the builder's lifetime, and
    /// `type_index` is `N`'s. `creator` returns an owned handle of an object of
    /// `N` that the field setters can fill.
    unsafe fn registering(
        type_index: i32,
        kind: Option<SEqHashKind>,
        creator: Option<TVMFFIObjectCreator>,
    ) -> Result<Self> {
        let info = TVMFFIGetTypeInfo(type_index);
        crate::ensure!(
            !info.is_null(),
            INTERNAL_ERROR,
            "no type info for `{}` (type index {type_index})",
            N::TYPE_KEY
        );
        let fresh = (*info).metadata.is_null();
        // `finish` writes `__ffi_type_final__` last: metadata without it is
        // left by a registration that failed partway.
        crate::ensure!(
            fresh
                || super::TypeAttrColumn::lookup("__ffi_type_final__")
                    .and_then(|column| column.get_raw_ffi_any(type_index))
                    .is_some_and(|cell| cell.type_index != TVMFFITypeIndex::kTVMFFINone as i32),
            INTERNAL_ERROR,
            "`{}` is partly registered: an earlier registration of it failed",
            N::TYPE_KEY
        );
        if fresh {
            let kind = match kind {
                Some(kind) => kind.raw(),
                None => inherited_kind(info),
            };
            let metadata = TVMFFITypeMetadata {
                doc: TVMFFIByteArray::from_str(""),
                creator,
                total_size: size_of::<N>() as i32,
                structural_eq_hash_kind: kind,
            };
            crate::check_safe_call!(TVMFFITypeRegisterMetadata(type_index, &metadata))?;
        }
        Ok(Self {
            type_index,
            fresh,
            creator,
            error: None,
            method_names: Vec::new(),
            _object: PhantomData,
        })
    }

    /// Registers the field of type `F` at byte offset `offset` in `N` under
    /// `name`, which must be unique in `N`, with its doc string (an empty one
    /// records none).
    ///
    /// # Safety
    ///
    /// `offset` is `std::mem::offset_of!` of a field of `N` whose type is `F`:
    /// the registered getter and setter read and write an `F` there.
    #[doc(hidden)]
    pub unsafe fn field<F: AnyCompatible>(
        self,
        name: &str,
        doc: &str,
        offset: usize,
        flags: FieldFlags,
    ) -> Result<Self> {
        if !self.fresh {
            return Ok(self);
        }
        // `__ffi_init__` leaves an `init_off` field as the creator made it, all-zero.
        if self.creator.is_some() && flags.bits() & FieldFlags::INIT_OFF.bits() != 0 {
            let zero = std::mem::MaybeUninit::<F>::zeroed();
            crate::ensure!(
                !unsafe { F::is_undefined(zero.as_ptr()) },
                VALUE_ERROR,
                "`init_off` field `{name}` of `{}` stays all-zero after `__ffi_init__`, which \
                 is no value of `{}`",
                N::TYPE_KEY,
                F::type_str()
            );
        }
        let type_str = F::type_str();
        crate::ensure!(
            F::has_field_layout(),
            TYPE_ERROR,
            "field `{name}` of `{}`: `Option` of a non-object value (`{type_str}`) has no FFI \
             field layout; use `tvm_ffi::Optional<T>`",
            N::TYPE_KEY
        );
        let metadata = type_schema_metadata(&F::type_schema());
        let info = TVMFFIFieldInfo {
            name: unsafe { TVMFFIByteArray::from_str(name) },
            doc: unsafe { TVMFFIByteArray::from_str(doc) },
            metadata: unsafe { TVMFFIByteArray::from_str(&metadata) },
            flags: flags.bits(),
            size: size_of::<F>() as i64,
            alignment: align_of::<F>() as i64,
            offset: offset as i64,
            getter: Some(field_getter::<F>),
            setter: field_setter::<F> as TVMFFIFieldSetter as *mut c_void,
            default_value_or_factory: TVMFFIAny::new(),
            field_static_type_index: field_static_type_index(&type_str),
        };
        // The table copies the strings before returning.
        crate::check_safe_call!(unsafe { TVMFFITypeRegisterField(self.type_index, &info) })?;
        Ok(self)
    }

    /// Registers `__ffi_shallow_copy__` from [`ShallowCopy`], which reflected
    /// mutation (copy on write through the field setters) needs.
    fn shallow_copy(self) -> Result<Self>
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

    /// Registers `func`, a typed function as for [`Function::from_typed`]
    /// whose first parameter is the object, as the method `name` of `N`,
    /// which must be unique in `N`, with its type schema and `doc`. A failed
    /// registration is reported when the type's registration finishes.
    pub fn def<F, I, O>(&mut self, name: &str, func: F, doc: &str) -> &mut Self
    where
        F: AsPackedCallable<I, O> + Send + Sync + 'static,
    {
        self.method(name, func, doc, false)
    }

    /// Registers `func` as the static method `name` of `N`, which takes no
    /// object; otherwise as [`ObjectDef::def`].
    pub fn def_static<F, I, O>(&mut self, name: &str, func: F, doc: &str) -> &mut Self
    where
        F: AsPackedCallable<I, O> + Send + Sync + 'static,
    {
        self.method(name, func, doc, true)
    }

    fn method<F, I, O>(&mut self, name: &str, func: F, doc: &str, is_static: bool) -> &mut Self
    where
        F: AsPackedCallable<I, O> + Send + Sync + 'static,
    {
        if !self.fresh || self.error.is_some() {
            return self;
        }
        if self.method_names.iter().any(|n| n == name) {
            self.error = Some(Error::new(
                VALUE_ERROR,
                &format!("duplicate method `{name}` of `{}`", N::TYPE_KEY),
                "",
            ));
            return self;
        }
        self.method_names.push(name.to_string());
        let metadata = type_schema_metadata(&F::type_schema());
        let func = Function::from_typed(func);
        let flags = if is_static {
            TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskIsStaticMethod as i64
        } else {
            0
        };
        let mut method = TVMFFIAny::new();
        // SAFETY: the strings outlive the call, which copies them, and the
        // table takes its own reference to the function.
        let registered = unsafe {
            <Function as AnyCompatible>::copy_to_any_view(&func, &mut method);
            let info = TVMFFIMethodInfo {
                name: TVMFFIByteArray::from_str(name),
                doc: TVMFFIByteArray::from_str(doc),
                metadata: TVMFFIByteArray::from_str(&metadata),
                flags,
                method,
            };
            crate::check_safe_call!(TVMFFITypeRegisterMethod(self.type_index, &info))
        };
        // A static `__ffi_init__` is also the type's `__ffi_init__` attribute, which
        // other languages call to construct it, as C++ `ObjectDef::def(init)` registers.
        let registered = registered.and_then(|()| {
            if is_static && name == "__ffi_init__" {
                register_type_attr(self.type_index, name, Any::from(func))
            } else {
                Ok(())
            }
        });
        if let Err(error) = registered {
            self.error = Some(Error::new(
                error.kind(),
                &format!("registering method `{name}`: {}", error.message()),
                "",
            ));
        }
        self
    }

    /// Registers `__ffi_new__` from the creator and
    /// `__ffi_init__` from the fields unless the type defines its own, then
    /// `__ffi_type_final__`, which no other registered metadata exposes; and
    /// reports the first failed method registration.
    fn finish(self) -> Result<()> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.fresh {
            return Ok(());
        }
        if let Some(creator) = self.creator {
            // Objects from this creator start zeroed: another language must set
            // every field before Rust code reads it.
            let new = Function::from_typed(move || -> Result<ObjectRef> {
                let mut handle: TVMFFIObjectHandle = std::ptr::null_mut();
                crate::check_safe_call!(unsafe { creator(&mut handle) })?;
                // SAFETY: the creator returns an owned handle, which `ObjectRef` views.
                Ok(unsafe { ObjectRef::from_data(ObjectArc::from_raw(handle.cast::<Object>())) })
            });
            register_type_attr(self.type_index, "__ffi_new__", Any::from(new))?;
            // C++ `ObjectDef` looks `ffi._RegisterFFIInit` up the same way, in case
            // it is not registered yet.
            let explicit_init = self.method_names.iter().any(|name| name == "__ffi_init__");
            if !explicit_init {
                if let Ok(register_init) = Function::get_global("ffi._RegisterFFIInit") {
                    register_init.call_tuple((self.type_index,))?;
                }
            }
        }
        register_type_attr(
            self.type_index,
            "__ffi_type_final__",
            Any::from(N::TYPE_FINAL),
        )
    }
}

/// Registers the reflection of `N` for `#[derive(Object)]`: its metadata,
/// then `fields` its fields, its `__ffi_shallow_copy__`, then `methods` its
/// methods, then its type attributes.
///
/// # Safety
///
/// As `ObjectDef::registering`; `fields` registers fields of `N` as
/// [`ObjectDef::field`] requires. Like every registration, it runs during
/// static initialization, before other threads read the type.
#[doc(hidden)]
pub unsafe fn register_object<N: ObjectCore + ShallowCopy>(
    type_index: i32,
    kind: Option<SEqHashKind>,
    creator: Option<TVMFFIObjectCreator>,
    fields: impl FnOnce(ObjectDef<N>) -> Result<ObjectDef<N>>,
    methods: impl FnOnce(&mut ObjectDef<N>),
) -> Result<()> {
    let mut def = fields(ObjectDef::registering(type_index, kind, creator)?)?.shallow_copy()?;
    methods(&mut def);
    def.finish()
}

/// Registers type attributes of the object type `T`, each as soon as
/// [`attr`](Self::attr) names it, for every language to read with
/// [`TypeAttrColumn`](crate::reflection::TypeAttrColumn).
///
/// As in C++, attributes register during static initialization, such as in a
/// [`static_init_block!`](crate::static_init_block), before other threads
/// read them: registrations from Rust lock each other out, but readers take
/// no lock.
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::Object;
/// use tvm_ffi::reflection::TypeAttrDef;
/// use tvm_ffi::reflection::TypeAttrColumn;
/// use tvm_ffi::{Object, ObjectCore};
///
/// #[repr(C)]
/// #[derive(Object)]
/// #[type_key = "my_ext.MyPair"]
/// #[type_register]
/// pub struct MyPairObj {
///     base_: Object,
/// }
///
/// TypeAttrDef::<MyPairObj>::new().attr("my_ext.arity", 2_i64)?;
///
/// let column = TypeAttrColumn::lookup("my_ext.arity").unwrap();
/// let arity: i64 = column.get(MyPairObj::type_index()).unwrap().try_into()?;
/// assert_eq!(arity, 2);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub struct TypeAttrDef<T: ObjectCore> {
    _type: PhantomData<fn() -> T>,
}

impl<T: ObjectCore> TypeAttrDef<T> {
    /// A builder for the type attributes of `T`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { _type: PhantomData }
    }

    /// Register the type attribute `name` of `T`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the attribute.
    /// * `value` - The value of the attribute, `Send + Sync` since every
    ///   thread reads it.
    ///
    /// # Returns
    ///
    /// The builder, for the next attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if `T` already has the attribute `name`, or a
    /// `TypeError` if `value` is an opaque pointer: register a native hook
    /// with [`native_fn_attr`](Self::native_fn_attr).
    pub fn attr(self, name: &str, value: impl Into<Any> + Send + Sync) -> Result<Self> {
        let value = value.into();
        ensure_not_opaque_ptr(name, &value)?;
        self.register(name, value)
    }

    /// Register the native hook `f` as the type attribute `name` of `T`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the attribute.
    /// * `f` - The hook.
    ///
    /// # Returns
    ///
    /// The builder, for the next attribute.
    ///
    /// # Errors
    ///
    /// Returns an error if `T` already has the attribute `name`.
    ///
    /// # Safety
    ///
    /// Every caller of the column `name` calls its hooks with exactly `Args`
    /// and reads `R`: a native caller calls the function pointer unchecked.
    pub unsafe fn native_fn_attr<Args, R>(
        self,
        name: &str,
        f: NativeFunction<Args, R>,
    ) -> Result<Self> {
        // SAFETY: the caller vouches for the readers of `name`.
        self.register(name, unsafe { f.into_any() })
    }

    fn register(self, name: &str, value: Any) -> Result<Self> {
        // Before the lock, which a first `type_index()` takes to register `T`.
        let type_index = T::type_index();
        let _registering = lock_type_registration();
        register_type_attr(type_index, name, value)?;
        Ok(self)
    }
}

/// Native callers of an attribute call an opaque pointer as a function, so
/// only [`NativeFunction`] may store one, through an `unsafe` registration.
pub(super) fn ensure_not_opaque_ptr(name: &str, value: &Any) -> Result<()> {
    crate::ensure!(
        value.type_index() != TVMFFITypeIndex::kTVMFFIOpaquePtr as i32,
        TYPE_ERROR,
        "attribute `{name}` cannot be an opaque pointer; register a native hook through \
         `NativeFunction`"
    );
    Ok(())
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
    let Some(node) = object.downcast_ref::<N>() else {
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
        unsafe { (*ObjectArc::as_ptr(ObjectRef::data(object)).cast::<TVMFFIObject>()).type_index };
    let info = unsafe { TVMFFIGetTypeInfo(type_index) };
    if info.is_null() {
        return format!("type index {type_index}");
    }
    unsafe { (*info).type_key.as_str().to_string() }
}

/// The caller holds the registration lock.
pub(super) fn register_type_attr(type_index: i32, name: &str, mut value: Any) -> Result<()> {
    let name = unsafe { TVMFFIByteArray::from_str(name) };
    // The table takes its own reference to the value.
    let value = Any::as_raw_ffi_any_mut(&mut value);
    crate::check_safe_call!(unsafe { TVMFFITypeRegisterAttr(type_index, &name, value) })
}

/// Reflection getter of a field of type `F`: writes an owned `Any` of the field's value.
/// `field` is the field's address inside the object.
unsafe extern "C" fn field_getter<F: AnyCompatible>(
    field: *mut c_void,
    result: *mut TVMFFIAny,
) -> i32 {
    let read = catch_unwind(AssertUnwindSafe(|| {
        let field = field.cast::<F>();
        // An undefined field reads as None, as a null `ObjectRef` does in C++.
        let value = if F::is_undefined(field) {
            Any::new()
        } else {
            (*field).to_any()
        };
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

/// Reflection setter of a field of type `F`: replaces the field with `value` converted to `F`.
/// `field` is the field's address inside the object.
unsafe extern "C" fn field_setter<F: AnyCompatible>(
    field: *mut c_void,
    value: *const TVMFFIAny,
) -> i32 {
    let written = catch_unwind(AssertUnwindSafe(|| {
        match F::try_cast_from_any_view(&*value) {
            Ok(converted) => {
                let field = field.cast::<F>();
                // An undefined field holds no value to drop.
                if F::is_undefined(field) {
                    field.write(converted);
                } else {
                    *field = converted;
                }
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

/// The static type index of a field, from the Rust type's `type_str()` (the names
/// `AnyCompatible` reports, such as `int`, `ffi.String`, `Array<my.Node>`).
// As C++ `TypeTraits<T>::field_static_type_index` states it.
pub(crate) fn field_static_type_index(type_str: &str) -> i32 {
    use TVMFFITypeIndex::*;
    /// The static index of a field holding any value (`kTVMFFIAny`).
    const ANY: i32 = -1;
    let origin = type_str.split('<').next().unwrap_or(type_str).trim();
    match origin {
        "None" => kTVMFFINone as i32,
        "int" => kTVMFFIInt as i32,
        "bool" => kTVMFFIBool as i32,
        "float" => kTVMFFIFloat as i32,
        "void*" => kTVMFFIOpaquePtr as i32,
        "DataType" => kTVMFFIDataType as i32,
        "Device" => kTVMFFIDevice as i32,
        "DLTensor*" => kTVMFFIDLTensorPtr as i32,
        // C++ `String`, `Bytes` and `BigInt` fields are typed `Any` statically (they have
        // inline forms), as is `Optional<T>`.
        "ffi.String" | "ffi.Bytes" | "BigInt" | "Optional" => ANY,
        "ffi.Function" => kTVMFFIFunction as i32,
        "ffi.Shape" => kTVMFFIShape as i32,
        "Array" => kTVMFFIArray as i32,
        "List" => kTVMFFIList as i32,
        // Any other object reference, `Map` included, is typed `Object` statically.
        _ => kTVMFFIStaticObjectBegin as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::field_static_type_index;
    use crate::type_traits::{type_schema_metadata, AnyCompatible};
    use crate::{Array, List, Map};

    #[test]
    fn field_static_type_indices_follow_the_cpp_type_traits() {
        assert_eq!(field_static_type_index("int"), 1);
        assert_eq!(field_static_type_index("ffi.String"), -1);
        assert_eq!(field_static_type_index("Optional<my.Node>"), -1);
        assert_eq!(field_static_type_index("my.Node"), 64);
        assert_eq!(field_static_type_index("Array<my.Node>"), 71);
        assert_eq!(field_static_type_index("List<int>"), 75);
        assert_eq!(field_static_type_index("Map<ffi.String, int>"), 64);
    }

    #[test]
    fn field_schemas_follow_the_cpp_type_traits() {
        assert_eq!(
            <Map<crate::String, i64>>::type_schema(),
            r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"int"}]}"#
        );
        assert_eq!(
            <List<i64>>::type_schema(),
            r#"{"type":"ffi.List","args":[{"type":"int"}]}"#
        );
        assert_eq!(
            <Option<Array<crate::String>>>::type_schema(),
            r#"{"type":"Optional","args":[{"type":"ffi.Array","args":[{"type":"ffi.String"}]}]}"#
        );
        assert_eq!(
            type_schema_metadata(r#"{"type":"int"}"#),
            r#"{"type_schema":"{\"type\":\"int\"}"}"#
        );
        // As C++ `EscapeStringJSON`, `/`, `\b` and DEL are escaped too.
        assert_eq!(
            type_schema_metadata("a/b\u{8}\u{7f}"),
            r#"{"type_schema":"a\/b\b\u007f"}"#
        );
    }
}
