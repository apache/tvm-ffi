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

//! Safe access to object reflection metadata.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::function_internal::{type_schema_metadata, AsPackedCallable};
use crate::object::TYPE_REGISTRATION;
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskIsStaticMethod, kTVMFFIFieldFlagBitMaskWritable,
};
use crate::tvm_ffi_sys::TVMFFISEqHashKind::kTVMFFISEqHashKindUnsupported;
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldGetter, TVMFFIFieldInfo, TVMFFIGetTypeAttrColumn,
    TVMFFIGetTypeInfo, TVMFFIMethodInfo, TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex,
    TVMFFITypeMetadata, TVMFFITypeRegisterAttr, TVMFFITypeRegisterField,
    TVMFFITypeRegisterMetadata, TVMFFITypeRegisterMethod,
};
use crate::type_traits::ContainerElement;
use crate::{Any, AnyCompatible, AnyView, Error, Function, ObjectCore, Result, TYPE_ERROR};

/// A registry-owned type-attribute column indexed by runtime type.
///
/// [`TypeAttrColumn::get`] returns owning copies. Registration must not race
/// with reads.
#[derive(Clone, Copy)]
pub struct TypeAttrColumn(NonNull<TVMFFITypeAttrColumn>);

// Type-attribute columns and their cells are registry-owned process-lifetime
// data. Once registration is complete, reading a cell does not mutate the
// registry and is safe from any thread.
unsafe impl Send for TypeAttrColumn {}
unsafe impl Sync for TypeAttrColumn {}

impl TypeAttrColumn {
    /// Look up a registered type-attribute column by name.
    pub fn new(name: &str) -> Option<Self> {
        unsafe {
            let name = TVMFFIByteArray::from_str(name);
            NonNull::new(TVMFFIGetTypeAttrColumn(&name).cast_mut()).map(Self)
        }
    }

    /// Return an owning copy of this attribute for `type_index`.
    pub fn get(self, type_index: i32) -> Option<Any> {
        let raw = self.get_raw(type_index)?;
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return None;
        }
        Some(Any::from(unsafe { AnyView::from_raw_ffi_any(raw) }))
    }

    pub(crate) unsafe fn from_non_null(pointer: NonNull<TVMFFITypeAttrColumn>) -> Self {
        Self(pointer)
    }

    pub(crate) fn as_ptr(self) -> *mut TVMFFITypeAttrColumn {
        self.0.as_ptr()
    }

    /// Copy one borrowed cell without taking ownership.
    pub(crate) fn get_raw(self, type_index: i32) -> Option<TVMFFIAny> {
        unsafe {
            let column = self.0.as_ref();
            let index = type_index - column.begin_index;
            if index < 0 || index >= column.size || column.data.is_null() {
                None
            } else {
                Some(*column.data.offset(index as isize))
            }
        }
    }
}

/// Look up one type attribute and copy it into an owning value.
pub fn get_type_attr(type_index: i32, attr_name: &str) -> Option<Any> {
    TypeAttrColumn::new(attr_name)?.get(type_index)
}

/// Resolves a reflected field once, then uses its registered C ABI getter.
#[derive(Clone, Copy)]
pub struct FieldGetter {
    owner_type_index: i32,
    owner_type_depth: i32,
    field_offset: i64,
    getter: TVMFFIFieldGetter,
}

impl FieldGetter {
    /// Resolve a reflected field declared by `type_index` or one of its bases.
    pub fn new(type_index: i32, field_name: &str) -> Result<Self> {
        let type_info = unsafe { TVMFFIGetTypeInfo(type_index) };
        if type_info.is_null() {
            return Err(Error::new(
                TYPE_ERROR,
                &format!("Cannot find type info for type_index={type_index}"),
                "",
            ));
        }

        let field = unsafe { find_field(type_info, field_name) }.ok_or_else(|| {
            let type_key = unsafe { (*type_info).type_key.as_str() };
            Error::new(
                TYPE_ERROR,
                &format!("Cannot find reflected field `{field_name}` in type `{type_key}`"),
                "",
            )
        })?;
        let field = unsafe { field.as_ref() };
        let getter = field.getter.ok_or_else(|| {
            Error::new(
                TYPE_ERROR,
                &format!("Reflected field `{}` has no getter", field.name.as_str()),
                "",
            )
        })?;
        Ok(Self {
            owner_type_index: type_index,
            owner_type_depth: unsafe { (*type_info).type_depth },
            field_offset: field.offset,
            getter,
        })
    }

    /// Read the field as an owning [`Any`].
    ///
    /// `object` may have the declared owner type or any registered subtype.
    pub fn get_any<N: ObjectCore>(&self, object: &N) -> Result<Any> {
        let object_pointer = std::ptr::from_ref(object);
        let header = object_pointer.cast::<TVMFFIObject>();
        let dynamic_type_index = unsafe { (*header).type_index };
        if !unsafe {
            is_type_or_subtype(
                dynamic_type_index,
                self.owner_type_index,
                self.owner_type_depth,
            )
        } {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "Cannot read a field of type_index={} from object type_index={dynamic_type_index}",
                    self.owner_type_index
                ),
                "",
            ));
        }

        let field_address = unsafe {
            object_pointer
                .cast::<u8>()
                .offset(self.field_offset as isize)
                .cast_mut()
                .cast::<c_void>()
        };
        let mut result = Any::new();
        if unsafe { (self.getter)(field_address, Any::as_data_ptr(&mut result)) } != 0 {
            return Err(Error::from_raised());
        }
        Ok(result)
    }

    /// Read and convert the field to `T`.
    pub fn get<N, T>(&self, object: &N) -> Result<T>
    where
        N: ObjectCore,
        T: TryFrom<Any, Error = Error>,
    {
        T::try_from(self.get_any(object)?)
    }
}

unsafe fn find_field(
    type_info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    field_name: &str,
) -> Option<NonNull<TVMFFIFieldInfo>> {
    // Prefer the most-derived declaration, then search nearest ancestors.
    if let Some(field) = find_field_at_level(type_info, field_name) {
        return Some(field);
    }
    for depth in (0..(*type_info).type_depth).rev() {
        let ancestor = *(*type_info).type_acenstors.add(depth as usize);
        if let Some(field) = find_field_at_level(ancestor, field_name) {
            return Some(field);
        }
    }
    None
}

unsafe fn find_field_at_level(
    type_info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    field_name: &str,
) -> Option<NonNull<TVMFFIFieldInfo>> {
    if type_info.is_null() || (*type_info).fields.is_null() {
        return None;
    }
    for index in 0..(*type_info).num_fields as usize {
        let field = (*type_info).fields.add(index);
        if (*field).name.as_str() == field_name {
            return NonNull::new(field.cast_mut());
        }
    }
    None
}

unsafe fn is_type_or_subtype(
    dynamic_type_index: i32,
    target_type_index: i32,
    target_type_depth: i32,
) -> bool {
    if dynamic_type_index == target_type_index {
        return true;
    }
    let dynamic_info = TVMFFIGetTypeInfo(dynamic_type_index);
    if dynamic_info.is_null()
        || (*dynamic_info).type_depth <= target_type_depth
        || (*dynamic_info).type_acenstors.is_null()
    {
        return false;
    }
    let ancestor = *(*dynamic_info)
        .type_acenstors
        .add(target_type_depth as usize);
    !ancestor.is_null() && (*ancestor).type_index == target_type_index
}

/// Registers the reflection of an object type defined in Rust, its fields,
/// methods and type metadata, as C++ `refl::ObjectDef<T>` does, so that other
/// languages read its fields and call its methods as for a C++ type.
///
/// `#[derive(Object)]` with `#[type_register]` creates one when it registers
/// the type: it registers the fields marked `#[def_ro]` or `#[def_rw]`, then
/// passes the `ObjectDef` to the function that `#[type_reflection(path)]`
/// names, which registers methods. This happens once, on first use of
/// `type_index()`, before any object of the type exists, as C++ registers
/// reflection while its library loads.
///
/// ```rust,ignore
/// use tvm_ffi::derive::{Object, ObjectRef};
/// use tvm_ffi::reflection::ObjectDef;
/// use tvm_ffi::{Object, ObjectArc, Result};
///
/// #[repr(C)]
/// #[derive(Object)]
/// #[type_key = "doc.Point"]
/// #[type_register]
/// #[type_reflection(PointObj::register_reflection)]
/// pub struct PointObj {
///     object: Object,
///     #[def_ro(doc = "The x coordinate")]
///     x: i64,
///     #[def_ro]
///     y: i64,
/// }
///
/// #[repr(C)]
/// #[derive(ObjectRef, Clone)]
/// pub struct Point {
///     data: ObjectArc<PointObj>,
/// }
///
/// impl PointObj {
///     fn register_reflection(def: &mut ObjectDef<Self>) {
///         def.def("norm1", |p: Point| -> Result<i64> { Ok(p.data.x.abs() + p.data.y.abs()) }, "")
///             .def_static(
///                 "origin",
///                 || -> Result<Point> {
///                     Ok(Point { data: ObjectArc::new(PointObj { object: Object::new(), x: 0, y: 0 }) })
///                 },
///                 "The origin.",
///             );
///     }
/// }
/// ```
///
/// Each method's metadata records its type schema, and each field's records
/// the field type's, as C++ does. The type's metadata records its size, and
/// `__ffi_type_final__` its finality.
///
/// Registration fails with a panic, as C++ `ObjectDef` throws: for instance,
/// for a name registered twice. The registering function must not use
/// `T::type_index()` or create objects of type `T`, since `T` is not
/// registered until it returns.
pub struct ObjectDef<T: ObjectCore> {
    type_index: i32,
    _marker: PhantomData<fn() -> T>,
}

/// Holds [`TYPE_REGISTRATION`] around one update of the type table.
fn register<R>(update: impl FnOnce() -> R) -> R {
    let _registering = TYPE_REGISTRATION.lock().unwrap_or_else(|e| e.into_inner());
    update()
}

impl<T: ObjectCore> ObjectDef<T> {
    /// Start the reflection of `T`, registered as `type_index`, by
    /// registering its type metadata, unless the type already has metadata,
    /// in which case another copy of this type, in another library, has
    /// registered its reflection and this returns `None`.
    ///
    /// # Safety
    ///
    /// `type_index` must be the type index `T` registered.
    #[doc(hidden)]
    pub unsafe fn begin(type_index: i32) -> Option<Self> {
        let metadata = TVMFFITypeMetadata {
            doc: TVMFFIByteArray::from_str(""),
            creator: None,
            total_size: i32::try_from(std::mem::size_of::<T>())
                .expect("object type too large to register"),
            structural_eq_hash_kind: kTVMFFISEqHashKindUnsupported as i32,
        };
        let registered = register(|| {
            if !(*TVMFFIGetTypeInfo(type_index)).metadata.is_null() {
                return false;
            }
            check_registration(
                TVMFFITypeRegisterMetadata(type_index, &metadata),
                T::TYPE_KEY,
            );
            true
        });
        registered.then_some(Self {
            type_index,
            _marker: PhantomData,
        })
    }

    /// Register a field of type `V` at byte `offset` from the object header,
    /// with a getter and a setter, as C++ `ObjectDef::def_ro` and `def_rw`
    /// do. A writable field can be set from other languages.
    ///
    /// # Safety
    ///
    /// `T` must have a field of type `V` at `offset`, and a writable field
    /// must be one that other languages may set while Rust holds a reference
    /// to the object.
    #[doc(hidden)]
    pub unsafe fn field<V: ContainerElement>(
        &mut self,
        name: &str,
        doc: &str,
        offset: usize,
        writable: bool,
    ) -> &mut Self {
        let metadata = type_schema_metadata(&V::container_type_schema())
            .unwrap_or_else(|error| panic!("{error}"));
        let info = TVMFFIFieldInfo {
            name: TVMFFIByteArray::from_str(name),
            doc: TVMFFIByteArray::from_str(doc),
            metadata: TVMFFIByteArray::from_str(&metadata),
            flags: if writable {
                kTVMFFIFieldFlagBitMaskWritable as i64
            } else {
                0
            },
            size: std::mem::size_of::<V>() as i64,
            alignment: std::mem::align_of::<V>() as i64,
            offset: offset as i64,
            getter: Some(field_getter::<V>),
            // As in C++, a read-only field has a setter too, for
            // deserialization and generated initializers.
            setter: field_setter::<V> as *mut c_void,
            default_value_or_factory: TVMFFIAny::new(),
            field_static_type_index: V::CONTAINER_FIELD_STATIC_TYPE_INDEX,
        };
        register(|| {
            check_registration(TVMFFITypeRegisterField(self.type_index, &info), T::TYPE_KEY)
        });
        self
    }

    /// Define a method, whose first parameter is the object, as C++
    /// `ObjectDef::def` does.
    ///
    /// # Arguments
    /// * `name` - The name of the method
    /// * `func` - The method, a typed function as for [`Function::from_typed`]
    /// * `doc` - The doc string of the method; an empty one records none
    pub fn def<F, I, O>(&mut self, name: &str, func: F, doc: &str) -> &mut Self
    where
        F: AsPackedCallable<I, O> + 'static,
    {
        self.method(name, func, doc, false)
    }

    /// Define a static method, as C++ `ObjectDef::def_static` does.
    ///
    /// # Arguments
    /// * `name` - The name of the method
    /// * `func` - The method, a typed function as for [`Function::from_typed`]
    /// * `doc` - The doc string of the method; an empty one records none
    pub fn def_static<F, I, O>(&mut self, name: &str, func: F, doc: &str) -> &mut Self
    where
        F: AsPackedCallable<I, O> + 'static,
    {
        self.method(name, func, doc, true)
    }

    fn method<F, I, O>(&mut self, name: &str, func: F, doc: &str, is_static: bool) -> &mut Self
    where
        F: AsPackedCallable<I, O> + 'static,
    {
        let metadata =
            type_schema_metadata(&F::type_schema()).unwrap_or_else(|error| panic!("{error}"));
        let func = Function::from_typed(func);
        let mut method = TVMFFIAny::new();
        unsafe {
            <Function as AnyCompatible>::copy_to_any_view(&func, &mut method);
            let info = TVMFFIMethodInfo {
                name: TVMFFIByteArray::from_str(name),
                doc: TVMFFIByteArray::from_str(doc),
                metadata: TVMFFIByteArray::from_str(&metadata),
                flags: if is_static {
                    kTVMFFIFieldFlagBitMaskIsStaticMethod as i64
                } else {
                    0
                },
                method,
            };
            // The type table keeps its own reference to the method.
            register(|| {
                check_registration(
                    TVMFFITypeRegisterMethod(self.type_index, &info),
                    T::TYPE_KEY,
                )
            });
        }
        self
    }

    /// Finish the reflection of `T` by registering `__ffi_type_final__`, as
    /// C++ `ObjectDef` does when it is dropped.
    #[doc(hidden)]
    pub fn finish(self) {
        let mut value = TVMFFIAny::new();
        unsafe {
            let name = TVMFFIByteArray::from_str("__ffi_type_final__");
            <bool as AnyCompatible>::copy_to_any_view(&T::TYPE_FINAL, &mut value);
            register(|| {
                check_registration(
                    TVMFFITypeRegisterAttr(self.type_index, &name, &value),
                    T::TYPE_KEY,
                )
            });
        }
    }
}

/// Panics with the raised error if a registration of `type_key` failed.
fn check_registration(ret_code: i32, type_key: &str) {
    if ret_code != 0 {
        let error = Error::from_raised();
        panic!("Failed to register the reflection of {type_key}: {error}");
    }
}

/// The getter of a reflected field of type `V`, as C++
/// `ReflectionDefBase::FieldGetter` defines it.
unsafe extern "C" fn field_getter<V: ContainerElement>(
    field: *mut c_void,
    result: *mut TVMFFIAny,
) -> i32 {
    let mut view = TVMFFIAny::new();
    V::container_copy_to_any_view(&*field.cast::<V>(), &mut view);
    *result = Any::into_raw_ffi_any(Any::from(AnyView::from_raw_ffi_any(view)));
    0
}

/// The setter of a reflected field of type `V`, as C++
/// `ReflectionDefBase::FieldSetter` defines it.
unsafe extern "C" fn field_setter<V: ContainerElement>(
    field: *mut c_void,
    value: *const TVMFFIAny,
) -> i32 {
    let value = &*value;
    let converted = if V::container_check_any_strict(value) {
        Ok(V::container_copy_from_any_view_after_check(value))
    } else {
        V::container_try_cast_from_any_view(value).map_err(|()| {
            Error::new(
                TYPE_ERROR,
                &format!(
                    "Cannot convert from type `{}` to `{}`",
                    V::container_get_mismatch_type_info(value),
                    V::container_type_str()
                ),
                "",
            )
        })
    };
    match converted {
        Ok(converted) => {
            *field.cast::<V>() = converted;
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}
