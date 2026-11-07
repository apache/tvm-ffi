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

//! Reads of registered types: methods, fields and type attributes.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{PoisonError, RwLock};

use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldFlagBitMask, TVMFFIFieldGetter, TVMFFIFieldInfo,
    TVMFFIGetTypeAttrColumn, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeAttrColumn,
    TVMFFITypeIndex, TVMFFITypeInfo, TVMFFITypeKeyToIndex,
};
use crate::type_traits::AnyCompatible;
use crate::{Any, AnyView, Array, Error, Function, ObjectCore, Result, TYPE_ERROR};

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

    /// The attribute of `type_index`, as an owned value, or `None` if unset.
    pub fn get(self, type_index: i32) -> Option<Any> {
        let raw = self.get_raw_ffi_any(type_index)?;
        if raw.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return None;
        }
        Some(Any::from(unsafe { AnyView::from_raw_ffi_any(raw) }))
    }

    /// The raw cell of `type_index`'s attribute, borrowed from the registry.
    ///
    /// The cell does not retain its value: the value must stay registered while
    /// the cell is used, and registration must not race with reads. Use
    /// [`Self::get`] for an owned [`Any`].
    ///
    /// Returns `None` when the index is outside the column or the column has no
    /// data. Unlike [`Self::get`], it returns a `None`-valued cell as `Some`.
    #[inline(always)]
    pub fn get_raw_ffi_any(self, type_index: i32) -> Option<TVMFFIAny> {
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

/// The type key registered for `type_index`, such as `"ffi.Array"`, or `None`
/// if no type has that index.
///
/// ```
/// use tvm_ffi::collections::array::ArrayObj;
/// use tvm_ffi::{reflection, ObjectCore};
///
/// assert_eq!(reflection::type_key(ArrayObj::type_index()), Some("ffi.Array"));
/// assert_eq!(reflection::type_key(-1), None);
/// ```
pub fn type_key(type_index: i32) -> Option<&'static str> {
    if !is_registered(type_index) {
        return None;
    }
    // SAFETY: the index is registered, and type information lives as long as
    // the process.
    let info: &'static TVMFFITypeInfo = unsafe { &*TVMFFIGetTypeInfo(type_index) };
    // SAFETY: a type key is UTF-8 text.
    Some(unsafe { info.type_key.as_str() })
}

/// The default value `T` registers for its field `name`, or `None` if `T`
/// registers no such field, the field has no default, or its default comes
/// from a factory.
///
/// Only fields `T` registers itself are searched, not those of its ancestors.
pub fn field_default<T: ObjectCore>(name: &str) -> Option<Any> {
    // SAFETY: `ObjectCore` types are registered, and type information lives
    // as long as the process.
    let info: &'static TVMFFITypeInfo = unsafe { &*TVMFFIGetTypeInfo(T::type_index()) };
    // SAFETY: a registered type lists `num_fields` initialized fields.
    let fields = unsafe {
        crate::util::slice_from_nullable_raw_parts(info.fields, info.num_fields as usize)
    };
    // SAFETY: a field name is UTF-8 text.
    let field = fields
        .iter()
        .find(|field| unsafe { field.name.as_str() } == name)?;
    let has_default = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskHasDefault as i64;
    let from_factory = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskDefaultFromFactory as i64;
    if field.flags & has_default == 0 || field.flags & from_factory != 0 {
        return None;
    }
    // SAFETY: the registry keeps the default alive; the view only borrows it
    // for the copy.
    Some(Any::from(unsafe {
        AnyView::from_raw_ffi_any(field.default_value_or_factory)
    }))
}

/// Whether a type is registered with index `type_index`. The registry aborts
/// the process on a lookup of any other index, so the registered indices are
/// listed once and again whenever an index is missing, as types register over
/// time.
fn is_registered(type_index: i32) -> bool {
    static REGISTERED: RwLock<Vec<bool>> = RwLock::new(Vec::new());
    let Ok(index) = usize::try_from(type_index) else {
        return false;
    };
    let read = REGISTERED.read().unwrap_or_else(PoisonError::into_inner);
    if read.get(index).copied().unwrap_or(false) {
        return true;
    }
    drop(read);
    let keys = crate::cached_global_func!("ffi.GetRegisteredTypeKeys")
        .call_tuple(())
        .and_then(Array::<crate::String>::try_from);
    let mut registered = REGISTERED.write().unwrap_or_else(PoisonError::into_inner);
    for key in keys.iter().flat_map(|keys| keys.iter()) {
        let mut key_index = 0;
        // SAFETY: the byte array borrows `key` for the call.
        let found = unsafe {
            TVMFFITypeKeyToIndex(&TVMFFIByteArray::from_str(key.as_str()), &mut key_index) == 0
        };
        if let (true, Ok(key_index)) = (found, usize::try_from(key_index)) {
            if registered.len() <= key_index {
                registered.resize(key_index + 1, false);
            }
            registered[key_index] = true;
        }
    }
    registered.get(index).copied().unwrap_or(false)
}

/// Look up a reflected method of a type by type key and method name.
///
/// Same as [`get_method_by_index`], but resolves `type_key` to a type index
/// first.
///
/// # Arguments
///
/// * `type_key` - The type key of the type that owns the method.
/// * `method_name` - The name of the method.
///
/// # Returns
///
/// The reflected method.
///
/// # Errors
///
/// Returns an error if the type or the method does not exist.
pub fn get_method(type_key: &str, method_name: &str) -> Result<Function> {
    unsafe {
        let type_key_arg = TVMFFIByteArray::from_str(type_key);
        let mut type_index: i32 = 0;
        crate::check_safe_call!(TVMFFITypeKeyToIndex(&type_key_arg, &mut type_index))?;
        get_method_by_index(type_index, method_name)
    }
}

/// Look up a reflected method of a type by type index and method name, as
/// [`get_method`] does by type key.
///
/// Reflected methods, registered in any language (such as with
/// [`ObjectDef::def`](super::ObjectDef::def)), live in the type's method
/// table rather than the global function table. The constructor is reachable
/// under the reserved name `__ffi_init__`. For instance methods, the first
/// argument is the object itself.
///
/// `type_index` must be a registered type index, obtained for example from a
/// live object via `Any::type_index` or from a type key; the C API treats an
/// unregistered index as a fatal error.
///
/// # Arguments
///
/// * `type_index` - The type index of the type that owns the method.
/// * `method_name` - The name of the method.
///
/// # Returns
///
/// The reflected method.
///
/// # Errors
///
/// Returns an error if the type has no method `method_name`.
pub fn get_method_by_index(type_index: i32, method_name: &str) -> Result<Function> {
    unsafe {
        let type_info = TVMFFIGetTypeInfo(type_index);
        if type_info.is_null() {
            crate::bail!(
                crate::error::TYPE_ERROR,
                "Cannot find type info for type_index={}",
                type_index
            );
        }
        let type_info = &*type_info;
        for i in 0..type_info.num_methods as usize {
            let method_info = &*type_info.methods.add(i);
            if method_info.name.as_str() != method_name {
                continue;
            }
            if !<Function as AnyCompatible>::check_any_strict(&method_info.method) {
                crate::bail!(
                    crate::error::TYPE_ERROR,
                    "Method `{}` of type `{}` is not a Function",
                    method_name,
                    type_info.type_key.as_str()
                );
            }
            // the table entry stores the method as a non-owning AnyView;
            // copy out a strong reference
            return Ok(<Function as AnyCompatible>::copy_from_any_view_after_check(
                &method_info.method,
            ));
        }
        crate::bail!(
            crate::error::TYPE_ERROR,
            "Cannot find method `{}` of type `{}`",
            method_name,
            type_info.type_key.as_str()
        );
    }
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
            // SAFETY: a field name is UTF-8 text.
            let name = unsafe { field.name.as_str() };
            Error::new(
                TYPE_ERROR,
                &format!("Reflected field `{name}` has no getter"),
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
        if unsafe { (self.getter)(field_address, Any::as_raw_ffi_any_mut(&mut result)) } != 0 {
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
