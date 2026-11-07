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

//! Registration of enum values and their attributes, kept in each enum type's
//! registry (the `__ffi_enum__` type attribute) that every language shares.

use std::marker::PhantomData;
use std::mem::{align_of, size_of};

use super::registry::{ensure_not_opaque_ptr, register_type_attr};
use super::{get_method, NativeFunction, TypeAttrColumn};
use crate::any::TryFromTemp;
use crate::derive::{Object, ObjectRef};
use crate::object::{
    is_instance_of, lock_type_registration, Object, ObjectArc, ObjectCore, ObjectRef,
};
use crate::tvm_ffi_sys::{TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeInfo};
use crate::type_traits::AnyCompatible;
use crate::{Any, AnyView, EnumObj, EnumRef, Error, List, Result, String, TYPE_ERROR, VALUE_ERROR};

/// Registers a value of the enum type `E`, then its attributes.
///
/// [`new`](Self::new) registers the value at once, and
/// [`set_attr`](Self::set_attr) sets each attribute on it, which
/// [`EnumRef::attr_map`] reads. The value's object is an [`EnumObj`], so `E`'s
/// object type adds no fields to its parent `EnumObj`.
///
/// As in C++, values and attributes register during static initialization,
/// such as in a [`static_init_block!`](crate::static_init_block), before other
/// threads read them: registrations from Rust lock each other out, but
/// readers take no lock.
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::{Object, ObjectRef};
/// use tvm_ffi::reflection::EnumDef;
/// use tvm_ffi::{EnumObj, EnumRef, ObjectArc};
///
/// #[repr(C)]
/// #[derive(Object)]
/// #[type_key = "my_ext.Color"]
/// #[type_register]
/// #[type_final]
/// pub struct ColorObj {
///     base_: EnumObj,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// pub struct Color {
///     data: ObjectArc<ColorObj>,
/// }
///
/// impl EnumRef for Color {}
///
/// let red = EnumDef::<Color>::new("red")?.set_attr("my_ext.hex", 0xff0000_i64)?.value();
///
/// let hex = Color::attr_map::<i64>("my_ext.hex")?.unwrap();
/// assert_eq!(hex.get(&red)?, Some(0xff0000));
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub struct EnumDef<E: EnumRef> {
    value: E,
}

impl<E: EnumRef> EnumDef<E> {
    /// Register the value `name` of `E`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the value.
    ///
    /// # Returns
    ///
    /// The builder of the value's attributes.
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `name` is empty or `E` already has a value named
    /// `name`, and a `TypeError` if `E` is not an enum type, is an integer enum
    /// (`ffi.IntEnum`), whose values name their own indices, or registers an
    /// object size other than that of [`EnumObj`]: an enum type whose objects
    /// have fields of their own.
    pub fn new(name: &str) -> Result<Self> {
        const {
            assert!(
                size_of::<E::ContainerType>() == size_of::<EnumObj>()
                    && align_of::<E::ContainerType>() == align_of::<EnumObj>(),
                "an enum type registered from Rust adds no fields to `EnumObj`"
            )
        };
        let type_index = E::ContainerType::type_index();
        let type_key = E::ContainerType::TYPE_KEY;
        if name.is_empty() {
            return Err(Error::new(
                VALUE_ERROR,
                &format!("a value name of `{type_key}` must be non-empty"),
                "",
            ));
        }
        if !is_instance_of::<EnumObj>(type_index) || is_int_enum(type_index) {
            return Err(Error::new(
                TYPE_ERROR,
                &format!("`{type_key}` is not an enum type whose values Rust can register"),
                "",
            ));
        }
        // The assert above sees only the Rust mirror, which may be opaque: a
        // registered object size must be exactly that of the `EnumObj`
        // allocated below. A type registered without one (such as by
        // `#[derive(Object)]`) is laid out as its Rust struct.
        // SAFETY: the type is registered, and its information is static.
        let metadata = unsafe { (*TVMFFIGetTypeInfo(type_index)).metadata };
        // SAFETY: registered metadata is static.
        let total_size = unsafe { metadata.as_ref() }.map_or(0, |metadata| metadata.total_size);
        if total_size != 0 && total_size as usize != size_of::<EnumObj>() {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "`{type_key}` objects have fields beyond `ffi.Enum`, so Rust cannot \
                     register its values"
                ),
                "",
            ));
        }
        let _registering = lock_type_registration();
        let state = ensure_state(type_index)?;
        let index = state.entries.len() as i64;
        let key = String::from(name);
        let indexes = AnyView::from(&state.indexes);
        if dict_get(indexes, AnyView::from(&key))?.is_some()
            || dict_get(indexes, AnyView::from(&index))?.is_some()
        {
            return Err(Error::new(
                VALUE_ERROR,
                &format!("Enum `{type_key}` already has a value named `{name}`"),
                "",
            ));
        }
        let object = ObjectArc::into_raw(ObjectArc::new(EnumObj::new(index, name)));
        // SAFETY: `E`'s object is an `EnumObj` with `E`'s type index (checked
        // above), and the object is not shared yet. The deleter stays that of
        // `EnumObj`, which drops the same fields.
        let value = unsafe {
            (*object.cast_mut().cast::<TVMFFIObject>()).type_index = type_index;
            E::from_data(ObjectArc::from_raw(object.cast()))
        };
        state.entries.push(Any::from(value.clone()))?;
        dict_set(indexes, AnyView::from(&index), AnyView::from(&value))?;
        dict_set(indexes, AnyView::from(&key), AnyView::from(&value))?;
        Ok(Self { value })
    }

    /// Set the attribute `name` of the value.
    ///
    /// See also [`EnumRef::attr_map`].
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
    /// Returns a `TypeError` if `value` is an opaque pointer: set a native
    /// hook with [`set_native_fn_attr`](Self::set_native_fn_attr).
    pub fn set_attr(self, name: &str, value: impl Into<Any> + Send + Sync) -> Result<Self> {
        let value = value.into();
        ensure_not_opaque_ptr(name, &value)?;
        self.set_attr_any(name, value)
    }

    /// Set the native hook `f` as the attribute `name` of the value.
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
    /// # Safety
    ///
    /// Every caller of the attribute `name` calls its hooks with exactly
    /// `Args` and reads `R`: a native caller calls the function pointer
    /// unchecked.
    pub unsafe fn set_native_fn_attr<Args, R>(
        self,
        name: &str,
        f: NativeFunction<Args, R>,
    ) -> Result<Self> {
        self.set_attr_any(name, f.into_any())
    }

    fn set_attr_any(self, name: &str, value: Any) -> Result<Self> {
        // Before the lock, which a first `type_index()` takes to register `E`.
        let type_index = E::ContainerType::type_index();
        let _registering = lock_type_registration();
        let state = ensure_state(type_index)?;
        let attrs = AnyView::from(&state.attrs);
        let key = String::from(name);
        let column = match dict_get(attrs, AnyView::from(&key))? {
            Some(column) => column,
            None => {
                let column = crate::cached_global_func!("ffi.Dict").call_packed(&[])?;
                dict_set(attrs, AnyView::from(&key), AnyView::from(&column))?;
                column
            }
        };
        dict_set(
            AnyView::from(&column),
            AnyView::from(&self.value),
            AnyView::from(&value),
        )?;
        Ok(self)
    }

    /// The registered value.
    pub fn value(&self) -> E {
        self.value.clone()
    }
}

/// One attribute of the values of the enum type `E`, as `V`, returned by
/// [`EnumRef::attr_map`].
pub struct EnumAttrMap<E, V> {
    column: Any,
    _types: PhantomData<fn() -> (E, V)>,
}

impl<E: EnumRef, V: AnyCompatible> EnumAttrMap<E, V> {
    pub(crate) fn lookup(name: &str) -> Result<Option<Self>> {
        let Some(state) = enum_state(E::ContainerType::type_index())? else {
            return Ok(None);
        };
        let key = String::from(name);
        let column = dict_get(AnyView::from(&state.attrs), AnyView::from(&key))?;
        Ok(column.map(|column| Self {
            column,
            _types: PhantomData,
        }))
    }

    /// The attribute of `value`, or `None` when it has none.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the attribute is not a `V`.
    pub fn get(&self, value: &E) -> Result<Option<V>> {
        dict_get(AnyView::from(&self.column), AnyView::from(value))?
            .map(any_into)
            .transpose()
    }
}

/// The type attribute that holds an enum type's registry.
const ENUM_STATE_ATTR: &str = "__ffi_enum__";

/// The registry of one enum type: its values in registration order, the
/// values by index and by name (`Dict<Any, ObjectRef>`), and the attribute
/// columns (`Dict<String, Dict<ObjectRef, Any>>`).
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.EnumState"]
#[type_final]
pub(crate) struct EnumStateObj {
    base_: Object,
    entries: List<Any>,
    // The crate has no `Dict`; `dict_get` and `dict_set` call its global
    // functions instead.
    pub(crate) indexes: ObjectRef,
    attrs: ObjectRef,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
pub(crate) struct EnumState {
    data: ObjectArc<EnumStateObj>,
}

/// The registry of the enum type `type_index`, or `None` before its first value.
pub(crate) fn enum_state(type_index: i32) -> Result<Option<EnumState>> {
    let Some(column) = TypeAttrColumn::lookup(ENUM_STATE_ATTR) else {
        return Ok(None);
    };
    column.get(type_index).map(EnumState::try_from).transpose()
}

/// The registry of the enum type `type_index`, created with its first value,
/// under the registration lock.
fn ensure_state(type_index: i32) -> Result<EnumState> {
    if let Some(state) = enum_state(type_index)? {
        return Ok(state);
    }
    let state: EnumState =
        any_into(get_method("ffi.EnumState", "__ffi_init__")?.call_packed(&[])?)?;
    register_type_attr(type_index, ENUM_STATE_ATTR, Any::from(state.clone()))?;
    Ok(state)
}

fn is_int_enum(type_index: i32) -> bool {
    let is_int_enum =
        |info: *const TVMFFITypeInfo| unsafe { (*info).type_key.as_str() == "ffi.IntEnum" };
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        !info.is_null()
            && (is_int_enum(info)
                || (0..(*info).type_depth as usize)
                    .any(|depth| is_int_enum(*(*info).type_acenstors.add(depth))))
    }
}

pub(crate) fn dict_get(dict: AnyView<'_>, key: AnyView<'_>) -> Result<Option<Any>> {
    // `ffi.DictGetItemOrMissing` saves a call but returns a sentinel object
    // that Rust cannot name.
    let count: i64 = crate::cached_global_func!("ffi.DictCount")
        .call_packed(&[dict, key])?
        .try_into()?;
    if count == 0 {
        return Ok(None);
    }
    crate::cached_global_func!("ffi.DictGetItem")
        .call_packed(&[dict, key])
        .map(Some)
}

fn dict_set(dict: AnyView<'_>, key: AnyView<'_>, value: AnyView<'_>) -> Result<()> {
    crate::cached_global_func!("ffi.DictSetItem").call_packed(&[dict, key, value])?;
    Ok(())
}

pub(crate) fn any_into<T: AnyCompatible>(value: Any) -> Result<T> {
    TryFromTemp::<T>::try_from(AnyView::from(&value)).map(TryFromTemp::into_value)
}
