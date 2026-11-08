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

//! Enums shared across languages: each value is a named singleton of an
//! object type whose parent is [`EnumObj`]. [`EnumRef`] looks values up;
//! [`reflection::EnumDef`](crate::reflection::EnumDef) registers them.

use crate::derive::Object;
use crate::object::{Object, ObjectCore, ObjectRefCore};
use crate::reflection::enum_def::{any_into, dict_get, enum_state};
use crate::reflection::EnumAttrMap;
use crate::type_traits::AnyCompatible;
use crate::{AnyView, Error, Result, String, VALUE_ERROR};

/// The object behind every enum value: its name and the index the registry
/// assigned it. An enum type's object embeds it as its parent,
/// `base_: EnumObj`.
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.Enum", immutable)]
pub struct EnumObj {
    base_: Object,
    // The registration order, which can differ between processes, so values
    // are identified by name only.
    #[allow(dead_code)]
    index: i64,
    name: String,
}

impl EnumObj {
    pub(crate) fn new(index: i64, name: &str) -> Self {
        Self {
            base_: Object::new(),
            index,
            name: String::from(name),
        }
    }
}

/// A reference to an enum value: one named singleton of an enum type,
/// registered once and shared with every language.
///
/// [`get`](Self::get) looks a value up by name. Each name has one value, so
/// equality is identity: compare values with
/// [`same_as`](ObjectRefCore::same_as).
/// [`EnumDef`](crate::reflection::EnumDef) registers values and their
/// attributes, and [`attr_map`](Self::attr_map) reads an attribute.
///
/// The stub generator implements it for the enum types it binds. A
/// hand-written reference implements it when its object's parent is
/// [`EnumObj`].
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::{Object, ObjectRef};
/// use tvm_ffi::reflection::EnumDef;
/// use tvm_ffi::{EnumRef, ObjectArc, ObjectRefCore};
/// use tvm_ffi::enum_ref::EnumObj;
///
/// #[repr(C)]
/// #[derive(Object)]
/// #[object(type_key = "my_ext.Color", register, final, immutable)]
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
/// EnumDef::<Color>::new("red")?;
///
/// let red = Color::get("red")?;
/// assert_eq!(red.name(), "red");
/// assert!(red.same_as(&Color::get("red")?));
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub trait EnumRef: ObjectRefCore {
    /// Look up the value registered as `name`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the value.
    ///
    /// # Returns
    ///
    /// The value, the same object on every call.
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if the enum type has no value named `name`.
    fn get(name: &str) -> Result<Self> {
        let key = String::from(name);
        let found = match enum_state(Self::ContainerType::type_index())? {
            Some(state) => dict_get(AnyView::from(&state.indexes), AnyView::from(&key))?,
            None => None,
        };
        let Some(value) = found else {
            return Err(Error::new(
                VALUE_ERROR,
                &format!(
                    "Enum `{}` has no value named `{name}`",
                    Self::ContainerType::TYPE_KEY
                ),
            ));
        };
        any_into(value)
    }

    /// The name of the value, unique within its enum type.
    ///
    /// # Panics
    ///
    /// Panics if the object of `Self` does not have [`EnumObj`] as an
    /// ancestor, which only an incorrect `impl EnumRef` allows.
    fn name(&self) -> &str {
        let object = self
            .downcast_ref::<EnumObj>()
            .expect("an `EnumRef` object has `EnumObj` as an ancestor");
        object.name.as_str()
    }

    /// The attribute `name` of the values of the enum type, or `None` when no
    /// value has it.
    ///
    /// The map holds the registry's column, so it also sees the attributes set
    /// after it was read. See also
    /// [`EnumDef::set_attr`](crate::reflection::EnumDef::set_attr).
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the attribute.
    ///
    /// # Returns
    ///
    /// The attribute of every value that has it, read as `V`.
    fn attr_map<V: AnyCompatible>(name: &str) -> Result<Option<EnumAttrMap<Self, V>>> {
        EnumAttrMap::lookup(name)
    }
}
