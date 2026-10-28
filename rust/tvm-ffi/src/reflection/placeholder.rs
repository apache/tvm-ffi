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

use crate::collections::list::List;
use crate::object::Object;
use crate::optional::{Optional, OptionalCompatible};
use crate::type_traits::ContainerElement;
use crate::{Any, Array, Bytes, Map, String};

/// A valid value of an object or field type, which an object made before
/// reflection sets its fields holds until then.
///
/// A C++ object can start empty, its references null, and the reflection then
/// fills it (the creator of `refl::ObjectDef`); a Rust object cannot, so a
/// creator of a Rust type (an `__ffi_new__` that a JSON graph or a generated
/// `__ffi_init__` fills) makes the object from placeholders. `#[rs_class(...,
/// placeholder)]` implements it for the object, from the placeholders of the
/// parent's object and of the fields. tvm-ffi implements it for `Object`, the
/// scalars, `String`, `Bytes`, `Any` and the containers; a field or parent type
/// of another crate implements it there.
pub trait Placeholder {
    /// The value.
    fn placeholder() -> Self;
}

impl Placeholder for Object {
    fn placeholder() -> Self {
        Object::new()
    }
}

macro_rules! default_placeholder {
    ($($ty:ty),* $(,)?) => {
        $(impl Placeholder for $ty {
            fn placeholder() -> Self {
                Self::default()
            }
        })*
    };
}

default_placeholder!(
    bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64, String, Bytes, Any,
);

impl<T: ContainerElement + Clone> Placeholder for Array<T> {
    fn placeholder() -> Self {
        Self::default()
    }
}

impl<T: ContainerElement> Placeholder for List<T> {
    fn placeholder() -> Self {
        Self::default()
    }
}

impl<K: ContainerElement, V: ContainerElement> Placeholder for Map<K, V> {
    fn placeholder() -> Self {
        Self::default()
    }
}

impl<T: OptionalCompatible> Placeholder for Optional<T> {
    fn placeholder() -> Self {
        Self::default()
    }
}

impl<T> Placeholder for Option<T> {
    fn placeholder() -> Self {
        None
    }
}
