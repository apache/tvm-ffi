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

//! Access paths, which locate a value nested in another.

use crate::Any;
use crate::derive::{Object, ObjectRef};
use crate::object::{Object, ObjectArc, ObjectRef};

/// What an [`AccessStep`] accesses: an attribute, an array item or a map
/// item, or one of those that is missing.
///
/// The values are part of the ABI.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccessKind {
    /// An object attribute; the key is its name.
    Attr = 0,
    /// An array item; the key is its index.
    ArrayItem = 1,
    /// A map item; the key is the map key.
    MapItem = 2,
    /// An attribute that does not exist, in an error report.
    AttrMissing = 3,
    /// An array index out of range, in an error report.
    ArrayItemMissing = 4,
    /// A map key that is absent, in an error report.
    MapItemMissing = 5,
}

/// The object behind an [`AccessStep`], laid out as the C++ `AccessStepObj`.
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.reflection.AccessStep", final)]
pub struct AccessStepObj {
    object: Object,
    /// What the step accesses.
    pub kind: AccessKind,
    /// The attribute name, array index or map key.
    pub key: Any,
}

/// One step of an [`AccessPath`]: an attribute, array index or map key.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct AccessStep {
    data: ObjectArc<AccessStepObj>,
}

/// The object behind an [`AccessPath`], laid out as the C++ `AccessPathObj`.
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.reflection.AccessPath", final)]
pub struct AccessPathObj {
    object: Object,
    /// The path without its last step; `None` at the root.
    pub parent: Option<ObjectRef>,
    /// The last step; `None` at the root.
    pub step: Option<AccessStep>,
    /// The number of steps; 0 at the root.
    pub depth: i32,
}

/// A path from a root value to a nested one, such as `root.field[0]`, as a
/// chain of parents; its methods are reached with `reflection::get_method`.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct AccessPath {
    data: ObjectArc<AccessPathObj>,
}
