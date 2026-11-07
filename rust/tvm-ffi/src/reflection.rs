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

//! Object reflection: look up the methods, fields and type attributes of
//! registered types, and register Rust object types with [`ObjectDef`].

mod access_path;
mod accessor;
mod registry;
mod unsafe_init;
pub use access_path::{AccessKind, AccessPath, AccessPathObj, AccessStep, AccessStepObj};
pub use accessor::{get_method, get_method_by_index, FieldGetter, TypeAttrColumn};
#[doc(hidden)]
pub use registry::register_object;
pub use registry::{FieldFlags, ObjectDef, SEqHashKind, ShallowCopy};
#[doc(hidden)]
pub use unsafe_init::{drop_if_defined, unsafe_init_creator};
pub use unsafe_init::{unsafe_init, UnsafeInit};
