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
pub mod any;
pub mod collections;
pub mod derive;
pub mod device;
pub mod dtype;
pub mod error;
pub mod extra;
pub mod function;
pub mod function_internal;
pub mod macros;
pub mod object;
pub mod string;
pub mod type_traits;
pub use tvm_ffi_sys;

pub use crate::any::{Any, AnyView};
pub use crate::collections::shape::Shape;
pub use crate::collections::tensor::{CPUNDAlloc, NDAllocator, Tensor};
pub use crate::device::{current_stream, with_stream};
pub use crate::dtype::DLDataTypeExt;
pub use crate::error::{Error, ErrorKind, Result};
pub use crate::error::{
    ATTRIBUTE_ERROR, INDEX_ERROR, KEY_ERROR, RUNTIME_ERROR, TYPE_ERROR, VALUE_ERROR,
};
pub use crate::extra::module::Module;
pub use crate::function::Function;
pub use crate::object::{Object, ObjectArc, ObjectCore, ObjectCoreWithExtraItems, ObjectRefCore};
pub use crate::string::{Bytes, String};
pub use crate::type_traits::AnyCompatible;

pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
pub use tvm_ffi_sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, TVMFFIAny, TVMFFIObject, TVMFFIStreamHandle,
};
