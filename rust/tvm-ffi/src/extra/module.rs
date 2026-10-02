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
use crate::derive::{Object, ObjectRef};
use crate::error::Result;
use crate::function::Function;
use crate::object::{Object, ObjectArc};
use crate::string::String;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// Module
//-----------------------------------------------------

/// A TVM FFI Module for loading dynamic libraries and retrieving functions.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Module"]
#[type_index(TypeIndex::kTVMFFIModule)]
#[type_final]
pub struct ModuleObj {
    object: Object,
}

/// ABI-stable owned Module for FFI operations.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Module {
    data: ObjectArc<ModuleObj>,
}

impl Module {
    /// Load a module from a dynamic library file.
    ///
    /// # Arguments
    /// * `file_name` - Path to the dynamic library file to load
    ///
    /// # Returns
    /// * `Result<Module>` - A `Module` instance on success
    pub fn load_from_file<Str: AsRef<str>>(file_name: Str) -> Result<Module> {
        let file_name = crate::string::String::from(file_name);
        crate::cached_global_func!("ffi.ModuleLoadFromFile")
            .call_tuple_with_len::<1, _>((file_name,))?
            .try_into()
    }

    /// Get a function from the module by name.
    ///
    /// # Arguments
    /// * `name` - The name of the function to retrieve
    ///
    /// # Returns
    /// * `Result<Function>` - A `Function` instance on success
    pub fn get_function<Str: AsRef<str>>(&self, name: Str) -> Result<Function> {
        let name = crate::string::String::from(name);
        crate::cached_global_func!("ffi.ModuleGetFunction")
            .call_tuple_with_len::<3, _>((self, name, true))?
            .try_into()
    }

    /// Check whether the module implements a function.
    ///
    /// # Arguments
    /// * `name` - The name of the function
    /// * `query_imports` - Whether to also check the modules this module imports
    ///
    /// # Returns
    /// * `Result<bool>` - Whether the function exists
    pub fn implements_function<Str: AsRef<str>>(
        &self,
        name: Str,
        query_imports: bool,
    ) -> Result<bool> {
        let name = String::from(name);
        crate::cached_global_func!("ffi.ModuleImplementsFunction")
            .call_tuple_with_len::<3, _>((self, name, query_imports))?
            .try_into()
    }

    /// Get the metadata of a function the module exports.
    ///
    /// The metadata is a JSON object whose `type_schema` entry is the
    /// function's type schema, as C++ `TVM_FFI_DLL_EXPORT_TYPED_FUNC` exports
    /// it with `TVM_FFI_DLL_EXPORT_INCLUDE_METADATA`.
    ///
    /// # Arguments
    /// * `name` - The name of the function
    /// * `query_imports` - Whether to also check the modules this module imports
    ///
    /// # Returns
    /// * `Result<Option<String>>` - The metadata as a JSON string, or `None`
    ///   if the function has no metadata
    pub fn get_function_metadata<Str: AsRef<str>>(
        &self,
        name: Str,
        query_imports: bool,
    ) -> Result<Option<String>> {
        let name = String::from(name);
        crate::cached_global_func!("ffi.ModuleGetFunctionMetadata")
            .call_tuple_with_len::<3, _>((self, name, query_imports))?
            .try_into()
    }

    /// Get the doc string of a function the module exports.
    ///
    /// The doc string is the one C++ `TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC`
    /// exports.
    ///
    /// # Arguments
    /// * `name` - The name of the function
    /// * `query_imports` - Whether to also check the modules this module imports
    ///
    /// # Returns
    /// * `Result<Option<String>>` - The doc string, or `None` if the function
    ///   has none
    pub fn get_function_doc<Str: AsRef<str>>(
        &self,
        name: Str,
        query_imports: bool,
    ) -> Result<Option<String>> {
        let name = String::from(name);
        crate::cached_global_func!("ffi.ModuleGetFunctionDoc")
            .call_tuple_with_len::<3, _>((self, name, query_imports))?
            .try_into()
    }
}
