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
//! [`Module`], a loaded library of functions.

use crate::derive::{Object, ObjectRef};
use crate::error::Result;
use crate::function::Function;
use crate::object::{Object, ObjectArc};
use crate::string::String;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// Module
//-----------------------------------------------------

/// The object behind a [`Module`].
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.Module", unsafe type_index = TypeIndex::kTVMFFIModule)]
pub struct ModuleObj {
    object: Object,
}

/// A collection of functions, such as a dynamic library, passed across
/// languages as `ffi.Module`.
#[repr(C)]
#[derive(ObjectRef, Clone)]
#[object_ref(skip_deref)]
pub struct Module {
    data: ObjectArc<ModuleObj>,
}

impl Module {
    /// Load a module from a dynamic library file.
    ///
    /// # Arguments
    /// * `file_name` - The path of the library.
    ///
    /// # Returns
    /// The module.
    ///
    /// # Errors
    /// If the library cannot be loaded.
    pub fn load_from_file<Str: AsRef<str>>(file_name: Str) -> Result<Module> {
        let file_name = crate::string::String::from(file_name.as_ref());
        crate::cached_global_func!("ffi.ModuleLoadFromFile")
            .call_tuple((file_name,))?
            .try_into()
    }

    /// Look up a function of the module by name.
    ///
    /// Only this module is searched, not the modules it imports.
    ///
    /// # Arguments
    /// * `name` - The name of the function.
    ///
    /// # Returns
    /// The function.
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` naming the function if the module has no function
    /// `name`, as [`Function::get_global`] does for a missing global function.
    pub fn get_function<Str: AsRef<str>>(&self, name: Str) -> Result<Function> {
        self.lookup_function(name.as_ref(), false)
    }

    /// As [`get_function`](Self::get_function), but also searching the modules
    /// this one imports, depth first in import order.
    pub fn get_function_with_imports<Str: AsRef<str>>(&self, name: Str) -> Result<Function> {
        self.lookup_function(name.as_ref(), true)
    }

    fn lookup_function(&self, name: &str, query_imports: bool) -> Result<Function> {
        let function = crate::cached_global_func!("ffi.ModuleGetFunction").call_tuple((
            self,
            name,
            query_imports,
        ))?;
        if function.is_none() {
            crate::bail!(
                crate::error::VALUE_ERROR,
                "Module has no function '{}'",
                name
            );
        }
        function.try_into()
    }

    /// Whether the module implements the function `name`.
    ///
    /// Only this module is searched, not the modules it imports.
    ///
    /// # Arguments
    /// * `name` - The name of the function.
    ///
    /// # Returns
    /// Whether the function exists.
    pub fn implements_function<Str: AsRef<str>>(&self, name: Str) -> Result<bool> {
        let name = String::from(name.as_ref());
        crate::cached_global_func!("ffi.ModuleImplementsFunction")
            .call_tuple((self, name, false))?
            .try_into()
    }

    /// As [`implements_function`](Self::implements_function), but also searching the modules
    /// this one imports, depth first in import order.
    pub fn implements_function_with_imports<Str: AsRef<str>>(&self, name: Str) -> Result<bool> {
        let name = String::from(name.as_ref());
        crate::cached_global_func!("ffi.ModuleImplementsFunction")
            .call_tuple((self, name, true))?
            .try_into()
    }

    /// Get the metadata of a function the module exports.
    ///
    /// Only this module is searched, not the modules it imports.
    ///
    /// The metadata is a JSON object whose `type_schema` entry is the
    /// function's signature, as an export with metadata enabled records it
    /// (see [`tvm_ffi_dll_export_typed_func!`](crate::tvm_ffi_dll_export_typed_func)).
    ///
    /// # Arguments
    /// * `name` - The name of the function.
    ///
    /// # Returns
    /// The metadata as a JSON string, or `None` if the function has none.
    pub fn get_function_metadata<Str: AsRef<str>>(&self, name: Str) -> Result<Option<String>> {
        let name = String::from(name.as_ref());
        crate::cached_global_func!("ffi.ModuleGetFunctionMetadata")
            .call_tuple((self, name, false))?
            .try_into()
    }

    /// As [`get_function_metadata`](Self::get_function_metadata), but also searching the modules
    /// this one imports, depth first in import order.
    pub fn get_function_metadata_with_imports<Str: AsRef<str>>(
        &self,
        name: Str,
    ) -> Result<Option<String>> {
        let name = String::from(name.as_ref());
        crate::cached_global_func!("ffi.ModuleGetFunctionMetadata")
            .call_tuple((self, name, true))?
            .try_into()
    }

    /// Get the doc string of a function the module exports.
    ///
    /// Only this module is searched, not the modules it imports.
    ///
    /// The doc string is the one the export recorded (see
    /// [`tvm_ffi_dll_export_typed_func_doc!`](crate::tvm_ffi_dll_export_typed_func_doc)).
    ///
    /// # Arguments
    /// * `name` - The name of the function.
    ///
    /// # Returns
    /// The doc string, or `None` if the function has none.
    pub fn get_function_doc<Str: AsRef<str>>(&self, name: Str) -> Result<Option<String>> {
        let name = String::from(name.as_ref());
        crate::cached_global_func!("ffi.ModuleGetFunctionDoc")
            .call_tuple((self, name, false))?
            .try_into()
    }

    /// As [`get_function_doc`](Self::get_function_doc), but also searching the modules
    /// this one imports, depth first in import order.
    pub fn get_function_doc_with_imports<Str: AsRef<str>>(
        &self,
        name: Str,
    ) -> Result<Option<String>> {
        let name = String::from(name.as_ref());
        crate::cached_global_func!("ffi.ModuleGetFunctionDoc")
            .call_tuple((self, name, true))?
            .try_into()
    }
}
