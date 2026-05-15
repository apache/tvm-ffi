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
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// Module
//-----------------------------------------------------

/// A TVM FFI Module for loading dynamic libraries and retrieving functions.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Module"]
#[type_index(TypeIndex::kTVMFFIModule)]
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
        static API_FUNC: std::sync::LazyLock<Function> =
            std::sync::LazyLock::new(|| Function::get_global("ffi.ModuleLoadFromFile").unwrap());
        let file_name = crate::string::String::from(file_name);
        (*API_FUNC)
            .call_tuple_with_len::<1, _>((file_name,))?
            .try_into()
    }

    /// Load a module from in-memory bytes by dispatching to the registered
    /// `ffi.Module.load_from_bytes.<kind>` loader.
    ///
    /// # Arguments
    /// * `kind` - The module kind. Examples: `"cuda"`, `"cubin"`, `"ptx"`,
    ///            `"rocm"`, or any custom kind for which a loader has been
    ///            registered via `register_global_func`.
    /// * `bytes` - The module payload bytes (e.g. PTX text or compiled cubin).
    ///
    /// # Returns
    /// * `Result<Module>` - A `Module` instance on success
    ///
    /// # Errors
    /// Returns `RuntimeError` if no loader is registered for `kind`.
    ///
    /// # Example
    /// ```no_run
    /// use tvm_ffi::{Bytes, Module, Result};
    /// fn load_ptx(ptx: &[u8]) -> Result<Module> {
    ///     Module::load_from_bytes("ptx", &Bytes::from(ptx))
    /// }
    /// ```
    pub fn load_from_bytes<Str: AsRef<str>>(
        kind: Str,
        bytes: &crate::string::Bytes,
    ) -> Result<Module> {
        static API_FUNC: std::sync::LazyLock<Function> = std::sync::LazyLock::new(|| {
            Function::get_global("ffi.ModuleLoadFromBytes").unwrap()
        });
        let kind = crate::string::String::from(kind);
        (*API_FUNC)
            .call_tuple_with_len::<2, _>((kind, bytes))?
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
        static API_FUNC: std::sync::LazyLock<Function> =
            std::sync::LazyLock::new(|| Function::get_global("ffi.ModuleGetFunction").unwrap());
        let name = crate::string::String::from(name);
        (*API_FUNC)
            .call_tuple_with_len::<3, _>((self, name, true))?
            .try_into()
    }
}
