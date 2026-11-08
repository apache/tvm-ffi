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
//! Module accessors read the functions, metadata and doc strings that
//! libtvm_ffi_testing exports with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` and
//! `TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC`.

use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFISafeCallType};
use tvm_ffi::*;

// The metadata and doc getters libtvm_ffi_testing exports, read directly.
extern "C" {
    fn __tvm_ffi__metadata_testing_dll_schema_id_int(
        handle: *mut std::ffi::c_void,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32;
    fn __tvm_ffi__doc_testing_dll_test_add_with_docstring(
        handle: *mut std::ffi::c_void,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32;
}

const SCHEMA_ID_INT: &str = "testing_dll_schema_id_int";
const ADD_WITH_DOCSTRING: &str = "testing_dll_test_add_with_docstring";
const MISSING: &str = "testing_dll_missing";

/// libtvm_ffi_testing, which the test binary links, so that the loader finds
/// it by name.
fn testing_module() -> Module {
    Module::load_from_file(format!("{DLL_PREFIX}tvm_ffi_testing{DLL_SUFFIX}")).unwrap()
}

/// A module without functions of its own that imports libtvm_ffi_testing.
fn importing_module() -> Module {
    let module: Module = Function::get_global("ffi.SystemLib")
        .unwrap()
        .call_tuple((String::from("testing.rust.test_module."),))
        .unwrap()
        .try_into()
        .unwrap();
    Function::get_global("ffi.ModuleImportModule")
        .unwrap()
        .call_tuple((&module, &testing_module()))
        .unwrap();
    module
}

/// Calls an exported metadata or doc getter.
fn call_getter(getter: TVMFFISafeCallType) -> String {
    // SAFETY: the getters take no arguments and do not use the handle.
    let getter = unsafe { Function::from_extern_c(std::ptr::null_mut(), getter, None).unwrap() };
    getter.call_packed(&[]).unwrap().try_into().unwrap()
}

#[test]
fn test_module_implements_function() {
    let module = testing_module();
    assert!(module.implements_function(SCHEMA_ID_INT).unwrap());
    assert!(!module.implements_function(MISSING).unwrap());
}

#[test]
fn test_module_get_function_metadata() {
    let module = testing_module();
    let metadata = module
        .get_function_metadata(SCHEMA_ID_INT)
        .unwrap()
        .unwrap();
    let exported = call_getter(__tvm_ffi__metadata_testing_dll_schema_id_int);
    assert_eq!(metadata.as_str(), exported.as_str());
    assert!(module.get_function_metadata(MISSING).unwrap().is_none());
}

#[test]
fn test_module_get_function_doc() {
    let module = testing_module();
    let doc = module
        .get_function_doc(ADD_WITH_DOCSTRING)
        .unwrap()
        .unwrap();
    let exported = call_getter(__tvm_ffi__doc_testing_dll_test_add_with_docstring);
    assert_eq!(doc.as_str(), exported.as_str());
    // A function exported without a doc string has none.
    assert!(module.get_function_doc(SCHEMA_ID_INT).unwrap().is_none());
}

#[test]
fn test_module_accessors_and_imports() {
    // The function lives in an imported module, which the accessors do not
    // search, as C++ `GetFunction(name)` and Python's default do not.
    let module = importing_module();
    assert!(!module.implements_function(SCHEMA_ID_INT).unwrap());
    assert!(module
        .get_function_metadata(SCHEMA_ID_INT)
        .unwrap()
        .is_none());
    assert!(module
        .get_function_doc(ADD_WITH_DOCSTRING)
        .unwrap()
        .is_none());
    // The `_with_imports` lookups find it there.
    assert!(module.get_function(SCHEMA_ID_INT).is_err());
    assert!(module.get_function_with_imports(SCHEMA_ID_INT).is_ok());
    assert!(module
        .implements_function_with_imports(SCHEMA_ID_INT)
        .unwrap());
    assert!(module
        .get_function_metadata_with_imports(SCHEMA_ID_INT)
        .unwrap()
        .is_some());
    assert!(module
        .get_function_doc_with_imports(ADD_WITH_DOCSTRING)
        .unwrap()
        .is_some());
}
