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
//! The metadata and doc strings that Rust records for exported and global
//! functions match the ones C++ records for the same signatures in
//! `libtvm_ffi_testing`, byte for byte. Exported functions have them with the
//! `export-metadata` feature, as C++ ones with
//! `TVM_FFI_DLL_EXPORT_INCLUDE_METADATA`; global functions always do.

use tvm_ffi::function_internal::AsPackedCallable;
#[cfg(feature = "export-metadata")]
use tvm_ffi::tvm_ffi_sys::TVMFFIAny;
use tvm_ffi::*;

// Exported by libtvm_ffi_testing with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` and
// `TVM_FFI_DLL_EXPORT_INCLUDE_METADATA`, for `int64_t(int64_t)`.
#[cfg(feature = "export-metadata")]
extern "C" {
    fn __tvm_ffi__metadata_testing_dll_schema_id_int(
        handle: *mut std::ffi::c_void,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32;
}

/// Calls an exported metadata or doc getter.
#[cfg(feature = "export-metadata")]
fn call_getter(getter: tvm_ffi::tvm_ffi_sys::TVMFFISafeCallType) -> String {
    // SAFETY: the getters take no arguments and do not use the handle.
    let getter = unsafe { Function::from_extern_c(std::ptr::null_mut(), getter, None) };
    getter.call_packed(&[]).unwrap().try_into().unwrap()
}

/// The dummy call keeps `libtvm_ffi_testing`, which registers the C++
/// functions compared with, linked.
fn global_metadata(name: &str) -> String {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    Function::get_global("ffi.GetGlobalFuncMetadata")
        .unwrap()
        .call_tuple_with_len::<1, _>((String::from(name),))
        .unwrap()
        .try_into()
        .unwrap()
}

fn rust_schema_id_int(x: i64) -> Result<i64> {
    Ok(x)
}
tvm_ffi_dll_export_typed_func!(rust_schema_id_int, rust_schema_id_int);

#[cfg(feature = "export-metadata")]
#[test]
fn test_exported_metadata_matches_cpp() {
    let rust = call_getter(__tvm_ffi__metadata_rust_schema_id_int);
    let cpp = call_getter(__tvm_ffi__metadata_testing_dll_schema_id_int);
    assert_eq!(rust.as_str(), cpp.as_str());
}

fn rust_add(a: i64, b: i64) -> Result<i64> {
    Ok(a + b)
}
tvm_ffi_dll_export_typed_func!(rust_add, rust_add);
tvm_ffi_dll_export_typed_func_doc!(
    rust_add,
    "Add two integers and return the sum.\n\n\"a\" and \"b\" are integers."
);

#[cfg(feature = "export-metadata")]
#[test]
fn test_exported_doc_is_verbatim() {
    let doc = call_getter(__tvm_ffi__doc_rust_add);
    assert_eq!(
        doc.as_str(),
        "Add two integers and return the sum.\n\n\"a\" and \"b\" are integers."
    );
}

/// A callable whose signature is not known, as a crate may define to export a
/// packed function.
struct Packed(fn(&[AnyView]) -> Result<Any>);
struct PackedArgs;

impl AsPackedCallable<PackedArgs, Any> for Packed {
    fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any> {
        (self.0)(packed_args)
    }
}

fn rust_packed(_args: &[AnyView]) -> Result<Any> {
    Ok(Any::new())
}
tvm_ffi_dll_export_typed_func!(rust_packed, Packed(rust_packed));

#[cfg(feature = "export-metadata")]
#[test]
fn test_exported_packed_metadata_is_untyped() {
    let rust = call_getter(__tvm_ffi__metadata_rust_packed);
    assert_eq!(
        rust.as_str(),
        global_metadata("testing.schema_packed").as_str()
    );
}

// Without `export-metadata`, the exports define no metadata or doc getters:
// these symbols of the same names would otherwise be defined twice.
#[cfg(not(feature = "export-metadata"))]
mod without_export_metadata {
    #[no_mangle]
    pub extern "C" fn __tvm_ffi__metadata_rust_schema_id_int() {}
    #[no_mangle]
    pub extern "C" fn __tvm_ffi__metadata_rust_packed() {}
    #[no_mangle]
    pub extern "C" fn __tvm_ffi__doc_rust_add() {}
}

#[cfg(not(feature = "export-metadata"))]
#[test]
fn test_exports_without_metadata() {
    // The functions themselves are exported.
    // SAFETY: rust_add takes two arguments and does not use the handle.
    let add = unsafe { Function::from_extern_c(std::ptr::null_mut(), __tvm_ffi_rust_add, None) };
    let sum = add.call_tuple_with_len::<2, _>((1i64, 2i64)).unwrap();
    assert_eq!(i64::try_from(sum).unwrap(), 3);
    without_export_metadata::__tvm_ffi__metadata_rust_schema_id_int();
    without_export_metadata::__tvm_ffi__metadata_rust_packed();
    without_export_metadata::__tvm_ffi__doc_rust_add();
}

/// Registers `func` as `testing.rust.<name>` and checks that its metadata is
/// the one C++ records for `testing.<name>`.
fn check_global<F, I, O>(name: &str, func: F)
where
    F: AsPackedCallable<I, O> + 'static,
{
    let rust_name = format!("testing.rust.{name}");
    Function::register_global_typed(&rust_name, func, "").unwrap();
    assert_eq!(
        global_metadata(&rust_name).as_str(),
        global_metadata(&format!("testing.{name}")).as_str(),
        "{name}"
    );
}

#[test]
fn test_global_metadata_matches_cpp() {
    check_global("schema_id_float", |x: f64| -> Result<f64> { Ok(x) });
    check_global("schema_id_bool", |x: bool| -> Result<bool> { Ok(x) });
    check_global("schema_id_device", |x: DLDevice| -> Result<DLDevice> {
        Ok(x)
    });
    check_global("schema_id_dtype", |x: DLDataType| -> Result<DLDataType> {
        Ok(x)
    });
    check_global("schema_id_string", |x: String| -> Result<String> { Ok(x) });
    check_global("schema_id_bytes", |x: Bytes| -> Result<Bytes> { Ok(x) });
    check_global("schema_id_func", |x: Function| -> Result<Function> {
        Ok(x)
    });
    check_global(
        "schema_id_object",
        |x: object::ObjectRef| -> Result<object::ObjectRef> { Ok(x) },
    );
    check_global("schema_id_tensor", |x: Tensor| -> Result<Tensor> { Ok(x) });
    check_global("schema_tensor_view_input", |_: TensorView| -> Result<()> {
        Ok(())
    });
    check_global(
        "schema_id_opt_int",
        |x: Option<i64>| -> Result<Option<i64>> { Ok(x) },
    );
    check_global(
        "schema_id_opt_str",
        |x: Option<String>| -> Result<Option<String>> { Ok(x) },
    );
    check_global(
        "schema_id_opt_obj",
        |x: Option<object::ObjectRef>| -> Result<Option<object::ObjectRef>> { Ok(x) },
    );
    check_global("schema_id_arr_int", |x: Array<i64>| -> Result<Array<i64>> {
        Ok(x)
    });
    check_global(
        "schema_id_arr_str",
        |x: Array<String>| -> Result<Array<String>> { Ok(x) },
    );
    check_global(
        "schema_id_arr_obj",
        |x: Array<object::ObjectRef>| -> Result<Array<object::ObjectRef>> { Ok(x) },
    );
    check_global(
        "schema_id_map_str_int",
        |x: Map<String, i64>| -> Result<Map<String, i64>> { Ok(x) },
    );
    check_global(
        "schema_id_map_str_str",
        |x: Map<String, String>| -> Result<Map<String, String>> { Ok(x) },
    );
    check_global(
        "schema_id_map_str_obj",
        |x: Map<String, object::ObjectRef>| -> Result<Map<String, object::ObjectRef>> { Ok(x) },
    );
    check_global(
        "schema_arr_map_opt",
        |_: Array<Option<i64>>,
         mp: Map<String, Array<i64>>,
         _: Option<String>|
         -> Result<Map<String, Array<i64>>> { Ok(mp) },
    );
    check_global("schema_no_args", || -> Result<i64> { Ok(1) });
    check_global("schema_no_return", |_: i64| -> Result<()> { Ok(()) });
    check_global("schema_no_args_no_return", || -> Result<()> { Ok(()) });
}

#[test]
fn test_global_untyped_metadata_matches_cpp() {
    let func = Function::from_packed(|_: &[AnyView]| -> Result<Any> { Ok(Any::new()) });
    Function::register_global("testing.rust.schema_packed", func).unwrap();
    assert_eq!(
        global_metadata("testing.rust.schema_packed").as_str(),
        global_metadata("testing.schema_packed").as_str()
    );
}

#[test]
fn test_global_typed_registration_calls_the_function() {
    Function::register_global_typed(
        "testing.rust.add",
        |a: i64, b: i64| -> Result<i64> { Ok(a + b) },
        "Add two integers.",
    )
    .unwrap();
    let add = Function::get_global("testing.rust.add").unwrap();
    let sum = add.call_tuple_with_len::<2, _>((1i64, 2i64)).unwrap();
    assert_eq!(i64::try_from(sum).unwrap(), 3);
}
