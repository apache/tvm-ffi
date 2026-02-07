// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use libloading::Library;
use std::path::PathBuf;
use tvm_ffi::Array;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIByteArray, TVMFFIGetTypeInfo, TVMFFITypeInfo, TVMFFITypeKeyToIndex,
};
use tvm_ffi::{Function, Result as FfiResult, String as FfiString};

pub(crate) fn load_dlls(paths: &[PathBuf]) -> Result<Vec<Library>, Box<dyn std::error::Error>> {
    let mut libs = Vec::new();
    for path in paths {
        let lib = unsafe { Library::new(path) }?;
        libs.push(lib);
    }
    Ok(libs)
}

pub(crate) fn list_global_function_names() -> FfiResult<Vec<String>> {
    let functor_func = Function::get_global("ffi.FunctionListGlobalNamesFunctor")?;
    let functor_any = functor_func.call_tuple_with_len::<0, _>(())?;
    let functor: Function = functor_any.try_into()?;
    let count_any = functor.call_tuple_with_len::<1, _>((-1i64,))?;
    let count: i64 = count_any.try_into()?;
    let mut out = Vec::new();
    for idx in 0..count {
        let name_any = functor.call_tuple_with_len::<1, _>((idx,))?;
        let name: FfiString = name_any.try_into()?;
        out.push(name.as_str().to_string());
    }
    Ok(out)
}

pub(crate) fn list_registered_type_keys() -> FfiResult<Vec<String>> {
    let get_keys = Function::get_global("ffi.GetRegisteredTypeKeys")?;
    let keys_any = get_keys.call_tuple_with_len::<0, _>(())?;
    let mut out = Vec::new();
    let keys: Array<FfiString> = keys_any.try_into()?;
    for key in &keys {
        out.push(key.as_str().to_string());
    }
    Ok(out)
}

pub(crate) fn get_type_info(type_key: &str) -> Option<&'static TVMFFITypeInfo> {
    unsafe {
        let key = TVMFFIByteArray::from_str(type_key);
        let mut tindex = 0;
        if TVMFFITypeKeyToIndex(&key, &mut tindex) != 0 {
            return None;
        }
        let info = TVMFFIGetTypeInfo(tindex);
        if info.is_null() {
            None
        } else {
            Some(&*info)
        }
    }
}

pub(crate) fn get_global_func_metadata(name: &str) -> FfiResult<Option<String>> {
    let func = Function::get_global("ffi.GetGlobalFuncMetadata")?;
    let name_arg = FfiString::from(name);
    let meta_any = func.call_tuple_with_len::<1, _>((name_arg,))?;
    let meta: FfiString = meta_any.try_into()?;
    Ok(Some(meta.as_str().to_string()))
}

pub(crate) fn byte_array_to_string_opt(value: &TVMFFIByteArray) -> Option<String> {
    if value.data.is_null() || value.size == 0 {
        return None;
    }
    let slice = unsafe { std::slice::from_raw_parts(value.data, value.size) };
    Some(String::from_utf8_lossy(slice).to_string())
}
