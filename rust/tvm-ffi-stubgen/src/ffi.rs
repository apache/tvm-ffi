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
use std::sync::LazyLock;
use tvm_ffi::function_internal::{ArgIntoRef, IntoArgHolder};
use tvm_ffi::{Any, Error, Function, Result as FfiResult, String as FfiString, TYPE_ERROR};
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIGetTypeInfo, TVMFFIObjectHandle, TVMFFITypeIndex,
    TVMFFITypeInfo, TVMFFITypeKeyToIndex,
};

#[repr(C)]
#[derive(Debug)]
pub(crate) struct Array {
    handle: TVMFFIObjectHandle,
}

extern "C" {
    fn TVMFFIObjectIncRef(handle: TVMFFIObjectHandle) -> i32;
    fn TVMFFIObjectDecRef(handle: TVMFFIObjectHandle) -> i32;
}

impl Clone for Array {
    fn clone(&self) -> Self {
        unsafe {
            TVMFFIObjectIncRef(self.handle);
        }
        Self { handle: self.handle }
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            TVMFFIObjectDecRef(self.handle);
        }
    }
}

unsafe impl tvm_ffi::type_traits::AnyCompatible for Array {
    fn type_str() -> String {
        "ffi.Array".to_string()
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TVMFFITypeIndex::kTVMFFIArray as i32;
        data.small_str_len = 0;
        data.data_union.v_obj = src.handle as *mut tvm_ffi::tvm_ffi_sys::TVMFFIObject;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TVMFFITypeIndex::kTVMFFIArray as i32;
        data.small_str_len = 0;
        data.data_union.v_obj = src.handle as *mut tvm_ffi::tvm_ffi_sys::TVMFFIObject;
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TVMFFITypeIndex::kTVMFFIArray as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        let handle = data.data_union.v_obj as TVMFFIObjectHandle;
        TVMFFIObjectIncRef(handle);
        Self { handle }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        let handle = data.data_union.v_obj as TVMFFIObjectHandle;
        Self { handle }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TVMFFITypeIndex::kTVMFFIArray as i32 {
            Ok(Self::copy_from_any_view_after_check(data))
        } else {
            Err(())
        }
    }
}

impl ArgIntoRef for Array {
    type Target = Array;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl<'a> ArgIntoRef for &'a Array {
    type Target = Array;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl IntoArgHolder for Array {
    type Target = Array;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}

impl<'a> IntoArgHolder for &'a Array {
    type Target = &'a Array;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}

impl TryFrom<Any> for Array {
    type Error = Error;
    fn try_from(value: Any) -> Result<Self, Self::Error> {
        if let Some(ret) = value.try_as::<Array>() {
            Ok(ret)
        } else {
            Err(Error::new(TYPE_ERROR, "Expected ffi.Array", ""))
        }
    }
}

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
    let keys: Array = keys_any.try_into()?;
    let size = array_size(&keys)?;
    let mut out = Vec::new();
    for idx in 0..size {
        let item = array_get_item(&keys, idx)?;
        let key: FfiString = item.try_into()?;
        out.push(key.as_str().to_string());
    }
    Ok(out)
}

pub(crate) fn array_size(array: &Array) -> FfiResult<i64> {
    static FUNC: LazyLock<Function> =
        LazyLock::new(|| Function::get_global("ffi.ArraySize").expect("ffi.ArraySize missing"));
    let func = &*FUNC;
    let size_any = func.call_tuple_with_len::<1, _>((array,))?;
    size_any.try_into()
}

pub(crate) fn array_get_item(array: &Array, index: i64) -> FfiResult<Any> {
    static FUNC: LazyLock<Function> = LazyLock::new(|| {
        Function::get_global("ffi.ArrayGetItem").expect("ffi.ArrayGetItem missing")
    });
    let func = &*FUNC;
    func.call_tuple_with_len::<2, _>((array, index))
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
