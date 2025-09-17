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
use crate::error::Result;
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::dlpack::DLDevice;
use tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex as TypeIndex};
use tvm_ffi_sys::{TVMFFIEnvGetStream, TVMFFIEnvSetStream, TVMFFIStreamHandle};

/// Get the current stream for a device
pub fn current_stream(device: &DLDevice) -> TVMFFIStreamHandle {
    unsafe { TVMFFIEnvGetStream(device.device_type as i32, device.device_id) }
}
/// call f with the stream set to the given stream
pub fn with_stream<T>(
    device: &DLDevice,
    stream: TVMFFIStreamHandle,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let mut prev_stream: TVMFFIStreamHandle = std::ptr::null_mut();
    unsafe {
        crate::check_safe_call!(TVMFFIEnvSetStream(
            device.device_type as i32,
            device.device_id,
            stream,
            &mut prev_stream as *mut TVMFFIStreamHandle
        ))?;
    }
    let result = f()?;
    unsafe {
        crate::check_safe_call!(TVMFFIEnvSetStream(
            device.device_type as i32,
            device.device_id,
            prev_stream,
            std::ptr::null_mut()
        ))?;
    }
    Ok(result)
}

/// AnyCompatible for DLDevice
unsafe impl AnyCompatible for DLDevice {
    fn type_str() -> String {
        // make it consistent with c++ representation
        "Device".to_string()
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIDevice as i32;
        data.small_str_len = 0;
        data.data_union.v_uint64 = 0;
        data.data_union.v_device = *src;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIDevice as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = 0;
        data.data_union.v_device = src;
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        return data.type_index == TypeIndex::kTVMFFIDevice as i32;
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        data.data_union.v_device
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        data.data_union.v_device
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIDevice as i32 {
            Ok(data.data_union.v_device)
        } else {
            Err(())
        }
    }
}
