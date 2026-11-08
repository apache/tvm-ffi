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
//! The device stream of the FFI environment: [`current_stream`] and [`with_stream`].

use crate::error::Result;
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::dlpack::DLDevice;
use tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex as TypeIndex};
use tvm_ffi_sys::{TVMFFIEnvGetStream, TVMFFIEnvSetStream, TVMFFIStreamHandle};

/// The current stream of `device`, as the caller or [`with_stream`] set it;
/// null if none is set.
pub fn current_stream(device: &DLDevice) -> TVMFFIStreamHandle {
    unsafe { TVMFFIEnvGetStream(device.device_type, device.device_id) }
}

/// Call `f` with the device stream temporarily set to `stream`.
///
/// The previous stream is restored however `f` exits: when it returns a
/// value, when it returns an error, and when it panics.
///
/// # Safety
///
/// `stream` must be a valid stream handle for the given device, or null.
pub unsafe fn with_stream<T>(
    device: &DLDevice,
    stream: TVMFFIStreamHandle,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let mut prev_stream: TVMFFIStreamHandle = std::ptr::null_mut();
    unsafe {
        crate::check_safe_call!(TVMFFIEnvSetStream(
            device.device_type,
            device.device_id,
            stream,
            &mut prev_stream as *mut TVMFFIStreamHandle
        ))?;
    }
    let restore = RestoreStream {
        device: *device,
        stream: prev_stream,
    };
    let result = f();
    let restored = restore.restore();
    let value = result?;
    restored?;
    Ok(value)
}

/// Restores a device's previous stream when dropped, so that an unwinding
/// panic in `with_stream`'s closure still restores it.
struct RestoreStream {
    device: DLDevice,
    stream: TVMFFIStreamHandle,
}

impl RestoreStream {
    fn set(&self) -> i32 {
        unsafe {
            TVMFFIEnvSetStream(
                self.device.device_type,
                self.device.device_id,
                self.stream,
                std::ptr::null_mut(),
            )
        }
    }

    /// Restores the stream now, reporting a failure.
    fn restore(self) -> Result<()> {
        let restore = std::mem::ManuallyDrop::new(self);
        crate::check_safe_call!(restore.set())
    }
}

impl Drop for RestoreStream {
    fn drop(&mut self) {
        // Only reached while unwinding; a failure cannot be reported there.
        self.set();
    }
}

unsafe impl AnyCompatible for DLDevice {
    #[inline]
    unsafe fn is_undefined(field: *const Self) -> bool {
        // No device type is zero.
        unsafe { *std::ptr::addr_of!((*field).device_type) == 0 }
    }

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
        data.type_index == TypeIndex::kTVMFFIDevice as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        unsafe { data.data_union.v_device }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        unsafe { data.data_union.v_device }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIDevice as i32 {
            return Ok(unsafe { data.data_union.v_device });
        }
        let text = unsafe { <crate::String as AnyCompatible>::try_cast_from_any_view(data)? };
        parse_device(text.as_str()).ok_or(())
    }
}

/// The device a string such as `"cuda:1"` names, as C++ converts one: a
/// device name, an optional `:index`, and anything after a space ignored.
#[cold]
fn parse_device(text: &str) -> Option<DLDevice> {
    use tvm_ffi_sys::dlpack::DLDeviceType::*;
    let text = text.split(' ').next().unwrap_or_default();
    let (name, index) = match text.split_once(':') {
        Some((name, index)) => (name, Some(index)),
        None => (text, None),
    };
    let device_type = match name {
        "cpu" => kDLCPU,
        "cuda" => kDLCUDA,
        "opencl" => kDLOpenCL,
        "vulkan" => kDLVulkan,
        "metal" | "mps" => kDLMetal,
        "vpi" => kDLVPI,
        "rocm" => kDLROCM,
        "ext_dev" => kDLExtDev,
        "oneapi" | "xpu" => kDLOneAPI,
        "hexagon" => kDLHexagon,
        "wgpu" | "webgpu" => kDLWebGPU,
        "maia" => kDLMAIA,
        "trn" => kDLTrn,
        "tpu" => kDLTPU,
        "tpu_host" => kDLTPUHost,
        "ascend" => kDLAscend,
        _ => return None,
    };
    let device_id = match index {
        // Digits only, as C++ parses them: no sign.
        Some(index) if !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) => {
            index.parse::<i32>().ok()?
        }
        Some(_) => return None,
        None => 0,
    };
    Some(DLDevice {
        device_type: device_type as i32,
        device_id,
    })
}
