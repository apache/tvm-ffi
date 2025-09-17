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
// NOTE: we manually write the C ABI as they are reasonably minimal
// and we need to ensure clear control of the atomic access etc.
#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::os::raw::c_char;

use crate::c_api::TVMFFIObjectHandle;
use crate::dlpack::DLTensor;

// ----------------------------------------------------------------------------
// Stream context
// Focusing on minimalistic thread-local context recording stream being used.
// We explicitly not handle allocation/de-allocation of stream here.
// ----------------------------------------------------------------------------

/// The type of the stream handle.
pub type TVMFFIStreamHandle = *mut c_void;

/// DLPack tensor allocator function type
pub type DLPackTensorAllocator = unsafe extern "C" fn(
    prototype: *mut DLTensor,
    out: *mut *mut c_void, // DLManagedTensorVersioned**
    error_ctx: *mut c_void,
    set_error: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char),
) -> i32;

unsafe extern "C" {
    pub fn TVMFFIEnvSetStream(
        device_type: i32,
        device_id: i32,
        stream: TVMFFIStreamHandle,
        opt_out_original_stream: *mut TVMFFIStreamHandle,
    ) -> i32;

    pub fn TVMFFIEnvGetStream(device_type: i32, device_id: i32) -> TVMFFIStreamHandle;

    pub fn TVMFFIEnvSetTensorAllocator(
        allocator: DLPackTensorAllocator,
        write_to_global_context: i32,
        opt_out_original_allocator: *mut DLPackTensorAllocator,
    ) -> i32;

    pub fn TVMFFIEnvGetTensorAllocator() -> DLPackTensorAllocator;

    pub fn TVMFFIEnvCheckSignals() -> i32;

    pub fn TVMFFIEnvRegisterCAPI(name: *const c_char, symbol: *mut c_void) -> i32;

    pub fn TVMFFIEnvModLookupFromImports(
        library_ctx: TVMFFIObjectHandle,
        func_name: *const c_char,
        out: *mut TVMFFIObjectHandle,
    ) -> i32;

    pub fn TVMFFIEnvModRegisterContextSymbol(name: *const c_char, symbol: *mut c_void) -> i32;

    pub fn TVMFFIEnvModRegisterSystemLibSymbol(name: *const c_char, symbol: *mut c_void) -> i32;
}
