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
// DLPack C ABI declarations
// NOTE: we manually write the C ABI as they are reasonably minimal
// and we need to ensure clear control of the atomic access etc.
#![allow(non_camel_case_types)]

// DLPack related declarations
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
    kDLMAIA = 17,
    kDLTrn = 18,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

/// DLPack data type code enum
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLOpaqueHandle = 3,
    kDLBool = 6,
    kDLFloat8_e3m4 = 7,
    kDLFloat8_e4m3 = 8,
    kDLFloat8_e4m3b11fnuz = 9,
    kDLFloat8_e4m3fn = 10,
    kDLFloat8_e4m3fnuz = 11,
    kDLFloat8_e5m2 = 12,
    kDLFloat8_e5m2fnuz = 13,
    kDLFloat8_e8m0fnu = 14,
    kDLFloat6_e2m3fn = 15,
    kDLFloat6_e3m2fn = 16,
    kDLFloat4_e2m1fn = 17,
}

/// DLPack data type struct
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

/// DLPack tensor struct - plain C tensor object, does not manage memory
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DLTensor {
    /// The data pointer points to the allocated data
    pub data: *mut core::ffi::c_void,
    /// The device of the tensor
    pub device: DLDevice,
    /// Number of dimensions
    pub ndim: i32,
    /// The data type of the pointer
    pub dtype: DLDataType,
    /// The shape of the tensor
    pub shape: *mut i64,
    /// Strides of the tensor (in number of elements, not bytes)
    /// Can be NULL, indicating tensor is compact and row-majored
    pub strides: *mut i64,
    /// The offset in bytes to the beginning pointer to data
    pub byte_offset: u64,
}

impl DLDevice {
    pub fn new(device_type: DLDeviceType, device_id: i32) -> Self {
        Self {
            device_type: device_type,
            device_id: device_id,
        }
    }
}

impl DLDataType {
    pub fn new(code: DLDataTypeCode, bits: u8, lanes: u16) -> Self {
        Self {
            code: code as u8,
            bits: bits,
            lanes: lanes,
        }
    }
}
