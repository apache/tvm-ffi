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
use tvm_ffi::*;

#[test]
fn test_device_stream() {
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let dummy_stream = 2 as TVMFFIStreamHandle;
    with_stream(&device, dummy_stream, || {
        assert_eq!(current_stream(&device), dummy_stream);
        Ok(())
    })
    .unwrap();
    assert_eq!(current_stream(&device), std::ptr::null_mut());
}

#[test]
fn test_device_any_conversion() {
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let any_device: Any = device.into();
    assert_eq!(any_device.type_index(), TypeIndex::kTVMFFIDevice as i32);
    let converted_device: DLDevice = any_device.try_into().unwrap();
    assert_eq!(converted_device.device_type, device.device_type);
    assert_eq!(converted_device.device_id, device.device_id);
}
