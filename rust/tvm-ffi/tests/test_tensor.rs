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

// ============================================================================
// Tensor Tests
// ============================================================================

#[test]
fn test_tensor_basic() {
    // Create a tensor using CPUNDAlloc
    let shape = [2, 3, 4];
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, &shape, dtype, device);

    // Test accessor methods
    assert_eq!(tensor.shape(), &shape);
    assert_eq!(tensor.ndim(), 3);
    assert_eq!(tensor.dtype().code, DLDataTypeCode::kDLFloat as u8);
    assert_eq!(tensor.dtype().bits, 32 as u8);
    assert_eq!(tensor.device().device_type, DLDeviceType::kDLCPU);

    // Test strides (should be calculated correctly for row-major layout)
    let strides = tensor.strides();
    assert_eq!(strides.len(), 3);
    // For shape [2, 3, 4], strides should be [12, 4, 1] (row-major)
    assert_eq!(strides[0], 12); // 3 * 4
    assert_eq!(strides[1], 4); // 4
    assert_eq!(strides[2], 1); // 1
}

#[test]
fn test_tensor_data_as_slice_f32() {
    // Create a tensor using CPUNDAlloc with f32 data type
    let shape = [2, 3, 4];
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, &shape, dtype, device);

    // Test data_as_slice for f32
    let data_slice = tensor.data_as_slice::<f32>().unwrap();
    assert_eq!(data_slice.len(), 24); // 2 * 3 * 4 = 24 elements

    // Test data_as_slice_mut for f32
    let data_slice_mut = tensor.data_as_slice_mut::<f32>().unwrap();
    assert_eq!(data_slice_mut.len(), 24);

    // Test that we can write to the tensor data
    for i in 0..data_slice_mut.len() {
        data_slice_mut[i] = i as f32;
    }

    // Test that we can read the written data
    let data_slice_read = tensor.data_as_slice::<f32>().unwrap();
    for i in 0..data_slice_read.len() {
        assert_eq!(data_slice_read[i], i as f32);
    }
}

#[test]
fn test_tensor_data_as_slice_type_mismatch() {
    // Create a tensor with f32 data type
    let shape = [2, 3];
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, &shape, dtype, device);

    // Test that trying to access as f64 (wrong type) fails
    let result = tensor.data_as_slice::<f64>();
    assert!(result.is_err());

    // Test that trying to access as i32 (wrong type) fails
    let result = tensor.data_as_slice::<i32>();
    assert!(result.is_err());
}

#[test]
fn test_any_tensor() {
    let shape = [2, 3, 4];
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, &shape, dtype, device);
    let any = Any::from(tensor.clone());
    let any_view = AnyView::from(&tensor);

    assert_eq!(any.type_index(), TypeIndex::kTVMFFITensor as i32);
    let converted = Tensor::try_from(any).unwrap();
    assert_eq!(converted.shape(), &shape);
    assert_eq!(converted.dtype().code, DLDataTypeCode::kDLFloat as u8);

    assert_eq!(any_view.type_index(), TypeIndex::kTVMFFITensor as i32);
    let converted_view = Tensor::try_from(any_view).unwrap();
    assert_eq!(converted_view.shape(), &shape);
    assert_eq!(converted_view.dtype().code, DLDataTypeCode::kDLFloat as u8);
}
