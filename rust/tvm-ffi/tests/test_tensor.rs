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

// ============================================================================
// Environment allocator tests
// ============================================================================

mod env_alloc {
    use std::ffi::{c_char, c_void, CStr};
    use tvm_ffi::tvm_ffi_sys::{
        dlpack::DLTensor, DLPackManagedTensorAllocator, TVMFFIEnvSetDLPackManagedTensorAllocator,
    };
    use tvm_ffi::*;

    /// DLPack's `DLManagedTensorVersioned`.
    #[repr(C)]
    struct ManagedTensor {
        version: [u32; 2],
        manager_ctx: *mut c_void,
        deleter: Option<unsafe extern "C" fn(*mut ManagedTensor)>,
        flags: u64,
        dl_tensor: DLTensor,
    }

    /// The memory of a tensor made by `cpu_allocator`.
    struct Owned {
        shape: Vec<i64>,
        data: Vec<u8>,
    }

    unsafe extern "C" fn delete(tensor: *mut ManagedTensor) {
        let tensor = Box::from_raw(tensor);
        drop(Box::from_raw(tensor.manager_ctx as *mut Owned));
    }

    unsafe extern "C" fn cpu_allocator(
        prototype: *mut DLTensor,
        out: *mut *mut c_void,
        _error_ctx: *mut c_void,
        _set_error: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char),
    ) -> i32 {
        let prototype = &*prototype;
        let shape = std::slice::from_raw_parts(prototype.shape, prototype.ndim as usize).to_vec();
        let bytes = shape.iter().product::<i64>() as usize * (prototype.dtype.bits as usize / 8);
        let mut owned = Box::new(Owned {
            shape,
            data: vec![0; bytes],
        });
        let dl_tensor = DLTensor {
            data: owned.data.as_mut_ptr() as *mut c_void,
            device: prototype.device,
            ndim: prototype.ndim,
            dtype: prototype.dtype,
            shape: owned.shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let tensor = Box::new(ManagedTensor {
            version: [1, 1],
            manager_ctx: Box::into_raw(owned) as *mut c_void,
            deleter: Some(delete),
            flags: 0,
            dl_tensor,
        });
        *out = Box::into_raw(tensor) as *mut c_void;
        0
    }

    unsafe extern "C" fn failing_allocator(
        _prototype: *mut DLTensor,
        _out: *mut *mut c_void,
        error_ctx: *mut c_void,
        set_error: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char),
    ) -> i32 {
        let kind = CStr::from_bytes_with_nul(b"MemoryError\0").unwrap();
        let message = CStr::from_bytes_with_nul(b"out of test memory\0").unwrap();
        set_error(error_ctx, kind.as_ptr(), message.as_ptr());
        -1
    }

    /// Runs `f` with `allocator` set on this thread, then restores the
    /// original allocator.
    fn with_allocator<R>(
        allocator: Option<DLPackManagedTensorAllocator>,
        f: impl FnOnce() -> R,
    ) -> R {
        let mut original = None;
        unsafe {
            assert_eq!(
                TVMFFIEnvSetDLPackManagedTensorAllocator(allocator, 0, &mut original),
                0
            );
        }
        let result = f();
        unsafe {
            assert_eq!(
                TVMFFIEnvSetDLPackManagedTensorAllocator(original, 0, std::ptr::null_mut()),
                0
            );
        }
        result
    }

    fn f32_on_cpu() -> (DLDataType, DLDevice) {
        (
            DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
            DLDevice::new(DLDeviceType::kDLCPU, 0),
        )
    }

    #[test]
    fn allocates_with_the_environment_allocator() {
        let (dtype, device) = f32_on_cpu();
        let tensor = with_allocator(Some(cpu_allocator), || {
            Tensor::from_env_alloc(&[1, 2, 3], dtype, device)
        })
        .unwrap();
        assert_eq!(tensor.shape(), &[1, 2, 3]);
        assert_eq!(tensor.strides(), &[6, 3, 1]);
        assert_eq!(tensor.dtype(), dtype);
        assert_eq!(tensor.device(), device);
        assert!(!tensor.data_ptr().is_null());
        tensor.data_as_slice::<f32>().unwrap();
    }

    #[test]
    fn raises_the_allocator_error() {
        let (dtype, device) = f32_on_cpu();
        let error = with_allocator(Some(failing_allocator), || {
            Tensor::from_env_alloc(&[4], dtype, device)
        })
        .err()
        .unwrap();
        assert_eq!(error.kind().as_str(), "MemoryError");
        assert_eq!(error.message(), "out of test memory");
    }

    #[test]
    fn fails_without_an_allocator() {
        let (dtype, device) = f32_on_cpu();
        let error = with_allocator(None, || Tensor::from_env_alloc(&[4], dtype, device))
            .err()
            .unwrap();
        assert_eq!(error.kind(), RUNTIME_ERROR);
    }
}
