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
use tvm_ffi::tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor};
use tvm_ffi::*;

/// A 2x3 f32 DLTensor over `data`, with the given strides, as a C caller
/// would build it.
fn dltensor(data: &mut [f32; 6], shape: &mut [i64; 2], strides: *mut i64) -> DLTensor {
    DLTensor {
        data: data.as_mut_ptr() as *mut core::ffi::c_void,
        device: DLDevice::new(DLDeviceType::kDLCPU, 0),
        ndim: 2,
        dtype: DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
        shape: shape.as_mut_ptr(),
        strides,
        byte_offset: 0,
    }
}

/// `(ndim, shape[1], strides[0], first element)` of the argument.
fn describe() -> Function {
    Function::from_typed(|t: TensorView| -> Result<Array<i64>> {
        let first = unsafe { *(t.data_ptr() as *const f32) } as i64;
        Ok(Array::new(vec![
            t.ndim() as i64,
            t.shape()[1],
            t.strides()[0],
            first,
        ]))
    })
}

fn call(f: &Function, arg: AnyView) -> Result<Vec<i64>> {
    let out = Array::<i64>::try_from(f.call_packed(&[arg])?)?;
    (0..out.len()).map(|i| Ok(i64::try_from(out[i])?)).collect()
}

#[test]
fn test_tensor_view_from_dltensor_ptr() {
    let (mut data, mut shape, mut strides) = ([7.0f32, 1., 2., 3., 4., 5.], [2, 3], [3i64, 1]);
    let raw = dltensor(&mut data, &mut shape, strides.as_mut_ptr());
    let view = unsafe { TensorView::from_raw(&raw) };
    let arg = AnyView::from(&view);
    assert_eq!(arg.type_index(), TypeIndex::kTVMFFIDLTensorPtr as i32);
    assert_eq!(call(&describe(), arg).unwrap(), vec![2, 3, 3, 7]);
}

#[test]
fn test_tensor_view_from_tensor_object() {
    let tensor = Tensor::from_slice(&[9.0f32, 1., 2., 3., 4., 5.], &[2, 3]).unwrap();
    let arg = AnyView::from(&tensor);
    assert_eq!(arg.type_index(), TypeIndex::kTVMFFITensor as i32);
    assert_eq!(call(&describe(), arg).unwrap(), vec![2, 3, 3, 9]);
    let view = TensorView::from(&tensor);
    assert_eq!(view.shape(), tensor.shape());
    assert_eq!(view.strides(), tensor.strides());
    assert_eq!(
        view.data_ptr() as *const core::ffi::c_void,
        tensor.data_ptr()
    );
}

#[test]
fn test_tensor_view_rejects_other_types() {
    let err = call(&describe(), AnyView::from(&1i32)).unwrap_err();
    assert!(err.message().contains("DLTensor*"), "{}", err.message());
}

#[test]
fn test_tensor_view_contiguity() {
    let (mut data, mut shape) = ([0f32; 6], [2i64, 3]);
    let mut compact_strides = [3i64, 1];
    let compact = dltensor(&mut data, &mut shape, compact_strides.as_mut_ptr());
    let view = unsafe { TensorView::from_raw(&compact) };
    assert!(view.is_contiguous());
    assert_eq!(view.numel(), 6);
    let mut padded_strides = [4i64, 1];
    let padded = dltensor(&mut data, &mut shape, padded_strides.as_mut_ptr());
    assert!(!unsafe { TensorView::from_raw(&padded) }.is_contiguous());
}

#[test]
#[should_panic(expected = "null strides")]
fn test_tensor_view_null_strides() {
    let (mut data, mut shape) = ([0f32; 6], [2i64, 3]);
    let compact = dltensor(&mut data, &mut shape, std::ptr::null_mut());
    unsafe { TensorView::from_raw(&compact) }.strides();
}
