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
use crate::collections::shape::Shape;
use crate::derive::{Object, ObjectRef};
use crate::dtype::AsDLDataType;
use crate::dtype::DLDataTypeExt;
use crate::error::Result;
use crate::object::{Object, ObjectArc, ObjectCore, ObjectCoreWithExtraItems};
use tvm_ffi_sys::dlpack::{DLDataType, DLDevice, DLDeviceType, DLTensor};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// NDAllocator Trait
//-----------------------------------------------------
/// Trait for n-dimensional array allocators
pub unsafe trait NDAllocator: 'static {
    /// The minimum alignment of the data allocated by the allocator
    const MIN_ALIGN: usize;
    /// Allocate data for the given DLTensor
    ///
    /// # Arguments
    /// * `tensor` - The DLTensor to allocate data for
    ///
    /// This method should fill in the data pointer of the DLTensor.
    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut core::ffi::c_void;

    /// Free data for the given DLTensor
    ///
    /// # Arguments
    /// * `tensor` - The DLTensor to free data for
    ///
    /// This method should free the data pointer of the DLTensor.
    unsafe fn free_data(&mut self, tensor: &DLTensor);
}

/// DLTensorExt trait
/// This trait provides methods to get the number of elements and the item size of a DLTensor
pub trait DLTensorExt {
    fn numel(&self) -> usize;
    fn item_size(&self) -> usize;
}

impl DLTensorExt for DLTensor {
    fn numel(&self) -> usize {
        unsafe {
            std::slice::from_raw_parts(self.shape, self.ndim as usize)
                .iter()
                .product::<i64>() as usize
        }
    }

    fn item_size(&self) -> usize {
        (self.dtype.bits as usize * self.dtype.lanes as usize + 7) / 8
    }
}

//-----------------------------------------------------
// Shape
//-----------------------------------------------------
// ShapeObj for heap-allocated shape
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Tensor"]
#[type_index(TypeIndex::kTVMFFITensor)]
pub struct TensorObj {
    object: Object,
    dltensor: DLTensor,
}

/// ABI stable owned Shape for ffi
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Tensor {
    data: ObjectArc<TensorObj>,
}

impl Tensor {
    /// Get the data pointer of the Tensor
    ///
    /// # Returns
    /// * `*mut core::ffi::c_void` - The data pointer of the Tensor
    pub fn data_ptr(&self) -> *const core::ffi::c_void {
        self.data.dltensor.data
    }
    /// Get the data pointer of the Tensor
    ///
    /// # Returns
    /// * `*mut core::ffi::c_void` - The data pointer of the Tensor
    pub fn data_ptr_mut(&mut self) -> *mut core::ffi::c_void {
        self.data.dltensor.data
    }
    /// Check if the Tensor is contiguous
    ///
    /// # Returns
    /// * `bool` - True if the Tensor is contiguous, false otherwise
    pub fn is_contiguous(&self) -> bool {
        let strides = self.strides();
        let shape = self.shape();
        let mut expected_stride = 1;
        for i in (0..self.ndim()).rev() {
            if strides[i] != expected_stride {
                return false;
            }
            expected_stride *= shape[i];
        }
        true
    }

    pub fn data_as_slice<T: AsDLDataType>(&self) -> Result<&[T]> {
        let dtype = T::DL_DATA_TYPE;
        if self.dtype() != dtype {
            crate::bail!(
                crate::error::TYPE_ERROR,
                "Data type mismatch {} vs {}",
                self.dtype().to_string(),
                dtype.to_string()
            );
        }
        if self.device().device_type != DLDeviceType::kDLCPU {
            crate::bail!(crate::error::RUNTIME_ERROR, "Tensor is not on CPU");
        }
        crate::ensure!(
            self.is_contiguous(),
            crate::error::RUNTIME_ERROR,
            "Tensor is not contiguous"
        );

        unsafe {
            Ok(std::slice::from_raw_parts(
                self.data.dltensor.data as *const T,
                self.numel(),
            ))
        }
    }
    /// Get the data as a mutable slice
    ///
    /// Note that we do allow mutable data access to copies of the Tensor,
    /// as in the case of low-level deep learning frameworks.
    ///
    /// # Arguments
    /// * `T` - The type of the data
    ///
    /// # Returns
    /// * `Result<&mut [T]>` - The data as a mutable slice
    pub fn data_as_slice_mut<T: AsDLDataType>(&self) -> Result<&mut [T]> {
        let dtype = T::DL_DATA_TYPE;
        if self.dtype() != dtype {
            crate::bail!(
                crate::error::TYPE_ERROR,
                "Data type mismatch: expected {}, got {}",
                dtype.to_string(),
                self.dtype().to_string()
            );
        }
        if self.device().device_type != DLDeviceType::kDLCPU {
            crate::bail!(crate::error::RUNTIME_ERROR, "Tensor is not on CPU");
        }
        crate::ensure!(
            self.is_contiguous(),
            crate::error::RUNTIME_ERROR,
            "Tensor is not contiguous"
        );
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                self.data.dltensor.data as *mut T,
                self.numel(),
            ))
        }
    }

    pub fn shape(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.data.dltensor.shape, self.ndim()) }
    }

    pub fn ndim(&self) -> usize {
        self.data.dltensor.ndim as usize
    }

    pub fn numel(&self) -> usize {
        self.data.dltensor.numel()
    }

    pub fn strides(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.data.dltensor.strides, self.ndim()) }
    }

    pub fn dtype(&self) -> DLDataType {
        self.data.dltensor.dtype
    }

    pub fn device(&self) -> DLDevice {
        self.data.dltensor.device
    }
}

struct TensorObjFromNDAlloc<TNDAlloc>
where
    TNDAlloc: NDAllocator,
{
    base: TensorObj,
    alloc: TNDAlloc,
}

unsafe impl<TNDAlloc: NDAllocator> ObjectCore for TensorObjFromNDAlloc<TNDAlloc> {
    const TYPE_KEY: &'static str = TensorObj::TYPE_KEY;
    fn type_index() -> i32 {
        TensorObj::type_index()
    }
    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        TensorObj::object_header_mut(&mut this.base)
    }
}

unsafe impl<TNDAlloc: NDAllocator> ObjectCoreWithExtraItems for TensorObjFromNDAlloc<TNDAlloc> {
    type ExtraItem = i64;
    fn extra_items_count(this: &Self) -> usize {
        (this.base.dltensor.ndim * 2) as usize
    }
}

impl<TNDAlloc: NDAllocator> Drop for TensorObjFromNDAlloc<TNDAlloc> {
    fn drop(&mut self) {
        unsafe {
            self.alloc.free_data(&self.base.dltensor);
        }
    }
}

impl Tensor {
    // Create a Tensor from a NDAllocator
    ///
    /// # Arguments
    /// * `alloc` - The NDAllocator
    /// * `shape` - The shape of the Tensor
    /// * `dtype` - The data type of the Tensor
    /// * `device` - The device of the Tensor
    ///
    /// # Returns
    /// * `Tensor` - The created Tensor
    pub fn from_nd_alloc<TNDAlloc>(
        alloc: TNDAlloc,
        shape: &[i64],
        dtype: DLDataType,
        device: DLDevice,
    ) -> Self
    where
        TNDAlloc: NDAllocator,
    {
        let tensor_obj = TensorObjFromNDAlloc {
            base: TensorObj {
                object: Object::new(),
                dltensor: DLTensor {
                    data: std::ptr::null_mut(),
                    device: device,
                    ndim: shape.len() as i32,
                    dtype: dtype,
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
            },
            alloc: alloc,
        };
        unsafe {
            let mut obj_arc = ObjectArc::new_with_extra_items(tensor_obj);
            obj_arc.base.dltensor.shape =
                TensorObjFromNDAlloc::extra_items(&obj_arc).as_ptr() as *mut i64;
            obj_arc.base.dltensor.strides = obj_arc.base.dltensor.shape.add(shape.len());
            let extra_items = TensorObjFromNDAlloc::extra_items_mut(&mut obj_arc);
            extra_items[..shape.len()].copy_from_slice(shape);
            Shape::fill_strides_from_shape(shape, &mut extra_items[shape.len()..]);
            let dltensor_ptr = &obj_arc.base.dltensor as *const DLTensor;
            obj_arc.base.dltensor.data = obj_arc.alloc.alloc_data(&*dltensor_ptr);
            Self {
                data: ObjectArc::from_raw(ObjectArc::into_raw(obj_arc) as *mut TensorObj),
            }
        }
    }
    /// Create a Tensor from a slice
    ///
    /// # Arguments
    /// * `slice` - The slice to create the Tensor from
    /// * `shape` - The shape of the Tensor
    ///
    /// # Returns
    /// * `Tensor` - The created Tensor
    pub fn from_slice<T: AsDLDataType>(slice: &[T], shape: &[i64]) -> Result<Self> {
        let dtype = T::DL_DATA_TYPE;
        let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
        let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, shape, dtype, device);
        if tensor.numel() != slice.len() {
            crate::bail!(crate::error::VALUE_ERROR, "Slice length mismatch");
        }
        tensor.data_as_slice_mut::<T>()?.copy_from_slice(slice);
        Ok(tensor)
    }
}

/// Example CPU NDAllocator
/// This allocator allocates data on the CPU
pub struct CPUNDAlloc {}

unsafe impl NDAllocator for CPUNDAlloc {
    const MIN_ALIGN: usize = 64;

    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut core::ffi::c_void {
        let numel = prototype.numel() as usize;
        let item_size = prototype.item_size();
        let size = numel * item_size as usize;
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        let ptr = std::alloc::alloc(layout);
        ptr as *mut core::ffi::c_void
    }

    unsafe fn free_data(&mut self, tensor: &DLTensor) {
        let numel = tensor.numel() as usize;
        let item_size = tensor.item_size();
        let size = numel * item_size;
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        std::alloc::dealloc(tensor.data as *mut u8, layout);
    }
}
