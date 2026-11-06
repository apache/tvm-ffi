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
use crate::any::AnyView;
use crate::collections::shape::Shape;
use crate::derive::{Object, ObjectRef};
use crate::dtype::AsDLDataType;
use crate::dtype::DLDataTypeExt;
use crate::error::Result;
use crate::object::{Object, ObjectArc, ObjectCore, ObjectCoreWithExtraItems};
use tvm_ffi_sys::dlpack::{DLDataType, DLDevice, DLDeviceType, DLTensor};
use tvm_ffi_sys::TVMFFIAny;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIEnvTensorAlloc, TVMFFIObjectHandle};

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
        // A scalar (`ndim == 0`) has one element and may have a null shape.
        if self.ndim <= 0 {
            return 1;
        }
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

/// A reference-counted n-dimensional array, shared with C++ and Python and
/// exchanged with other frameworks through DLPack.
///
/// # Examples
///
/// ```
/// use tvm_ffi::Tensor;
///
/// let tensor = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
/// assert_eq!(tensor.shape(), &[2, 3]);
/// assert_eq!(tensor.numel(), 6);
/// assert_eq!(tensor.data_as_slice::<f32>()?[4], 5.0);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Tensor {
    data: ObjectArc<TensorObj>,
}

impl Tensor {
    /// The data pointer, as in the `DLTensor`; `byte_offset` is not applied.
    pub fn data_ptr(&self) -> *const core::ffi::c_void {
        self.data.dltensor.data
    }

    /// The data pointer for writing, as in the `DLTensor`.
    ///
    /// A tensor is a shared handle, so `&mut self` would not make writes
    /// exclusive; getting the pointer is safe, and writing through it is the
    /// caller's `unsafe` responsibility, as for
    /// [`data_as_slice_mut`](Self::data_as_slice_mut).
    pub fn data_ptr_mut(&self) -> *mut core::ffi::c_void {
        self.data.dltensor.data
    }
    /// Whether the elements are laid out in row-major order without gaps, as
    /// C++ `tvm::ffi::IsContiguous` decides; see [`TensorView::is_contiguous`].
    pub fn is_contiguous(&self) -> bool {
        TensorView::from(self).is_contiguous()
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
            // An empty tensor may have a null data pointer, which a slice cannot.
            if self.numel() == 0 {
                return Ok(&[]);
            }
            Ok(std::slice::from_raw_parts(
                self.data.dltensor.data as *const T,
                self.numel(),
            ))
        }
    }
    /// Returns the tensor data as a mutable slice.
    ///
    /// This method takes `&self` rather than `&mut self` by design: like
    /// `std::fs::File::write`, the *metadata* of a Tensor (shape, dtype,
    /// device) is governed by Rust's ownership rules, but writing to the
    /// underlying data buffer (CPU memory or a GPU pointer) is a side-effect
    /// outside Rust's aliasing model.  Most C/CUDA kernel APIs accept a
    /// non-mut Tensor and mutate its data content, so requiring `&mut self`
    /// here would force artificial mutability annotations throughout the
    /// deep-learning stack with no real safety benefit.
    ///
    /// # Safety
    ///
    /// While the returned slice is alive, nothing else may access the data:
    /// no other slice from this method or [`data_as_slice`](Self::data_as_slice)
    /// on this tensor or a clone of it, and no kernel or foreign caller writing
    /// the buffer. Rust cannot check this because the method takes `&self`, as
    /// `std::slice::from_raw_parts_mut` cannot check it for a raw pointer.
    #[allow(clippy::wrong_self_convention, clippy::mut_from_ref)]
    pub unsafe fn data_as_slice_mut<T: AsDLDataType>(&self) -> Result<&mut [T]> {
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
            if self.numel() == 0 {
                return Ok(&mut []);
            }
            Ok(std::slice::from_raw_parts_mut(
                self.data.dltensor.data as *mut T,
                self.numel(),
            ))
        }
    }

    pub fn shape(&self) -> &[i64] {
        let t = &self.data.dltensor;
        // A scalar may have a null shape, which a slice cannot.
        if t.ndim == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) }
    }

    pub fn ndim(&self) -> usize {
        self.data.dltensor.ndim as usize
    }

    pub fn numel(&self) -> usize {
        self.data.dltensor.numel()
    }

    /// The strides, in elements.
    ///
    /// # Panics
    /// If the tensor has null strides and at least one dimension, as
    /// [`TensorView::strides`] does.
    pub fn strides(&self) -> &[i64] {
        let t = &self.data.dltensor;
        if t.ndim == 0 {
            return &[];
        }
        assert!(
            !t.strides.is_null(),
            "Tensor::strides of a DLTensor with null strides"
        );
        unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) }
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
    const TYPE_DEPTH: i32 = TensorObj::TYPE_DEPTH;
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
            // The fresh allocation is unique and has not been exposed.
            let object = &mut *ObjectArc::as_mut_ptr(&mut obj_arc);
            object.base.dltensor.shape =
                TensorObjFromNDAlloc::extra_items(object).as_ptr() as *mut i64;
            object.base.dltensor.strides = object.base.dltensor.shape.add(shape.len());
            let extra_items = TensorObjFromNDAlloc::extra_items_mut(object);
            extra_items[..shape.len()].copy_from_slice(shape);
            Shape::fill_strides_from_shape(shape, &mut extra_items[shape.len()..]);
            let dltensor_ptr = &object.base.dltensor as *const DLTensor;
            object.base.dltensor.data = object.alloc.alloc_data(&*dltensor_ptr);
            Self {
                data: ObjectArc::from_raw(ObjectArc::into_raw(obj_arc) as *mut TensorObj),
            }
        }
    }
    /// Create a Tensor with the environment allocator of this thread, as C++
    /// `Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ...)` does.
    ///
    /// Kernel libraries allocate intermediate tensors this way, so that they
    /// come from the caller's allocator, such as the one a framework sets
    /// with `TVMFFIEnvSetDLPackManagedTensorAllocator`.
    ///
    /// # Arguments
    /// * `shape` - The shape of the Tensor
    /// * `dtype` - The data type of the Tensor
    /// * `device` - The device of the Tensor
    ///
    /// # Returns
    /// * `Result<Tensor>` - The created Tensor, or the error the allocator
    ///   raised, which is a `RuntimeError` when no allocator is set
    pub fn from_env_alloc(shape: &[i64], dtype: DLDataType, device: DLDevice) -> Result<Self> {
        let mut prototype = DLTensor {
            data: std::ptr::null_mut(),
            device,
            ndim: shape.len() as i32,
            dtype,
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut out: TVMFFIObjectHandle = std::ptr::null_mut();
        unsafe {
            if TVMFFIEnvTensorAlloc(&mut prototype, &mut out) != 0 {
                return Err(crate::error::Error::from_raised());
            }
            Ok(Self {
                data: ObjectArc::from_raw(out as *const TensorObj),
            })
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
        // SAFETY: the tensor was just allocated, so nothing else can access it.
        unsafe { tensor.data_as_slice_mut::<T>()? }.copy_from_slice(slice);
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
        // A zero-size allocation is undefined behavior; an empty tensor gets an
        // aligned dangling pointer, which `free_data` does not release.
        if size == 0 {
            return Self::MIN_ALIGN as *mut core::ffi::c_void;
        }
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        ptr as *mut core::ffi::c_void
    }

    unsafe fn free_data(&mut self, tensor: &DLTensor) {
        let numel = tensor.numel() as usize;
        let item_size = tensor.item_size();
        let size = numel * item_size;
        if size == 0 {
            return;
        }
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        std::alloc::dealloc(tensor.data as *mut u8, layout);
    }
}

/// A non-owning view of a tensor: a pointer to a `DLTensor`.
///
/// This mirrors `tvm::ffi::TensorView` in C++. A function argument of this
/// type accepts both a borrowed `DLTensor*` (type index `kTVMFFIDLTensorPtr`,
/// how C and C++ callers pass tensors) and a `Tensor` object (type index
/// `kTVMFFITensor`), and is passed on as a `DLTensor*`. Prefer it over
/// [`Tensor`] for arguments of exported functions that only read a tensor's
/// metadata and data.
///
/// The view does not keep anything alive. The caller must ensure that the
/// `DLTensor` it points to, and the memory that the `DLTensor` points to
/// (data, shape and strides), outlive every use of the view. For an argument
/// of an exported function, that is the duration of the call. For the same
/// reason, as in C++, a view cannot be moved into an owned [`Any`](crate::Any);
/// use [`Tensor`] instead.
///
/// The lifetime ties a view to what it borrows, so it cannot outlive its
/// tensor, and a view cannot be held in an owned `Any`:
///
/// ```compile_fail
/// use tvm_ffi::{Tensor, TensorView};
/// let view = {
///     let tensor = Tensor::from_slice(&[1.0_f32], &[1]).unwrap();
///     TensorView::from(&tensor)
/// };
/// view.ndim();
/// ```
///
/// ```compile_fail
/// use tvm_ffi::{Any, Tensor, TensorView};
/// let tensor = Tensor::from_slice(&[1.0_f32], &[1]).unwrap();
/// let _ = Any::from(TensorView::from(&tensor));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    tensor: *const DLTensor,
    _borrow: std::marker::PhantomData<&'a DLTensor>,
}

impl<'a> TensorView<'a> {
    /// Creates a view of a `DLTensor`.
    ///
    /// # Safety
    /// `tensor` must be non-null and valid for every use of the view.
    pub unsafe fn from_raw(tensor: *const DLTensor) -> Self {
        assert!(!tensor.is_null(), "TensorView of a null DLTensor");
        Self {
            tensor,
            _borrow: std::marker::PhantomData,
        }
    }

    /// The viewed `DLTensor`.
    pub fn as_raw(&self) -> *const DLTensor {
        self.tensor
    }

    fn dltensor(&self) -> &'a DLTensor {
        // SAFETY: the view borrows a live `DLTensor` for `'a`.
        unsafe { &*self.tensor }
    }

    /// The data pointer, as in the `DLTensor`; `byte_offset` is not applied.
    pub fn data_ptr(&self) -> *mut core::ffi::c_void {
        self.dltensor().data
    }

    pub fn device(&self) -> DLDevice {
        self.dltensor().device
    }

    pub fn dtype(&self) -> DLDataType {
        self.dltensor().dtype
    }

    pub fn ndim(&self) -> usize {
        self.dltensor().ndim as usize
    }

    pub fn shape(&self) -> &'a [i64] {
        let t = self.dltensor();
        if t.ndim == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) }
    }

    pub fn numel(&self) -> usize {
        self.shape().iter().product::<i64>() as usize
    }

    /// The strides, in elements.
    ///
    /// # Panics
    /// If the `DLTensor` has null strides and at least one dimension, which
    /// DLPack forbids since v1.2, as C++ `TensorView::strides` checks.
    pub fn strides(&self) -> &'a [i64] {
        let t = self.dltensor();
        if t.ndim == 0 {
            return &[];
        }
        assert!(
            !t.strides.is_null(),
            "TensorView::strides of a DLTensor with null strides"
        );
        unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) }
    }

    /// Whether the elements are laid out in row-major order without gaps, as
    /// C++ `tvm::ffi::IsContiguous` decides: a `DLTensor` with null strides
    /// or no elements is contiguous, and a dimension of extent 1 may have any
    /// stride.
    pub fn is_contiguous(&self) -> bool {
        if self.dltensor().strides.is_null() {
            return true;
        }
        let shape = self.shape();
        if shape.contains(&0) {
            return true;
        }
        let mut expected_stride = 1;
        for (size, stride) in shape.iter().zip(self.strides()).rev() {
            if *size == 1 {
                continue;
            }
            if *stride != expected_stride {
                return false;
            }
            expected_stride *= size;
        }
        true
    }
}

impl<'a> From<&'a Tensor> for TensorView<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        Self {
            tensor: &tensor.data.dltensor as *const DLTensor,
            _borrow: std::marker::PhantomData,
        }
    }
}

impl<'a> TensorView<'a> {
    /// The view of `value`'s cell, when it holds a `DLTensor*` or a tensor object.
    fn from_cell(value: &TVMFFIAny) -> Option<Self> {
        let tensor = if value.type_index == TypeIndex::kTVMFFIDLTensorPtr as i32 {
            unsafe { value.data_union.v_ptr as *const DLTensor }
        } else if value.type_index == TypeIndex::kTVMFFITensor as i32 {
            // A tensor object's DLTensor follows its object header.
            unsafe { &(*(value.data_union.v_obj as *const TensorObj)).dltensor as *const DLTensor }
        } else {
            return None;
        };
        Some(Self {
            tensor,
            _borrow: std::marker::PhantomData,
        })
    }

    /// The `TypeError` for a value that is not a tensor.
    fn mismatch(value: &TVMFFIAny, context: &str) -> crate::error::Error {
        let message = format!(
            "{context}Cannot convert from type `{}` to `DLTensor*`",
            crate::object::type_key_of_index(value.type_index)
        );
        crate::error::Error::new(crate::error::TYPE_ERROR, &message, "")
    }
}

// A view is not `AnyCompatible`: that trait builds values from a cell without
// a lifetime, so a view could outlive what it borrows, and a view cannot be
// held in an owned `Any` (C++ `TypeTraits<DLTensor*>::MoveToAny` throws). It
// instead converts from an `AnyView` with that view's lifetime, is a typed
// function argument, and passes as a `DLTensor*` argument.

// As C++ `AnyView::cast<TensorView>`: from a `DLTensor*` or a tensor object.
impl<'a> TryFrom<AnyView<'a>> for TensorView<'a> {
    type Error = crate::error::Error;
    #[inline]
    fn try_from(value: AnyView<'a>) -> Result<Self> {
        let cell = value.as_raw_ffi_any();
        Self::from_cell(cell).ok_or_else(|| Self::mismatch(cell, ""))
    }
}

// A typed function argument. A closure `|t: TensorView| ..` is generic over
// the view's lifetime, so the view cannot be kept past the call.
impl crate::type_traits::TypeSchema for TensorView<'_> {
    fn type_schema() -> String {
        crate::type_traits::type_schema("DLTensor*", &[])
    }
}

impl crate::any::ArgTryFromAnyView for TensorView<'_> {
    fn try_from_any_view(value: &AnyView, arg_index: usize) -> Result<Self> {
        let cell = value.as_raw_ffi_any();
        Self::from_cell(cell)
            .ok_or_else(|| Self::mismatch(cell, &format!("Argument #{arg_index}: ")))
    }
}

// A view of the `DLTensor*`, for `call_packed` and other `AnyView` APIs.
impl<'a> From<&TensorView<'a>> for AnyView<'a> {
    fn from(view: &TensorView<'a>) -> Self {
        let mut cell = TVMFFIAny::new();
        cell.type_index = TypeIndex::kTVMFFIDLTensorPtr as i32;
        cell.data_union.v_ptr = view.tensor as *mut core::ffi::c_void;
        // SAFETY: the view keeps its `DLTensor` alive for `'a`.
        unsafe { AnyView::from_raw_ffi_any(cell) }
    }
}

// Passed to a function as a `DLTensor*`.
impl<'a> crate::function_internal::IntoArgHolder for TensorView<'a> {
    type Target = TensorView<'a>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}

impl<'a, 'b> crate::function_internal::IntoArgHolder for &'b TensorView<'a> {
    type Target = &'b TensorView<'a>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}

impl<'a> crate::function_internal::ArgIntoRef for TensorView<'a> {
    type Target = TensorView<'a>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl<'a> crate::function_internal::ArgIntoRef for &TensorView<'a> {
    type Target = TensorView<'a>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl crate::function_internal::PackedArg for TensorView<'_> {
    fn as_packed_arg(&self) -> AnyView<'_> {
        AnyView::from(self)
    }
}
