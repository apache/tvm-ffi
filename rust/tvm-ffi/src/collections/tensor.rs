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
//! [`Tensor`], an owned n-dimensional array, and [`TensorView`], a borrowed one.

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
/// Allocates the data buffer of a tensor made by [`Tensor::from_nd_alloc`].
///
/// # Safety
/// `alloc_data` must return a pointer aligned to `MIN_ALIGN` and valid for the
/// prototype's data, `numel() * bits * lanes` bits rounded up to whole bytes
/// (sub-byte types are packed), and `free_data` must release
/// exactly what `alloc_data` returned. `free_data` may be called for a tensor
/// whose `data` is null, when `alloc_data` panicked, and must then do nothing.
pub unsafe trait NDAllocator: 'static {
    /// The minimum alignment of the allocated data, in bytes.
    const MIN_ALIGN: usize;
    /// Allocate the data of a tensor.
    ///
    /// # Arguments
    /// * `prototype` - The tensor to allocate for; its data pointer is still null.
    ///
    /// # Returns
    /// The data pointer, which the tensor stores.
    ///
    /// # Safety
    /// `prototype` must have a valid shape and dtype.
    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut core::ffi::c_void;

    /// Free the data of a tensor, when the tensor is dropped.
    ///
    /// # Arguments
    /// * `tensor` - The tensor whose `data` to free.
    ///
    /// # Safety
    /// `tensor.data` must be null or come from `alloc_data` on this allocator
    /// with the same shape and dtype, and must not be used afterwards.
    unsafe fn free_data(&mut self, tensor: &DLTensor);
}

/// Size queries on a `DLTensor`.
pub trait DLTensorExt {
    /// The number of elements: the product of the shape, 1 for a scalar.
    ///
    /// # Panics
    /// If an extent is negative or the product overflows `usize`.
    fn numel(&self) -> usize;
}

impl DLTensorExt for DLTensor {
    fn numel(&self) -> usize {
        // A scalar (`ndim == 0`) has one element and may have a null shape.
        let shape =
            unsafe { crate::util::slice_from_nullable_raw_parts(self.shape, self.ndim as usize) };
        checked_numel(shape).expect("tensor shape has a negative extent or overflows usize")
    }
}

/// The product of `shape`; `None` if an extent is negative or it overflows.
fn checked_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |numel, &extent| {
        numel.checked_mul(usize::try_from(extent).ok()?)
    })
}

/// The bytes of `tensor`'s data, with sub-byte types packed as C++
/// `GetDataSize` does; it fits in `isize`, as an allocation must.
fn data_size(tensor: &DLTensor) -> usize {
    let bits = tensor.dtype.bits as usize * tensor.dtype.lanes as usize;
    tensor
        .numel()
        .checked_mul(bits)
        .map(|total| total.div_ceil(8))
        .filter(|&size| size <= isize::MAX as usize)
        .expect("tensor byte size overflows isize")
}

//-----------------------------------------------------
// Tensor
//-----------------------------------------------------
/// The object behind a [`Tensor`]: an object header followed by a `DLTensor`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Tensor"]
#[type_index(unsafe TypeIndex::kTVMFFITensor)]
pub struct TensorObj {
    object: Object,
    dltensor: DLTensor,
}

/// A reference-counted n-dimensional array, passed across languages as
/// `ffi.Tensor` and exchanged with other frameworks through DLPack.
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
#[object_ref(skip_deref)]
pub struct Tensor {
    data: ObjectArc<TensorObj>,
}

impl Tensor {
    /// The data pointer, as in the `DLTensor`; `byte_offset` is not applied.
    ///
    /// A tensor is a shared handle, so `&mut self` would not make writes
    /// exclusive; getting the pointer is safe, and writing through it is the
    /// caller's `unsafe` responsibility, as for
    /// [`data_as_slice_mut`](Self::data_as_slice_mut).
    pub fn data_ptr(&self) -> *mut core::ffi::c_void {
        self.data.dltensor.data
    }

    /// Whether the elements are laid out in row-major order without gaps; see
    /// [`TensorView::is_contiguous`].
    pub fn is_contiguous(&self) -> bool {
        TensorView::from(self).is_contiguous()
    }

    /// The elements as a slice, starting `byte_offset` bytes into the data.
    ///
    /// # Errors
    /// A `TypeError` if `T` is not the tensor's dtype, and a `RuntimeError`
    /// if the tensor is not on the CPU, not contiguous, or its data is not
    /// aligned for `T`.
    pub fn data_as_slice<T: AsDLDataType>(&self) -> Result<&[T]> {
        let (data, len) = self.cpu_elements::<T>()?;
        unsafe { Ok(crate::util::slice_from_nullable_raw_parts(data, len)) }
    }
    /// The elements as a mutable slice, starting `byte_offset` bytes into the
    /// data.
    ///
    /// It takes `&self`: a tensor is a shared handle, and kernels write the
    /// data of tensors they receive by shared reference, so `&mut self` would
    /// not make the write exclusive.
    ///
    /// # Errors
    /// As [`data_as_slice`](Self::data_as_slice).
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
        let (data, len) = self.cpu_elements::<T>()?;
        unsafe { Ok(crate::util::slice_from_nullable_raw_parts_mut(data, len)) }
    }

    /// The first element and element count, checked for a slice of `T`.
    fn cpu_elements<T: AsDLDataType>(&self) -> Result<(*mut T, usize)> {
        let dtype = T::DL_DATA_TYPE;
        if self.dtype() != dtype {
            crate::bail!(
                crate::error::TYPE_ERROR,
                "Data type mismatch: expected {}, got {}",
                dtype.to_string(),
                self.dtype().to_string()
            );
        }
        if self.device().device_type != DLDeviceType::kDLCPU as i32 {
            crate::bail!(crate::error::RUNTIME_ERROR, "Tensor is not on CPU");
        }
        crate::ensure!(
            self.is_contiguous(),
            crate::error::RUNTIME_ERROR,
            "Tensor is not contiguous"
        );
        let len = self.numel();
        let t = &self.data.dltensor;
        let data = t
            .data
            .cast::<u8>()
            .wrapping_add(t.byte_offset as usize)
            .cast::<T>();
        // Imported DLPack data may sit at any address.
        crate::ensure!(
            len == 0 || data.is_aligned(),
            crate::error::RUNTIME_ERROR,
            "Tensor data is not aligned for {}",
            dtype.to_string()
        );
        Ok((data, len))
    }

    /// The extent of each dimension.
    pub fn shape(&self) -> &[i64] {
        let t = &self.data.dltensor;
        unsafe { crate::util::slice_from_nullable_raw_parts(t.shape, t.ndim as usize) }
    }

    /// The number of dimensions.
    pub fn ndim(&self) -> usize {
        self.data.dltensor.ndim as usize
    }

    /// The number of elements.
    ///
    /// # Panics
    /// As [`DLTensorExt::numel`].
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

    /// The element type.
    pub fn dtype(&self) -> DLDataType {
        self.data.dltensor.dtype
    }

    /// The device that holds the data.
    pub fn device(&self) -> DLDevice {
        self.data.dltensor.device
    }
}

#[repr(C)]
struct TensorObjFromNDAlloc<TNDAlloc>
where
    TNDAlloc: NDAllocator,
{
    base_: TensorObj,
    alloc: TNDAlloc,
}

unsafe impl<TNDAlloc: NDAllocator> ObjectCore for TensorObjFromNDAlloc<TNDAlloc> {
    type Parent = TensorObj;
    const TYPE_KEY: &'static str = TensorObj::TYPE_KEY;
    const TYPE_DEPTH: i32 = TensorObj::TYPE_DEPTH;
    fn type_index() -> i32 {
        TensorObj::type_index()
    }
    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        TensorObj::object_header_mut(&mut this.base_)
    }
}

unsafe impl<TNDAlloc: NDAllocator> ObjectCoreWithExtraItems for TensorObjFromNDAlloc<TNDAlloc> {
    type ExtraItem = i64;
    fn extra_items_count(this: &Self) -> usize {
        (this.base_.dltensor.ndim * 2) as usize
    }
}

impl<TNDAlloc: NDAllocator> Drop for TensorObjFromNDAlloc<TNDAlloc> {
    fn drop(&mut self) {
        unsafe {
            self.alloc.free_data(&self.base_.dltensor);
        }
    }
}

impl Tensor {
    /// Create a contiguous tensor whose data `alloc` allocates and frees.
    ///
    /// # Arguments
    /// * `alloc` - The allocator, kept by the tensor until it is dropped.
    /// * `shape` - The shape of the tensor.
    /// * `dtype` - The element type.
    /// * `device` - The device of the data.
    ///
    /// # Returns
    /// The new tensor, with row-major strides.
    ///
    /// # Panics
    /// If an extent of `shape` is negative or the byte size overflows `isize`.
    pub fn from_nd_alloc<TNDAlloc>(
        alloc: TNDAlloc,
        shape: &[i64],
        dtype: DLDataType,
        device: DLDevice,
    ) -> Self
    where
        TNDAlloc: NDAllocator,
    {
        let dltensor = DLTensor {
            data: std::ptr::null_mut(),
            device,
            ndim: shape.len() as i32,
            dtype,
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // Allocators size the data from the shape, so check it before allocating.
        data_size(&dltensor);
        let tensor_obj = TensorObjFromNDAlloc {
            base_: TensorObj {
                object: Object::new(),
                dltensor: DLTensor {
                    shape: std::ptr::null_mut(),
                    ..dltensor
                },
            },
            alloc,
        };
        unsafe {
            let mut obj_arc = ObjectArc::new_with_extra_items(tensor_obj);
            // The fresh allocation is unique and has not been exposed.
            let object = &mut *ObjectArc::as_mut_ptr(&mut obj_arc);
            object.base_.dltensor.shape =
                TensorObjFromNDAlloc::extra_items(object).as_ptr() as *mut i64;
            object.base_.dltensor.strides = object.base_.dltensor.shape.add(shape.len());
            let extra_items = TensorObjFromNDAlloc::extra_items_mut(object);
            extra_items[..shape.len()].copy_from_slice(shape);
            Shape::fill_strides_from_shape(shape, &mut extra_items[shape.len()..]);
            let dltensor_ptr = &object.base_.dltensor as *const DLTensor;
            object.base_.dltensor.data = object.alloc.alloc_data(&*dltensor_ptr);
            Self {
                data: ObjectArc::from_raw(ObjectArc::into_raw(obj_arc) as *mut TensorObj),
            }
        }
    }
    /// Create a contiguous tensor with this thread's environment allocator.
    ///
    /// Kernel libraries allocate intermediate tensors this way, so that they
    /// come from the caller's allocator, such as the one a framework sets
    /// with `TVMFFIEnvSetDLPackManagedTensorAllocator`.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor.
    /// * `dtype` - The element type.
    /// * `device` - The device of the data.
    ///
    /// # Returns
    /// The new tensor.
    ///
    /// # Errors
    /// The error the allocator raises, or a `RuntimeError` when no allocator
    /// is set.
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

    /// Create a CPU tensor holding a copy of `slice`.
    ///
    /// # Arguments
    /// * `slice` - The elements, in row-major order.
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    /// The new tensor, allocated with [`CPUNDAlloc`].
    ///
    /// # Errors
    /// A `ValueError` if `slice` does not have as many elements as `shape`,
    /// or `shape` has a negative extent.
    pub fn from_slice<T: AsDLDataType>(slice: &[T], shape: &[i64]) -> Result<Self> {
        if checked_numel(shape) != Some(slice.len()) {
            crate::bail!(crate::error::VALUE_ERROR, "Slice length mismatch");
        }
        let dtype = T::DL_DATA_TYPE;
        let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
        let tensor = Tensor::from_nd_alloc(CPUNDAlloc, shape, dtype, device);
        // SAFETY: the tensor was just allocated, so nothing else can access it.
        unsafe { tensor.data_as_slice_mut::<T>()? }.copy_from_slice(slice);
        Ok(tensor)
    }
}

/// An [`NDAllocator`] for CPU memory from the Rust global allocator, aligned
/// to 64 bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct CPUNDAlloc;

unsafe impl NDAllocator for CPUNDAlloc {
    const MIN_ALIGN: usize = 64;

    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut core::ffi::c_void {
        let size = data_size(prototype);
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
        let size = data_size(tensor);
        if size == 0 {
            return;
        }
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN).unwrap();
        std::alloc::dealloc(tensor.data as *mut u8, layout);
    }
}

/// A non-owning view of a tensor: a pointer to a `DLTensor`.
///
/// A function argument of this type accepts both a borrowed `DLTensor*` (type
/// index `kTVMFFIDLTensorPtr`, how C callers pass tensors) and a `Tensor`
/// object (type index `kTVMFFITensor`), and is passed on as a `DLTensor*`.
/// Prefer it over [`Tensor`] for arguments of exported functions that only
/// read a tensor's metadata and data.
///
/// The view does not keep anything alive: its lifetime `'a` ties it to what
/// it borrows. For the same reason, a view cannot be moved into an owned
/// [`Any`](crate::Any), which could outlive it; use [`Tensor`] instead.
/// Neither misuse compiles:
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
///
/// A view received as an argument of an exported function is valid only
/// during the call. The function may name any lifetime for it,
/// `TensorView<'static>` included, so the compiler does not stop it from
/// being kept past the call; doing so is undefined behavior.
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

    /// The device that holds the data.
    pub fn device(&self) -> DLDevice {
        self.dltensor().device
    }

    /// The element type.
    pub fn dtype(&self) -> DLDataType {
        self.dltensor().dtype
    }

    /// The number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dltensor().ndim as usize
    }

    /// The extent of each dimension.
    pub fn shape(&self) -> &'a [i64] {
        let t = self.dltensor();
        unsafe { crate::util::slice_from_nullable_raw_parts(t.shape, t.ndim as usize) }
    }

    /// The number of elements.
    ///
    /// # Panics
    /// As [`DLTensorExt::numel`].
    pub fn numel(&self) -> usize {
        self.dltensor().numel()
    }

    /// The strides, in elements.
    ///
    /// # Panics
    /// If the `DLTensor` has null strides and at least one dimension, which
    /// DLPack forbids since v1.2.
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

    /// Whether the elements are laid out in row-major order without gaps: a
    /// `DLTensor` with null strides or no elements is contiguous, and a
    /// dimension of extent 1 may have any stride.
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

// A typed function argument, valid only during the call; the lifetime is not
// checked against it (see the type docs).
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
