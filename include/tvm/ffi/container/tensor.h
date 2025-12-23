
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

/*!
 * \file tvm/ffi/container/tensor.h
 * \brief Container to store a Tensor.
 */
#ifndef TVM_FFI_CONTAINER_TENSOR_H_
#define TVM_FFI_CONTAINER_TENSOR_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/type_traits.h>

#include <optional>
#include <string>
#include <utility>

namespace tvm {
namespace ffi {

class Tensor;

/*!
 * \brief Check if the device uses direct address, where address of data indicate alignment.
 * \param device The input device.
 * \return True if the device uses direct address, false otherwise.
 */
inline bool IsDirectAddressDevice(const DLDevice& device) {
  return device.device_type <= kDLCUDAHost || device.device_type == kDLCUDAManaged ||
         device.device_type == kDLROCM || device.device_type == kDLROCMHost;
}

/*!
 * \brief check if a DLTensor is contiguous.
 * \param arr The input DLTensor.
 * \return The check result.
 */
inline bool IsContiguous(const DLTensor& arr) {
  if (arr.strides == nullptr) return true;
  int64_t expected_stride = 1;
  for (int32_t i = arr.ndim; i != 0; --i) {
    int32_t k = i - 1;
    if (arr.shape[k] == 1) {
      // Skip stride check if shape[k] is 1, where the dimension is contiguous
      // regardless of the value of stride.
      //
      // For example, PyTorch will normalize stride to 1 if shape is 1 when exporting
      // to DLPack.
      // More context: https://github.com/pytorch/pytorch/pull/83158
      continue;
    }
    if (arr.strides[k] != expected_stride) return false;
    expected_stride *= arr.shape[k];
  }
  return true;
}

/**
 * \brief Check if the data in the DLTensor is aligned to the given alignment.
 * \param arr The input DLTensor.
 * \param alignment The alignment to check.
 * \return True if the data is aligned to the given alignment, false otherwise.
 */
inline bool IsAligned(const DLTensor& arr, size_t alignment) {
  if (IsDirectAddressDevice(arr.device)) {
    return (reinterpret_cast<size_t>(static_cast<char*>(arr.data) + arr.byte_offset) % alignment ==
            0);
  } else {
    return arr.byte_offset % alignment == 0;
  }
}

/*!
 * \brief return the total number of bytes needed to store packed data
 *
 * \param numel the number of elements in the array
 * \param dtype the data type of the array
 * \return the total number of bytes needed to store packed data
 */
inline size_t GetDataSize(size_t numel, DLDataType dtype) {
  // compatible handling sub-byte uint1(bool), which usually stored as uint8_t
  // TODO(tqchen): revisit and switch to kDLBool
  if (dtype.code == kDLUInt && dtype.bits == 1 && dtype.lanes == 1) {
    return numel;
  }
  // for other sub-byte types, packing is preferred
  return (numel * dtype.bits * dtype.lanes + 7) / 8;
}

/*!
 * \brief return the size of data the DLTensor holds, in terms of number of bytes
 *
 *  \param arr the input DLTensor
 *  \return number of bytes of data in the DLTensor.
 */
inline size_t GetDataSize(const DLTensor& arr) {
  size_t size = 1;
  for (int i = 0; i < arr.ndim; ++i) {
    size *= static_cast<size_t>(arr.shape[i]);
  }
  return GetDataSize(size, arr.dtype);
}

/*! \brief An object representing a Tensor. */
class TensorObj : public Object, public DLTensor {
 public:
  /// \cond Doxygen_Suppress
  static constexpr const uint32_t _type_index = TypeIndex::kTVMFFITensor;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFITensor, TensorObj, Object);
  /// \endcond

  /*!
   * \brief Move a Tensor to a DLPack managed tensor.
   * \return The converted DLPack managed tensor.
   */
  DLManagedTensor* ToDLPack() const {
    TensorObj* self = const_cast<TensorObj*>(this);
    DLManagedTensor* ret = new DLManagedTensor();
    ret->dl_tensor = *static_cast<DLTensor*>(self);
    ret->manager_ctx = self;
    ret->deleter = DLManagedTensorDeleter<DLManagedTensor>;
    details::ObjectUnsafe::IncRefObjectHandle(self);
    return ret;
  }

  /*!
   * \brief Move a Tensor to a DLPack managed tensor.
   * \return The converted DLPack managed tensor.
   */
  DLManagedTensorVersioned* ToDLPackVersioned() const {
    TensorObj* self = const_cast<TensorObj*>(this);
    DLManagedTensorVersioned* ret = new DLManagedTensorVersioned();
    ret->version.major = DLPACK_MAJOR_VERSION;
    ret->version.minor = DLPACK_MINOR_VERSION;
    ret->dl_tensor = *static_cast<DLTensor*>(self);
    ret->manager_ctx = self;
    ret->deleter = DLManagedTensorDeleter<DLManagedTensorVersioned>;
    details::ObjectUnsafe::IncRefObjectHandle(self);
    return ret;
  }

 protected:
  /*!
   * \brief Deleter for DLManagedTensor.
   * \param tensor The DLManagedTensor to be deleted.
   */
  template <typename TDLManagedTensor>
  static void DLManagedTensorDeleter(TDLManagedTensor* tensor) {
    TensorObj* obj = static_cast<TensorObj*>(tensor->manager_ctx);
    details::ObjectUnsafe::DecRefObjectHandle(obj);
    delete tensor;
  }

  friend class Tensor;
};

namespace details {
/*!
 *\brief Helper class to create an TensorObj from an NDAllocator
 *
 * The underlying allocator needs to be implemented by user.
 */
template <typename TNDAlloc>
class TensorObjFromNDAlloc : public TensorObj {
 public:
  using Self = TensorObjFromNDAlloc<TNDAlloc>;

  template <typename... ExtraArgs>
  TensorObjFromNDAlloc(TNDAlloc alloc, ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                       ExtraArgs&&... extra_args)
      : alloc_(std::move(alloc)) {
    this->device = device;
    this->ndim = static_cast<int>(shape.size());
    this->dtype = dtype;
    this->byte_offset = 0;
    // inplace alloc shape and strides after data structure
    this->shape = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(this) + sizeof(Self));
    this->strides = this->shape + shape.size();
    std::copy(shape.begin(), shape.end(), this->shape);
    details::FillStridesFromShape(shape, this->strides);
    // call allocator to alloc data
    alloc_.AllocData(static_cast<DLTensor*>(this), std::forward<ExtraArgs>(extra_args)...);
  }

  template <typename... ExtraArgs>
  TensorObjFromNDAlloc(TNDAlloc alloc, const DLTensor& prototype, ExtraArgs&&... extra_args)
      : alloc_(std::move(alloc)) {
    *static_cast<DLTensor*>(this) = prototype;
    this->shape = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(this) + sizeof(Self));
    this->strides = this->shape + prototype.ndim;
    TVM_FFI_ICHECK_NOTNULL(prototype.strides);
    std::copy(prototype.shape, prototype.shape + prototype.ndim, this->shape);
    std::copy(prototype.strides, prototype.strides + prototype.ndim, this->strides);
    // call allocator to alloc data
    alloc_.AllocData(static_cast<DLTensor*>(this), std::forward<ExtraArgs>(extra_args)...);
  }

  ~TensorObjFromNDAlloc() { alloc_.FreeData(static_cast<DLTensor*>(this)); }

 private:
  TNDAlloc alloc_;
};

/*! \brief helper class to import from DLPack legacy DLManagedTensor */
template <typename TDLPackManagedTensor>
class TensorObjFromDLPack : public TensorObj {
 public:
  using Self = TensorObjFromDLPack<TDLPackManagedTensor>;

  explicit TensorObjFromDLPack(TDLPackManagedTensor* tensor, bool extra_strides_at_tail)
      : tensor_(tensor) {
    *static_cast<DLTensor*>(this) = tensor_->dl_tensor;
    if (extra_strides_at_tail) {
      this->strides = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(this) + sizeof(Self));
      details::FillStridesFromShape(ShapeView(tensor_->dl_tensor.shape, tensor_->dl_tensor.ndim),
                                    this->strides);
    }
  }

  ~TensorObjFromDLPack() {
    // run DLPack deleter if needed.
    if (tensor_->deleter != nullptr) {
      (*tensor_->deleter)(tensor_);
    }
  }

 private:
  TDLPackManagedTensor* tensor_;
};
}  // namespace details

/*!
 * \brief Managed Tensor (n-dimensional array).
 *  The tensor is backed by reference counted blocks.
 *
 * \note This class can be subclassed to implement downstream customized
 *       Tensor types that are backed by the same TensorObj storage type.
 */
class Tensor : public ObjectRef {
 public:
  /*!
   * \brief Default constructor.
   */
  Tensor() = default;
  /*!
   * \brief Constructor from a ObjectPtr<TensorObj>.
   * \param n The ObjectPtr<TensorObj>.
   */
  explicit Tensor(::tvm::ffi::ObjectPtr<TensorObj> n) : ObjectRef(std::move(n)) {}
  /*!
   * \brief Constructor from a UnsafeInit tag.
   * \param tag The UnsafeInit tag.
   */
  explicit Tensor(::tvm::ffi::UnsafeInit tag) : ObjectRef(tag) {}
  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(Tensor)
  /// \endcond
  /*!
   * \brief Get the data pointer of the Tensor.
   * \return The data pointer of the Tensor.
   */
  void* data_ptr() const { return get()->data; }

  /*!
   * \brief Get the device of the Tensor.
   * \return The device of the Tensor.
   */
  DLDevice device() const { return get()->device; }

  /*!
   * \brief Get the number of dimensions in the Tensor.
   * \return The number of dimensions in the Tensor.
   */
  int32_t ndim() const { return get()->ndim; }

  /*!
   * \brief Get the data type of the Tensor.
   * \return The data type of the Tensor.
   */
  DLDataType dtype() const { return get()->dtype; }

  /*!
   * \brief Get the shape of the Tensor.
   * \return The shape of the Tensor.
   */
  ShapeView shape() const {
    const TensorObj* obj = get();
    return tvm::ffi::ShapeView(obj->shape, obj->ndim);
  }

  /*!
   * \brief Get the strides of the Tensor.
   * \return The strides of the Tensor.
   */
  ShapeView strides() const {
    const TensorObj* obj = get();
    TVM_FFI_ICHECK(obj->strides != nullptr || obj->ndim == 0);
    return ShapeView(obj->strides, obj->ndim);
  }

  /*!
   * \brief Get the size of the idx-th dimension. If the idx is negative,
   * it gets the size of last idx-th dimension.
   * \param idx The index of the size.
   * \return The size of the idx-th dimension.
   */
  int64_t size(int64_t idx) const {
    const TensorObj* ptr = get();
    return ptr->shape[idx >= 0 ? idx : (ptr->ndim + idx)];
  }

  /*!
   * \brief Get the stride of the idx-th dimension. If the idx is negative,
   * it gets the stride of last idx-th dimension.
   * \param idx The index of the stride.
   * \return The stride of the idx-th dimension.
   */
  int64_t stride(int64_t idx) const {
    const TensorObj* ptr = get();
    return ptr->strides[idx >= 0 ? idx : (ptr->ndim + idx)];
  }

  /*!
   * \brief Get the number of elements in the Tensor.
   * \return The number of elements in the Tensor.
   */
  int64_t numel() const { return this->shape().Product(); }
  /*!
   * \brief Get the byte offset of the Tensor.
   * \return The byte offset of the Tensor.
   */
  uint64_t byte_offset() const { return get()->byte_offset; }
  /*!
   * \brief Check if the Tensor is contiguous.
   * \return True if the Tensor is contiguous, false otherwise.
   */
  bool IsContiguous() const { return tvm::ffi::IsContiguous(*get()); }
  /*!
   * \brief Check if the Tensor data is aligned to the given alignment.
   * \param alignment The alignment to check.
   * \return True if the Tensor data is aligned to the given alignment, false otherwise.
   */
  bool IsAligned(size_t alignment) const { return tvm::ffi::IsAligned(*get(), alignment); }

  /*!
   * \brief Create a new Tensor as a strided view of the current Tensor.
   * \param shape The shape of the new Tensor.
   * \param strides The strides of the new Tensor.
   * \param element_offset The element offset of the new Tensor in the unit of dtype elements.
   * \return The new Tensor.
   * \note element_offset is in the unit of dtype elements not bytes.
   */
  Tensor as_strided(ShapeView shape, ShapeView strides,
                    std::optional<int64_t> element_offset = std::nullopt) const {
    DLTensor prototype;
    prototype = *static_cast<const DLTensor*>(get());
    prototype.shape = const_cast<int64_t*>(shape.data());
    prototype.ndim = static_cast<int>(shape.size());
    prototype.strides = const_cast<int64_t*>(strides.data());
    int64_t elem_offset_as_i64 = element_offset.value_or(0);

    TVM_FFI_ICHECK_GE(elem_offset_as_i64, 0);
    prototype.byte_offset += GetDataSize(static_cast<size_t>(elem_offset_as_i64), prototype.dtype);

    if (prototype.byte_offset != 0 && IsDirectAddressDevice(prototype.device)) {
      // If the device supports direct address, we can just add the byte offset to the data pointer.
      prototype.data =
          reinterpret_cast<void*>(reinterpret_cast<char*>(prototype.data) + prototype.byte_offset);
      prototype.byte_offset = 0;
    }

    TVMFFIObjectHandle out;
    Object* obj_handle = const_cast<TensorObj*>(get());
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITensorCreateUnsafeView(obj_handle, &prototype, &out));
    return Tensor(
        details::ObjectUnsafe::ObjectPtrFromOwned<TensorObj>(static_cast<TVMFFIObject*>(out)));
  }
  /*!
   * \brief Create a Tensor from a NDAllocator.
   *
   * \note When building a kernel library, we always recommend use FromEnvAlloc when possible to
   * allocate intermediate Tensors. When a loaded module returns an allocated tensor to the caller,
   * we need to keep the module alive before the returned tensors get freed, because its
   * deleter is defined within the module. FromNDAlloc can be used by C++ applications and runtimes
   * to create Tensors.
   *
   * Example usage:
   * \code{.cpp}
   * // CPU Allocator
   * struct CPUNDAlloc {
   *   void AllocData(DLTensor* tensor) { tensor->data = malloc(ffi::GetDataSize(*tensor)); }
   *   void FreeData(DLTensor* tensor) { free(tensor->data); }
   * };
   *
   * // CUDA Allocator
   * struct CUDANDAlloc {
   *   void AllocData(DLTensor* tensor) {
   *     size_t data_size = ffi::GetDataSize(*tensor);
   *     void* ptr = nullptr;
   *     cudaError_t err = cudaMalloc(&ptr, data_size);
   *     TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
   *     tensor->data = ptr;
   *   }
   *   void FreeData(DLTensor* tensor) {
   *     if (tensor->data != nullptr) {
   *       cudaError_t err = cudaFree(tensor->data);
   *       TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
   *       tensor->data = nullptr;
   *     }
   *   }
   * };
   *
   * // NVSHMEM Allocator
   * struct NVSHMEMNDAlloc {
   *   void AllocData(DLTensor* tensor) {
   *     size_t size = tvm::ffi::GetDataSize(*tensor);
   *     tensor->data = nvshmem_malloc(size);
   *     TVM_FFI_ICHECK_NE(tensor->data, nullptr) << "nvshmem_malloc failed. size: " << size;
   *   }
   *   void FreeData(DLTensor* tensor) { nvshmem_free(tensor->data); }
   * };
   *
   * // Allocator usage
   * ffi::Tensor cpu_tensor = ffi::Tensor::FromNDAlloc(CPUNDAlloc(), ...);
   * ffi::Tensor cuda_tensor = ffi::Tensor::FromNDAlloc(CUDANDAlloc(), ...);
   * ffi::Tensor nvshmem_tensor = ffi::Tensor::FromNDAlloc(NVSHMEMNDAlloc(), ...);
   * \endcode
   *
   * \param alloc The NDAllocator.
   * \param shape The shape of the Tensor.
   * \param dtype The data type of the Tensor.
   * \param device The device of the Tensor.
   * \param extra_args Extra arguments to be forwarded to TNDAlloc.
   * \return The created Tensor.
   * \tparam TNDAlloc The type of the NDAllocator, impelments Alloc and Free.
   * \tparam ExtraArgs Extra arguments to be passed to Alloc.
   */
  template <typename TNDAlloc, typename... ExtraArgs>
  static Tensor FromNDAlloc(TNDAlloc alloc, ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                            ExtraArgs&&... extra_args) {
    // inplace alloc shape and strides after data structure (as a result why multiply 2)
    size_t num_extra_i64_at_tail = shape.size() * 2;
    return Tensor(make_inplace_array_object<details::TensorObjFromNDAlloc<TNDAlloc>, int64_t>(
        num_extra_i64_at_tail, alloc, shape, dtype, device,
        std::forward<ExtraArgs>(extra_args)...));
  }

  /*!
   * \brief A variant of FromNDAlloc that allows explicit passing a strides.
   *
   * \note This function needs to ensure that strides are well-defined
   *       with respect to the allocated compact shape.
   *
   * \param alloc The NDAllocator.
   * \param shape The shape of the Tensor.
   * \param strides The strides of the Tensor.
   * \param dtype The data type of the Tensor.
   * \param device The device of the Tensor.
   * \param extra_args Extra arguments to be forwarded to TNDAlloc.
   * \return The created Tensor.
   * \tparam TNDAlloc The type of the NDAllocator, impelments Alloc and Free.
   * \tparam ExtraArgs Extra arguments to be passed to Alloc.
   */
  template <typename TNDAlloc, typename... ExtraArgs>
  static Tensor FromNDAllocStrided(TNDAlloc alloc, ffi::ShapeView shape, ffi::ShapeView strides,
                                   DLDataType dtype, DLDevice device, ExtraArgs&&... extra_args) {
    TVM_FFI_CHECK(shape.size() == strides.size(), ValueError)
        << "shape and strides must have the same size.";
    // inplace alloc shape and strides after data structure (as a result why multiply 2)
    size_t num_extra_i64_at_tail = shape.size() * 2;
    DLTensor prototype;
    prototype.data = nullptr;
    prototype.device = device;
    prototype.dtype = dtype;
    prototype.shape = const_cast<int64_t*>(shape.data());
    prototype.ndim = static_cast<int>(shape.size());
    prototype.strides = const_cast<int64_t*>(strides.data());
    prototype.byte_offset = 0;
    return Tensor(make_inplace_array_object<details::TensorObjFromNDAlloc<TNDAlloc>, int64_t>(
        num_extra_i64_at_tail, alloc, prototype, std::forward<ExtraArgs>(extra_args)...));
  }
  /*!
   * \brief Create a Tensor from the TVMFFIEnvTensorAlloc API
   *
   * This function can be used together with TVMFFIEnvSetDLPackManagedTensorAllocator
   * in the extra/c_env_api.h to create a Tensor from the thread-local environment allocator.
   * We explicitly pass TVMFFIEnvTensorAlloc to maintain explicit dependency on extra/c_env_api.h
   *
   * \code{.cpp}
   * ffi::Tensor tensor = ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape, dtype, device);
   * \endcode
   *
   * \param env_alloc TVMFFIEnvTensorAlloc function pointer.
   * \param shape The shape of the Tensor.
   * \param dtype The data type of the Tensor.
   * \param device The device of the Tensor.
   * \return The created Tensor.
   *
   * \sa TVMFFIEnvTensorAlloc
   */
  static Tensor FromEnvAlloc(int (*env_alloc)(DLTensor*, TVMFFIObjectHandle*), ffi::ShapeView shape,
                             DLDataType dtype, DLDevice device) {
    TVMFFIObjectHandle out;
    DLTensor prototype{};
    prototype.device = device;
    prototype.dtype = dtype;
    prototype.shape = const_cast<int64_t*>(shape.data());
    prototype.ndim = static_cast<int>(shape.size());
    TVM_FFI_CHECK_SAFE_CALL(env_alloc(&prototype, &out));
    return Tensor(
        details::ObjectUnsafe::ObjectPtrFromOwned<TensorObj>(static_cast<TVMFFIObject*>(out)));
  }
  /*!
   * \brief Create a Tensor from a DLPack managed tensor, pre v1.0 API.
   * \param tensor The input DLPack managed tensor.
   * \param require_alignment The minimum alignment requored of the data + byte_offset.
   * \param require_contiguous Boolean flag indicating if we need to check for contiguity.
   * \note This function will not run any checks on flags.
   * \return The created Tensor.
   */
  static Tensor FromDLPack(DLManagedTensor* tensor, size_t require_alignment = 0,
                           bool require_contiguous = false) {
    if (require_alignment != 0 && !ffi::IsAligned(tensor->dl_tensor, require_alignment)) {
      TVM_FFI_THROW(RuntimeError) << "FromDLPack: Data is not aligned to " << require_alignment
                                  << " bytes.";
    }
    if (require_contiguous && !ffi::IsContiguous(tensor->dl_tensor)) {
      TVM_FFI_THROW(RuntimeError) << "FromDLPack: Tensor is not contiguous.";
    }
    if (tensor->dl_tensor.strides != nullptr || tensor->dl_tensor.ndim == 0) {
      return Tensor(make_object<details::TensorObjFromDLPack<DLManagedTensor>>(
          tensor, /*extra_strides_at_tail=*/false));
    } else {
      return Tensor(
          make_inplace_array_object<details::TensorObjFromDLPack<DLManagedTensor>, int64_t>(
              tensor->dl_tensor.ndim, tensor, /*extra_strides_at_tail=*/true));
    }
  }

  /*!
   * \brief Create a Tensor from a DLPack managed tensor, post v1.0 API.
   * \param tensor The input DLPack managed tensor.
   * \param require_alignment The minimum alignment requored of the data + byte_offset.
   * \param require_contiguous Boolean flag indicating if we need to check for contiguity.
   * \return The created Tensor.
   */
  static Tensor FromDLPackVersioned(DLManagedTensorVersioned* tensor, size_t require_alignment = 0,
                                    bool require_contiguous = false) {
    if (require_alignment != 0 && !ffi::IsAligned(tensor->dl_tensor, require_alignment)) {
      TVM_FFI_THROW(RuntimeError) << "FromDLPack: Data is not aligned to " << require_alignment
                                  << " bytes.";
    }
    if (require_contiguous && !ffi::IsContiguous(tensor->dl_tensor)) {
      TVM_FFI_THROW(RuntimeError) << "FromDLPack: Tensor is not contiguous.";
    }
    if (tensor->flags & DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED) {
      TVM_FFI_THROW(RuntimeError) << "Subbyte type padded is not yet supported";
    }
    if (tensor->dl_tensor.strides != nullptr || tensor->dl_tensor.ndim == 0) {
      return Tensor(make_object<details::TensorObjFromDLPack<DLManagedTensorVersioned>>(
          tensor, /*extra_strides_at_tail=*/false));
    } else {
      return Tensor(
          make_inplace_array_object<details::TensorObjFromDLPack<DLManagedTensorVersioned>,
                                    int64_t>(tensor->dl_tensor.ndim, tensor,
                                             /*extra_strides_at_tail=*/true));
    }
  }

  /*!
   * \brief Convert the Tensor to a DLPack managed tensor.
   * \return The converted DLPack managed tensor.
   */
  DLManagedTensor* ToDLPack() const { return get_mutable()->ToDLPack(); }

  /*!
   * \brief Convert the Tensor to a DLPack managed tensor.
   * \return The converted DLPack managed tensor.
   */
  DLManagedTensorVersioned* ToDLPackVersioned() const { return get_mutable()->ToDLPackVersioned(); }
  /*!
   * \brief Get the underlying DLTensor pointer.
   * \return The underlying DLTensor pointer.
   */
  const DLTensor* GetDLTensorPtr() const { return get(); }
  /// \cond Doxygen_Suppress
  [[maybe_unused]] static constexpr bool _type_is_nullable = true;
  using ContainerType = TensorObj;
  /// \endcond

  // the following code are convenient APIs redirections created to provide aten-style api
  /*!
   * \brief This functions redirects to ndim().
   * \return The number of dimensions in the Tensor.
   */
  inline int32_t dim() { return ndim(); }
  /*!
   * \brief This functions redirects to shape().
   * \return The shape of the Tensor.
   */
  inline ShapeView sizes() const { return shape(); }
  /*!
   * \brief This functions redirects to IsContiguous().
   * \return True if the Tensor is contiguous, false otherwise.
   */
  inline bool is_contiguous() const { return IsContiguous(); }

 protected:
  /*!
   * \brief Get const internal container pointer.
   * \return a const container pointer.
   */
  const TensorObj* get() const { return static_cast<const TensorObj*>(ObjectRef::get()); }
  /*!
   * \brief Get mutable internal container pointer.
   * \return a mutable container pointer.
   */
  TensorObj* get_mutable() const { return const_cast<TensorObj*>(get()); }
};

/*!
 * \brief A non-owning view of a Tensor.
 *
 * This class stores a light-weight non-owning view of a Tensor.
 * This is useful for accessing tensor data without retaining a strong reference to the Tensor.
 * Since the caller may not always be able to pass in a strong referenced tensor.
 *
 * It is the user's responsibility to ensure
 * that the underlying tensor data outlives the `TensorView`.
 * This responsibility extends to all data pointed to by the underlying DLTensor.
 * This includes not only the tensor elements in data but also the memory for shape and strides.
 *
 * When exposing a function that expects only expects a TensorView, we recommend using
 * ffi::TensorView as the argument type instead of ffi::Tensor.
 */
class TensorView {
 public:
  /*!
   * \brief Create a TensorView from a Tensor.
   * \param tensor The input Tensor.
   */
  TensorView(const Tensor& tensor) {  // NOLINT(*)
    TVM_FFI_ICHECK(tensor.defined());
    tensor_ = *tensor.GetDLTensorPtr();
  }  // NOLINT(*)
  /*!
   * \brief Create a TensorView from a DLTensor.
   * \param tensor The input DLTensor.
   */
  TensorView(const DLTensor* tensor) {  // NOLINT(*)
    TVM_FFI_ICHECK(tensor != nullptr);
    tensor_ = *tensor;
  }
  /*!
   * \brief Copy constructor.
   * \param tensor The input TensorView.
   */
  TensorView(const TensorView& tensor) = default;
  /*!
   * \brief Move constructor.
   * \param tensor The input TensorView.
   */
  TensorView(TensorView&& tensor) = default;
  /*!
   * \brief Copy assignment operator.
   * \param tensor The input TensorView.
   * \return The created TensorView.
   */
  TensorView& operator=(const TensorView& tensor) = default;
  /*!
   * \brief Move assignment operator.
   * \param tensor The input TensorView.
   * \return The created TensorView.
   */
  TensorView& operator=(TensorView&& tensor) = default;
  /*!
   * \brief Assignment operator from a Tensor.
   * \param tensor The input Tensor.
   * \return The created TensorView.
   */
  TensorView& operator=(const Tensor& tensor) {
    TVM_FFI_ICHECK(tensor.defined());
    tensor_ = *tensor.GetDLTensorPtr();
    return *this;
  }

  // explicitly delete move constructor
  TensorView(Tensor&& tensor) = delete;  // NOLINT(*)
  // delete move assignment operator from owned tensor
  TensorView& operator=(Tensor&& tensor) = delete;
  /*!
   * \brief Get the data pointer of the Tensor.
   * \return The data pointer of the Tensor.
   */
  void* data_ptr() const { return tensor_.data; }
  /*!
   * \brief Get the device of the Tensor.
   * \return The device of the Tensor.
   */
  DLDevice device() const { return tensor_.device; }
  /*!
   * \brief Get the number of dimensions in the Tensor.
   * \return The number of dimensions in the Tensor.
   */
  int32_t ndim() const { return tensor_.ndim; }
  /*!
   * \brief Get the data type of the Tensor.
   * \return The data type of the Tensor.
   */
  DLDataType dtype() const { return tensor_.dtype; }
  /*!
   * \brief Get the shape of the Tensor.
   * \return The shape of the Tensor.
   */
  ShapeView shape() const { return ShapeView(tensor_.shape, tensor_.ndim); }

  /*!
   * \brief Get the number of elements in the Tensor.
   * \return The number of elements in the Tensor.
   */
  int64_t numel() const { return this->shape().Product(); }

  /*!
   * \brief Get the strides of the Tensor.
   * \return The strides of the Tensor.
   */
  ShapeView strides() const {
    TVM_FFI_ICHECK(tensor_.strides != nullptr || tensor_.ndim == 0);
    return ShapeView(tensor_.strides, tensor_.ndim);
  }

  /*!
   * \brief Get the size of the idx-th dimension. If the idx is negative,
   * it gets the size of last idx-th dimension.
   * \param idx The index of the size.
   * \return The size of the idx-th dimension.
   */
  int64_t size(int64_t idx) const { return tensor_.shape[idx >= 0 ? idx : tensor_.ndim + idx]; }

  /*!
   * \brief Get the stride of the idx-th dimension. If the idx is negative,
   * it gets the stride of last idx-th dimension.
   * \param idx The index of the stride.
   * \return The stride of the idx-th dimension.
   */
  int64_t stride(int64_t idx) const { return tensor_.strides[idx >= 0 ? idx : tensor_.ndim + idx]; }

  /*!
   * \brief Get the byte offset of the Tensor.
   * \return The byte offset of the Tensor.
   */
  uint64_t byte_offset() const { return tensor_.byte_offset; }

  /*!
   * \brief Check if the Tensor is contiguous.
   * \return True if the Tensor is contiguous, false otherwise.
   */
  bool IsContiguous() const { return tvm::ffi::IsContiguous(tensor_); }

  // the following code are convenient APIs redirections created to provide aten-style api
  /*!
   * \brief This functions redirects to ndim().
   * \return The number of dimensions in the Tensor.
   */
  int32_t dim() const { return ndim(); }
  /*!
   * \brief This functions redirects to shape().
   * \return The shape of the Tensor.
   */
  ShapeView sizes() const { return shape(); }
  /*!
   * \brief This functions redirects to IsContiguous().
   * \return True if the Tensor is contiguous, false otherwise.
   */
  bool is_contiguous() const { return IsContiguous(); }
  /*!
   * \brief Create a new TensorView as a strided view of the current TensorView.
   *
   * Use this function with extreme caution. The user must ensure that the shape and strides
   * arrays, as well as the data pointer, remain valid for the lifetime of the returned TensorView.
   *
   * One common anti-pattern is to create temporary shape/strides arrays and pass them in, but
   * then deallocate the temporary arrays immediately after the call to as_strided, which
   * causes the returned TensorView to point to invalid memory.
   *
   * \param shape The shape of the new TensorView.
   * \param strides The strides of the new TensorView.
   * \param element_offset The element offset of the new TensorView in units of dtype elements, not
   * bytes.
   * \return The new TensorView.
   *
   * \note The caller must ensure that the shape and strides arrays remain valid for the lifetime
   *       of the returned TensorView.
   */
  TensorView as_strided(ShapeView shape, ShapeView strides,
                        std::optional<int64_t> element_offset = std::nullopt) const {
    DLTensor prototype = tensor_;
    prototype.shape = const_cast<int64_t*>(shape.data());
    prototype.ndim = static_cast<int>(shape.size());
    prototype.strides = const_cast<int64_t*>(strides.data());
    TVM_FFI_ICHECK_EQ(shape.size(), strides.size());
    int64_t elem_offset_as_i64 = element_offset.value_or(0);
    TVM_FFI_ICHECK_GE(elem_offset_as_i64, 0);
    prototype.byte_offset += GetDataSize(static_cast<size_t>(elem_offset_as_i64), prototype.dtype);
    if (prototype.byte_offset != 0 && IsDirectAddressDevice(prototype.device)) {
      // If the device supports direct address, we can just add the byte offset to the data pointer.
      prototype.data =
          reinterpret_cast<void*>(reinterpret_cast<char*>(prototype.data) + prototype.byte_offset);
      prototype.byte_offset = 0;
    }
    return TensorView(&prototype);
  }

 private:
  DLTensor tensor_;
  template <typename, typename>
  friend struct TypeTraits;
};

/*!
 * \brief Get the data size of the Tensor.
 * \param tensor The input Tensor.
 * \return The data size of the Tensor.
 */
inline size_t GetDataSize(const Tensor& tensor) {
  return GetDataSize(tensor.numel(), tensor.dtype());
}

/*!
 * \brief Get the data size of the TensorView.
 * \param tensor The input TensorView.
 * \return The data size of the TensorView.
 */
inline size_t GetDataSize(const TensorView& tensor) {
  return GetDataSize(tensor.numel(), tensor.dtype());
}

// TensorView type, allow implicit casting from DLTensor*
// NOTE: we deliberately do not support MoveToAny and MoveFromAny since it does not retain ownership
template <>
struct TypeTraits<TensorView> : public TypeTraitsBase {
  static constexpr bool storage_enabled = false;
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDLTensorPtr;

  TVM_FFI_INLINE static void CopyToAnyView(const TensorView& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDLTensorPtr;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_ptr = const_cast<DLTensor*>(&(src.tensor_));
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDLTensorPtr;
  }

  TVM_FFI_INLINE static TensorView CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return TensorView(static_cast<DLTensor*>(src->v_ptr));
  }

  TVM_FFI_INLINE static std::optional<TensorView> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDLTensorPtr) {
      return TensorView(static_cast<DLTensor*>(src->v_ptr));
    } else if (src->type_index == TypeIndex::kTVMFFITensor) {
      return TensorView(TVMFFITensorGetDLTensorPtr(src->v_obj));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIDLTensorPtr; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIDLTensorPtr) + R"("})";
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_CONTAINER_TENSOR_H_
