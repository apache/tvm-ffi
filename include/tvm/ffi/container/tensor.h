
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

#include <atomic>
#include <memory>
#include <string>
#include <utility>

namespace tvm {
namespace ffi {

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
inline size_t GetDataSize(int64_t numel, DLDataType dtype) {
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
      : alloc_(alloc) {
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
    TVM_FFI_ICHECK(obj->strides != nullptr);
    return ShapeView(obj->strides, obj->ndim);
  }

  /*!
   * \brief Get the data pointer of the Tensor.
   * \return The data pointer of the Tensor.
   */
  void* data_ptr() const { return (*this)->data; }

  /*!
   * \brief Get the number of dimensions in the Tensor.
   * \return The number of dimensions in the Tensor.
   */
  int32_t ndim() const { return (*this)->ndim; }

  /*!
   * \brief Get the number of elements in the Tensor.
   * \return The number of elements in the Tensor.
   */
  int64_t numel() const { return this->shape().Product(); }

  /*!
   * \brief Get the data type of the Tensor.
   * \return The data type of the Tensor.
   */
  DLDataType dtype() const { return (*this)->dtype; }
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
   * \brief Create a Tensor from a NDAllocator.
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
   * \brief Create a Tensor from a DLPackTensorAllocator
   *
   * This function can be used together with TVMFFIEnvSetTensorAllocator
   * in the extra/c_env_api.h to create Tensor from the thread-local
   * environment allocator.
   *
   * \code
   *
   * ffi::Tensor tensor = ffi::Tensor::FromDLPackAlloc(
   *   TVMFFIEnvGetTensorAllocator(), shape, dtype, device
   * );
   * \endcode
   *
   * \param allocator The DLPack allocator.
   * \param shape The shape of the Tensor.
   * \param dtype The data type of the Tensor.
   * \param device The device of the Tensor.
   * \return The created Tensor.
   */
  static Tensor FromDLPackAlloc(DLPackTensorAllocator allocator, ffi::Shape shape, DLDataType dtype,
                                DLDevice device) {
    if (allocator == nullptr) {
      TVM_FFI_THROW(RuntimeError)
          << "FromDLPackAlloc: allocator is nullptr, "
          << "likely because TVMFFIEnvSetTensorAllocator has not been called.";
    }
    DLTensor prototype;
    prototype.device = device;
    prototype.dtype = dtype;
    prototype.shape = const_cast<int64_t*>(shape.data());
    prototype.ndim = static_cast<int>(shape.size());
    prototype.strides = nullptr;
    prototype.byte_offset = 0;
    prototype.data = nullptr;
    DLManagedTensorVersioned* tensor = nullptr;
    // error context to be used to propagate error
    struct ErrorContext {
      std::string kind;
      std::string message;
      static void SetError(void* error_ctx, const char* kind, const char* message) {
        ErrorContext* error_context = static_cast<ErrorContext*>(error_ctx);
        error_context->kind = kind;
        error_context->message = message;
      }
    };
    ErrorContext error_context;
    int ret = (*allocator)(&prototype, &tensor, &error_context, ErrorContext::SetError);
    if (ret != 0) {
      throw ffi::Error(error_context.kind, error_context.message,
                       TVMFFIBacktrace(__FILE__, __LINE__, __func__, 0));
    }
    if (tensor->dl_tensor.strides != nullptr) {
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
    if (tensor->dl_tensor.strides != nullptr) {
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
    if (tensor->dl_tensor.strides != nullptr) {
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

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tensor, ObjectRef, TensorObj);
  /// \endcond

 protected:
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
    tensor_ = *tensor.operator->();
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
    tensor_ = *tensor.operator->();
    return *this;
  }

  // explicitly delete move constructor
  TensorView(Tensor&& tensor) = delete;  // NOLINT(*)
  // delete move assignment operator from owned tensor
  TensorView& operator=(Tensor&& tensor) = delete;
  /*!
   * \brief Get the underlying DLTensor pointer.
   * \return The underlying DLTensor pointer.
   */
  const DLTensor* operator->() const { return &tensor_; }

  /*!
   * \brief Get the shape of the Tensor.
   * \return The shape of the Tensor.
   */
  ShapeView shape() const { return ShapeView(tensor_.shape, tensor_.ndim); }

  /*!
   * \brief Get the strides of the Tensor.
   * \return The strides of the Tensor.
   */
  ShapeView strides() const {
    TVM_FFI_ICHECK(tensor_.strides != nullptr);
    return ShapeView(tensor_.strides, tensor_.ndim);
  }

  /*!
   * \brief Get the data pointer of the Tensor.
   * \return The data pointer of the Tensor.
   */
  void* data_ptr() const { return tensor_.data; }

  /*!
   * \brief Get the number of dimensions in the Tensor.
   * \return The number of dimensions in the Tensor.
   */
  int32_t ndim() const { return tensor_.ndim; }

  /*!
   * \brief Get the number of elements in the Tensor.
   * \return The number of elements in the Tensor.
   */
  int64_t numel() const { return this->shape().Product(); }

  /*!
   * \brief Get the data type of the Tensor.
   * \return The data type of the Tensor.
   */
  DLDataType dtype() const { return tensor_.dtype; }

  /*!
   * \brief Check if the Tensor is contiguous.
   * \return True if the Tensor is contiguous, false otherwise.
   */
  bool IsContiguous() const { return tvm::ffi::IsContiguous(tensor_); }

 private:
  DLTensor tensor_;
};

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
    result->v_ptr = const_cast<DLTensor*>(src.operator->());
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
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_CONTAINER_TENSOR_H_
