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
 * \file tvm/ffi/type_traits.h
 * \brief Type trait helpers for FFI values and lookup-only ABI objects.
 */
#ifndef TVM_FFI_TYPE_TRAITS_H_
#define TVM_FFI_TYPE_TRAITS_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

class Any;

/*!
 * \brief TypeIndex enum, alias of TVMFFITypeIndex.
 */
using TypeIndex = TVMFFITypeIndex;
/*!
 * \brief TypeInfo, alias of TVMFFITypeInfo.
 */
using TypeInfo = TVMFFITypeInfo;

/*!
 * \brief Known type keys for pre-defined types.
 */
struct StaticTypeKey {
  /*! \brief The type key for Any */
  static constexpr const char* kTVMFFIAny = "Any";
  /*! \brief The type key for None */
  static constexpr const char* kTVMFFINone = "None";
  /*! \brief The type key for bool */
  static constexpr const char* kTVMFFIBool = "bool";
  /*! \brief The type key for int */
  static constexpr const char* kTVMFFIInt = "int";
  /*! \brief The type key for float */
  static constexpr const char* kTVMFFIFloat = "float";
  /*! \brief The type key for void* */
  static constexpr const char* kTVMFFIOpaquePtr = "void*";
  /*! \brief The type key for DataType */
  static constexpr const char* kTVMFFIDataType = "DataType";
  /*! \brief The type key for Device */
  static constexpr const char* kTVMFFIDevice = "Device";
  /*! \brief The type key for DLTensor* */
  static constexpr const char* kTVMFFIDLTensorPtr = "DLTensor*";
  /*! \brief The type key for const char* */
  static constexpr const char* kTVMFFIRawStr = "const char*";
  /*! \brief The type key for TVMFFIByteArray* */
  static constexpr const char* kTVMFFIByteArrayPtr = "TVMFFIByteArray*";
  /*! \brief The type key for ObjectRValueRef */
  static constexpr const char* kTVMFFIObjectRValueRef = "ObjectRValueRef";
  /*! \brief The type key for SmallStr */
  static constexpr const char* kTVMFFISmallStr = "ffi.SmallStr";
  /*! \brief The type key for SmallBytes */
  static constexpr const char* kTVMFFISmallBytes = "ffi.SmallBytes";
  /*! \brief The type key for Error */
  static constexpr const char* kTVMFFIError = "ffi.Error";
  /*! \brief The type key for Bytes */
  static constexpr const char* kTVMFFIBytes = "ffi.Bytes";
  /*! \brief The type key for String */
  static constexpr const char* kTVMFFIStr = "ffi.String";
  /*! \brief The type key for Shape */
  static constexpr const char* kTVMFFIShape = "ffi.Shape";
  /*! \brief The type key for Tensor */
  static constexpr const char* kTVMFFITensor = "ffi.Tensor";
  /*! \brief The type key for Object */
  static constexpr const char* kTVMFFIObject = "ffi.Object";
  /*! \brief The type key for Function */
  static constexpr const char* kTVMFFIFunction = "ffi.Function";
  /*! \brief The type key for Array */
  static constexpr const char* kTVMFFIArray = "ffi.Array";
  /*! \brief The type key for List */
  static constexpr const char* kTVMFFIList = "ffi.List";
  /*! \brief The type key for Map */
  static constexpr const char* kTVMFFIMap = "ffi.Map";
  /*! \brief The type key for Module */
  static constexpr const char* kTVMFFIModule = "ffi.Module";
  /*! \brief The type key for Dict */
  static constexpr const char* kTVMFFIDict = "ffi.Dict";
  /*! \brief The type key for VisitInterrupt */
  static constexpr const char* kTVMFFIVisitInterrupt = "ffi.VisitInterrupt";
  /*! \brief The type key for OpaquePyObject */
  static constexpr const char* kTVMFFIOpaquePyObject = "ffi.OpaquePyObject";
};

/*!
 * \brief Get type key from type index
 * \param type_index The input type index
 * \return the type key
 */
inline std::string TypeIndexToTypeKey(int32_t type_index) {
  const TypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  return std::string(type_info->type_key.data, type_info->type_key.size);
}

/*!
 * \brief Whether TargetType subsumes SourceType for direct storage reuse.
 *
 * The target type is first and the source type is second. The result is true
 * exactly when every SourceType value can reuse TargetType storage without
 * conversion.
 *
 * \tparam TargetType The target storage type.
 * \tparam SourceType The source value type.
 */
template <typename TargetType, typename SourceType, typename = void>
inline constexpr bool type_subsumes_v =
    std::is_base_of_v<TargetType, SourceType> || std::is_same_v<TargetType, SourceType>;

/// \cond Doxygen_Suppress
/*! \brief Whether Any subsumes SourceType for direct storage reuse. */
template <typename SourceType>
inline constexpr bool type_subsumes_v<Any, SourceType> = true;
/// \endcond

/*!
 * \brief TypeTraits that specifies the conversion behavior from/to FFI Any.
 *
 * The function specifications of TypeTraits<T>
 *
 * - CopyToAnyView: Convert a value T to AnyView
 * - MoveToAny: Move a value to Any
 * - CheckAnyStrict: Check if a Any stores a result of CopyToAnyView of current T.
 * - CopyFromAnyViewAfterCheck: Copy a value T from Any view after we pass CheckAnyStrict.
 * - MoveFromAnyAfterCheck: Move a value T from Any storage after we pass CheckAnyStrict.
 * - TryCastFromAnyView: Convert a AnyView to a T, we may apply type conversion.
 * - GetMismatchTypeInfo: Get the type key of a type when TryCastFromAnyView fails.
 * - TypeStr: Get the type key of a type
 *
 * It is possible that CheckAnyStrict is false but TryCastFromAnyView still works.
 *
 * For example, when Any x stores int, TypeTraits<float>::CheckAnyStrict(x) will be false,
 * but TypeTraits<float>::TryCastFromAnyView(x) will return a corresponding float value
 * via type conversion.
 *
 * CheckAnyStrict is mainly used in recursive container such as Array<T> to
 * decide if a new Array needed to be created via recursive conversion,
 * or we can use the current container as is when converting to Array<T>.
 *
 * A container array: Array<T> satisfies the following invariant:
 * - `all(TypeTraits<T>::CheckAnyStrict(x) for x in the array)`.
 */
template <typename, typename = void>
struct TypeTraits {
  /*! \brief Whether the type is enabled in FFI. */
  static constexpr bool convert_enabled = false;
  /*! \brief Whether the type can appear as a storage type in Container */
  static constexpr bool storage_enabled = false;
};

/*!
 * \brief TypeTraits that removes const and reference keywords.
 * \tparam T the original type
 */
template <typename T>
using TypeTraitsNoCR = TypeTraits<std::remove_const_t<std::remove_reference_t<T>>>;

template <typename T>
inline constexpr bool use_default_type_traits_v = true;

struct TypeTraitsBase {
  static constexpr bool convert_enabled = true;
  static constexpr bool storage_enabled = true;
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;
  // get mismatched type when result mismatches the trait.
  // this function is called after TryCastFromAnyView fails
  // to get more detailed type information in runtime
  // especially when the error involves nested container type
  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* source) {
    return TypeIndexToTypeKey(source->type_index);
  }
};

/*!
 * \brief Trait that maps a type to its field static type index
 * \tparam T the type
 * \return the field static type index
 */
template <typename T, typename = void>
struct TypeToFieldStaticTypeIndex {
  /*! \brief The field static type index of the type  */
  static constexpr int32_t value = TypeIndex::kTVMFFIAny;
};

template <typename T>
struct TypeToFieldStaticTypeIndex<T, std::enable_if_t<TypeTraits<T>::convert_enabled>> {
  static constexpr int32_t value = TypeTraits<T>::field_static_type_index;
};

/*!
 * \brief Trait that maps a type to its runtime type index
 * \tparam T the type
 * \return the runtime type index
 */
template <typename T, typename = void>
struct TypeToRuntimeTypeIndex {
  /*!
   * \brief Get the runtime type index of the type
   * \return the runtime type index
   */
  static int32_t v() { return TypeToFieldStaticTypeIndex<T>::value; }
};

// None
template <>
struct TypeTraits<std::nullptr_t> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFINone;
  TVM_FFI_INLINE static void CopyToAnyView(const std::nullptr_t&, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    result->zero_padding = 0;
    // invariant: the pointer field also equals nullptr
    // this will simplify same_as comparisons and hash
    result->v_int64 = 0;
  }

  TVM_FFI_INLINE static void MoveToAny(std::nullptr_t, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    result->zero_padding = 0;
    // invariant: the pointer field also equals nullptr
    // this will simplify same_as comparisons and hash
    result->v_int64 = 0;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFINone;
  }

  TVM_FFI_INLINE static std::nullptr_t CopyFromAnyViewAfterCheck(const TVMFFIAny*) {
    return nullptr;
  }

  TVM_FFI_INLINE static std::nullptr_t MoveFromAnyAfterCheck(TVMFFIAny*) { return nullptr; }

  TVM_FFI_INLINE static std::optional<std::nullptr_t> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return nullptr;
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFINone; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFINone) + R"("})";
  }
};

/**
 * \brief A type that forbids implicit conversion from int to bool
 *
 * This type is used to prevent implicit conversion from int to bool.
 */
class StrictBool {
 public:
  /*!
   * \brief Constructor
   * \param value The value of the strict bool.
   */
  StrictBool(bool value) : value_(value) {}  // NOLINT(google-explicit-constructor)
  /*!
   *\brief Convert the strict bool to bool.
   * \return The value of the strict bool.
   */
  operator bool() const { return value_; }  // NOLINT(google-explicit-constructor)

 private:
  bool value_;
};

template <>
struct TypeTraits<StrictBool> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;

  TVM_FFI_INLINE static void CopyToAnyView(const StrictBool& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIBool;
    result->zero_padding = 0;
    result->v_int64 = static_cast<bool>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(StrictBool src, TVMFFIAny* result) {
    CopyToAnyView(src, result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIBool;
  }

  TVM_FFI_INLINE static StrictBool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIBool);
    return static_cast<bool>(src->v_int64);
  }

  TVM_FFI_INLINE static StrictBool MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<StrictBool> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIBool) {
      return StrictBool(static_cast<bool>(src->v_int64));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIBool) + R"("})";
  }
};

// Bool type, allow implicit casting from int
template <>
struct TypeTraits<bool> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;

  TVM_FFI_INLINE static void CopyToAnyView(const bool& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIBool;
    result->zero_padding = 0;
    result->v_int64 = static_cast<int64_t>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(bool src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIBool;
  }

  TVM_FFI_INLINE static bool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIBool);
    return static_cast<bool>(src->v_int64);
  }

  TVM_FFI_INLINE static bool MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<bool> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
      return static_cast<bool>(src->v_int64);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIBool) + R"("})";
  }
};

template <typename Int>
struct TypeTraitsIntBase : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIInt;

  TVM_FFI_INLINE static void CopyInt64ToAnyView(int64_t src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIInt;
    result->zero_padding = 0;
    result->v_int64 = src;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
    return src->type_index == TypeIndex::kTVMFFIInt;
  }

  TVM_FFI_INLINE static Int CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIInt);
    return static_cast<Int>(src->v_int64);
  }

  TVM_FFI_INLINE static Int MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Int> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
      return Int(src->v_int64);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIInt; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIInt) + R"("})";
  }
};

// Integer POD values
template <typename Int>
struct TypeTraits<Int, std::enable_if_t<std::is_integral_v<Int>>> : public TypeTraitsIntBase<Int> {
  TVM_FFI_INLINE static void CopyToAnyView(const Int& src, TVMFFIAny* result) {
    TypeTraitsIntBase<Int>::CopyInt64ToAnyView(static_cast<int64_t>(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Int src, TVMFFIAny* result) { CopyToAnyView(src, result); }
};

/// \cond Doxygen_Suppress

// trait to check if a type is an integeral enum
// note that we need this trait so we can confirm underlying_type_t is an integral type
// to avoid potential undefined behavior
template <typename T, bool = std::is_enum_v<T>>
constexpr bool is_integeral_enum_v = false;

template <typename T>
constexpr bool is_integeral_enum_v<T, true> = std::is_integral_v<std::underlying_type_t<T>>;

/// \endcond

// Enum Integer POD values
template <typename IntEnum>
struct TypeTraits<IntEnum, std::enable_if_t<is_integeral_enum_v<IntEnum>>>
    : public TypeTraitsIntBase<IntEnum> {
  TVM_FFI_INLINE static void CopyToAnyView(const IntEnum& src, TVMFFIAny* result) {
    TypeTraitsIntBase<IntEnum>::CopyInt64ToAnyView(static_cast<int64_t>(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(IntEnum src, TVMFFIAny* result) {
    CopyToAnyView(src, result);
  }
};

// Float POD values
template <typename Float>
struct TypeTraits<Float, std::enable_if_t<std::is_floating_point_v<Float>>>
    : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFloat;

  TVM_FFI_INLINE static void CopyToAnyView(const Float& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIFloat;
    result->zero_padding = 0;
    result->v_float64 = static_cast<double>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(Float src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
    return src->type_index == TypeIndex::kTVMFFIFloat;
  }

  TVM_FFI_INLINE static Float CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIFloat);
    return static_cast<Float>(src->v_float64);
  }

  TVM_FFI_INLINE static Float MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Float> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIFloat) {
      return Float(src->v_float64);
    } else if (src->type_index == TypeIndex::kTVMFFIInt ||
               src->type_index == TypeIndex::kTVMFFIBool) {
      return Float(src->v_int64);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIFloat; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIFloat) + R"("})";
  }
};

// void*
template <>
struct TypeTraits<void*> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIOpaquePtr;

  TVM_FFI_INLINE static void CopyToAnyView(void* src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIOpaquePtr;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_ptr = src;
  }

  TVM_FFI_INLINE static void MoveToAny(void* src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
    return src->type_index == TypeIndex::kTVMFFIOpaquePtr;
  }

  TVM_FFI_INLINE static void* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    TVM_FFI_UNSAFE_ASSUME(src->type_index == TypeIndex::kTVMFFIOpaquePtr);
    return src->v_ptr;
  }

  TVM_FFI_INLINE static void* MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<void*> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIOpaquePtr) {
      return static_cast<void*>(src->v_ptr);
    }
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return static_cast<void*>(nullptr);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIOpaquePtr; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIOpaquePtr) + R"("})";
  }
};

/*!
 * \brief Helper class that convert to T only via the FallbackTypes
 *
 * The conversion will go through the FallbackTypes in the order
 * specified in the template parameter.
 * \tparam T The type of the target value.
 * \tparam FallbackTypes The type of the fallback value.
 * \note TypeTraits<T> must be derived from this class and define
 *     ConvertFallbackValue(FallbackType)->T for each FallbackType
 */
template <typename T, typename... FallbackTypes>
struct FallbackOnlyTraitsBase : public TypeTraitsBase {
  // disable container for FallbackOnlyTraitsBase
  /// \cond Doxygen_Suppress
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static std::optional<T> TryCastFromAnyView(const TVMFFIAny* src) {
    return TryFallbackTypes<FallbackTypes...>(src);
  }

  template <typename FallbackType, typename... Rest>
  TVM_FFI_INLINE static std::optional<T> TryFallbackTypes(const TVMFFIAny* src) {
    static_assert(!std::is_same_v<bool, FallbackType>,
                  "Using bool as FallbackType can cause bug because int will be detected as bool, "
                  "use tvm::ffi::StrictBool instead");
    if (auto opt_fallback = TypeTraits<FallbackType>::TryCastFromAnyView(src)) {
      return TypeTraits<T>::ConvertFallbackValue(*std::move(opt_fallback));
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryFallbackTypes<Rest...>(src);
    }
    return std::nullopt;
  }
  /// \endcond
};

}  // namespace ffi
}  // namespace tvm

/*!
 * \brief Declare lookup-only metadata for an ABI object.
 *
 * Unlike TVM_FFI_DECLARE_OBJECT_INFO, this macro only looks up an existing
 * registered type key and never allocates or registers a type index. Invoke it
 * in the public section of an object class with a string literal key.
 *
 * \note Include `tvm/ffi/function.h` or `tvm/ffi/tvm_ffi.h` before expanding
 * this macro so safe-call errors can be propagated as C++ exceptions.
 *
 * \param RegisteredKey The registered type key of the ABI object.
 * \param TypeDepth The reflected depth of the type in the object hierarchy.
 */
#define TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP(RegisteredKey, TypeDepth)           \
  static constexpr const char* _type_key = RegisteredKey;                      \
  static constexpr int32_t _type_depth = TypeDepth;                            \
  static constexpr bool _type_mutable = true;                                  \
  static constexpr bool _type_final = false;                                   \
  static constexpr uint32_t _type_child_slots = 0;                             \
  static constexpr bool _type_child_slots_can_overflow = true;                 \
  static int32_t RuntimeTypeIndex() {                                          \
    static const int32_t type_index = []() {                                   \
      constexpr TVMFFIByteArray key{RegisteredKey, sizeof(RegisteredKey) - 1}; \
      int32_t result = -1;                                                     \
      TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&key, &result));            \
      return result;                                                           \
    }();                                                                       \
    return type_index;                                                         \
  }

#endif  // TVM_FFI_TYPE_TRAITS_H_
