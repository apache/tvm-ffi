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
 * \file tvm/ffi/extra/stl.h
 * \brief STL container support.
 * \note This file is an extra extension of TVM FFI,
 * which provides support for STL containers in C++ exported functions.
 *
 * Whenever possible, prefer using tvm/ffi/container/ implementations,
 * such as `tvm::ffi::Array` and `tvm::ffi::Tuple`, over STL containers
 * in exported functions for better performance and compatibility.
 */
#ifndef TVM_FFI_EXTRA_STL_H_
#define TVM_FFI_EXTRA_STL_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace tvm {
namespace ffi {
namespace details {

template <typename Type>
struct STLTypeTrait : public TypeTraitsBase {
 public:
  using TypeTraitsBase::convert_enabled;
  using TypeTraitsBase::storage_enabled;

 protected:
  // we always copy STL types into an Object first, then move the ObjectPtr to Any.
  TVM_FFI_INLINE static void MoveToAnyImpl(ObjectPtr<Type>&& src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(src));
    result->type_index = obj_ptr->type_index;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  // we always construct STL types from an Object first, then copy from the ObjectPtr in Any.
  TVM_FFI_INLINE static ObjectPtr<Type> CopyFromAnyViewAfterCheckImpl(const TVMFFIAny* src) {
    return details::ObjectUnsafe::ObjectPtrFromUnowned<Type>(src->v_obj);
  }
};

struct ContainerTemplate {};

}  // namespace details

template <>
struct TypeTraits<details::ContainerTemplate> : public details::STLTypeTrait<::tvm::ffi::ArrayObj> {
 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;

 private:
  template <std::size_t... Is, typename Tuple>
  TVM_FFI_INLINE static ObjectPtr<ArrayObj> CopyToTupleImpl(std::index_sequence<Is...>,
                                                            Tuple&& src) {
    auto array = ArrayObj::Empty(static_cast<std::int64_t>(sizeof...(Is)));
    auto dst = array->MutableBegin();
    // increase size after each new to ensure exception safety
    ((::new (dst++) Any(std::get<Is>(std::forward<Tuple>(src))), array->size_++), ...);
    return array;
  }

  template <typename Iter>
  TVM_FFI_INLINE static ObjectPtr<ArrayObj> CopyToArrayImpl(Iter src, std::size_t size) {
    auto array = ArrayObj::Empty(static_cast<std::int64_t>(size));
    auto dst = array->MutableBegin();
    // increase size after each new to ensure exception safety
    for (std::size_t i = 0; i < size; ++i) {
      ::new (dst++) Any(*(src++));
      array->size_++;
    }
    return array;
  }

 protected:
  template <typename Tuple>
  TVM_FFI_INLINE static ObjectPtr<ArrayObj> CopyToTuple(const Tuple& src) {
    return CopyToTupleImpl(std::make_index_sequence<std::tuple_size_v<Tuple>>{}, src);
  }

  template <typename Tuple>
  TVM_FFI_INLINE static ObjectPtr<ArrayObj> MoveToTuple(Tuple&& src) {
    return CopyToTupleImpl(std::make_index_sequence<std::tuple_size_v<Tuple>>{},
                           std::forward<Tuple>(src));
  }

  template <typename Range>
  TVM_FFI_INLINE static ObjectPtr<ArrayObj> CopyToArray(const Range& src) {
    return CopyToArrayImpl(std::begin(src), std::size(src));
  }

  template <typename Range>
  TVM_FFI_INLINE static ObjectPtr<ArrayObj> MoveToArray(Range&& src) {
    return CopyToArrayImpl(std::make_move_iterator(std::begin(src)), std::size(src));
  }

  /// NOTE: STL types are not natively movable from Any, so we always make a new copy.
  template <typename T>
  TVM_FFI_INLINE static T CopyFromAny(const Any& value) {
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(value);
  }
};

template <typename T, std::size_t Nm>
struct TypeTraits<std::array<T, Nm>> : public TypeTraits<details::ContainerTemplate> {
 public:
  using Self = std::array<T, Nm>;
  static_assert(Nm > 0, "Zero-length std::array is not supported.");

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return MoveToAnyImpl(CopyToArray(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return MoveToAnyImpl(MoveToArray(std::move(src)), result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    const ArrayObj& n = *reinterpret_cast<const ArrayObj*>(src->v_obj);
    // check static length first
    if (n.size_ != Nm) return false;
    // then check element type
    if constexpr (std::is_same_v<T, Any>) {
      return true;
    } else {
      return std::all_of(n.begin(), n.end(), details::AnyUnsafe::CheckAnyStrict<T>);
    }
  }

  TVM_FFI_INLINE static Self CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    auto array = CopyFromAnyViewAfterCheckImpl(src);
    auto begin = array->MutableBegin();
    Self result;  // no initialization to avoid overhead
    for (std::size_t i = 0; i < Nm; ++i) {
      result[i] = CopyFromAny<T>(begin[i]);
    }
    return result;
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyStrict(src)) return std::nullopt;
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "std::array<" + details::Type2Str<T>::v() + ", " + std::to_string(Nm) + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"std::array","args":[)" + details::TypeSchema<T>::v() + "," +
           std::to_string(Nm) + "]}";
  }
};

template <typename T>
struct TypeTraits<std::vector<T>> : public TypeTraits<details::ContainerTemplate> {
 public:
  using Self = std::vector<T>;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return MoveToAnyImpl(CopyToArray(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return MoveToAnyImpl(MoveToArray(std::move(src)), result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    const ArrayObj& n = *reinterpret_cast<const ArrayObj*>(src->v_obj);
    if constexpr (std::is_same_v<T, Any>) {
      return true;
    } else {
      return std::all_of(n.begin(), n.end(), details::AnyUnsafe::CheckAnyStrict<T>);
    }
  }

  TVM_FFI_INLINE static Self CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    auto array = CopyFromAnyViewAfterCheckImpl(src);
    auto begin = array->MutableBegin();
    auto result = Self{};
    auto length = array->size_;
    result.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
      result.emplace_back(CopyFromAny<T>(begin[i]));
    }
    return result;
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyStrict(src)) return std::nullopt;
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "std::vector<" + details::Type2Str<T>::v() + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"std::vector","args":[)" + details::TypeSchema<T>::v() + "]}";
  }
};

template <typename T>
struct TypeTraits<std::optional<T>> : public TypeTraitsBase {
 public:
  using Self = std::optional<T>;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::CopyToAnyView(*src, result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }
  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::MoveToAny(std::move(*src), result);
    } else {
      TypeTraits<std::nullptr_t>::MoveToAny(nullptr, result);
    }
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return true;
    return TypeTraits<T>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static Self CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Self{std::nullopt};
    return TypeTraits<T>::CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Self{std::nullopt};
    return TypeTraits<T>::MoveFromAnyAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Self{std::nullopt};
    auto result = std::optional<Self>{};
    if (std::optional<T> opt = TypeTraits<T>::TryCastFromAnyView(src)) {
      /// NOTE: std::optional<T> is just what we want (Self).
      result.emplace(std::move(opt));
    } else {
      result.reset();  // failed to cast, indicate failure
    }
    return result;
  }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraits<T>::GetMismatchTypeInfo(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "std::optional<" + TypeTraits<T>::TypeStr() + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"std::optional","args":[)" + details::TypeSchema<T>::v() + "]}";
  }
};

template <typename... Args>
struct TypeTraits<std::variant<Args...>> : public TypeTraitsBase {
 private:
  using Self = std::variant<Args...>;
  using ArgTuple = std::tuple<Args...>;
  static constexpr std::size_t Nm = sizeof...(Args);

  template <std::size_t Is = 0>
  static Self CopyUnsafeAux(const TVMFFIAny* src) {
    if constexpr (Is >= Nm) {
      TVM_FFI_ICHECK(false) << "Unreachable: variant TryCast failed.";
      throw;  // unreachable
    } else {
      using ElemType = std::tuple_element_t<Is, ArgTuple>;
      if (TypeTraits<ElemType>::CheckAnyStrict(src)) {
        return Self{TypeTraits<ElemType>::CopyFromAnyViewAfterCheck(src)};
      } else {
        return CopyUnsafeAux<Is + 1>(src);
      }
    }
  }

  template <std::size_t Is = 0>
  static std::optional<Self> TryCastAux(const TVMFFIAny* src) {
    if constexpr (Is >= Nm) {
      return std::nullopt;
    } else {
      using ElemType = std::tuple_element_t<Is, ArgTuple>;
      if (TypeTraits<ElemType>::CheckAnyStrict(src)) {
        return Self{TypeTraits<ElemType>::CopyFromAnyViewAfterCheck(src)};
      } else {
        return TryCastAux<Is + 1>(src);
      }
    }
  }

 public:
  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return std::visit(
        [&](const auto& value) {
          using ValueType = std::decay_t<decltype(value)>;
          TypeTraits<ValueType>::CopyToAnyView(value, result);
        },
        src);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return std::visit(
        [&](auto&& value) {
          using ValueType = std::decay_t<decltype(value)>;
          TypeTraits<ValueType>::MoveToAny(std::forward<ValueType>(value), result);
        },
        std::move(src));
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return (TypeTraits<Args>::CheckAnyStrict(src) || ...);
  }

  TVM_FFI_INLINE static Self CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    // find the first possible type to copy
    return CopyUnsafeAux(src);
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // find the first possible type to copy
    return CopyUnsafeAux(src);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    // find the first possible type to copy
    return TryCastAux(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    std::stringstream os;
    os << "std::variant<";
    const char* sep = "";
    ((os << sep << details::Type2Str<Args>::v(), sep = ", "), ...);
    os << ">";
    return std::move(os).str();
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    std::stringstream os;
    os << R"({"type":"std::variant","args":[)";
    const char* sep = "";
    ((os << sep << details::TypeSchema<Args>::v(), sep = ", "), ...);
    os << "]}";
    return std::move(os).str();
  }
};

template <typename... Args>
struct TypeTraits<std::tuple<Args...>> : public TypeTraits<details::ContainerTemplate> {
 private:
  using Self = std::tuple<Args...>;
  static constexpr std::size_t Nm = sizeof...(Args);
  static_assert(Nm > 0, "Zero-length std::tuple is not supported.");

  template <std::size_t... Is>
  static bool CheckSubTypeAux(std::index_sequence<Is...>, const ArrayObj& n) {
    return (... && details::AnyUnsafe::CheckAnyStrict<std::tuple_element_t<Is, Self>>(n[Is]));
  }

  template <std::size_t... Is>
  static Self ConstructTupleAux(std::index_sequence<Is...>, const ArrayObj& n) {
    return Self{CopyFromAny<std::tuple_element_t<Is, Self>>(n[Is])...};
  }

 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return MoveToAnyImpl(CopyToTuple(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return MoveToAnyImpl(MoveToTuple(std::move(src)), result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    const ArrayObj& n = *reinterpret_cast<const ArrayObj*>(src->v_obj);
    // check static length first
    if (n.size_ != Nm) return false;
    // then check element type
    return CheckSubTypeAux(std::make_index_sequence<Nm>{}, n);
  }

  TVM_FFI_INLINE static Self CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    auto array = CopyFromAnyViewAfterCheckImpl(src);
    return ConstructTupleAux(std::make_index_sequence<Nm>{}, *array);
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyStrict(src)) return std::nullopt;
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    std::stringstream os;
    os << "std::tuple<";
    const char* sep = "";
    ((os << sep << details::Type2Str<Args>::v(), sep = ", "), ...);
    os << ">";
    return std::move(os).str();
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    std::stringstream os;
    os << R"({"type":"std::tuple","args":[)";
    const char* sep = "";
    ((os << sep << details::TypeSchema<Args>::v(), sep = ", "), ...);
    os << "]}";
    return std::move(os).str();
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STL_H_
