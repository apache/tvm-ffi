/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   htT://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/ffi/container/stl.h
 * \brief STL container support.
 *
 */
#ifndef TVM_FFI_CONTAINER_STL_H_
#define TVM_FFI_CONTAINER_STL_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "tvm/ffi/c_api.h"
#include "tvm/ffi/error.h"

namespace tvm {
namespace ffi {

template <typename T, std::size_t Nm>
struct TypeTraits<std::array<T, Nm>> : public TypeTraitsBase {
 private:
  using Self = std::array<T, Nm>;
  using Array = ::tvm::ffi::Array<T>;
  static_assert(Nm > 0, "Zero-length std::array is not supported.");

 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return TypeTraits<Array>::MoveToAny({src.begin(), src.end()}, result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return TypeTraits<Array>::MoveToAny(
        {std::make_move_iterator(src.begin()), std::make_move_iterator(src.end())}, result);
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
    auto array = TypeTraits<Array>::CopyFromAnyViewAfterCheck(src);
    Self result;  // no initialization to avoid overhead
    std::copy_n(std::make_move_iterator(array.begin()), Nm, result.begin());
    return result;
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    auto array = TypeTraits<Array>::MoveFromAnyAfterCheck(src);
    Self result;  // no initialization to avoid overhead
    std::copy_n(std::make_move_iterator(array.begin()), Nm, result.begin());
    return result;
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
struct TypeTraits<std::vector<T>> : public TypeTraitsBase {
 private:
  using Self = std::vector<T>;
  using Array = ::tvm::ffi::Array<T>;

 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return TypeTraits<Array>::MoveToAny({src.begin(), src.end()}, result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return TypeTraits<Array>::MoveToAny(
        {std::make_move_iterator(src.begin()), std::make_move_iterator(src.end())}, result);
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
    auto array = TypeTraits<Array>::CopyFromAnyViewAfterCheck(src);
    return Self{std::make_move_iterator(array.begin()), std::make_move_iterator(array.end())};
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    auto array = TypeTraits<Array>::MoveFromAnyAfterCheck(src);
    return Self{std::make_move_iterator(array.begin()), std::make_move_iterator(array.end())};
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
 private:
  using Self = std::optional<T>;

 public:
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
struct TypeTraits<std::tuple<Args...>> : public TypeTraitsBase {
 private:
  using Self = std::tuple<Args...>;
  using Tuple = ::tvm::ffi::Tuple<Args...>;
  static constexpr std::size_t Nm = sizeof...(Args);
  static_assert(Nm > 0, "Zero-length std::tuple is not supported.");

  template <std::size_t... Is>
  static bool CheckTupleSubType(std::index_sequence<Is...>, const ArrayObj& n) {
    return (... && details::AnyUnsafe::CheckAnyStrict<std::tuple_element_t<Is, Self>>(n[Is]));
  }

  template <std::size_t... Is>
  static Tuple CopyToFFITupleAux(std::index_sequence<Is...>, const Self& tuple) {
    return Tuple{std::get<Is>(tuple)...};
  }

  template <std::size_t... Is>
  static Tuple MoveToFFITupleAux(std::index_sequence<Is...>, Self&& tuple) {
    return Tuple{std::get<Is>(std::move(tuple))...};
  }

  template <std::size_t... Is>
  static Self MoveToSTDTupleAux(std::index_sequence<Is...>, Tuple&& tuple) {
    return Self{std::move(tuple.template get<Is>())...};
  }

  static Tuple CopyToFFITuple(const Self& tuple) {
    return CopyToFFITupleAux(std::make_index_sequence<Nm>{}, tuple);
  }

  static Tuple MoveToFFITuple(Self&& tuple) {
    return MoveToFFITupleAux(std::make_index_sequence<Nm>{}, std::move(tuple));
  }

  static Self MoveToSTDTuple(Tuple&& tuple) {
    return MoveToSTDTupleAux(std::make_index_sequence<Nm>{}, std::move(tuple));
  }

 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return TypeTraits<Tuple>::MoveToAny(CopyToFFITuple(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return TypeTraits<Tuple>::MoveToAny(MoveToFFITuple(std::move(src)), result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    const ArrayObj& n = *reinterpret_cast<const ArrayObj*>(src->v_obj);
    // check static length first
    if (n.size_ != Nm) return false;
    // then check element type
    return CheckTupleSubType(std::make_index_sequence<Nm>{}, n);
  }

  TVM_FFI_INLINE static Self CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return MoveToSTDTuple(TypeTraits<Tuple>::CopyFromAnyViewAfterCheck(src));
  }

  TVM_FFI_INLINE static Self MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return MoveToSTDTuple(TypeTraits<Tuple>::MoveFromAnyAfterCheck(src));
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

template <typename... Args>
struct TypeTraits<std::variant<Args...>> : public TypeTraitsBase {
 private:
  using Self = std::variant<Args...>;
  using Tuple = std::tuple<Args...>;
  static constexpr std::size_t Nm = sizeof...(Args);

  template <std::size_t Is = 0>
  static Self MoveUnsafeAux(TVMFFIAny* src) {
    if constexpr (Is >= Nm) {
      TVM_FFI_ICHECK(false) << "Unreachable: variant TryCast failed.";
    } else {
      using ElemType = std::tuple_element_t<Is, Tuple>;
      if (TypeTraits<ElemType>::CheckAnyStrict(src)) {
        return Self{TypeTraits<ElemType>::MoveFromAnyAfterCheck(src)};
      } else {
        return MoveUnsafeAux<Is + 1>(src);
      }
    }
  }

  template <std::size_t Is = 0>
  static Self CopyUnsafeAux(const TVMFFIAny* src) {
    if constexpr (Is >= Nm) {
      TVM_FFI_ICHECK(false) << "Unreachable: variant TryCast failed.";
      throw;  // unreachable
    } else {
      using ElemType = std::tuple_element_t<Is, Tuple>;
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
      using ElemType = std::tuple_element_t<Is, Tuple>;
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
    // find the first possible type to move
    return MoveUnsafeAux(src);
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

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_CONTAINER_STL_H_
