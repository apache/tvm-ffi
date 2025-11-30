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
 * such as `tvm::ffi::Array` and `tvm::ffi::Tuple`, over STL containers.
 *
 * Native ffi objects comes with stable data layout and can be directly accessed
 * through compiled languages (Rust) and DSLs(via LLVM) with raw pointer access
 * for better performance and compatibility.
 */
#ifndef TVM_FFI_EXTRA_STL_H_
#define TVM_FFI_EXTRA_STL_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "tvm/ffi/function.h"

namespace tvm {
namespace ffi {
namespace details {

struct STLTypeMismatch : public std::exception {
  const char* what() const noexcept override { return "STL type mismatch"; }
};

struct STLTypeTrait : public TypeTraitsBase {
 public:
  static constexpr bool storage_enabled = false;

 protected:
  /// NOTE: we always copy STL types into an Object first, then move the ObjectPtr to Any.
  template <typename T>
  TVM_FFI_INLINE static void MoveToAnyImpl(ObjectPtr<T>&& src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(src));
    result->type_index = obj_ptr->type_index;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  /// NOTE: we always construct STL types from an Object first, then copy from the ObjectPtr in Any.
  template <typename T>
  TVM_FFI_INLINE static ObjectPtr<T> CopyFromAnyImpl(const TVMFFIAny* src) {
    return ObjectUnsafe::ObjectPtrFromUnowned<T>(src->v_obj);
  }

  /// NOTE: STL types are not natively movable from Any, so we always make a new copy.
  template <typename T>
  TVM_FFI_INLINE static T ConstructFromAny(const Any& value) {
    using TypeTrait = TypeTraits<T>;
    if constexpr (std::is_same_v<T, Any>) {
      return value;
    } else {
      /// NOTE: Not all type support `CheckArgStrict` (e.g. std::string),
      /// so we use `TryCast` instead (without any prior check).
      auto opt = TypeTrait::TryCastFromAnyView(AnyUnsafe::TVMFFIAnyPtrFromAny(value));
      if (!opt.has_value()) {
        throw STLTypeMismatch{};
      }
      return std::move(*opt);
    }
  }
};

struct ListTemplate {};
struct MapTemplate {};

}  // namespace details

template <>
struct TypeTraits<details::ListTemplate> : public details::STLTypeTrait {
 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;

 private:
  template <std::size_t... Is, typename Tuple>
  TVM_FFI_INLINE static ObjectPtr<ArrayObj> CopyToTupleImpl(std::index_sequence<Is...>,
                                                            Tuple&& src) {
    auto array = ArrayObj::Empty(static_cast<std::int64_t>(sizeof...(Is)));
    auto dst = array->MutableBegin();
    // increase size after each new to ensure exception safety
    std::apply(
        [&](auto&&... elems) {
          ((::new (dst++) Any(std::forward<decltype(elems)>(elems)), array->size_++), ...);
        },
        std::forward<Tuple>(src));
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
};

template <>
struct TypeTraits<details::MapTemplate> : public details::STLTypeTrait {
 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIMap;

 protected:
  template <typename MapType>
  TVM_FFI_INLINE static ObjectPtr<Object> CopyToMap(const MapType& src) {
    return MapObj::CreateFromRange(std::begin(src), std::end(src));
  }

  template <typename MapType>
  TVM_FFI_INLINE static ObjectPtr<Object> MoveToMap(MapType&& src) {
    return MapObj::CreateFromRange(std::make_move_iterator(std::begin(src)),
                                   std::make_move_iterator(std::end(src)));
  }

  template <typename MapType, bool CanReserve>
  TVM_FFI_INLINE static MapType ConstructMap(const TVMFFIAny* src) {
    using KeyType = typename MapType::key_type;
    using ValueType = typename MapType::mapped_type;
    auto result = MapType{};
    auto map_obj = CopyFromAnyImpl<MapObj>(src);
    if constexpr (CanReserve) {
      result.reserve(map_obj->size());
    }
    for (const auto& [key, value] : *map_obj) {
      result.try_emplace(ConstructFromAny<KeyType>(key), ConstructFromAny<ValueType>(value));
    }
    return result;
  }
};

template <typename T, std::size_t Nm>
struct TypeTraits<std::array<T, Nm>> : public TypeTraits<details::ListTemplate> {
 private:
  using Self = std::array<T, Nm>;

  TVM_FFI_INLINE static bool CheckAnyFast(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    const ArrayObj& n = *reinterpret_cast<const ArrayObj*>(src->v_obj);
    return n.size_ == Nm;
  }

 public:
  static_assert(Nm > 0, "Zero-length std::array is not supported.");

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return MoveToAnyImpl(CopyToArray(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return MoveToAnyImpl(MoveToArray(std::move(src)), result);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyFast(src)) return std::nullopt;
    try {
      auto array = CopyFromAnyImpl<ArrayObj>(src);
      auto begin = array->MutableBegin();
      Self result;  // no initialization to avoid overhead
      for (std::size_t i = 0; i < Nm; ++i) {
        result[i] = ConstructFromAny<T>(begin[i]);
      }
      return result;
    } catch (const details::STLTypeMismatch&) {
      return std::nullopt;
    }
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
struct TypeTraits<std::vector<T>> : public TypeTraits<details::ListTemplate> {
 private:
  using Self = std::vector<T>;

  TVM_FFI_INLINE static bool CheckAnyFast(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIArray;
  }

 public:
  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return MoveToAnyImpl(CopyToArray(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return MoveToAnyImpl(MoveToArray(std::move(src)), result);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyFast(src)) return std::nullopt;
    try {
      auto array = CopyFromAnyImpl<ArrayObj>(src);
      auto begin = array->MutableBegin();
      auto result = Self{};
      int64_t length = array->size_;
      result.reserve(length);
      for (int64_t i = 0; i < length; ++i) {
        result.emplace_back(ConstructFromAny<T>(begin[i]));
      }
      return result;
    } catch (const details::STLTypeMismatch&) {
      return std::nullopt;
    }
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
  static constexpr std::size_t Nm = sizeof...(Args);

  template <std::size_t Is = 0>
  TVM_FFI_INLINE static Self CopyUnsafeAux(const TVMFFIAny* src) {
    if constexpr (Is >= Nm) {
      TVM_FFI_ICHECK(false) << "Unreachable: variant TryCast failed.";
      throw;  // unreachable
    } else {
      using ElemType = std::variant_alternative_t<Is, Self>;
      if (TypeTraits<ElemType>::CheckAnyStrict(src)) {
        return Self{std::in_place_index<Is>, TypeTraits<ElemType>::CopyFromAnyViewAfterCheck(src)};
      } else {
        return CopyUnsafeAux<Is + 1>(src);
      }
    }
  }

  template <std::size_t Is = 0>
  TVM_FFI_INLINE static Self MoveUnsafeAux(const TVMFFIAny* src) {
    if constexpr (Is >= Nm) {
      TVM_FFI_ICHECK(false) << "Unreachable: variant TryCast failed.";
      throw;  // unreachable
    } else {
      using ElemType = std::variant_alternative_t<Is, Self>;
      if (TypeTraits<ElemType>::CheckAnyStrict(src)) {
        return Self{std::in_place_index<Is>, TypeTraits<ElemType>::MoveFromAnyAfterCheck(src)};
      } else {
        return MoveUnsafeAux<Is + 1>(src);
      }
    }
  }

  template <std::size_t Is = 0>
  TVM_FFI_INLINE static std::optional<Self> TryCastAux(const TVMFFIAny* src) {
    if constexpr (Is >= Nm) {
      return std::nullopt;
    } else {
      using ElemType = std::variant_alternative_t<Is, Self>;
      if (auto opt = TypeTraits<ElemType>::TryCastFromAnyView(src)) {
        return Self{std::in_place_index<Is>, std::move(*opt)};
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
    // try to find the first possible type to copy
    return TryCastAux(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    std::ostringstream os;
    os << "std::variant<";
    const char* sep = "";
    ((os << sep << details::Type2Str<Args>::v(), sep = ", "), ...);
    os << ">";
    return std::move(os).str();
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream os;
    os << R"({"type":"std::variant","args":[)";
    const char* sep = "";
    ((os << sep << details::TypeSchema<Args>::v(), sep = ", "), ...);
    os << "]}";
    return std::move(os).str();
  }
};

template <typename... Args>
struct TypeTraits<std::tuple<Args...>> : public TypeTraits<details::ListTemplate> {
 private:
  using Self = std::tuple<Args...>;
  static constexpr std::size_t Nm = sizeof...(Args);
  static_assert(Nm > 0, "Zero-length std::tuple is not supported.");

  TVM_FFI_INLINE static bool CheckAnyFast(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    const ArrayObj& n = *reinterpret_cast<const ArrayObj*>(src->v_obj);
    return n.size_ == Nm;
  }

  template <std::size_t... Is>
  TVM_FFI_INLINE static Self ConstructTupleAux(std::index_sequence<Is...>, const ArrayObj& n) {
    return Self{ConstructFromAny<std::tuple_element_t<Is, Self>>(n[Is])...};
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

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyFast(src)) return std::nullopt;
    try {
      auto array = CopyFromAnyImpl<ArrayObj>(src);
      return ConstructTupleAux(std::make_index_sequence<Nm>{}, *array);
    } catch (const details::STLTypeMismatch&) {
      return std::nullopt;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    std::ostringstream os;
    os << "std::tuple<";
    const char* sep = "";
    ((os << sep << details::Type2Str<Args>::v(), sep = ", "), ...);
    os << ">";
    return std::move(os).str();
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream os;
    os << R"({"type":"std::tuple","args":[)";
    const char* sep = "";
    ((os << sep << details::TypeSchema<Args>::v(), sep = ", "), ...);
    os << "]}";
    return std::move(os).str();
  }
};

template <typename K, typename V>
struct TypeTraits<std::map<K, V>> : public TypeTraits<details::MapTemplate> {
 private:
  using Self = std::map<K, V>;
  TVM_FFI_INLINE static bool CheckAnyFast(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIMap;
  }

 public:
  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return MoveToAnyImpl(CopyToMap(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return MoveToAnyImpl(MoveToMap(std::move(src)), result);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyFast(src)) return std::nullopt;
    try {
      return ConstructMap<Self, /*CanReserve=*/false>(src);
    } catch (const details::STLTypeMismatch&) {
      return std::nullopt;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "std::map<" + details::Type2Str<K>::v() + ", " + details::Type2Str<V>::v() + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"std::map","args":[)" + details::TypeSchema<K>::v() + "," +
           details::TypeSchema<V>::v() + "]}";
  }
};

template <typename K, typename V>
struct TypeTraits<std::unordered_map<K, V>> : public TypeTraits<details::MapTemplate> {
 private:
  using Self = std::unordered_map<K, V>;
  TVM_FFI_INLINE static bool CheckAnyFast(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIMap;
  }

 public:
  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return MoveToAnyImpl(CopyToMap(src), result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return MoveToAnyImpl(MoveToMap(std::move(src)), result);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    if (!CheckAnyFast(src)) return std::nullopt;
    try {
      return ConstructMap<Self, /*CanReserve=*/true>(src);
    } catch (const details::STLTypeMismatch&) {
      return std::nullopt;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "std::unordered_map<" + details::Type2Str<K>::v() + ", " + details::Type2Str<V>::v() +
           ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"std::unordered_map","args":[)" + details::TypeSchema<K>::v() + "," +
           details::TypeSchema<V>::v() + "]}";
  }
};

template <typename Ret, typename... Args>
struct TypeTraits<std::function<Ret(Args...)>> : TypeTraitsBase {
 private:
  using Self = std::function<Ret(Args...)>;
  using Function = TypedFunction<Ret(Args...)>;
  using ProxyTrait = TypeTraits<Function>;

  TVM_FFI_INLINE static Self GetFunc(Function&& f) {
    return [fn = std::move(f)](Args... args) -> Ret { return fn(std::forward<Args>(args)...); };
  }

 public:
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFunction;
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static void CopyToAnyView(const Self& src, TVMFFIAny* result) {
    return ProxyTrait::MoveToAny(Function{src}, result);
  }

  TVM_FFI_INLINE static void MoveToAny(Self&& src, TVMFFIAny* result) {
    return ProxyTrait::MoveToAny(Function{std::move(src)}, result);
  }

  TVM_FFI_INLINE static std::optional<Self> TryCastFromAnyView(const TVMFFIAny* src) {
    auto opt = ProxyTrait::TryCastFromAnyView(src);
    if (opt.has_value()) {
      return GetFunc(std::move(*opt));
    } else {
      return std::nullopt;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    std::ostringstream os;
    os << "std::function<" << details::Type2Str<Ret>::v() << "(";
    const char* sep = "";
    ((os << sep << details::Type2Str<Args>::v(), sep = ", "), ...);
    os << ")>";
    return std::move(os).str();
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream os;
    os << R"({"type":"std::function","args":[)" << details::TypeSchema<Ret>::v() << ",[";
    const char* sep = "";
    ((os << sep << details::TypeSchema<Args>::v(), sep = ", "), ...);
    os << "]]}";
    return std::move(os).str();
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STL_H_
