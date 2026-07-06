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
 * \file tvm/ffi/container/variant.h
 * \brief Runtime variant container types.
 */
#ifndef TVM_FFI_CONTAINER_VARIANT_H_
#define TVM_FFI_CONTAINER_VARIANT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/container_details.h>
#include <tvm/ffi/optional.h>

#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {
/*!
 * \brief A typed variant container.
 *
 * \note Variant is always backed by a single Any (TVMFFIAny). Even when every
 *       alternative derives from ObjectRef, Variant is not ObjectRef-derived;
 *       this keeps the layout independent of the contained types
 *       (sizeof(Variant<...>) == sizeof(Any)).
 */
template <typename... V>
class Variant {
 public:
  /// \cond Doxygen_Suppress
  static_assert(details::all_storage_enabled_v<V...>,
                "All types used in Variant<...> must be compatible with Any");
  /// \cond Doxygen_Suppress
  static constexpr bool _type_container_is_exact = false;
  /// \endcond
  /*
   * \brief Helper utility to check if the type can be contained in the variant
   */
  template <typename T>
  static constexpr bool variant_contains_v = (type_subsumes_v<V, T> || ...);
  /* \brief Helper utility for SFINAE if the type is part of the variant */
  template <typename T>
  using enable_if_variant_contains_t = std::enable_if_t<variant_contains_v<T>>;
  /// \endcond
  /*!
   * \brief Constructor from another variant
   * \param other The other variant
   */
  Variant(const Variant<V...>& other) = default;
  /*!
   * \brief Constructor from another variant
   * \param other The other variant
   */
  Variant(Variant<V...>&& other) noexcept = default;

  /*!
   * \brief Assignment from another variant
   * \param other The other variant
   */
  Variant& operator=(const Variant<V...>& other) = default;

  /*!
   * \brief Assignment from another variant
   * \param other The other variant
   */
  Variant& operator=(Variant<V...>&& other) noexcept = default;

  /*!
   * \brief Constructor from a contained value
   * \param other The value to store
   */
  template <typename T, typename = enable_if_variant_contains_t<T>>
  Variant(T other) : data_(std::move(other)) {}  // NOLINT(*)

  /*!
   * \brief Assignment from another variant
   * \param other The other variant
   */
  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE Variant& operator=(T other) {
    return operator=(Variant(std::move(other)));
  }

  /*!
   * \brief Try to cast to a type T, return std::nullopt if the cast is not possible.
   * \return The casted value, or std::nullopt if the cast is not possible.
   * \tparam T The type to cast to.
   */
  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE std::optional<T> as() const {
    return ToAnyView().template as<T>();
  }

  /*!
   * \brief Shortcut of as Object to cast to a const pointer when T is an Object.
   *
   * \tparam T The object type.
   * \return The requested pointer, returns nullptr if type mismatches.
   */
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Object, T>>>
  TVM_FFI_INLINE const T* as() const {
    return ToAnyView().template as<const T*>().value_or(nullptr);
  }

  /*!
   * \brief Get the value of the variant in type T, throws an exception if cast fails.
   * \return The value of the variant
   * \tparam T The type to get.
   */
  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE T get() const& {
    return ToAnyView().template cast<T>();
  }

  /*!
   * \brief Get the value of the variant in type T, throws an exception if cast fails.
   * \return The value of the variant
   * \tparam T The type to get.
   */
  template <typename T, typename = enable_if_variant_contains_t<T>>
  TVM_FFI_INLINE T get() && {
    return std::move(*this).MoveToAny().template cast<T>();
  }

  /*!
   * \brief Get the type key of the variant
   * \return The type key of the variant
   */
  TVM_FFI_INLINE std::string GetTypeKey() const { return ToAnyView().GetTypeKey(); }

  /*!
   * \brief Shallow-compare with another variant.
   * \param other The other variant.
   * \return Whether the two hold the same underlying value.
   */
  TVM_FFI_INLINE bool same_as(const Variant<V...>& other) const {
    return data_.same_as(other.data_);
  }

 private:
  friend struct TypeTraits<Variant<V...>>;
  friend struct ObjectPtrHash;
  friend struct ObjectPtrEqual;
  // constructor from any
  explicit Variant(Any data) : data_(std::move(data)) {}
  /*!
   * \brief Get the object pointer from the variant
   * \note This function is only available if all types used in Variant<...> are derived from
   * ObjectRef
   */
  TVM_FFI_INLINE Object* GetObjectPtrForHashEqual() const {
    constexpr bool all_object_v = (std::is_base_of_v<ObjectRef, V> && ...);
    static_assert(all_object_v,
                  "All types used in Variant<...> must be derived from ObjectRef "
                  "to enable ObjectPtrHash/ObjectPtrEqual");
    return details::AnyUnsafe::ObjectPtrFromAnyAfterCheck(this->data_);
  }
  TVM_FFI_INLINE AnyView ToAnyView() const { return data_.operator AnyView(); }
  TVM_FFI_INLINE Any MoveToAny() && { return std::move(data_); }
  /*! \brief The underlying Any backing store. */
  Any data_;
};

template <typename... V>
inline constexpr bool use_default_type_traits_v<Variant<V...>> = false;

template <typename... V>
struct TypeTraits<Variant<V...>> : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(const Variant<V...>& src, TVMFFIAny* result) {
    *result = src.ToAnyView().CopyToTVMFFIAny();
  }

  TVM_FFI_INLINE static void MoveToAny(Variant<V...> src, TVMFFIAny* result) {
    *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src).MoveToAny());
  }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraitsBase::GetMismatchTypeInfo(src);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return (TypeTraits<V>::CheckAnyStrict(src) || ...);
  }

  TVM_FFI_INLINE static Variant<V...> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return Variant<V...>(Any(AnyView::CopyFromTVMFFIAny(*src)));
  }

  TVM_FFI_INLINE static Variant<V...> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return Variant<V...>(details::AnyUnsafe::MoveTVMFFIAnyToAny(src));
  }

  TVM_FFI_INLINE static std::optional<Variant<V...>> TryCastFromAnyView(const TVMFFIAny* src) {
    // fast path, storage is already in the right type
    if (CheckAnyStrict(src)) {
      return CopyFromAnyViewAfterCheck(src);
    }
    // More expensive path, try to convert to each type, in order of declaration
    return TryVariantTypes<V...>(src);
  }

  template <typename VariantType, typename... Rest>
  TVM_FFI_INLINE static std::optional<Variant<V...>> TryVariantTypes(const TVMFFIAny* src) {
    if (auto opt_convert = TypeTraits<VariantType>::TryCastFromAnyView(src)) {
      return Variant<V...>(*std::move(opt_convert));
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryVariantTypes<Rest...>(src);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return details::ContainerTypeStr<V...>("Variant"); }
  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream oss;
    oss << R"({"type":"Variant","args":[)";
    const char* sep = "";
    ((oss << sep << details::TypeSchema<V>::v(), sep = ","), ...);
    oss << "]}";
    return oss.str();
  }
};

template <typename... V>
TVM_FFI_INLINE size_t ObjectPtrHash::operator()(const Variant<V...>& a) const {
  return std::hash<Object*>()(a.GetObjectPtrForHashEqual());
}

template <typename... V>
TVM_FFI_INLINE bool ObjectPtrEqual::operator()(const Variant<V...>& a,
                                               const Variant<V...>& b) const {
  return a.GetObjectPtrForHashEqual() == b.GetObjectPtrForHashEqual();
}

/// \cond Doxygen_Suppress
/*! \brief Whether Variant storage subsumes a source type through one alternative. */
template <typename... V, typename T>
inline constexpr bool type_subsumes_v<Variant<V...>, T> = (type_subsumes_v<V, T> || ...);
/// \endcond
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_VARIANT_H_
