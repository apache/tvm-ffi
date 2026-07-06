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
 * \file tvm/ffi/optional.h
 * \brief Runtime Optional container types.
 * \note Optional<T> uses a hybrid representation. For types that enable Any
 *       storage (`TypeTraits<T>::storage_enabled`), it is backed by a single
 *       TVMFFIAny (Any) with nullopt represented as kTVMFFINone, mirroring
 *       Variant<...>; the layout is then independent of T (sizeof == sizeof(Any))
 *       which keeps the ABI stable. For types that do not enable storage (e.g.
 *       non-owning view types) it falls back to std::optional<T>.
 */
#ifndef TVM_FFI_OPTIONAL_H_
#define TVM_FFI_OPTIONAL_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <optional>
#include <string>
#include <utility>

namespace tvm {
namespace ffi {

// Note: We place optional in tvm/ffi instead of tvm/ffi/container
// because optional itself is an inherent core component of the FFI system.
/// \cond Doxygen_Suppress
template <typename T>
inline constexpr bool is_optional_type_v = false;

template <typename T>
inline constexpr bool is_optional_type_v<Optional<T>> = true;
/// \endcond

// Fallback specialization for types that do NOT enable Any storage
// (`TypeTraits<T>::storage_enabled == false`), such as non-owning view types
// that cannot be moved into an Any. These simply reuse std::optional<T>.
template <typename T>
class Optional<T, std::enable_if_t<!TypeTraits<T>::storage_enabled>> {
 public:
  // default constructors.
  Optional() = default;
  // NOLINTBEGIN(google-explicit-constructor)
  Optional(const Optional& other) = default;
  Optional(Optional&& other) noexcept = default;
  Optional(std::optional<T> other) : data_(std::move(other)) {}
  Optional(std::nullopt_t) {}
  Optional(T other) : data_(std::move(other)) {}
  // NOLINTEND(google-explicit-constructor)

  Optional& operator=(const Optional& other) = default;
  Optional& operator=(Optional&& other) noexcept = default;

  TVM_FFI_INLINE Optional& operator=(T other) {
    data_ = std::move(other);
    return *this;
  }

  TVM_FFI_INLINE Optional& operator=(std::nullopt_t) {
    data_ = std::nullopt;
    return *this;
  }

  TVM_FFI_INLINE const T& value() const& {
    if (TVM_FFI_PREDICT_FALSE(!data_.has_value())) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return *data_;
  }

  TVM_FFI_INLINE T&& value() && {
    if (TVM_FFI_PREDICT_FALSE(!data_.has_value())) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return *std::move(data_);
  }

  template <typename U = std::remove_cv_t<T>>
  TVM_FFI_INLINE T value_or(U&& default_value) const {
    return data_.value_or(std::forward<U>(default_value));
  }

  TVM_FFI_INLINE explicit operator bool() const noexcept { return data_.has_value(); }

  TVM_FFI_INLINE bool has_value() const noexcept { return data_.has_value(); }

  TVM_FFI_INLINE bool operator==(const Optional& other) const { return data_ == other.data_; }
  TVM_FFI_INLINE bool operator!=(const Optional& other) const { return data_ != other.data_; }
  template <typename U>
  TVM_FFI_INLINE bool operator==(const U& other) const {
    return data_ == other;
  }
  template <typename U>
  TVM_FFI_INLINE bool operator!=(const U& other) const {
    return data_ != other;
  }

  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  /*!
   * \brief Direct access to the value.
   * \note only use this function after checking has_value()
   */
  TVM_FFI_INLINE T&& operator*() && noexcept { return *std::move(data_); }
  /*!
   * \brief Direct access to the value.
   * \note only use this function after checking has_value()
   */
  TVM_FFI_INLINE const T& operator*() const& noexcept { return *data_; }
  // NOLINTEND(bugprone-unchecked-optional-access)

 private:
  std::optional<T> data_;
};

/*!
 * \brief Optional container backed by a single TVMFFIAny (Any) for storage-enabled T.
 *
 * Mirrors the Variant<...> implementation: the value is stored in a single Any
 * and nullopt is represented as kTVMFFINone. The layout is therefore independent
 * of T (sizeof(Optional<T>) == sizeof(Any)) which keeps the ABI stable across
 * contained types.
 *
 * \note None/null is treated as nullopt. For a nullable T, a null value reads
 *       back as "no value" (has_value() == false). This gives one stable
 *       cross-language ABI representation, consistent with Python's None. If a
 *       use case must distinguish nullopt from a stored null, use std::optional
 *       instead.
 *
 * \tparam T The underlying value type (must enable Any storage).
 */
template <typename T>
class Optional<T, std::enable_if_t<TypeTraits<T>::storage_enabled>> {
 public:
  /*! \brief default constructor, represents nullopt (Any() is kTVMFFINone). */
  Optional() = default;
  // NOLINTBEGIN(google-explicit-constructor)
  /*! \brief construct nullopt from std::nullopt. */
  Optional(std::nullopt_t) {}
  /*! \brief copy constructor. */
  Optional(const Optional& other) = default;
  /*! \brief move constructor. */
  Optional(Optional&& other) noexcept = default;
  /*! \brief construct from a value of type T (copy). */
  Optional(const T& value) : data_(value) {}
  /*! \brief construct from a value of type T (move). */
  Optional(T&& value) : data_(std::move(value)) {}
  /*! \brief construct from a std::optional<T>. */
  Optional(std::optional<T> other) {
    if (other.has_value()) {
      data_ = Any(*std::move(other));
    }
  }
  // NOLINTEND(google-explicit-constructor)

  /*! \brief copy assignment. */
  Optional& operator=(const Optional& other) = default;
  /*! \brief move assignment. */
  Optional& operator=(Optional&& other) noexcept = default;

  TVM_FFI_INLINE Optional& operator=(T other) {
    data_ = Any(std::move(other));
    return *this;
  }

  TVM_FFI_INLINE Optional& operator=(std::nullopt_t) {
    data_.reset();
    return *this;
  }

  TVM_FFI_INLINE Optional& operator=(std::nullptr_t) {
    data_.reset();
    return *this;
  }

  TVM_FFI_INLINE T value() const& {
    if (TVM_FFI_PREDICT_FALSE(!has_value())) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    // The invariant guarantees the stored value is exactly a T, so decode it
    // directly with the low-level after-check path (no conversion/cast).
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data_);
  }

  TVM_FFI_INLINE T value() && {
    if (TVM_FFI_PREDICT_FALSE(!has_value())) {
      TVM_FFI_THROW(RuntimeError) << "Back optional access";
    }
    return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
  }

  template <typename U = std::remove_cv_t<T>>
  TVM_FFI_INLINE T value_or(U&& default_value) const {
    if (has_value()) {
      return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data_);
    }
    return T(std::forward<U>(default_value));
  }

  TVM_FFI_INLINE explicit operator bool() const noexcept { return has_value(); }

  TVM_FFI_INLINE bool has_value() const noexcept {
    return data_.type_index() != TypeIndex::kTVMFFINone;
  }

  /*!
   * \brief Try to reinterpret the stored value as a type U (strict, no conversion).
   * \tparam U The type to reinterpret to.
   * \return std::optional<U> for ObjectRef-like U, or const U* when U is an Object type.
   * \note Returns an empty result when nullopt or the strict type check fails.
   */
  template <typename U>
  TVM_FFI_INLINE auto as() const {
    return data_.template as<U>();
  }

  /*!
   * \brief Direct access to the value.
   * \note only use this function after checking has_value()
   */
  TVM_FFI_INLINE T operator*() const& {
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data_);
  }
  /*!
   * \brief Direct access to the value, moved out of the storage.
   * \note only use this function after checking has_value()
   */
  TVM_FFI_INLINE T operator*() && {
    return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
  }

  // comparison with nullopt / nullptr
  TVM_FFI_INLINE bool operator==(std::nullopt_t) const noexcept { return !has_value(); }
  TVM_FFI_INLINE bool operator!=(std::nullopt_t) const noexcept { return has_value(); }
  TVM_FFI_INLINE bool operator==(std::nullptr_t) const noexcept { return !has_value(); }
  TVM_FFI_INLINE bool operator!=(std::nullptr_t) const noexcept { return has_value(); }

  // comparison with another Optional<T>
  TVM_FFI_INLINE auto operator==(const Optional& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(std::declval<T>() == std::declval<T>());
    if (data_.same_as(other.data_)) return RetType(true);
    if (has_value() && other.has_value()) return **this == *other;
    return RetType(false);
  }
  TVM_FFI_INLINE auto operator!=(const Optional& other) const {
    using RetType = decltype(std::declval<T>() != std::declval<T>());
    if (data_.same_as(other.data_)) return RetType(false);
    if (has_value() && other.has_value()) return **this != *other;
    return RetType(true);
  }

  // comparison with a std::optional<T>
  TVM_FFI_INLINE auto operator==(const std::optional<T>& other) const {
    using RetType = decltype(std::declval<T>() == std::declval<T>());
    if (has_value() && other.has_value()) return **this == *other;
    return RetType(!has_value() && !other.has_value());
  }
  TVM_FFI_INLINE auto operator!=(const std::optional<T>& other) const {
    using RetType = decltype(std::declval<T>() != std::declval<T>());
    if (has_value() && other.has_value()) return **this != *other;
    return RetType(has_value() != other.has_value());
  }

  // comparison with a value of another type U
  template <typename U, typename = std::enable_if_t<!is_optional_type_v<U> &&
                                                    !std::is_same_v<U, std::nullopt_t> &&
                                                    !std::is_same_v<U, std::nullptr_t>>>
  TVM_FFI_INLINE auto operator==(const U& other) const {
    using RetType = decltype(std::declval<T>() == std::declval<U>());
    if constexpr (std::is_base_of_v<ObjectRef, T> && std::is_base_of_v<ObjectRef, U>) {
      // support case where sub-class returns a symbolic ref type.
      if (data_.same_as(other)) return RetType(true);
    }
    if (!has_value()) return RetType(false);
    return **this == other;
  }
  template <typename U, typename = std::enable_if_t<!is_optional_type_v<U> &&
                                                    !std::is_same_v<U, std::nullopt_t> &&
                                                    !std::is_same_v<U, std::nullptr_t>>>
  TVM_FFI_INLINE auto operator!=(const U& other) const {
    using RetType = decltype(std::declval<T>() != std::declval<U>());
    if constexpr (std::is_base_of_v<ObjectRef, T> && std::is_base_of_v<ObjectRef, U>) {
      if (data_.same_as(other)) return RetType(false);
    }
    if (!has_value()) return RetType(true);
    return **this != other;
  }

  /*!
   * \brief Shallow-compare with another Optional<T>.
   * \return Whether the two refer to the same underlying value.
   */
  TVM_FFI_INLINE bool same_as(const Optional& other) const { return data_.same_as(other.data_); }

  /*!
   * \brief Shallow-compare with a value of an ObjectRef type.
   * \return Whether the two refer to the same underlying object.
   */
  template <typename U = T, typename = std::enable_if_t<std::is_base_of_v<ObjectRef, U>>>
  TVM_FFI_INLINE bool same_as(const U& other) const {
    return data_.same_as(other);
  }

 private:
  friend struct TypeTraits<Optional<T>>;
  // construct directly from an Any backing store.
  explicit Optional(Any data) : data_(std::move(data)) {}
  TVM_FFI_INLINE AnyView ToAnyView() const { return data_.operator AnyView(); }
  TVM_FFI_INLINE Any MoveToAny() && { return std::move(data_); }
  /*! \brief The underlying Any backing store, kTVMFFINone represents nullopt. */
  Any data_;
};

template <typename T>
inline constexpr bool use_default_type_traits_v<Optional<T>> = false;

template <typename T>
struct TypeTraits<Optional<T>> : public TypeTraitsBase {
  // storage_enabled propagates from T: Optional<T> can live in an Any exactly
  // when T can. This keeps nested Optional<Optional<T>> and Optional<T> used
  // inside Variant<...>/containers Any-backed iff T is storage-enabled.
  static constexpr bool storage_enabled = TypeTraits<T>::storage_enabled;

  TVM_FFI_INLINE static void CopyToAnyView(const Optional<T>& src, TVMFFIAny* result) {
    if constexpr (TypeTraits<T>::storage_enabled) {
      // Storage-enabled: the Any already holds the exact representation.
      *result = src.ToAnyView().CopyToTVMFFIAny();
    } else {
      if (src.has_value()) {
        TypeTraits<T>::CopyToAnyView(*src, result);
      } else {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
      }
    }
  }

  TVM_FFI_INLINE static void MoveToAny(Optional<T> src, TVMFFIAny* result) {
    if constexpr (TypeTraits<T>::storage_enabled) {
      *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src).MoveToAny());
    } else {
      if (src.has_value()) {
        TypeTraits<T>::MoveToAny(*std::move(src), result);
      } else {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
      }
    }
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return true;
    return TypeTraits<T>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static Optional<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if constexpr (TypeTraits<T>::storage_enabled) {
      return Optional<T>(Any(AnyView::CopyFromTVMFFIAny(*src)));
    } else {
      if (src->type_index == TypeIndex::kTVMFFINone) return Optional<T>(std::nullopt);
      return Optional<T>(TypeTraits<T>::CopyFromAnyViewAfterCheck(src));
    }
  }

  TVM_FFI_INLINE static Optional<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if constexpr (TypeTraits<T>::storage_enabled) {
      return Optional<T>(details::AnyUnsafe::MoveTVMFFIAnyToAny(src));
    } else {
      if (src->type_index == TypeIndex::kTVMFFINone) return Optional<T>(std::nullopt);
      return Optional<T>(TypeTraits<T>::MoveFromAnyAfterCheck(src));
    }
  }

  TVM_FFI_INLINE static std::optional<Optional<T>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Optional<T>(std::nullopt);
    if (std::optional<T> opt = TypeTraits<T>::TryCastFromAnyView(src)) {
      return Optional<T>(*std::move(opt));
    }
    // Important to be explicit here because nullopt can convert to
    // std::optional<T>(nullopt), which would incorrectly signal success.
    return std::optional<Optional<T>>(std::nullopt);
  }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraits<T>::GetMismatchTypeInfo(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "Optional<" + TypeTraits<T>::TypeStr() + ">";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"Optional","args":[)" + details::TypeSchema<T>::v() + "]}";
  }
};
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_OPTIONAL_H_
