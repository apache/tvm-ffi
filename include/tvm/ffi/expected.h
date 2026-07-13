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
 * \file tvm/ffi/expected.h
 * \brief Runtime Expected container type for exception-free error handling.
 */
#ifndef TVM_FFI_EXPECTED_H_
#define TVM_FFI_EXPECTED_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/error.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Wrapper to explicitly construct an Expected in the error state.
 * \tparam E The error type, must derive from Error.
 */
template <typename E = Error>
class Unexpected {
  static_assert(std::is_base_of_v<Error, std::remove_cv_t<E>>,
                "Unexpected<E> requires E to be Error or a subclass of Error.");

 public:
  /*! \brief Construct from an error value. */
  explicit Unexpected(E error) : error_(std::move(error)) {}

  /*! \brief Access the stored error. */
  const E& error() const& noexcept { return error_; }
  /*! \brief Access the stored error. */
  E& error() & noexcept { return error_; }
  /*! \brief Access the stored error (rvalue). */
  const E&& error() const&& noexcept { return std::move(error_); }
  /*! \brief Access the stored error (rvalue). */
  E&& error() && noexcept { return std::move(error_); }

 private:
  E error_;
};

#ifndef TVM_FFI_DOXYGEN_MODE
template <typename E>
Unexpected(E) -> Unexpected<E>;
#endif

namespace details {

struct ExpectedUnsafe;

}  // namespace details

/*!
 * \brief Expected<T> provides exception-free error handling for FFI functions.
 *
 * Expected<T> is similar to Rust's Result<T, Error> or C++23's std::expected.
 * It can hold either a success value of type T or an error of type Error.
 *
 * \tparam T The success type. Must be Any-compatible and cannot be Error.
 *
 * Usage:
 * \code
 * Expected<int> divide(int a, int b) {
 *   if (b == 0) {
 *     return Error("ValueError", "Division by zero");
 *   }
 *   return a / b;
 * }
 *
 * Expected<int> result = divide(10, 2);
 * if (result.is_ok()) {
 *   int value = result.value();
 * } else {
 *   Error err = result.error();
 * }
 * \endcode
 */
template <typename T>
class Expected {
 public:
  static_assert(!details::is_qualified_object_v<T>,
                "Expected<TObject> requires an unqualified Object subclass");
  /*! \brief The normalized success type stored by Expected. */
  using value_type = details::object_ptr_type_t<T>;
  static_assert(!std::is_same_v<value_type, Error>,
                "Expected<Error> is not allowed. Use Error directly.");

  /*!
   * \brief Implicit constructor from a success value.
   * \param value The success value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(value_type value) : data_(Any(std::move(value))) {}

  /*!
   * \brief Implicit constructor from an error.
   * \param error The error value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(Error error) : data_(Any(std::move(error))) {}

  /*! \brief Implicit constructor from an Unexpected wrapper. */
  template <typename E, typename = std::enable_if_t<std::is_base_of_v<Error, std::remove_cv_t<E>>>>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(Unexpected<E> unexpected) : data_(Any(std::move(unexpected).error())) {}

  /*! \brief Convert between equivalent shorthand and explicit ObjectPtr success types. */
  template <typename U,
            std::enable_if_t<!std::is_same_v<T, U> &&
                                 std::is_same_v<value_type, typename Expected<U>::value_type>,
                             int> = 0>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(const Expected<U>& other)
      : data_(other.is_err() ? Any(other.error()) : Any(other.value())) {}

  /*! \brief Move between equivalent shorthand and explicit ObjectPtr success types. */
  template <typename U,
            std::enable_if_t<!std::is_same_v<T, U> &&
                                 std::is_same_v<value_type, typename Expected<U>::value_type>,
                             int> = 0>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(Expected<U>&& other)
      : data_(other.is_err() ? Any(std::move(other).error()) : Any(std::move(other).value())) {}

  template <typename U,
            std::enable_if_t<!std::is_same_v<T, U> &&
                                 std::is_same_v<value_type, typename Expected<U>::value_type>,
                             int> = 0>
  TVM_FFI_INLINE Expected& operator=(const Expected<U>& other) {
    data_ = other.is_err() ? Any(other.error()) : Any(other.value());
    return *this;
  }

  template <typename U,
            std::enable_if_t<!std::is_same_v<T, U> &&
                                 std::is_same_v<value_type, typename Expected<U>::value_type>,
                             int> = 0>
  TVM_FFI_INLINE Expected& operator=(Expected<U>&& other) {
    data_ = other.is_err() ? Any(std::move(other).error()) : Any(std::move(other).value());
    return *this;
  }

  /*! \brief Return the raw stored type index. */
  TVM_FFI_INLINE int32_t type_index() const noexcept { return data_.type_index(); }

  /*! \brief Returns true if the Expected contains a success value. */
  TVM_FFI_INLINE bool is_ok() const noexcept {
    return data_.type_index() != TypeIndex::kTVMFFIError;
  }

  /*! \brief Returns true if the Expected contains an error. */
  TVM_FFI_INLINE bool is_err() const noexcept {
    return data_.type_index() == TypeIndex::kTVMFFIError;
  }

  /*! \brief Alias for is_ok(). */
  TVM_FFI_INLINE bool has_value() const noexcept { return is_ok(); }

  /*! \brief Returns the success value, or throws the contained error. */
  TVM_FFI_INLINE value_type value() const& {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::CopyFromAnyViewAfterCheck<value_type>(data_);
    }
    throw details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
  }

  /*! \brief Returns the success value (moved out), or throws the contained error. */
  TVM_FFI_INLINE value_type value() && {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::MoveFromAnyAfterCheck<value_type>(std::move(data_));
    }
    throw details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
  }

  /*! \brief Returns the contained error, or throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() const& {
    // No branch hint: error() is itself a cold path — callers only invoke it
    // after observing !is_ok(), so the branch direction here doesn't matter.
    if (is_ok()) {
      TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    }
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data_);
  }

  /*! \brief Returns the contained error (moved out), or throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() && {
    // No branch hint: error() is itself a cold path — callers only invoke it
    // after observing !is_ok(), so the branch direction here doesn't matter.
    if (is_ok()) {
      TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    }
    return details::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(data_));
  }

  /*!
   * \brief Returns the success value, or \p default_value if the Expected holds an error.
   */
  template <typename U = std::remove_cv_t<value_type>>
  TVM_FFI_INLINE value_type value_or(U&& default_value) const& {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::CopyFromAnyViewAfterCheck<value_type>(data_);
    }
    return value_type(std::forward<U>(default_value));
  }

  /*!
   * \brief Returns the success value (moved out), or \p default_value if the Expected holds an
   * error.
   */
  template <typename U = std::remove_cv_t<value_type>>
  TVM_FFI_INLINE value_type value_or(U&& default_value) && {
    if (TVM_FFI_PREDICT_TRUE(is_ok())) {
      return details::AnyUnsafe::MoveFromAnyAfterCheck<value_type>(std::move(data_));
    }
    return value_type(std::forward<U>(default_value));
  }

 private:
  Expected() = default;

  friend struct details::ExpectedUnsafe;

  Any data_;  // Invariant: holds a T (type_index != kTVMFFIError) or an Error.
};

namespace details {

/*!
 * \brief Unsafe raw-storage helpers for Expected.
 *
 * These helpers bypass normal value checking and are intended for ABI boundaries
 * that already know the underlying Any storage holds either a valid T or Error.
 */
struct ExpectedUnsafe {
  /*!
   * \brief Move a raw TVMFFIAny into Expected storage.
   * \tparam T The Expected success type.
   * \param raw The raw FFI value to move.
   * \return Expected backed by moved Any storage.
   */
  template <typename T>
  TVM_FFI_INLINE static Expected<T> MoveFromTVMFFIAny(TVMFFIAny raw) {
    Expected<T> result;
    result.data_ = AnyUnsafe::MoveTVMFFIAnyToAny(&raw);
    return result;
  }

  /*!
   * \brief Move Expected storage to a raw TVMFFIAny.
   * \tparam T The Expected success type.
   * \param result The Expected value to move from.
   * \return Raw FFI value containing moved underlying Any storage.
   */
  template <typename T>
  TVM_FFI_INLINE static TVMFFIAny MoveToTVMFFIAny(Expected<T>&& result) {
    return AnyUnsafe::MoveAnyToTVMFFIAny(std::move(result.data_));
  }

  /*!
   * \brief Return the underlying Any storage.
   * \tparam T The Expected success type.
   * \param result The Expected value to inspect.
   * \return Const reference to the raw Any storage.
   */
  template <typename T>
  TVM_FFI_INLINE static const Any& GetData(const Expected<T>& result) noexcept {
    return result.data_;
  }

  /*!
   * \brief Read an Expected success value as a compatible raw storage type.
   * \tparam T The type to read from the underlying Any storage.
   * \tparam U The Expected success type.
   * \param result The Expected value to read from.
   * \return The stored value decoded as T.
   *
   * \note This assumes \p result stores T-compatible Any storage, or Error.
   */
  template <typename T, typename U>
  TVM_FFI_INLINE static T ValueAs(const Expected<U>& result) {
    const Any& data = result.data_;
    if (TVM_FFI_PREDICT_TRUE(data.type_index() != TypeIndex::kTVMFFIError)) {
      return AnyUnsafe::CopyFromAnyViewAfterCheck<T>(data);
    }
    throw AnyUnsafe::CopyFromAnyViewAfterCheck<Error>(data);
  }
};

}  // namespace details

// TypeTraits specialization for Expected<T>
template <typename T>
inline constexpr bool use_default_type_traits_v<Expected<T>> = false;

template <typename T>
struct TypeTraits<Expected<T>> : public TypeTraitsBase {
  using value_type = typename Expected<T>::value_type;

  TVM_FFI_INLINE static void CopyToAnyView(const Expected<T>& src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::CopyToAnyView(src.error(), result);
    } else {
      TypeTraits<value_type>::CopyToAnyView(src.value(), result);
    }
  }

  TVM_FFI_INLINE static void MoveToAny(Expected<T> src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::MoveToAny(std::move(src).error(), result);
    } else {
      TypeTraits<value_type>::MoveToAny(std::move(src).value(), result);
    }
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<value_type>::CheckAnyStrict(src) || TypeTraits<Error>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static Expected<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (TypeTraits<value_type>::CheckAnyStrict(src)) {
      return TypeTraits<value_type>::CopyFromAnyViewAfterCheck(src);
    }
    return TypeTraits<Error>::CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static Expected<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if (TypeTraits<value_type>::CheckAnyStrict(src)) {
      return TypeTraits<value_type>::MoveFromAnyAfterCheck(src);
    }
    return TypeTraits<Error>::MoveFromAnyAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Expected<T>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (auto opt = TypeTraits<value_type>::TryCastFromAnyView(src)) {
      return Expected<T>(*std::move(opt));
    }
    if (auto opt_err = TypeTraits<Error>::TryCastFromAnyView(src)) {
      return Expected<T>(*std::move(opt_err));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "Expected<" + TypeTraits<value_type>::TypeStr() + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"Expected","args":[)" + details::TypeSchema<value_type>::v() +
           R"(,{"type":"ffi.Error"}]})";
  }
};

/// \cond Doxygen_Suppress
/*! \brief Whether target Expected storage subsumes source Expected storage. */
template <typename T, typename U>
inline constexpr bool type_subsumes_v<Expected<T>, Expected<U>> =
    type_subsumes_v<typename Expected<T>::value_type, typename Expected<U>::value_type>;
/// \endcond

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXPECTED_H_
