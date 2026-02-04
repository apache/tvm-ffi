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
  static_assert(!std::is_same_v<T, Error>, "Expected<Error> is not allowed. Use Error directly.");

  /*!
   * \brief Implicit constructor from a success value.
   * \param value The success value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  Expected(T value) : data_(Any(std::move(value))) {}

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

  /*!
   * \brief Check if the Expected contains a success value.
   * \return True if contains success value, false if contains error.
   * \note Checks for Error first to handle cases where T is a base class of Error.
   */
  TVM_FFI_INLINE bool is_ok() const { return !data_.as<Error>().has_value(); }

  /*!
   * \brief Check if the Expected contains an error.
   * \return True if contains error, false if contains success value.
   */
  TVM_FFI_INLINE bool is_err() const { return !is_ok(); }

  /*!
   * \brief Alias for is_ok().
   * \return True if contains success value.
   */
  TVM_FFI_INLINE bool has_value() const { return is_ok(); }

  /*! \brief Access the success value. Throws the contained error if is_err(). */
  TVM_FFI_INLINE T value() const& {
    if (is_err()) throw data_.cast<Error>();
    return data_.cast<T>();
  }
  /*! \brief Access the success value (rvalue). Throws the contained error if is_err(). */
  TVM_FFI_INLINE T value() && {
    if (is_err()) throw std::move(data_).template cast<Error>();
    return std::move(data_).template cast<T>();
  }

  /*! \brief Access the error. Throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() const& {
    if (!is_err()) TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    return data_.cast<Error>();
  }
  /*! \brief Access the error (rvalue). Throws RuntimeError if is_ok(). */
  TVM_FFI_INLINE Error error() && {
    if (!is_err()) TVM_FFI_THROW(RuntimeError) << "Bad expected access: contains value, not error";
    return std::move(data_).template cast<Error>();
  }

  /*!
   * \brief Get the success value or a default value.
   * \param default_value The value to return if Expected contains an error.
   * \return The success value if present, otherwise the default value.
   */
  template <typename U = std::remove_cv_t<T>>
  TVM_FFI_INLINE T value_or(U&& default_value) const {
    if (is_ok()) {
      return data_.cast<T>();
    }
    return T(std::forward<U>(default_value));
  }

 private:
  Any data_;  // Holds either T or Error
};

// TypeTraits specialization for Expected<T>
template <typename T>
inline constexpr bool use_default_type_traits_v<Expected<T>> = false;

template <typename T>
struct TypeTraits<Expected<T>> : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(const Expected<T>& src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::CopyToAnyView(src.error(), result);
    } else {
      TypeTraits<T>::CopyToAnyView(src.value(), result);
    }
  }

  TVM_FFI_INLINE static void MoveToAny(Expected<T> src, TVMFFIAny* result) {
    if (src.is_err()) {
      TypeTraits<Error>::MoveToAny(std::move(src).error(), result);
    } else {
      TypeTraits<T>::MoveToAny(std::move(src).value(), result);
    }
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<T>::CheckAnyStrict(src) || TypeTraits<Error>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static Expected<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (TypeTraits<T>::CheckAnyStrict(src)) {
      return TypeTraits<T>::CopyFromAnyViewAfterCheck(src);
    }
    return TypeTraits<Error>::CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static Expected<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if (TypeTraits<T>::CheckAnyStrict(src)) {
      return TypeTraits<T>::MoveFromAnyAfterCheck(src);
    }
    return TypeTraits<Error>::MoveFromAnyAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Expected<T>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (auto opt = TypeTraits<T>::TryCastFromAnyView(src)) {
      return Expected<T>(*std::move(opt));
    }
    if (auto opt_err = TypeTraits<Error>::TryCastFromAnyView(src)) {
      return Expected<T>(*std::move(opt_err));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "Expected<" + TypeTraits<T>::TypeStr() + ">";
  }

  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"Expected","args":[)" + details::TypeSchema<T>::v() +
           R"(,{"type":"ffi.Error"}]})";
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXPECTED_H_
