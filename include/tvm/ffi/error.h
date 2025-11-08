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

/*
 * \file tvm/ffi/error.h
 * \brief Error handling component.
 */
#ifndef TVM_FFI_ERROR_H_
#define TVM_FFI_ERROR_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/*!
 * \brief Macro defines whether we enable libbacktrace
 */
#ifndef TVM_FFI_USE_LIBBACKTRACE
#define TVM_FFI_USE_LIBBACKTRACE 1
#endif

/*!
 * \brief Macro defines whether to install signal handler
 *   and print backtrace during segfault
 */
#ifndef TVM_FFI_BACKTRACE_ON_SEGFAULT
#define TVM_FFI_BACKTRACE_ON_SEGFAULT 1
#endif

#ifndef TVM_FFI_ALWAYS_LOG_BEFORE_THROW
#define TVM_FFI_ALWAYS_LOG_BEFORE_THROW 0
#endif

namespace tvm {
namespace ffi {

/*!
 * \brief Error already set in frontend env.
 *
 *  This error can be thrown by EnvCheckSignals to indicate
 *  that there is an error set in the frontend environment(e.g.
 *  python interpreter). The TVM FFI should catch this error
 *  and return a proper code to tell the frontend caller about
 *  this fact.
 *
 * \code
 *
 * void ExampleLongRunningFunction() {
 *   if (TVMFFIEnvCheckSignals() != 0) {
 *     throw ::tvm::ffi::EnvErrorAlreadySet();
 *   }
 *   // do work here
 * }
 *
 * \endcode
 */
struct EnvErrorAlreadySet : public std::exception {};

/*!
 * \brief Error object class.
 */
class ErrorObj : public Object, public TVMFFIErrorCell {
 public:
  /// \cond Doxygen_Suppress
  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIError;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIError, ErrorObj, Object);
  /// \endcond
};

namespace details {
class ErrorObjFromStd : public ErrorObj {
 public:
  ErrorObjFromStd(std::string kind, std::string message, std::string backtrace)
      : kind_data_(std::move(kind)),
        message_data_(std::move(message)),
        backtrace_data_(std::move(backtrace)) {
    this->kind = TVMFFIByteArray{kind_data_.data(), kind_data_.length()};
    this->message = TVMFFIByteArray{message_data_.data(), message_data_.length()};
    this->backtrace = TVMFFIByteArray{backtrace_data_.data(), backtrace_data_.length()};
    this->update_backtrace = UpdateBacktrace;
  }

 private:
  /*!
   * \brief Update the backtrace of the error object.
   * \param backtrace The backtrace to update.
   * \param update_mode The mode to update the backtrace,
   *        can be either kTVMFFIBacktraceUpdateModeReplace, kTVMFFIBacktraceUpdateModeAppend.
   */
  static void UpdateBacktrace(TVMFFIObjectHandle self, const TVMFFIByteArray* backtrace_str,
                              int32_t update_mode) {
    ErrorObjFromStd* obj = static_cast<ErrorObjFromStd*>(self);
    if (update_mode == kTVMFFIBacktraceUpdateModeReplace) {
      obj->backtrace_data_.resize(backtrace_str->size);
      std::memcpy(obj->backtrace_data_.data(), backtrace_str->data, backtrace_str->size);
      obj->backtrace = TVMFFIByteArray{obj->backtrace_data_.data(), obj->backtrace_data_.length()};
    } else {
      obj->backtrace_data_.append(backtrace_str->data, backtrace_str->size);
      obj->backtrace = TVMFFIByteArray{obj->backtrace_data_.data(), obj->backtrace_data_.length()};
    }
  }

  std::string kind_data_;
  std::string message_data_;
  std::string backtrace_data_;
};
}  // namespace details

/*!
 * \brief Managed reference to ErrorObj
 * \sa Error Object
 */
class Error : public ObjectRef, public std::exception {
 public:
  /*!
   * \brief Constructor
   * \param kind The kind of the error.
   * \param message The message of the error.
   * \param backtrace The backtrace of the error.
   */
  Error(std::string kind, std::string message, std::string backtrace) {
    data_ = make_object<details::ErrorObjFromStd>(std::move(kind), std::move(message),
                                                  std::move(backtrace));
  }

  /*!
   * \brief Constructor
   * \param kind The kind of the error.
   * \param message The message of the error.
   * \param backtrace The backtrace of the error.
   */
  Error(std::string kind, std::string message, const TVMFFIByteArray* backtrace)
      : Error(std::move(kind), std::move(message), std::string(backtrace->data, backtrace->size)) {}

  /*!
   * \brief Get the kind of the error object.
   * \return The kind of the error object.
   */
  std::string kind() const {
    ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
    return std::string(obj->kind.data, obj->kind.size);
  }

  /*!
   * \brief Get the message of the error object.
   * \return The message of the error object.
   */
  std::string message() const {
    ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
    return std::string(obj->message.data, obj->message.size);
  }

  /*!
   * \brief Get the backtrace of the error object.
   * \return The backtrace of the error object.
   * \note Consider use TracebackMostRecentCallLast for pythonic style traceback.
   *
   * \sa TracebackMostRecentCallLast
   */
  std::string backtrace() const {
    ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
    return std::string(obj->backtrace.data, obj->backtrace.size);
  }

  /*!
   * \brief Get the traceback in the order of most recent call last.
   *
   * \return The traceback of the error object.
   */
  std::string TracebackMostRecentCallLast() const {
    // add placeholder for the first line
    std::vector<int64_t> line_breakers = {-1};
    ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
    for (size_t i = 0; i < obj->backtrace.size; i++) {
      if (obj->backtrace.data[i] == '\n') {
        line_breakers.push_back(static_cast<int64_t>(i));
      }
    }
    std::string result;
    result.reserve(obj->backtrace.size);
    for (size_t i = line_breakers.size() - 1; i > 0; --i) {
      int64_t line_start = line_breakers[i - 1] + 1;
      int64_t line_end = line_breakers[i];
      if (line_start == line_end) continue;
      result.append(obj->backtrace.data + line_start, line_end - line_start);
      result.append("\n");
    }
    return result;
  }

  /*!
   * \brief Update the backtrace of the error object.
   * \param backtrace_str The backtrace to update.
   * \param update_mode The mode to update the backtrace,
   *        can be either kTVMFFIBacktraceUpdateModeReplace, kTVMFFIBacktraceUpdateModeAppend.
   */
  void UpdateBacktrace(const TVMFFIByteArray* backtrace_str, int32_t update_mode) {
    ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
    obj->update_backtrace(obj, backtrace_str, update_mode);
  }

  /*!
   * \brief Get the full message of the error, including kind, message and traceback.
   * \return The full message of the error object.
   */
  std::string FullMessage() const {
    ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
    return (std::string("Traceback (most recent call last):\n") + TracebackMostRecentCallLast() +
            std::string(obj->kind.data, obj->kind.size) + std::string(": ") +
            std::string(obj->message.data, obj->message.size) + '\n');
  }

  /*!
   * \brief Get the error message
   * \return The error message
   * \note To get the full message including kind and traceback, use FullMessage() instead.
   */
  const char* what() const noexcept(true) override {
    ErrorObj* obj = static_cast<ErrorObj*>(data_.get());
    return obj->message.data;
  }

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Error, ObjectRef, ErrorObj);
  /// \endcond
};

namespace details {

class ErrorBuilder {
 public:
  explicit ErrorBuilder(std::string kind, std::string backtrace, bool log_before_throw)
      : kind_(std::move(kind)),
        backtrace_(std::move(backtrace)),
        log_before_throw_(log_before_throw) {}

  explicit ErrorBuilder(std::string kind, const TVMFFIByteArray* backtrace, bool log_before_throw)
      : ErrorBuilder(std::move(kind), std::string(backtrace->data, backtrace->size),
                     log_before_throw) {}

// MSVC disable warning in error builder as it is exepected
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4722)
#endif
  // avoid inline to reduce binary size, error throw path do not need to be fast
  [[noreturn]] ~ErrorBuilder() noexcept(false) {
    ::tvm::ffi::Error error(std::move(kind_), stream_.str(), std::move(backtrace_));
    if (log_before_throw_) {
      std::cerr << error.FullMessage();
    }
    throw error;
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  std::ostringstream& stream() { return stream_; }

 protected:
  std::string kind_;
  std::ostringstream stream_;
  std::string backtrace_;
  bool log_before_throw_;
};

}  // namespace details

/*!
 * \brief Helper macro to throw an error with backtrace and message
 *
 * \code
 *
 *   void ThrowError() {
 *     TVM_FFI_THROW(RuntimeError) << "error message";
 *   }
 *
 * \endcode
 */
#define TVM_FFI_THROW(ErrorKind)                                                              \
  ::tvm::ffi::details::ErrorBuilder(#ErrorKind,                                               \
                                    TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0), \
                                    TVM_FFI_ALWAYS_LOG_BEFORE_THROW)                          \
      .stream()

/*!
 * \brief Explicitly log error in stderr and then throw the error.
 *
 * \note This is only necessary on startup functions where we know error
 *  cannot be caught, and it is better to have a clear log message.
 *  In most cases, we should use use TVM_FFI_THROW.
 */
#define TVM_FFI_LOG_AND_THROW(ErrorKind)                                          \
  ::tvm::ffi::details::ErrorBuilder(                                              \
      #ErrorKind, TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0), true) \
      .stream()

// Glog style checks with TVM_FFI prefix
// NOTE: we explicitly avoid glog style generic macros (LOG/CHECK) in tvm ffi
// to avoid potential conflict of downstream users who might have their own GLOG style macros
namespace details {

template <typename X, typename Y>
TVM_FFI_INLINE std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
  std::ostringstream os;
  os << " (" << x << " vs. " << y << ") ";  // CHECK_XX(x, y) requires x and y can be serialized to
                                            // string. Use CHECK(x OP y) otherwise.
  return std::make_unique<std::string>(os.str());
}

#define TVM_FFI_CHECK_FUNC(name, op)                                                   \
  template <typename X, typename Y>                                                    \
  TVM_FFI_INLINE std::unique_ptr<std::string> LogCheck##name(const X& x, const Y& y) { \
    if (x op y) return nullptr;                                                        \
    return LogCheckFormat(x, y);                                                       \
  }                                                                                    \
  TVM_FFI_INLINE std::unique_ptr<std::string> LogCheck##name(int x, int y) {           \
    return LogCheck##name<int, int>(x, y);                                             \
  }

// Inline _Pragma in macros does not work reliably on old version of MSVC and
// GCC. We wrap all comparisons in a function so that we can use #pragma to
// silence bad comparison warnings.
#if defined(__GNUC__) || defined(__clang__)  // GCC and Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#elif defined(_MSC_VER)  // MSVC
#pragma warning(push)
#pragma warning(disable : 4389)  // '==' : signed/unsigned mismatch
#endif

TVM_FFI_CHECK_FUNC(_LT, <)
TVM_FFI_CHECK_FUNC(_GT, >)
TVM_FFI_CHECK_FUNC(_LE, <=)
TVM_FFI_CHECK_FUNC(_GE, >=)
TVM_FFI_CHECK_FUNC(_EQ, ==)
TVM_FFI_CHECK_FUNC(_NE, !=)

#if defined(__GNUC__) || defined(__clang__)  // GCC and Clang
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)  // MSVC
#pragma warning(pop)
#endif
}  // namespace details

#define TVM_FFI_ICHECK_BINARY_OP(name, op, x, y)                                              \
  if (auto __tvm_ffi_log_err = /* NOLINT(bugprone-reserved-identifier) */                     \
      ::tvm::ffi::details::LogCheck##name(x, y))                                              \
  TVM_FFI_THROW(InternalError) << "Check failed: " << #x " " #op " " #y << *__tvm_ffi_log_err \
                               << ": "

#define TVM_FFI_ICHECK(x) \
  if (!(x)) TVM_FFI_THROW(InternalError) << "Check failed: (" #x << ") is false: "

#define TVM_FFI_CHECK(cond, ErrorKind) \
  if (!(cond)) TVM_FFI_THROW(ErrorKind) << "Check failed: (" #cond << ") is false: "

#define TVM_FFI_ICHECK_LT(x, y) TVM_FFI_ICHECK_BINARY_OP(_LT, <, x, y)
#define TVM_FFI_ICHECK_GT(x, y) TVM_FFI_ICHECK_BINARY_OP(_GT, >, x, y)
#define TVM_FFI_ICHECK_LE(x, y) TVM_FFI_ICHECK_BINARY_OP(_LE, <=, x, y)
#define TVM_FFI_ICHECK_GE(x, y) TVM_FFI_ICHECK_BINARY_OP(_GE, >=, x, y)
#define TVM_FFI_ICHECK_EQ(x, y) TVM_FFI_ICHECK_BINARY_OP(_EQ, ==, x, y)
#define TVM_FFI_ICHECK_NE(x, y) TVM_FFI_ICHECK_BINARY_OP(_NE, !=, x, y)
#define TVM_FFI_ICHECK_NOTNULL(x)                                                 \
  ((x) == nullptr ? TVM_FFI_THROW(InternalError) << "Check not null: " #x << ' ', \
   (x)            : (x))  // NOLINT(*)
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_ERROR_H_
